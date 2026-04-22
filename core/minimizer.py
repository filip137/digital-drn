from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .resistive import NonlinearResistiveLayer


def _load_lambertw():
    if hasattr(torch, "special") and hasattr(torch.special, "lambertw"):
        return torch.special.lambertw
    try:
        import torchlambertw.special as tw_special

        return tw_special.lambertw
    except Exception as exc:
        def _missing_lambertw(*args, **kwargs):
            raise ImportError(
                "LambertW backend unavailable; install torchlambertw or use a PyTorch build "
                "with torch.special.lambertw."
            ) from exc

        return _missing_lambertw


lambertw = _load_lambertw()


class LayerUpdater(ABC):
    def __init__(self, layer, fn):
        self._layer = layer
        self.grad = fn.grad_layer_fn(layer)

    @abstractmethod
    def pre_activate(self):
        raise NotImplementedError


class Minimizer:
    def __init__(self, fn, updaters, num_iterations, mode, voltage_amp, current_amp):
        source_fn = getattr(fn, "_energy_fn", fn)
        self._layers = fn.layers()
        self._params = fn.params()
        self._updaters = list(updaters)
        self._num_iterations = num_iterations
        self._mode = mode
        self._set_mode()
        if voltage_amp is None:
            voltage_amp = getattr(source_fn, "_voltage_amp", getattr(source_fn, "voltage_amp", None))
        if current_amp is None:
            current_amp = getattr(source_fn, "_current_amp", getattr(source_fn, "current_amp", None))
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        self._mode = mode
        self._set_mode()

    @property
    def num_iterations(self):
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, num_iterations):
        self._num_iterations = num_iterations
        self._set_mode()

    @property
    def voltage_amp(self):
        return self._voltage_amp

    @property
    def current_amp(self):
        return self._current_amp

    def compute_equilibrium(self):
        for layer_group in self._list_layers:
            self.step(layer_group)
        return {layer.name: layer.state for layer in self._layers}

    def compute_trajectory(self):
        trajectories = {layer.name: [layer.state] for layer in self._layers}
        for param in self._params:
            trajectories[param.name] = [param.state]
        for layer_group in self._list_layers:
            for param in self._params:
                param.state = param.state + torch.zeros_like(param.state)
                trajectories[param.name].append(param.state)
            self.step(layer_group)
            for layer in self._layers:
                trajectories[layer.name].append(layer.state)
        return trajectories

    def step(self, layer_group):
        pre_activations = [updater.pre_activate() for updater in layer_group]
        for updater, pre_activation in zip(layer_group, pre_activations):
            updater._layer.state = pre_activation
            updater._layer.state = updater._layer.activate()

    def _set_mode(self):
        if self._mode == "forward":
            self._list_layers = [[updater] for updater in self._updaters] * self._num_iterations
        elif self._mode == "backward":
            self._list_layers = [[updater] for updater in reversed(self._updaters)] * self._num_iterations
        elif self._mode == "synchronous":
            self._list_layers = [self._updaters] * self._num_iterations
        elif self._mode == "asynchronous":
            self._list_layers = [self._updaters[::2], self._updaters[1::2]] * self._num_iterations
        else:
            raise ValueError(f"expected 'forward', 'backward', 'synchronous' or 'asynchronous' but got {self._mode}")


class QuadraticUpdater(LayerUpdater):
    def __init__(self, layer, fn):
        super().__init__(layer, fn)
        self._a = fn.a_coef_fn(layer)
        self._b = fn.b_coef_fn(layer)

    def pre_activate(self):
        return -self._b() / (2.0 * self._a())


class AdaptiveQuadraticUpdater(QuadraticUpdater):
    def __init__(self, layer, fn, diode_params):
        super().__init__(layer, fn)
        self._diode_conductance = diode_params["diode_conductance"]
        self._v_off = float(diode_params.get("v_off", 0.0))

    def pre_activate(self):
        b = self._b()
        a = self._a()
        if torch.is_tensor(a):
            a = a.expand_as(b)
        if not isinstance(self._layer, NonlinearResistiveLayer):
            return -b / (2.0 * a)

        preamp_voltage = -b / (2.0 * a)
        dimension = self._layer._shape[0] // 2
        excitatory_pre = preamp_voltage[:, :dimension]
        inhibitory_pre = preamp_voltage[:, dimension:]
        excitatory_violations = excitatory_pre > self._v_off
        inhibitory_violations = inhibitory_pre < -self._v_off

        if excitatory_violations.any() or inhibitory_violations.any():
            a_add = torch.zeros_like(a)
            b_sub = torch.zeros_like(b)
            if excitatory_violations.any():
                a_add[:, :dimension] = torch.where(
                    excitatory_violations,
                    a_add[:, :dimension] + 0.5 * self._diode_conductance,
                    a_add[:, :dimension],
                )
                b_sub[:, :dimension] = torch.where(
                    excitatory_violations,
                    b_sub[:, :dimension] - self._diode_conductance * self._v_off,
                    b_sub[:, :dimension],
                )
            if inhibitory_violations.any():
                a_add[:, dimension:] = torch.where(
                    inhibitory_violations,
                    a_add[:, dimension:] + 0.5 * self._diode_conductance,
                    a_add[:, dimension:],
                )
                b_sub[:, dimension:] = torch.where(
                    inhibitory_violations,
                    b_sub[:, dimension:] + self._diode_conductance * self._v_off,
                    b_sub[:, dimension:],
                )
            return -(b + b_sub) / (2.0 * (a + a_add))

        return preamp_voltage


class QuadraticDoubleDiodeUpdaterOffset(QuadraticUpdater):
    def __init__(self, layer, fn, diode_params):
        super().__init__(layer, fn)
        self._diode_conductance = diode_params["diode_conductance"]
        self._v_off = diode_params.get("v_off", 0.0)

    def pre_activate(self):
        b = self._b()
        a = self._a().expand_as(b)
        if not isinstance(self._layer, NonlinearResistiveLayer):
            return -b / (2.0 * a)

        c = torch.as_tensor(self._diode_conductance, dtype=b.dtype, device=b.device)
        v_off = torch.as_tensor(self._v_off, dtype=b.dtype, device=b.device)
        v_lin = -b / (2.0 * a)
        mask_lin = v_lin.abs() <= v_off

        A = c
        Bp = 2.0 * a - 2.0 * c * v_off
        Cp = b + c * v_off**2
        disc_p = (Bp**2 - 4.0 * A * Cp).clamp(min=0)
        v_pos1 = (-Bp + torch.sqrt(disc_p)) / (2.0 * A)
        v_pos2 = (-Bp - torch.sqrt(disc_p)) / (2.0 * A)
        v_pos = torch.where(v_pos1 >= v_off, v_pos1, v_pos2)

        Bn = 2.0 * c * v_off - 2.0 * a
        Cn = c * v_off**2 - b
        disc_n = (Bn**2 - 4.0 * A * Cn).clamp(min=0)
        v_neg1 = (-Bn + torch.sqrt(disc_n)) / (2.0 * A)
        v_neg2 = (-Bn - torch.sqrt(disc_n)) / (2.0 * A)
        v_neg = torch.where(v_neg1 <= -v_off, v_neg1, v_neg2)

        v = v_lin.clone()
        v = torch.where((b < 0) & ~mask_lin, v_pos, v)
        v = torch.where((b > 0) & ~mask_lin, v_neg, v)
        return v


class ExponentialDoubleDiodeUpdater(QuadraticUpdater):
    def __init__(self, layer, fn, diode_params):
        super().__init__(layer, fn)
        self._Is = diode_params["I_s"]
        self._Vt = diode_params["V_t"]
        self._v_off = diode_params["V_off"]

    @staticmethod
    def _lambertw_large(z, terms=4):
        L1 = torch.log(z)
        L2 = torch.log(L1)
        w = L1 - L2
        if terms >= 2:
            w += L2 / L1
        if terms >= 3:
            w += (L2 * (-2 + L2)) / (2 * L1**2)
        if terms >= 4:
            w += (L2 * (6 - 9 * L2 + 2 * L2**2)) / (6 * L1**3)
        return w

    def _lambert_hidden(self, a, b, I_s, v_off, vt, z_thresh=1e10, polish=True, exp_clip=10000.0):
        out_dtype = a.dtype
        device = a.device
        a64 = a.to(torch.float64)
        b64 = b.to(torch.float64)
        I_s64 = torch.as_tensor(I_s, dtype=torch.float64, device=device)
        vt64 = torch.as_tensor(vt, dtype=torch.float64, device=device)
        v_off64 = torch.as_tensor(v_off, dtype=torch.float64, device=device)

        A = torch.where(b64 <= 0, (b64 - I_s64) / (2.0 * a64), (b64 + I_s64) / (2.0 * a64))
        exp_arg_add = (-(A + v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
        exp_arg_rev = ((A - v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
        z_add = (I_s64 / (2.0 * a64 * vt64)) * torch.exp(exp_arg_add)
        z_rev = (I_s64 / (2.0 * a64 * vt64)) * torch.exp(exp_arg_rev)
        z = torch.where(b64 > 0, z_rev, z_add)

        use_asym = z > z_thresh
        W0 = torch.empty_like(z, dtype=torch.float64, device=device)
        if (~use_asym).any():
            W0[~use_asym] = lambertw(torch.clamp(z[~use_asym], max=float(z_thresh))).real.to(torch.float64)
        if use_asym.any():
            W0[use_asym] = self._lambertw_large(torch.clamp(z[use_asym], min=1.0)).to(torch.float64)

        x = torch.where(b64 > 0, vt64 * W0 - A, -vt64 * W0 - A)

        if polish:
            for _ in range(64):
                mask_pos = b64 <= 0
                mask_neg = ~mask_pos
                f = torch.zeros_like(x)
                df = torch.zeros_like(x)
                if mask_pos.any():
                    xp = x[mask_pos]
                    ap = a64[mask_pos]
                    bp = b64[mask_pos]
                    ep = torch.exp(((xp - v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip))
                    f[mask_pos] = 2 * ap * xp + (bp - I_s64) + I_s64 * ep
                    df[mask_pos] = 2 * ap + (I_s64 / vt64) * ep
                if mask_neg.any():
                    xn = x[mask_neg]
                    an = a64[mask_neg]
                    bn = b64[mask_neg]
                    en = torch.exp(((-xn - v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip))
                    f[mask_neg] = 2 * an * xn + (bn + I_s64) - I_s64 * en
                    df[mask_neg] = 2 * an + (I_s64 / vt64) * en
                x = x - f / df
                if torch.norm(f) < 1e-7 * x.shape[0]:
                    break

        return x.to(out_dtype)

    def pre_activate(self):
        b = self._b()
        a = self._a().expand_as(b)
        if isinstance(self._layer, NonlinearResistiveLayer):
            return self._lambert_hidden(a, b, self._Is, self._v_off, self._Vt)
        return -b / (2.0 * a)


class ExponentialSingleDiodeUpdater(QuadraticUpdater):
    def __init__(self, layer, fn, diode_params):
        super().__init__(layer, fn)
        self._Is = diode_params["I_s"]
        self._Vt = diode_params["V_t"]
        self._v_off = diode_params.get("V_off")

    @staticmethod
    def _lambertw_large(z, terms=4):
        L1 = torch.log(z)
        L2 = torch.log(L1)
        w = L1 - L2
        if terms >= 2:
            w = w + L2 / L1
        if terms >= 3:
            w = w + (L2 * (-2.0 + L2)) / (2.0 * L1**2)
        if terms >= 4:
            w = w + (L2 * (6.0 - 9.0 * L2 + 2.0 * L2**2)) / (6.0 * L1**3)
        return w

    def _lambert_single_forward(
        self,
        a,
        b,
        I_s,
        v_off,
        vT,
        z_thresh=1e10,
        polish=True,
        abs_tol=1e-10,
        rel_tol=1e-10,
        exp_clip=10000.0,
        a_min=1e-30,
        max_inner_iter=32,
    ):
        out_dtype = a.dtype
        device = a.device
        a64 = torch.clamp(a.to(torch.float64), min=a_min)
        b64 = b.to(torch.float64)
        I_s64 = torch.as_tensor(I_s, dtype=torch.float64, device=device)
        vT64 = torch.as_tensor(vT, dtype=torch.float64, device=device)
        voff64 = torch.as_tensor(v_off, dtype=torch.float64, device=device)
        A = (b64 - I_s64) / (2.0 * a64)
        z = (I_s64 / (2.0 * a64 * vT64)) * torch.exp(-(A + voff64) / vT64)
        use_asym = z > z_thresh
        W0 = torch.empty_like(z, dtype=torch.float64, device=device)
        if (~use_asym).any():
            W0[~use_asym] = lambertw(torch.clamp(z[~use_asym], max=float(z_thresh))).real.to(torch.float64)
        if use_asym.any():
            W0[use_asym] = self._lambertw_large(torch.clamp(z[use_asym], min=1.0)).to(torch.float64)
        x = -A - vT64 * W0
        if polish:
            for _ in range(max_inner_iter):
                arg = ((x - voff64) / vT64).clamp(min=-exp_clip, max=exp_clip)
                e = torch.exp(arg)
                f = 2.0 * a64 * x + b64 + I_s64 * (e - 1.0)
                df = 2.0 * a64 + (I_s64 / vT64) * e
                x = x - f / df
                max_abs_f = torch.max(torch.abs(f))
                scale = torch.max(2.0 * torch.abs(a64) * torch.abs(x) + torch.abs(b64) + torch.abs(I_s64) + 1.0)
                if (max_abs_f < abs_tol) and (max_abs_f / (scale + 1e-30) < rel_tol):
                    break
        return x.to(out_dtype)

    def _lambert_single_reverse(
        self,
        a,
        b,
        I_s,
        v_off,
        vT,
        z_thresh=1e10,
        polish=True,
        abs_tol=1e-10,
        rel_tol=1e-10,
        exp_clip=10000.0,
        a_min=1e-30,
        max_inner_iter=32,
    ):
        out_dtype = a.dtype
        device = a.device
        a64 = torch.clamp(a.to(torch.float64), min=a_min)
        b64 = b.to(torch.float64)
        I_s64 = torch.as_tensor(I_s, dtype=torch.float64, device=device)
        vT64 = torch.as_tensor(vT, dtype=torch.float64, device=device)
        voff64 = torch.as_tensor(v_off, dtype=torch.float64, device=device)
        A = (b64 + I_s64) / (2.0 * a64)
        z = (I_s64 / (2.0 * a64 * vT64)) * torch.exp((-voff64 + A) / vT64)
        use_asym = z > z_thresh
        W0 = torch.empty_like(z, dtype=torch.float64, device=device)
        if (~use_asym).any():
            W0[~use_asym] = lambertw(torch.clamp(z[~use_asym], max=float(z_thresh))).real.to(torch.float64)
        if use_asym.any():
            W0[use_asym] = self._lambertw_large(torch.clamp(z[use_asym], min=1.0)).to(torch.float64)
        x = -A + vT64 * W0
        if polish:
            for _ in range(max_inner_iter):
                arg = (-(x + voff64) / vT64).clamp(min=-exp_clip, max=exp_clip)
                e = torch.exp(arg)
                f = 2.0 * a64 * x + b64 + I_s64 - I_s64 * e
                df = 2.0 * a64 + (I_s64 / vT64) * e
                x = x - f / df
                max_abs_f = torch.max(torch.abs(f))
                scale = torch.max(2.0 * torch.abs(a64) * torch.abs(x) + torch.abs(b64) + torch.abs(I_s64) + 1.0)
                if (max_abs_f < abs_tol) and (max_abs_f / (scale + 1e-30) < rel_tol):
                    break
        return x.to(out_dtype)

    def pre_activate(self):
        b = self._b()
        a = self._a().expand_as(b)
        if not isinstance(self._layer, NonlinearResistiveLayer):
            return -b / (2.0 * a)

        features = b.shape[1]
        reverse_mask_nodes = torch.arange(features, device=b.device) >= (features // 2)
        reverse_mask = reverse_mask_nodes.unsqueeze(0).expand_as(b)
        v_fwd = self._lambert_single_forward(a, b, self._Is, self._v_off, self._Vt)
        v_rev = self._lambert_single_reverse(a, b, self._Is, self._v_off, self._Vt)
        return torch.where(reverse_mask, v_rev, v_fwd)


class HardSigmoidUpdater(QuadraticUpdater):
    def __init__(self, layer, fn, diode_params):
        super().__init__(layer, fn)
        self.g_on = diode_params.get("g_on")
        self.g_off = diode_params.get("g_off")
        self._vmin = float(diode_params["v_min"])
        self._vmax = float(diode_params["v_max"])

    def pre_activate(self):
        a = self._a()
        b = self._b()
        if not isinstance(self._layer, NonlinearResistiveLayer):
            return -b / (2.0 * a)

        # The off-region interaction energy is 0.5 * g_off * v^2, so under the
        # 2*a*v + b quadratic convention it contributes +0.5 * g_off to a.
        a_off = a + 0.5 * self.g_off
        v_free = -b / (2.0 * a_off)
        below = v_free < self._vmin
        above = v_free > self._vmax
        if not (below.any() or above.any()):
            return v_free

        a_add = torch.zeros_like(a)
        b_sub = torch.zeros_like(b)
        if below.any():
            a_add = torch.where(below, a_add + 0.5 * self.g_on, a_add)
            b_sub = torch.where(below, b_sub - self.g_on * self._vmin, b_sub)
        if above.any():
            a_add = torch.where(above, a_add + 0.5 * self.g_on, a_add)
            b_sub = torch.where(above, b_sub - self.g_on * self._vmax, b_sub)
        return -(b + b_sub) / (2.0 * (a + a_add))


class QuadraticMinimizer(Minimizer):
    def __init__(
        self,
        fn,
        free_layers,
        num_iterations,
        mode,
        non_linearity,
        quadratic_diode_param,
        exponential_diode_param,
        voltage_amp,
        current_amp,
        hard_sigmoid_param=None,
    ):
        quadratic_params = dict(quadratic_diode_param)
        exponential_params = dict(exponential_diode_param)
        hard_sigmoid_params = dict(hard_sigmoid_param or {})
        if not hard_sigmoid_params and "diode_conductance" in quadratic_params:
            hard_sigmoid_params["g_on"] = quadratic_params["diode_conductance"]
            hard_sigmoid_params["g_off"] = quadratic_params["diode_conductance"]
        if "g_on" not in hard_sigmoid_params and "diode_conductance" in quadratic_params:
            hard_sigmoid_params["g_on"] = quadratic_params["diode_conductance"]
        if "g_off" not in hard_sigmoid_params and "g_on" in hard_sigmoid_params:
            hard_sigmoid_params["g_off"] = hard_sigmoid_params["g_on"]
        if "v_min" not in hard_sigmoid_params and "v_min" in quadratic_params:
            hard_sigmoid_params["v_min"] = quadratic_params["v_min"]
        if "v_max" not in hard_sigmoid_params and "v_max" in quadratic_params:
            hard_sigmoid_params["v_max"] = quadratic_params["v_max"]

        if non_linearity in {"perfect_diode", "linear"}:
            updaters = [QuadraticUpdater(layer, fn) for layer in free_layers]
        elif non_linearity == "lpw_diode":
            updaters = [AdaptiveQuadraticUpdater(layer, fn, quadratic_params) for layer in free_layers]
        elif non_linearity == "double_diode_quadratic":
            updaters = [QuadraticDoubleDiodeUpdaterOffset(layer, fn, quadratic_params) for layer in free_layers]
        elif non_linearity == "double_diode_exponential":
            updaters = [ExponentialDoubleDiodeUpdater(layer, fn, exponential_params) for layer in free_layers]
        elif non_linearity == "single_diode_exponential":
            updaters = [ExponentialSingleDiodeUpdater(layer, fn, exponential_params) for layer in free_layers]
        elif non_linearity == "hard_sigmoid":
            updaters = [HardSigmoidUpdater(layer, fn, hard_sigmoid_params) for layer in free_layers]
        else:
            raise ValueError(
                "non_linearity must be 'perfect_diode', 'double_diode_quadratic', "
                f"'lpw_diode', 'double_diode_exponential', 'hard_sigmoid', or 'linear'; got {non_linearity}"
            )

        super().__init__(fn, updaters, num_iterations, mode, voltage_amp, current_amp)
        for updater in self._updaters:
            updater.voltage_amp = self.voltage_amp
            updater.current_amp = self.current_amp

        self._non_linearity = non_linearity
        self._quadratic_params = quadratic_params
        self._exponential_params = exponential_params
        self._hard_sigmoid_params = hard_sigmoid_params
