from __future__ import annotations

import torch
import torch.nn.functional as F

from .interaction import QFunction
from .layer import InputLayer, Layer, LinearLayer


def _should_apply_scale(scale) -> bool:
    if torch.is_tensor(scale):
        return True
    return abs(scale - 1.0) > 1.0e-12


class ResistiveInputLayer(InputLayer):
    def __init__(self, shape, gain, batch_size=1, device=None):
        super().__init__(shape, batch_size=batch_size, device=device)
        self._gain = gain

    def set_input(self, input_values, mode="train"):
        if mode == "train":
            self._state = self._gain * torch.cat((input_values, -input_values), 1)
        elif mode == "debug":
            self._state = self._gain * torch.cat((input_values, torch.zeros_like(input_values)), 1)
        else:
            raise ValueError(f"Unsupported input mode '{mode}'.")


class NonlinearResistiveLayer(Layer):
    def __init__(self, shape, batch_size=1, device=None, non_linearity=None):
        super().__init__(shape, batch_size=batch_size, device=device)
        self.non_linearity = non_linearity

    def activate(self):
        if self.non_linearity == "perfect_diode":
            activation_mode = "hard_clip"
        elif self.non_linearity in {
            "hard_sigmoid",
            "linear",
            "lpw_diode",
            "double_diode_quadratic",
            "double_diode_exponential",
            "single_diode_exponential",
            "experimental",
        }:
            activation_mode = "no_clip"
        else:
            raise ValueError(f"Unknown non-linearity: {self.non_linearity}")

        dimension = self._shape[0] // 2
        if activation_mode == "hard_clip":
            excitatory = self._state[:, :dimension].clamp(min=0.0, max=None)
            inhibitory = self._state[:, dimension:].clamp(min=None, max=0.0)
        else:
            excitatory = self._state[:, :dimension]
            inhibitory = self._state[:, dimension:]
        return torch.cat((excitatory, inhibitory), 1)


class ConvLayer(NonlinearResistiveLayer):
    pass


class DenseResistive(QFunction):
    def __init__(
        self,
        layer_pre,
        layer_post,
        dense_weight,
        voltage_amp,
        current_amp,
        amplify_first_free_layer=True,
    ):
        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = dense_weight
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp
        self._amplify_first_free_layer = amplify_first_free_layer
        super().__init__([layer_pre, layer_post], [dense_weight])

    def _layer_index(self, layer):
        return int(layer._name.rsplit("_", 1)[-1])

    def _pre_amp_exponent(self):
        layer_pre_index = self._layer_index(self._layer_pre)
        if self._amplify_first_free_layer:
            return layer_pre_index
        return max(layer_pre_index - 1, 0)

    def _pre_voltage_scale(self):
        if self._pre_amp_exponent() <= 0:
            return 1.0
        return self._voltage_amp

    def eval(self):
        layer_pre = self._layer_pre.state.clone()
        pre_voltage_scale = self._pre_voltage_scale()
        if _should_apply_scale(pre_voltage_scale):
            layer_pre = layer_pre * pre_voltage_scale
        layer_post = self._layer_post.state
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.shape)
        for _ in range(dims_post):
            layer_pre = layer_pre.unsqueeze(-1)
        for _ in range(dims_pre):
            layer_post = layer_post.unsqueeze(1)
        weight = self._weight.get().unsqueeze(0)
        scale = (self._current_amp / self._voltage_amp) ** self._pre_amp_exponent()
        return 0.5 * ((layer_pre - self._current_amp * layer_post) ** 2).mul(weight).flatten(start_dim=1).sum(dim=1) * scale

    def a_coef_fn(self, layer):
        if layer is self._layer_pre:
            return self._a_coef_layer_pre
        if layer is self._layer_post:
            return self._a_coef_layer_post
        raise ValueError("Requested coefficient for an unmanaged layer.")

    def b_coef_fn(self, layer):
        if layer is self._layer_pre:
            return self._b_coef_layer_pre
        if layer is self._layer_post:
            return self._b_coef_layer_post
        raise ValueError("Requested coefficient for an unmanaged layer.")

    def grad_param_fn(self, param):
        if param is not self._weight:
            raise ValueError("Requested gradient for an unmanaged parameter.")
        return self._grad_weight

    def _b_coef_layer_pre(self):
        layer_post = self._layer_post.state
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.shape)
        weight = self._weight.get()
        dim_weight = len(weight.shape)
        permutation = tuple(range(dims_pre, dim_weight)) + tuple(range(dims_pre))
        return -torch.tensordot(layer_post, weight.permute(permutation), dims=dims_post) * self._current_amp

    def _a_coef_layer_pre(self):
        dims_pre = len(self._layer_pre.shape)
        a_coef = 0.5 * self._weight.get().flatten(start_dim=dims_pre).sum(dim=-1).unsqueeze(0)
        return a_coef * self._pre_voltage_scale() * self._current_amp

    def _b_coef_layer_post(self):
        layer_pre = self._layer_pre.state
        dims_pre = len(self._layer_pre.shape)
        b_coef = -torch.tensordot(layer_pre, self._weight.get(), dims=dims_pre)
        pre_voltage_scale = self._pre_voltage_scale()
        if _should_apply_scale(pre_voltage_scale):
            b_coef = b_coef * pre_voltage_scale
        return b_coef

    def _a_coef_layer_post(self):
        dims = len(self._layer_pre.shape) - 1
        return 0.5 * self._weight.get().flatten(end_dim=dims).sum(dim=0).unsqueeze(0)

    def _grad_weight(self):
        layer_pre = self._layer_pre.state.clone()
        pre_voltage_scale = self._pre_voltage_scale()
        if _should_apply_scale(pre_voltage_scale):
            layer_pre *= pre_voltage_scale
        layer_post = self._layer_post.state
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.shape)
        for _ in range(dims_post):
            layer_pre = layer_pre.unsqueeze(-1)
        for _ in range(dims_pre):
            layer_post = layer_post.unsqueeze(1)
        scale = (self._current_amp / self._voltage_amp) ** self._pre_amp_exponent()
        return 0.5 * ((layer_pre - self._current_amp * layer_post) ** 2).mean(dim=0) * scale


class ConvResistive(QFunction):
    def __init__(
        self,
        layer_pre,
        layer_post,
        conv_weight,
        padding,
        stride,
        dilation,
        voltage_amp,
        current_amp,
        amplify_first_free_layer=True,
    ):
        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = conv_weight
        self._P = padding
        self._S = stride
        self._D = dilation
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp
        self._amplify_first_free_layer = amplify_first_free_layer
        super().__init__([layer_pre, layer_post], [conv_weight])

    def _layer_index(self, layer):
        return int(layer._name.rsplit("_", 1)[-1])

    def _pre_amp_exponent(self):
        layer_pre_index = self._layer_index(self._layer_pre)
        if self._amplify_first_free_layer:
            return layer_pre_index
        return max(layer_pre_index - 1, 0)

    def _pre_voltage_scale(self):
        if self._pre_amp_exponent() <= 0:
            return 1.0
        return self._voltage_amp

    def _conv_geometry(self):
        weight = self._weight.get()
        c_out, c_in, kh, kw = weight.shape
        x = self._layer_pre.state
        _, _, h_in, w_in = x.shape
        h_out = (h_in + 2 * self._P - self._D * (kh - 1) - 1) // self._S + 1
        w_out = (w_in + 2 * self._P - self._D * (kw - 1) - 1) // self._S + 1
        return weight, c_out, c_in, kh, kw, h_in, w_in, h_out, w_out

    def eval(self, per_sample=True):
        weight, c_out, c_in, kh, kw, *_ = self._conv_geometry()
        layer_pre = self._layer_pre.state.clone()
        pre_voltage_scale = self._pre_voltage_scale()
        if _should_apply_scale(pre_voltage_scale):
            layer_pre = layer_pre * pre_voltage_scale
        layer_post = self._layer_post.state
        layer_post_scaled = layer_post * self._current_amp

        cols = F.unfold(layer_pre, (kh, kw), padding=self._P, stride=self._S, dilation=self._D)
        n, c_out, h_out, w_out = layer_post_scaled.shape
        k = c_in * kh * kw
        patches = cols.transpose(1, 2).unsqueeze(2)
        kernels = weight.view(1, 1, c_out, k)
        targets = layer_post_scaled.view(n, c_out, h_out * w_out).transpose(1, 2).unsqueeze(-1)

        diff2 = (patches - targets).pow(2)
        weighted = diff2 * kernels
        e_per = 0.5 * weighted.sum(dim=(1, 2, 3))
        return e_per if per_sample else e_per.sum()

    def im2col(self):
        x = self._layer_pre.state
        n = x.shape[0]
        weight, c_out, _, kh, kw, _, _, h_out, w_out = self._conv_geometry()
        cols = F.unfold(x, (kh, kw), padding=self._P, stride=self._S, dilation=self._D)
        w_flat = weight.view(c_out, -1)
        y = torch.matmul(cols.transpose(1, 2), w_flat.t())
        y = y.transpose(1, 2).reshape(n, c_out, h_out, w_out)
        return y

    def col2im(self):
        weight, c_out, c_in, kh, kw, h_in, w_in, h_out, w_out = self._conv_geometry()
        y = self._layer_post.state
        n = y.shape[0]
        l = h_out * w_out
        y_cols = y.reshape(n, c_out, l)
        w_flat = weight.view(c_out, -1)
        cols_pre = torch.matmul(w_flat.t().unsqueeze(0), y_cols)
        return F.fold(
            cols_pre,
            output_size=(h_in, w_in),
            kernel_size=(kh, kw),
            padding=self._P,
            stride=self._S,
            dilation=self._D,
        )

    def a_im2col(self):
        weight, c_out, _, _, _, _, _, h_out, w_out = self._conv_geometry()
        a_per_ch = weight.view(c_out, -1).sum(dim=1).view(1, c_out, 1, 1)
        return a_per_ch.expand(1, c_out, h_out, w_out)

    def a_col2im(self):
        weight, c_out, _, kh, kw, h_in, w_in, h_out, w_out = self._conv_geometry()
        y = self._layer_post.state
        y_ones = torch.ones_like(y)
        n = y_ones.shape[0]
        l = h_out * w_out
        y_cols = y_ones.reshape(n, c_out, l)
        w_flat = weight.view(c_out, -1)
        cols_pre = torch.matmul(w_flat.t().unsqueeze(0), y_cols)
        return F.fold(
            cols_pre,
            output_size=(h_in, w_in),
            kernel_size=(kh, kw),
            padding=self._P,
            stride=self._S,
            dilation=self._D,
        )

    def a_coef_fn(self, layer):
        if layer is self._layer_pre:
            return self._a_coef_layer_pre
        if layer is self._layer_post:
            return self._a_coef_layer_post
        raise ValueError("Requested coefficient for an unmanaged layer.")

    def b_coef_fn(self, layer):
        if layer is self._layer_pre:
            return self._b_coef_layer_pre
        if layer is self._layer_post:
            return self._b_coef_layer_post
        raise ValueError("Requested coefficient for an unmanaged layer.")

    def _b_coef_layer_pre(self):
        return -self.col2im() * self._current_amp

    def _b_coef_layer_post(self):
        b_coef = -self.im2col()
        pre_voltage_scale = self._pre_voltage_scale()
        if _should_apply_scale(pre_voltage_scale):
            b_coef = b_coef * pre_voltage_scale
        return b_coef

    def _a_coef_layer_pre(self):
        a_map = self.a_col2im() * self._pre_voltage_scale() * self._current_amp
        return 0.5 * a_map

    def _a_coef_layer_post(self):
        return 0.5 * self.a_im2col()

    def _grad_weight(self):
        weight, c_out, c_in, kh, kw, *_ = self._conv_geometry()
        x = self._layer_pre.state.clone()
        pre_voltage_scale = self._pre_voltage_scale()
        if _should_apply_scale(pre_voltage_scale):
            x = x * pre_voltage_scale
        y = self._layer_post.state.clone()
        y_rescaled = y * self._current_amp

        cols = F.unfold(x, (kh, kw), padding=self._P, stride=self._S, dilation=self._D)
        n, c_out, h_out, w_out = y.shape
        patches = cols.transpose(1, 2)
        targets = y_rescaled.reshape(n, c_out, h_out * w_out).transpose(1, 2)

        # Avoid the broadcasted [N, L, C_out, C_in * kh * kw] tensor from
        # (patches - targets)^2, which is prohibitive for wide conv blocks.
        # For grad[o, p] = 0.5 * mean_n sum_l (patch[n,l,p] - target[n,l,o])^2,
        # expand the square analytically:
        #   patch^2 - 2 * patch * target + target^2
        # and compute the cross term with an einsum / GEMM-sized contraction.
        patch_sq = patches.pow(2).sum(dim=1).mean(dim=0)
        target_sq = targets.pow(2).sum(dim=1).mean(dim=0)
        cross = torch.einsum("nlp,nlo->op", patches, targets) / n
        grad_weight = 0.5 * (
            patch_sq.unsqueeze(0) - 2.0 * cross + target_sq.unsqueeze(1)
        )

        scale = (self._current_amp / self._voltage_amp) ** self._pre_amp_exponent()
        return grad_weight.view(c_out, c_in, kh, kw) * scale

    def grad_param_fn(self, param):
        if param is not self._weight:
            raise ValueError("Requested gradient for an unmanaged parameter.")
        return self._grad_weight
