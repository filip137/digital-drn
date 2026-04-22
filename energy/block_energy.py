from typing import Mapping, Sequence

import torch

from .augmented import AugmentedFunction
from .block_interactions import FFCurrentInteraction
from ..core.interaction import (
    BiasInteraction,
    DoubleExponentialNonLinearInteraction,
    DoubleQuadraticNonLinearInteraction,
    HardSigmoidNonLinearInteraction,
    LpwNonLinearInteraction,
    SingleExponentialNonLinearInteraction,
    SumSeparableFunction,
)
from ..core.layer import LinearLayer
from ..core.minimizer import QuadraticMinimizer
from ..core.parameter import Bias, ConvWeight, DenseWeight
from ..core.resistive import ConvLayer, ConvResistive, DenseResistive, NonlinearResistiveLayer


def _normalize_weight_gains(weight_gains, count):
    if count <= 0:
        return []
    if isinstance(weight_gains, (int, float)):
        return [float(weight_gains)] * count
    gains = list(weight_gains)
    if len(gains) != count:
        raise ValueError(f"Expected {count} weight gains, got {len(gains)}.")
    return gains


class BaseBlockEnergy(SumSeparableFunction):
    """Shared runtime and helper layer for block-local DRN energies."""

    def __init__(
        self,
        layers,
        params,
        interactions,
        *,
        free_layers,
        output_layer,
        drive,
        non_linearity: str,
        voltage_amp: float,
        current_amp: float,
        quadratic_diode_param: Mapping | None = None,
        exponential_diode_param: Mapping | None = None,
        hard_sigmoid_param: Mapping | None = None,
        device=None,
    ) -> None:
        self._free_layers = list(free_layers)
        self._output_layer = output_layer
        self._drive = drive
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp
        self._non_linearity = non_linearity
        self._quadratic_diode_param = dict(quadratic_diode_param or {})
        self._exponential_diode_param = dict(exponential_diode_param or {})
        self._hard_sigmoid_param = dict(hard_sigmoid_param or {})
        self._interactions = list(interactions)
        super().__init__(layers, params, self._interactions)

        if device is not None:
            self.set_device(device)

    def build_augmented_energy(self, *, nudging_mode: str = "current") -> AugmentedFunction:
        return AugmentedFunction(self, nudging_mode=nudging_mode)

    def build_minimizer(
        self,
        *,
        fn=None,
        num_iterations: int = 6,
        mode: str = "asynchronous",
    ) -> QuadraticMinimizer:
        return QuadraticMinimizer(
            fn=self if fn is None else fn,
            free_layers=self.free_layers(),
            num_iterations=num_iterations,
            mode=mode,
            non_linearity=self._non_linearity,
            quadratic_diode_param=self._quadratic_diode_param,
            exponential_diode_param=self._exponential_diode_param,
            voltage_amp=self._voltage_amp,
            current_amp=self._current_amp,
            hard_sigmoid_param=self._hard_sigmoid_param,
        )

    @staticmethod
    def _build_non_linear_interactions(
        layers,
        non_linearity,
        quadratic_diode_param,
        exponential_diode_param,
        hard_sigmoid_param,
        voltage_amp,
        current_amp,
    ):
        if non_linearity in ("perfect_diode", "linear"):
            return []
        if non_linearity == "lpw_diode":
            return [
                LpwNonLinearInteraction(layer, quadratic_diode_param, voltage_amp=voltage_amp, current_amp=current_amp)
                for layer in layers
            ]
        if non_linearity == "hard_sigmoid":
            return [
                HardSigmoidNonLinearInteraction(
                    layer,
                    hard_sigmoid_param,
                    voltage_amp=voltage_amp,
                    current_amp=current_amp,
                )
                for layer in layers
            ]
        if non_linearity == "double_diode_quadratic":
            return [
                DoubleQuadraticNonLinearInteraction(
                    layer,
                    quadratic_diode_param,
                    voltage_amp=voltage_amp,
                    current_amp=current_amp,
                )
                for layer in layers
            ]
        if non_linearity == "double_diode_exponential":
            return [
                DoubleExponentialNonLinearInteraction(
                    layer,
                    exponential_diode_param,
                    voltage_amp=voltage_amp,
                    current_amp=current_amp,
                )
                for layer in layers
            ]
        if non_linearity == "single_diode_exponential":
            return [
                SingleExponentialNonLinearInteraction(
                    layer,
                    exponential_diode_param,
                    voltage_amp=voltage_amp,
                    current_amp=current_amp,
                )
                for layer in layers
            ]
        raise ValueError(f"Unsupported non_linearity '{non_linearity}'.")

    @property
    def drive(self):
        return self._drive

    def set_device(self, device):
        self._device = torch.device(device)

        for layer in self._layers:
            requires_grad = layer.state.requires_grad
            layer.state = layer.state.detach().to(self._device)
            layer.state.requires_grad_(requires_grad)

        with torch.no_grad():
            for param in self._params:
                requires_grad = param.state.requires_grad
                param.state = param.state.detach().to(self._device)
                param.state.requires_grad_(requires_grad)
                param.clamp_()

        if self._drive.current is not None:
            self._drive.set_current(self._drive.current.to(self._device))

    def free_layers(self):
        return list(self._free_layers)

    def set_drive(self, current):
        expected_shape = (current.size(0),) + self._free_layers[0].shape
        if tuple(current.shape) != expected_shape:
            raise ValueError(f"Expected injected current with shape {expected_shape}, got {tuple(current.shape)}.")
        self._drive.set_current(current)

    def reset_free_layers(self, batch_size: int, device=None):
        target_device = device if device is not None else self._device
        for layer in self._free_layers:
            layer.init_state(batch_size, target_device)

    def output_state(self):
        return self._output_layer.state

    def output_layer(self):
        return self._output_layer

    def detach_state(self):
        for layer in self._free_layers:
            layer.state = layer.state.detach()
        self._drive.set_current(None)

    def zero_param_grad(self, set_to_none: bool = True):
        for param in self._params:
            if param.state.grad is None:
                continue
            if set_to_none:
                param.state.grad = None
            else:
                param.state.grad.zero_()

    def clamp_params_(self):
        with torch.no_grad():
            for param in self._params:
                param.clamp_()


class DenseDRNBlockEnergy(BaseBlockEnergy):
    """Dense-only block-local resistive energy with FF current injection."""

    def __init__(
        self,
        layer_dims: Sequence[int],
        *,
        non_linearity: str = "linear",
        weight_gains=0.1,
        bias_gain: float = 0.0,
        voltage_amp: float = 1.0,
        current_amp: float = 1.0,
        amplify_first_free_layer: bool = True,
        weight_min=None,
        weight_max=None,
        weight_init_mode: str = "kaiming_uniform",
        quadratic_diode_param: Mapping | None = None,
        exponential_diode_param: Mapping | None = None,
        hard_sigmoid_param: Mapping | None = None,
        device=None,
    ) -> None:
        layer_dims = list(layer_dims)
        if len(layer_dims) < 2:
            raise ValueError("layer_dims must contain at least one nonlinear layer and one output layer.")
        if any(dim <= 0 for dim in layer_dims):
            raise ValueError("All layer dimensions must be strictly positive.")
        if any(dim % 2 != 0 for dim in layer_dims[:-1]):
            raise ValueError("All nonlinear resistive layer dimensions must be even.")

        self._layer_dims = tuple(layer_dims)

        nonlinear_layers = [
            NonlinearResistiveLayer((dim,), device=device, non_linearity=non_linearity)
            for dim in layer_dims[:-1]
        ]
        output_layer = LinearLayer((layer_dims[-1],), device=device)
        layers = nonlinear_layers + [output_layer]

        for idx, layer in enumerate(layers, start=1):
            layer._name = f"Layer_{idx}"

        drive = FFCurrentInteraction(nonlinear_layers[0])

        dense_pairs = list(zip(layers[:-1], layers[1:]))
        gains = _normalize_weight_gains(weight_gains, len(dense_pairs))

        self._dense_weights = [
            DenseWeight(
                layer_pre.shape,
                layer_post.shape,
                gain,
                device=device,
                clamp=True,
                clamp_min=weight_min,
                clamp_max=weight_max,
                init_mode=weight_init_mode,
            )
            for (layer_pre, layer_post), gain in zip(dense_pairs, gains)
        ]
        self._biases = [Bias(layer.shape, bias_gain, device=device) for layer in nonlinear_layers]

        interactions = [drive]
        interactions.extend(BiasInteraction(layer, bias) for layer, bias in zip(nonlinear_layers, self._biases))
        interactions.extend(
            DenseResistive(
                layer_pre,
                layer_post,
                weight,
                voltage_amp,
                current_amp,
                amplify_first_free_layer=amplify_first_free_layer,
            )
            for (layer_pre, layer_post), weight in zip(dense_pairs, self._dense_weights)
        )
        interactions.extend(
            self._build_non_linear_interactions(
                nonlinear_layers,
                non_linearity,
                quadratic_diode_param or {},
                exponential_diode_param or {},
                hard_sigmoid_param or {},
                voltage_amp,
                current_amp,
            )
        )

        params = self._dense_weights + self._biases
        super().__init__(
            layers,
            params,
            interactions,
            free_layers=layers,
            output_layer=output_layer,
            drive=drive,
            non_linearity=non_linearity,
            voltage_amp=voltage_amp,
            current_amp=current_amp,
            quadratic_diode_param=quadratic_diode_param,
            exponential_diode_param=exponential_diode_param,
            hard_sigmoid_param=hard_sigmoid_param,
            device=device,
        )

    @property
    def dense_weights(self):
        return self._dense_weights

    @property
    def biases(self):
        return self._biases


class ConvDRNBlockEnergy(BaseBlockEnergy):
    """Fixed-resolution convolutional DRN energy with FF current injection."""

    def __init__(
        self,
        layer_shapes: Sequence[Sequence[int]],
        *,
        kernel_sizes,
        strides=1,
        paddings=0,
        dilations=1,
        non_linearity: str = "linear",
        output_non_linearity: str | None = None,
        weight_gains=0.1,
        bias_gain: float = 0.0,
        voltage_amp: float = 1.0,
        current_amp: float = 1.0,
        amplify_first_free_layer: bool = True,
        weight_min=None,
        weight_max=None,
        weight_init_mode: str = "kaiming_uniform",
        quadratic_diode_param: Mapping | None = None,
        exponential_diode_param: Mapping | None = None,
        hard_sigmoid_param: Mapping | None = None,
        device=None,
    ) -> None:
        layer_shapes = [tuple(shape) for shape in layer_shapes]
        if len(layer_shapes) < 1:
            raise ValueError("layer_shapes must contain at least one conv state.")
        if any(len(shape) != 3 for shape in layer_shapes):
            raise ValueError("Each conv layer shape must be (channels, height, width).")
        if any(shape[0] <= 0 or shape[1] <= 0 or shape[2] <= 0 for shape in layer_shapes):
            raise ValueError("All conv layer dimensions must be strictly positive.")
        if any(shape[0] % 2 != 0 for shape in layer_shapes):
            raise ValueError("All conv layer channel counts must be even.")

        num_pairs = len(layer_shapes) - 1
        gains = _normalize_weight_gains(weight_gains, num_pairs)

        def _norm_spec(value, name, default):
            if isinstance(value, (int, float)):
                return [value] * num_pairs
            values = list(value) if value is not None else [default] * num_pairs
            if len(values) != num_pairs:
                raise ValueError(f"Expected {num_pairs} values for {name}, got {len(values)}.")
            return values

        kernel_sizes = _norm_spec(kernel_sizes, "kernel_sizes", 3)
        strides = _norm_spec(strides, "strides", 1)
        paddings = _norm_spec(paddings, "paddings", 0)
        dilations = _norm_spec(dilations, "dilations", 1)

        self._layer_shapes = tuple(layer_shapes)
        if output_non_linearity is None:
            output_non_linearity = non_linearity
        layer_non_linearities = [non_linearity] * max(len(layer_shapes) - 1, 0) + [output_non_linearity]

        layers = [
            ConvLayer(shape, device=device, non_linearity=layer_non_linearity)
            for shape, layer_non_linearity in zip(layer_shapes, layer_non_linearities)
        ]
        for idx, layer in enumerate(layers, start=1):
            layer._name = f"Layer_{idx}"

        drive = FFCurrentInteraction(layers[0])

        conv_specs = list(zip(layer_shapes[:-1], layer_shapes[1:], kernel_sizes, strides, paddings, dilations, gains))
        self._conv_weights = []
        for pre_shape, post_shape, kernel_size, _, _, _, gain in conv_specs:
            if isinstance(kernel_size, int):
                kh = kw = kernel_size
            else:
                kh, kw = kernel_size
            if pre_shape[1:] != post_shape[1:]:
                raise ValueError("ConvDRNBlockEnergy currently requires fixed spatial resolution across layers.")
            weight_shape = (post_shape[0], pre_shape[0], kh, kw)
            self._conv_weights.append(
                ConvWeight(
                    weight_shape,
                    gain,
                    device=device,
                    clamp=True,
                    clamp_min=weight_min,
                    clamp_max=weight_max,
                    init_mode=weight_init_mode,
                )
            )

        self._biases = [Bias(layer.shape, bias_gain, device=device) for layer in layers]

        interactions = [drive]
        interactions.extend(BiasInteraction(layer, bias) for layer, bias in zip(layers, self._biases))
        interactions.extend(
            ConvResistive(
                layer_pre,
                layer_post,
                weight,
                padding,
                stride,
                dilation,
                voltage_amp,
                current_amp,
                amplify_first_free_layer=amplify_first_free_layer,
            )
            for (layer_pre, layer_post), weight, stride, padding, dilation in zip(
                zip(layers[:-1], layers[1:]),
                self._conv_weights,
                strides,
                paddings,
                dilations,
            )
        )
        for layer, layer_non_linearity in zip(layers, layer_non_linearities):
            interactions.extend(
                self._build_non_linear_interactions(
                    [layer],
                    layer_non_linearity,
                    quadratic_diode_param or {},
                    exponential_diode_param or {},
                    hard_sigmoid_param or {},
                    voltage_amp,
                    current_amp,
                )
            )

        params = self._conv_weights + self._biases
        super().__init__(
            layers,
            params,
            interactions,
            free_layers=layers,
            output_layer=layers[-1],
            drive=drive,
            non_linearity=non_linearity,
            voltage_amp=voltage_amp,
            current_amp=current_amp,
            quadratic_diode_param=quadratic_diode_param,
            exponential_diode_param=exponential_diode_param,
            hard_sigmoid_param=hard_sigmoid_param,
            device=device,
        )

    @property
    def conv_weights(self):
        return self._conv_weights

    @property
    def biases(self):
        return self._biases


class ConvDenseDRNBlockEnergy(BaseBlockEnergy):
    """Mixed analog block with one conv state and one dense readout state."""

    def __init__(
        self,
        conv_state_shape: Sequence[int],
        output_dim: int,
        *,
        non_linearity: str = "linear",
        weight_gains=0.1,
        bias_gain: float = 0.0,
        voltage_amp: float = 1.0,
        current_amp: float = 1.0,
        amplify_first_free_layer: bool = True,
        weight_min=None,
        weight_max=None,
        weight_init_mode: str = "kaiming_uniform",
        quadratic_diode_param: Mapping | None = None,
        exponential_diode_param: Mapping | None = None,
        hard_sigmoid_param: Mapping | None = None,
        device=None,
    ) -> None:
        conv_state_shape = tuple(conv_state_shape)
        if len(conv_state_shape) != 3:
            raise ValueError("conv_state_shape must be (channels, height, width).")
        if any(dim <= 0 for dim in conv_state_shape):
            raise ValueError("All conv state dimensions must be strictly positive.")
        if conv_state_shape[0] % 2 != 0:
            raise ValueError("Conv state channel count must be even.")
        if output_dim <= 0:
            raise ValueError("output_dim must be strictly positive.")

        gains = _normalize_weight_gains(weight_gains, 1)

        self._conv_state_shape = conv_state_shape
        self._output_dim = int(output_dim)

        conv_layer = ConvLayer(conv_state_shape, device=device, non_linearity=non_linearity)
        output_layer = LinearLayer((self._output_dim,), device=device)
        conv_layer._name = "Layer_1"
        output_layer._name = "Layer_2"

        layers = [conv_layer, output_layer]
        drive = FFCurrentInteraction(conv_layer)

        self._dense_weights = [
            DenseWeight(
                conv_layer.shape,
                output_layer.shape,
                gains[0],
                device=device,
                clamp=True,
                clamp_min=weight_min,
                clamp_max=weight_max,
                init_mode=weight_init_mode,
            )
        ]
        self._biases = [
            Bias(conv_layer.shape, bias_gain, device=device),
            Bias(output_layer.shape, bias_gain, device=device),
        ]

        interactions = [drive]
        interactions.extend(BiasInteraction(layer, bias) for layer, bias in zip(layers, self._biases))
        interactions.append(
            DenseResistive(
                conv_layer,
                output_layer,
                self._dense_weights[0],
                voltage_amp,
                current_amp,
                amplify_first_free_layer=amplify_first_free_layer,
            )
        )
        interactions.extend(
            self._build_non_linear_interactions(
                [conv_layer],
                non_linearity,
                quadratic_diode_param or {},
                exponential_diode_param or {},
                hard_sigmoid_param or {},
                voltage_amp,
                current_amp,
            )
        )

        params = self._dense_weights + self._biases
        super().__init__(
            layers,
            params,
            interactions,
            free_layers=layers,
            output_layer=output_layer,
            drive=drive,
            non_linearity=non_linearity,
            voltage_amp=voltage_amp,
            current_amp=current_amp,
            quadratic_diode_param=quadratic_diode_param,
            exponential_diode_param=exponential_diode_param,
            hard_sigmoid_param=hard_sigmoid_param,
            device=device,
        )

    @property
    def dense_weights(self):
        return self._dense_weights

    @property
    def biases(self):
        return self._biases
