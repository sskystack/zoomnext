"""Jittor counterparts for the common ZoomNeXt operators.

This file intentionally mirrors the structure of
`methods/zoomnext/ops.py` and `utils/ops/tensor_ops.py` so we can
compare small operator blocks before migrating the whole model.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import jittor as jt
from jittor import nn


def _to_2tuple(value) -> Tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Expected a 2-tuple/list, got {value!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def rescale_2x(x: jt.Var, scale_factor=2) -> jt.Var:
    return nn.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def resize_to(x: jt.Var, tgt_hw: tuple) -> jt.Var:
    return nn.interpolate(x, size=tgt_hw, mode="bilinear", align_corners=False)


def global_avgpool(x: jt.Var) -> jt.Var:
    return x.mean(dims=(-1, -2), keepdims=True)


class _LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.1):
        super().__init__()
        self.negative_slope = negative_slope

    def execute(self, x: jt.Var) -> jt.Var:
        return nn.leaky_relu(x, scale=self.negative_slope)


class _GELU(nn.Module):
    def execute(self, x: jt.Var) -> jt.Var:
        return 0.5 * x * (1.0 + jt.erf(x / math.sqrt(2.0)))


def _get_act_fn(act_name: str, inplace: bool = True) -> nn.Module:
    del inplace  # Jittor activations do not use PyTorch-style inplace flags here.
    if act_name == "relu":
        return nn.ReLU()
    if act_name == "leaklyrelu":
        return _LeakyReLU(negative_slope=0.1)
    if act_name == "gelu":
        return _GELU()
    if act_name == "sigmoid":
        return nn.Sigmoid()
    raise NotImplementedError(act_name)


class ConvBN(nn.Module):
    def __init__(self, in_dim, out_dim, k, s=1, p=0, d=1, g=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=k,
            stride=_to_2tuple(s),
            padding=_to_2tuple(p),
            dilation=_to_2tuple(d),
            groups=g,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_dim)

    def execute(self, x: jt.Var) -> jt.Var:
        return self.bn(self.conv(x))


class CBR(nn.Module):
    def __init__(self, in_dim, out_dim, k, s=1, p=0, d=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=k,
            stride=_to_2tuple(s),
            padding=_to_2tuple(p),
            dilation=_to_2tuple(d),
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def execute(self, x: jt.Var) -> jt.Var:
        return self.relu(self.bn(self.conv(x)))


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
        is_transposed=False,
    ):
        super().__init__()
        conv_cls = nn.ConvTranspose if is_transposed else nn.Conv2d
        self.add_module(
            "conv",
            conv_cls(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=_to_2tuple(stride),
                padding=_to_2tuple(padding),
                dilation=_to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(act_name, _get_act_fn(act_name=act_name))


class ConvGNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        gn_groups=8,
        bias=False,
        act_name="relu",
        inplace=True,
    ):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=_to_2tuple(stride),
                padding=_to_2tuple(padding),
                dilation=_to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module("gn", nn.GroupNorm(num_groups=gn_groups, num_channels=out_planes))
        if act_name is not None:
            self.add_module(act_name, _get_act_fn(act_name=act_name, inplace=inplace))


class PixelNormalizer(nn.Module):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        # Leading underscores keep these Vars from being treated as trainable parameters in Jittor.
        self._mean = jt.array(mean).reshape((3, 1, 1)).float32()
        self._std = jt.array(std).reshape((3, 1, 1)).float32()

    @property
    def mean(self) -> jt.Var:
        return self._mean

    @property
    def std(self) -> jt.Var:
        return self._std

    def __repr__(self) -> str:
        mean = [float(x) for x in self.mean.reshape((-1,)).numpy().tolist()]
        std = [float(x) for x in self.std.reshape((-1,)).numpy().tolist()]
        return f"{self.__class__.__name__}(mean={mean}, std={std})"

    def execute(self, x: jt.Var) -> jt.Var:
        x = x - self.mean
        x = x / self.std
        return x


class LayerNorm2d(nn.Module):
    """A channel-wise LayerNorm that mirrors the custom PyTorch implementation."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = jt.ones((num_channels,), dtype=jt.float32)
        self.bias = jt.zeros((num_channels,), dtype=jt.float32)
        self.eps = eps

    def execute(self, x: jt.Var) -> jt.Var:
        u = x.mean(dim=1, keepdims=True)
        s = ((x - u) ** 2).mean(dim=1, keepdims=True)
        x = (x - u) / jt.sqrt(s + self.eps)
        x = self.weight.reshape((1, -1, 1, 1)) * x + self.bias.reshape((1, -1, 1, 1))
        return x
