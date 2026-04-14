"""Jittor counterparts for the common ZoomNeXt layers.

This file is migrated progressively. Each layer is ported one by one and
validated against the PyTorch reference before the next layer lands.
"""

from __future__ import annotations

import jittor as jt
from jittor import nn

from .ops_jt import ConvBNReLU, resize_to


class SimpleASPP(nn.Module):
    def __init__(self, in_dim, out_dim, dilation=3):
        super().__init__()
        self.out_dim = out_dim
        self.conv1x1_1 = ConvBNReLU(in_dim, 2 * out_dim, 1)
        self.conv1x1_2 = ConvBNReLU(out_dim, out_dim, 1)
        self.conv3x3_1 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_2 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_3 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.fuse = nn.Sequential(
            ConvBNReLU(5 * out_dim, out_dim, 1),
            ConvBNReLU(out_dim, out_dim, 3, 1, 1),
        )

    def execute(self, x: jt.Var) -> jt.Var:
        y = self.conv1x1_1(x)
        y1 = y[:, : self.out_dim, :, :]
        y5 = y[:, self.out_dim :, :, :]

        y2 = self.conv3x3_1(y1)
        y3 = self.conv3x3_2(y2)
        y4 = self.conv3x3_3(y3)

        y0 = y5.mean(dims=(2, 3), keepdims=True)
        y0 = self.conv1x1_2(y0)
        y0 = resize_to(y0, tgt_hw=x.shape[-2:])
        return self.fuse(jt.concat([y0, y1, y2, y3, y4], dim=1))


class DifferenceAwareOps(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def execute(self, x: jt.Var) -> jt.Var:
        raise NotImplementedError("DifferenceAwareOps will be migrated in the next step.")


class RGPU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def execute(self, x: jt.Var) -> jt.Var:
        raise NotImplementedError("RGPU will be migrated in the next step.")


class MHSIU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def execute(self, l: jt.Var, m: jt.Var, s: jt.Var) -> jt.Var:
        del l, m, s
        raise NotImplementedError("MHSIU will be migrated in the next step.")
