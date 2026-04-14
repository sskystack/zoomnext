"""Jittor counterparts for the common ZoomNeXt layers.

This file is migrated progressively. Each layer is ported one by one and
validated against the PyTorch reference before the next layer lands.
"""

from __future__ import annotations

import math

import jittor as jt
from jittor import nn

from .ops_jt import ConvBNReLU, adaptive_avg_pool2d_pt, adaptive_max_pool2d_pt, resize_to


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
    def __init__(self, num_frames):
        super().__init__()
        self.num_frames = num_frames
        self.temperal_proj_norm = nn.LayerNorm(num_frames, elementwise_affine=False)
        self.temperal_proj_kv = nn.Linear(num_frames, 2 * num_frames, bias=False)

        conv1 = nn.Conv2d(num_frames, num_frames, 3, 1, 1, bias=False)
        relu = nn.ReLU()
        conv2 = nn.Conv2d(num_frames, num_frames, 3, 1, 1, bias=False)
        conv2.weight.assign(jt.zeros(conv2.weight.shape))

        self.temperal_proj = nn.Sequential()
        self.temperal_proj.add_module("0", conv1)
        self.temperal_proj.add_module("1", relu)
        self.temperal_proj.add_module("2", conv2)
        self.softmax_y = nn.Softmax(dim=2)

    def execute(self, x: jt.Var) -> jt.Var:
        if self.num_frames == 1:
            return x

        bt, c, h, w = x.shape
        batch_size = bt // self.num_frames

        unshifted_x_tmp = x.reshape((batch_size, self.num_frames, c, h, w))
        unshifted_x_tmp = jt.permute(unshifted_x_tmp, 0, 2, 3, 4, 1)

        shifted_x_tmp = jt.concat([unshifted_x_tmp[..., -1:], unshifted_x_tmp[..., :-1]], dim=-1)
        diff_q = shifted_x_tmp - unshifted_x_tmp
        diff_q = self.temperal_proj_norm(diff_q)

        diff_kv = self.temperal_proj_kv(diff_q)
        diff_k = diff_kv[..., : self.num_frames]
        diff_v = diff_kv[..., self.num_frames :]

        diff_qk = jt.linalg.einsum("bxhwt,byhwt->bxyt", diff_q, diff_k) * (h * w) ** -0.5
        temperal_diff = jt.linalg.einsum("bxyt,byhwt->bxhwt", self.softmax_y(diff_qk), diff_v)

        temperal_diff = jt.permute(temperal_diff, 0, 1, 4, 2, 3)
        temperal_diff = temperal_diff.reshape((batch_size * c, self.num_frames, h, w))
        shifted_x_tmp = self.temperal_proj(temperal_diff)
        shifted_x_tmp = shifted_x_tmp.reshape((batch_size, c, self.num_frames, h, w))
        shifted_x_tmp = jt.permute(shifted_x_tmp, 0, 2, 1, 3, 4)
        shifted_x_tmp = shifted_x_tmp.reshape((bt, c, h, w))
        return x + shifted_x_tmp


class RGPU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def execute(self, x: jt.Var) -> jt.Var:
        raise NotImplementedError("RGPU will be migrated in the next step.")


class MHSIU(nn.Module):
    def __init__(self, in_dim, num_groups=4):
        super().__init__()
        self.in_dim = in_dim
        self.num_groups = num_groups
        self.group_dim = in_dim // num_groups

        self.conv_l_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        self.conv_l = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        self.conv_lms = ConvBNReLU(3 * in_dim, 3 * in_dim, 1)
        self.initial_merge = ConvBNReLU(3 * in_dim, 3 * in_dim, 1)

        self.trans = nn.Sequential(
            ConvBNReLU(3 * self.group_dim, self.group_dim, 1),
            ConvBNReLU(self.group_dim, self.group_dim, 3, 1, 1),
            nn.Conv2d(self.group_dim, 3, 1),
            nn.Softmax(dim=1),
        )

    def execute(self, l: jt.Var, m: jt.Var, s: jt.Var) -> jt.Var:
        tgt_size = (int(m.shape[2]), int(m.shape[3]))

        l = self.conv_l_pre(l)
        l = adaptive_max_pool2d_pt(l, tgt_size) + adaptive_avg_pool2d_pt(l, tgt_size)
        s = self.conv_s_pre(s)
        s = resize_to(s, tgt_hw=tgt_size)

        l = self.conv_l(l)
        m = self.conv_m(m)
        s = self.conv_s(s)
        lms = jt.concat([l, m, s], dim=1)

        attn = self.conv_lms(lms)
        bt, _, h, w = attn.shape
        attn = attn.reshape((bt, 3, self.num_groups, self.group_dim, h, w))
        attn = jt.permute(attn, 0, 2, 1, 3, 4, 5)
        attn = attn.reshape((bt * self.num_groups, 3 * self.group_dim, h, w))
        attn = self.trans(attn)
        attn = attn.unsqueeze(dim=2)

        x = self.initial_merge(lms)
        x = x.reshape((bt, 3, self.num_groups, self.group_dim, h, w))
        x = jt.permute(x, 0, 2, 1, 3, 4, 5)
        x = x.reshape((bt * self.num_groups, 3, self.group_dim, h, w))
        x = (attn * x).sum(dim=1)
        x = x.reshape((bt, self.num_groups, self.group_dim, h, w))
        x = x.reshape((bt, self.num_groups * self.group_dim, h, w))
        return x
