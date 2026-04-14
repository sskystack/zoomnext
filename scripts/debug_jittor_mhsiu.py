#!/usr/bin/env python3
"""Compare PyTorch and Jittor MHSIU intermediates step by step."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.zoomnext.layers import MHSIU as TorchMHSIU
from methods.zoomnext.ops import resize_to as torch_resize_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug Jittor MHSIU intermediates against PyTorch.")
    parser.add_argument("--seed", type=int, default=20260414)
    parser.add_argument("--in-dim", type=int, default=16)
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--h-l", type=int, default=17)
    parser.add_argument("--w-l", type=int, default=19)
    parser.add_argument("--h-m", type=int, default=11)
    parser.add_argument("--w-m", type=int, default=13)
    parser.add_argument("--h-s", type=int, default=7)
    parser.add_argument("--w-s", type=int, default=9)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compare_arrays(name: str, pt_value, jt_value) -> dict:
    pt_arr = np.asarray(pt_value, dtype=np.float32)
    jt_arr = np.asarray(jt_value, dtype=np.float32)
    abs_err = np.abs(pt_arr - jt_arr)
    return {
        "name": name,
        "shape": list(pt_arr.shape),
        "max_abs_err": float(abs_err.max()),
        "mean_abs_err": float(abs_err.mean()),
    }


def torch_state_dict_to_numpy(state_dict: dict) -> dict:
    return {name: value.detach().cpu().numpy() for name, value in state_dict.items()}


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    try:
        import jittor as jt
    except ImportError as exc:
        raise SystemExit(
            "Jittor is not installed. Please install Jittor in the container before running this script."
        ) from exc

    from jittor_impl.models.layers_jt import MHSIU as JittorMHSIU
    from jittor_impl.models.ops_jt import (
        adaptive_avg_pool2d_pt as jittor_adaptive_avg_pool2d_pt,
        adaptive_max_pool2d_pt as jittor_adaptive_max_pool2d_pt,
        resize_to as jittor_resize_to,
    )

    jt.flags.use_cuda = 0

    l_np = np.random.randn(args.batch_size, args.in_dim, args.h_l, args.w_l).astype(np.float32)
    m_np = np.random.randn(args.batch_size, args.in_dim, args.h_m, args.w_m).astype(np.float32)
    s_np = np.random.randn(args.batch_size, args.in_dim, args.h_s, args.w_s).astype(np.float32)

    l_pt, m_pt, s_pt = torch.from_numpy(l_np), torch.from_numpy(m_np), torch.from_numpy(s_np)
    l_jt, m_jt, s_jt = jt.array(l_np), jt.array(m_np), jt.array(s_np)

    pt_model = TorchMHSIU(args.in_dim, args.num_groups)
    pt_model.eval()

    jt_model = JittorMHSIU(args.in_dim, args.num_groups)
    jt_model.eval()
    jt_model.load_state_dict(torch_state_dict_to_numpy(pt_model.state_dict()))

    pt = {}
    with torch.no_grad():
        tgt_size_pt = m_pt.shape[2:]
        pt["l_pre"] = pt_model.conv_l_pre(l_pt)
        pt["l_pool"] = torch.nn.functional.adaptive_max_pool2d(pt["l_pre"], tgt_size_pt) + torch.nn.functional.adaptive_avg_pool2d(pt["l_pre"], tgt_size_pt)
        pt["s_pre"] = pt_model.conv_s_pre(s_pt)
        pt["s_resize"] = torch_resize_to(pt["s_pre"], tgt_hw=tgt_size_pt)
        pt["l_branch"] = pt_model.conv_l(pt["l_pool"])
        pt["m_branch"] = pt_model.conv_m(m_pt)
        pt["s_branch"] = pt_model.conv_s(pt["s_resize"])
        pt["lms"] = torch.cat([pt["l_branch"], pt["m_branch"], pt["s_branch"]], dim=1)

        attn = pt_model.conv_lms(pt["lms"])
        bt, _, h, w = attn.shape
        group_dim = args.in_dim // args.num_groups
        pt["attn_conv"] = attn
        attn_r = attn.reshape(bt, 3, args.num_groups, group_dim, h, w).permute(0, 2, 1, 3, 4, 5).reshape(bt * args.num_groups, 3 * group_dim, h, w)
        pt["attn_reshape"] = attn_r
        attn_t = pt_model.trans(attn_r)
        pt["attn_trans"] = attn_t
        pt["attn_unsqueeze"] = attn_t.unsqueeze(2)

        x = pt_model.initial_merge(pt["lms"])
        pt["initial_merge"] = x
        x_r = x.reshape(bt, 3, args.num_groups, group_dim, h, w).permute(0, 2, 1, 3, 4, 5).reshape(bt * args.num_groups, 3, group_dim, h, w)
        pt["x_reshape"] = x_r
        x_sum = (pt["attn_unsqueeze"] * x_r).sum(dim=1)
        pt["x_sum"] = x_sum
        pt["final"] = x_sum.reshape(bt, args.num_groups * group_dim, h, w)

    jt_out = {}
    tgt_size_jt = (int(m_jt.shape[2]), int(m_jt.shape[3]))
    jt_out["l_pre"] = jt_model.conv_l_pre(l_jt)
    jt_out["l_pool"] = jittor_adaptive_max_pool2d_pt(jt_out["l_pre"], tgt_size_jt) + jittor_adaptive_avg_pool2d_pt(jt_out["l_pre"], tgt_size_jt)
    jt_out["s_pre"] = jt_model.conv_s_pre(s_jt)
    jt_out["s_resize"] = jittor_resize_to(jt_out["s_pre"], tgt_hw=tgt_size_jt)
    jt_out["l_branch"] = jt_model.conv_l(jt_out["l_pool"])
    jt_out["m_branch"] = jt_model.conv_m(m_jt)
    jt_out["s_branch"] = jt_model.conv_s(jt_out["s_resize"])
    jt_out["lms"] = jt.concat([jt_out["l_branch"], jt_out["m_branch"], jt_out["s_branch"]], dim=1)

    attn = jt_model.conv_lms(jt_out["lms"])
    bt, _, h, w = attn.shape
    group_dim = args.in_dim // args.num_groups
    jt_out["attn_conv"] = attn
    attn_r = attn.reshape((bt, 3, args.num_groups, group_dim, h, w))
    attn_r = jt.permute(attn_r, 0, 2, 1, 3, 4, 5)
    attn_r = attn_r.reshape((bt * args.num_groups, 3 * group_dim, h, w))
    jt_out["attn_reshape"] = attn_r
    attn_t = jt_model.trans(attn_r)
    jt_out["attn_trans"] = attn_t
    jt_out["attn_unsqueeze"] = attn_t.unsqueeze(dim=2)

    x = jt_model.initial_merge(jt_out["lms"])
    jt_out["initial_merge"] = x
    x_r = x.reshape((bt, 3, args.num_groups, group_dim, h, w))
    x_r = jt.permute(x_r, 0, 2, 1, 3, 4, 5)
    x_r = x_r.reshape((bt * args.num_groups, 3, group_dim, h, w))
    jt_out["x_reshape"] = x_r
    x_sum = (jt_out["attn_unsqueeze"] * x_r).sum(dim=1)
    jt_out["x_sum"] = x_sum
    jt_out["final"] = x_sum.reshape((bt, args.num_groups * group_dim, h, w))

    names = [
        "l_pre",
        "l_pool",
        "s_pre",
        "s_resize",
        "l_branch",
        "m_branch",
        "s_branch",
        "lms",
        "attn_conv",
        "attn_reshape",
        "attn_trans",
        "attn_unsqueeze",
        "initial_merge",
        "x_reshape",
        "x_sum",
        "final",
    ]
    reports = []
    for name in names:
        reports.append(compare_arrays(name, pt[name].detach().cpu().numpy(), jt_out[name].numpy()))

    print(json.dumps(reports, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
