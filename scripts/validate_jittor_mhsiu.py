#!/usr/bin/env python3
"""Validate the migrated Jittor MHSIU against the PyTorch reference."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Jittor MHSIU against PyTorch.")
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
    parser.add_argument("--tol-max", type=float, default=1e-5)
    parser.add_argument("--tol-mean", type=float, default=1e-6)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compare_arrays(name: str, pt_value: torch.Tensor, jt_value: np.ndarray) -> dict:
    pt_arr = pt_value.detach().cpu().numpy().astype(np.float32)
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

    with torch.no_grad():
        pt_output = pt_model(l_pt, m_pt, s_pt)
    jt_output = jt_model(l_jt, m_jt, s_jt).numpy()

    report = compare_arrays("MHSIU", pt_output, jt_output)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if report["max_abs_err"] > args.tol_max or report["mean_abs_err"] > args.tol_mean:
        print(
            "\nValidation failed: "
            f"max_abs_err={report['max_abs_err']:.6e}, mean_abs_err={report['mean_abs_err']:.6e}"
        )
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
