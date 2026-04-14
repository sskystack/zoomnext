#!/usr/bin/env python3
"""Validate the migrated Jittor RGPU against the PyTorch reference."""

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

from methods.zoomnext.layers import RGPU as TorchRGPU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Jittor RGPU against PyTorch.")
    parser.add_argument("--seed", type=int, default=20260414)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--in-c", type=int, default=16)
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--height", type=int, default=11)
    parser.add_argument("--width", type=int, default=13)
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


def map_torch_state_dict_to_jittor(state_dict: dict) -> dict:
    mapped = {}
    for name, value in state_dict.items():
        mapped_name = name
        mapped_name = mapped_name.replace("gate_genator.1.", "gate_conv1.")
        mapped_name = mapped_name.replace("gate_genator.3.", "gate_conv2.")
        mapped_name = mapped_name.replace("fuse.0.", "fuse_diff.")
        mapped_name = mapped_name.replace("fuse.1.", "fuse_conv.")
        for group_id in range(100):
            prefix = f"interact.{group_id}."
            if mapped_name.startswith(prefix):
                mapped_name = mapped_name.replace(prefix, f"interact_{group_id}.", 1)
                break
        mapped[mapped_name] = value.detach().cpu().numpy()
    return mapped


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    try:
        import jittor as jt
    except ImportError as exc:
        raise SystemExit(
            "Jittor is not installed. Please install Jittor in the container before running this script."
        ) from exc

    from jittor_impl.models.layers_jt import RGPU as JittorRGPU

    jt.flags.use_cuda = 0

    x_np = np.random.randn(
        args.batch_size * args.num_frames,
        args.in_c,
        args.height,
        args.width,
    ).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_jt = jt.array(x_np)

    pt_model = TorchRGPU(
        in_c=args.in_c,
        num_groups=args.num_groups,
        hidden_dim=args.hidden_dim,
        num_frames=args.num_frames,
    )
    pt_model.eval()

    jt_model = JittorRGPU(
        in_c=args.in_c,
        num_groups=args.num_groups,
        hidden_dim=args.hidden_dim,
        num_frames=args.num_frames,
    )
    jt_model.eval()
    jt_model.load_state_dict(map_torch_state_dict_to_jittor(pt_model.state_dict()))

    with torch.no_grad():
        pt_output = pt_model(x_pt)
    jt_output = jt_model(x_jt).numpy()

    report = compare_arrays("RGPU", pt_output, jt_output)
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
