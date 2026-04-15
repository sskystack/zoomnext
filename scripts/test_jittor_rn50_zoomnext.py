#!/usr/bin/env python3
"""Evaluate the Jittor RN50_ZoomNeXt checkpoint on image test datasets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from mmengine import Config
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jittor_impl.models import RN50_ZoomNeXt_JT
from utils import io, ops, py_utils, recorder


def append_text_log(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def extract_model_state(payload):
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def prepare_eval_dirs(args: argparse.Namespace, cfg) -> dict[str, Path]:
    exp_name = py_utils.construct_exp_name(model_name="RN50_ZoomNeXt_JT_EVAL", cfg=cfg)
    path_cfg = py_utils.construct_path(output_dir=args.output_dir, exp_name=exp_name)
    py_utils.pre_mkdir(path_cfg)

    run_dir = Path(path_cfg["pth_log"])
    save_dir = Path(path_cfg["save"])
    config_copy_path = Path(path_cfg["cfg_copy"])
    runner_copy_path = Path(path_cfg["trainer_copy"])
    log_path = Path(path_cfg["log"])
    result_path = run_dir / "eval_results.json"
    run_meta_path = run_dir / "run_meta.json"

    shutil.copy2(args.config, config_copy_path)
    shutil.copy2(Path(__file__).resolve(), runner_copy_path)

    run_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(Path(args.config).resolve()),
        "data_cfg": str(Path(args.data_cfg).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "load_from": str(Path(args.load_from).resolve()),
        "test_datasets": list(cfg.test.data.names),
        "effective_batch_size": int(cfg.test.batch_size),
        "save_results": bool(args.save_results),
        "metric_names": list(args.metric_names),
        "clip_range": cfg.test.clip_range,
        "use_cuda": args.use_cuda,
    }
    run_meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    append_text_log(log_path, f"=== Jittor evaluation run {run_meta['created_at']} ===")
    append_text_log(log_path, json.dumps(run_meta, ensure_ascii=False))

    return {
        "run_dir": run_dir,
        "save_dir": save_dir,
        "config_copy": config_copy_path,
        "runner_copy": runner_copy_path,
        "log": log_path,
        "result": result_path,
        "run_meta": run_meta_path,
    }


class ImageTestDatasetJT:
    def __init__(self, dataset_info: dict, shape: dict):
        self.shape = shape

        image_path = os.path.join(dataset_info["root"], dataset_info["image"]["path"])
        image_suffix = dataset_info["image"]["suffix"]
        mask_path = os.path.join(dataset_info["root"], dataset_info["mask"]["path"])
        mask_suffix = dataset_info["mask"]["suffix"]

        image_names = [p[: -len(image_suffix)] for p in sorted(os.listdir(image_path)) if p.endswith(image_suffix)]
        mask_names = [p[: -len(mask_suffix)] for p in sorted(os.listdir(mask_path)) if p.endswith(mask_suffix)]
        valid_names = sorted(set(image_names).intersection(mask_names))
        self.total_data_paths = [
            (os.path.join(image_path, n) + image_suffix, os.path.join(mask_path, n) + mask_suffix) for n in valid_names
        ]

    def __len__(self):
        return len(self.total_data_paths)

    def __getitem__(self, index):
        image_path, mask_path = self.total_data_paths[index]
        image = io.read_color_array(image_path)

        base_h = self.shape["h"]
        base_w = self.shape["w"]
        images = ops.ms_resize(image, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)

        image_s = np.transpose(images[0].astype(np.float32) / 255.0, (2, 0, 1))
        image_m = np.transpose(images[1].astype(np.float32) / 255.0, (2, 0, 1))
        image_l = np.transpose(images[2].astype(np.float32) / 255.0, (2, 0, 1))

        return {
            "data": {"image_s": image_s, "image_m": image_m, "image_l": image_l},
            "info": {"mask_path": mask_path, "group_name": "image"},
        }


def batch_indices(num_samples: int, batch_size: int):
    for start in range(0, num_samples, batch_size):
        yield list(range(start, min(start + batch_size, num_samples)))


def collate_samples(samples: list[dict]) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    batch_data = {
        key: np.stack([sample["data"][key] for sample in samples], axis=0).astype(np.float32)
        for key in samples[0]["data"]
    }
    batch_info = {
        "mask_path": [sample["info"]["mask_path"] for sample in samples],
        "group_name": [sample["info"]["group_name"] for sample in samples],
    }
    return batch_data, batch_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Jittor RN50_ZoomNeXt evaluation script")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--data-cfg", required=True, type=str)
    parser.add_argument("--load-from", required=True, type=str)
    parser.add_argument("--output-dir", type=str, default="outputs_jittor_eval")
    parser.add_argument("--test-datasets", nargs="+", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--metric-names",
        nargs="+",
        type=str,
        default=["sm", "wfm", "mae", "em", "fmeasure"],
        choices=recorder.GroupedMetricRecorder.supported_metrics,
    )
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--encoder-weight-path", type=str, default="pretrained_weights/resnet50-timm.pth")
    parser.add_argument("--use-cuda", action="store_true")
    return parser.parse_args()


def load_cfg(args: argparse.Namespace):
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(
        {
            "data_cfg": args.data_cfg,
            "output_dir": args.output_dir,
            "pretrained": args.pretrained,
            "save_results": args.save_results,
            "metric_names": list(args.metric_names),
        }
    )
    with open(args.data_cfg, mode="r", encoding="utf-8") as f:
        cfg.dataset_infos = yaml.safe_load(f)
    if args.test_datasets:
        cfg.test.data.names = list(args.test_datasets)
    if args.batch_size is not None:
        cfg.test.batch_size = int(args.batch_size)
    return cfg


def evaluate_dataset(jt, model, dataset, batch_size: int, metric_names: list[str], clip_range, save_path: str = ""):
    model.eval()
    all_metrics = recorder.GroupedMetricRecorder(metric_names=metric_names)

    progress = tqdm(total=len(dataset), ncols=100, desc="[JT-EVAL]", file=sys.stdout)
    for ids in batch_indices(len(dataset), batch_size):
        samples = [dataset[idx] for idx in ids]
        np_batch, batch_info = collate_samples(samples)
        jt_batch = {key: jt.array(value) for key, value in np_batch.items()}

        logits = model(data=jt_batch)
        probs = jt.sigmoid(logits).squeeze(1).numpy().astype(np.float32)
        probs = probs - probs.min()
        probs = probs / (probs.max() + 1e-8)

        mask_paths = batch_info["mask_path"]
        group_names = batch_info["group_name"]
        for pred_idx, pred in enumerate(probs):
            mask_path = mask_paths[pred_idx]
            mask_array = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_array[mask_array > 0] = 255
            mask_h, mask_w = mask_array.shape
            pred = ops.resize(pred, height=mask_h, width=mask_w)

            if clip_range is not None:
                pred = ops.clip_to_normalize(pred, clip_range=clip_range)

            group_name = group_names[pred_idx]
            if save_path:
                ops.save_array_as_image(
                    data_array=pred,
                    save_name=os.path.basename(mask_path),
                    save_dir=os.path.join(save_path, group_name),
                )

            pred = (pred * 255).astype(np.uint8)
            all_metrics.step(group_name=group_name, pre=pred, gt=mask_array, gt_path=mask_path)

        progress.update(len(ids))

    progress.close()
    return all_metrics.show()


def main() -> int:
    args = parse_args()
    cfg = load_cfg(args)

    try:
        import jittor as jt
    except ImportError as exc:
        raise SystemExit("Jittor is not installed. Please install Jittor before running evaluation.") from exc

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    run_paths = prepare_eval_dirs(args, cfg)

    model = RN50_ZoomNeXt_JT(
        pretrained=args.pretrained,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        siu_groups=4,
        hmu_groups=6,
        weight_path=args.encoder_weight_path,
    )
    model.load_state_dict(extract_model_state(jt.load(args.load_from)))
    model.eval()

    all_results = {}
    for te_name in cfg.test.data.names:
        dataset_info = cfg.dataset_infos[te_name]
        dataset = ImageTestDatasetJT(dataset_info=dataset_info, shape=cfg.test.data.shape)
        print(json.dumps({"dataset": te_name, "length": len(dataset)}, ensure_ascii=False))

        save_path = ""
        if args.save_results:
            save_path = str(run_paths["save_dir"] / te_name)
            print(json.dumps({"dataset": te_name, "save_dir": save_path}, ensure_ascii=False))

        metrics = evaluate_dataset(
            jt=jt,
            model=model,
            dataset=dataset,
            batch_size=int(cfg.test.batch_size),
            metric_names=list(args.metric_names),
            clip_range=cfg.test.clip_range,
            save_path=save_path,
        )
        all_results[te_name] = metrics
        message = {"dataset": te_name, "metrics": metrics}
        print(json.dumps(message, ensure_ascii=False))
        append_text_log(run_paths["log"], json.dumps(message, ensure_ascii=False))

    run_paths["result"].write_text(json.dumps(all_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {"result_path": str(run_paths["result"]), "datasets": list(cfg.test.data.names)}
    print(json.dumps(summary, ensure_ascii=False))
    append_text_log(run_paths["log"], json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
