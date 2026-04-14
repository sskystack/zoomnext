#!/usr/bin/env python3
"""Train the Jittor RN50_ZoomNeXt model with dataset configs."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import yaml
from mmengine import Config
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jittor_impl.models import RN50_ZoomNeXt_JT
from jittor_impl.models.zoomnext_jt import frozen_bn_stats_jt
from utils import io, ops


class ImageTrainDatasetJT:
    def __init__(self, dataset_infos: dict, shape: dict):
        self.shape = shape
        self.total_data_paths = []
        for dataset_name, dataset_info in dataset_infos.items():
            image_path = os.path.join(dataset_info["root"], dataset_info["image"]["path"])
            image_suffix = dataset_info["image"]["suffix"]
            mask_path = os.path.join(dataset_info["root"], dataset_info["mask"]["path"])
            mask_suffix = dataset_info["mask"]["suffix"]

            image_names = [p[: -len(image_suffix)] for p in sorted(os.listdir(image_path)) if p.endswith(image_suffix)]
            mask_names = [p[: -len(mask_suffix)] for p in sorted(os.listdir(mask_path)) if p.endswith(mask_suffix)]
            valid_names = sorted(set(image_names).intersection(mask_names))
            data_paths = [
                (os.path.join(image_path, n) + image_suffix, os.path.join(mask_path, n) + mask_suffix)
                for n in valid_names
            ]
            print(json.dumps({"dataset": dataset_name, "length": len(data_paths)}, ensure_ascii=False))
            self.total_data_paths.extend(data_paths)

        self.trains = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=90, p=0.5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REPLICATE),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.5),
            ]
        )

    def __len__(self):
        return len(self.total_data_paths)

    def __getitem__(self, index):
        image_path, mask_path = self.total_data_paths[index]
        image = io.read_color_array(image_path)
        mask = io.read_gray_array(mask_path, thr=0)
        if image.shape[:2] != mask.shape:
            h, w = mask.shape
            image = ops.resize(image, height=h, width=w)

        transformed = self.trains(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        base_h = self.shape["h"]
        base_w = self.shape["w"]
        images = ops.ms_resize(image, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)

        image_s = images[0].astype(np.float32) / 255.0
        image_m = images[1].astype(np.float32) / 255.0
        image_l = images[2].astype(np.float32) / 255.0
        image_s = np.transpose(image_s, (2, 0, 1))
        image_m = np.transpose(image_m, (2, 0, 1))
        image_l = np.transpose(image_l, (2, 0, 1))

        mask = ops.resize(mask, height=base_h, width=base_w).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return {
            "image_s": image_s,
            "image_m": image_m,
            "image_l": image_l,
            "mask": mask,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Jittor RN50_ZoomNeXt training script")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--data-cfg", required=True, type=str)
    parser.add_argument("--output-dir", type=str, default="outputs_jittor")
    parser.add_argument("--train-datasets", nargs="+", type=str)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--encoder-weight-path", type=str, default="pretrained_weights/resnet50-timm.pth")
    parser.add_argument("--resume-from", type=str)
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=112358)
    parser.add_argument("--use-cuda", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_cfg(args: argparse.Namespace):
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(
        {
            "data_cfg": args.data_cfg,
            "output_dir": args.output_dir,
            "pretrained": args.pretrained,
        }
    )
    with open(args.data_cfg, mode="r", encoding="utf-8") as f:
        cfg.dataset_infos = yaml.safe_load(f)
    return cfg


def batch_indices(num_samples: int, batch_size: int, *, seed: int, drop_last: bool = True):
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    for start in range(0, num_samples, batch_size):
        batch = indices[start : start + batch_size]
        if len(batch) < batch_size and drop_last:
            continue
        yield batch.tolist()


def collate_samples(samples: list[dict]) -> dict[str, np.ndarray]:
    return {key: np.stack([sample[key] for sample in samples], axis=0).astype(np.float32) for key in samples[0]}


def filter_trainable(params):
    return [param for param in params if getattr(param, "requires_grad", True)]


def group_params_jt(model, group_mode: str, initial_lr: float, optim_cfg: dict):
    if group_mode == "all":
        return [param for _, param in model.named_parameters() if getattr(param, "requires_grad", True)]
    if group_mode == "finetune":
        params_groups = model.get_grouped_params()
        return [
            {
                "params": filter_trainable(params_groups["pretrained"]),
                "lr": optim_cfg.get("diff_factor", 0.1) * initial_lr,
            },
            {
                "params": filter_trainable(params_groups["retrained"]),
                "lr": initial_lr,
            },
        ]
    raise NotImplementedError(f"Unsupported group_mode for current Jittor script: {group_mode}")


def construct_optimizer_jt(jt, model, initial_lr: float, mode: str, group_mode: str, optim_cfg: dict):
    params = group_params_jt(model, group_mode=group_mode, initial_lr=initial_lr, optim_cfg=optim_cfg)
    if mode == "adam":
        optimizer = jt.optim.Adam(
            params=params,
            lr=initial_lr,
            betas=tuple(optim_cfg.get("betas", (0.9, 0.999))),
            weight_decay=optim_cfg.get("weight_decay", 0),
        )
    elif mode == "adamw":
        optimizer = jt.optim.AdamW(
            params=params,
            lr=initial_lr,
            betas=tuple(optim_cfg.get("betas", (0.9, 0.999))),
            weight_decay=optim_cfg.get("weight_decay", 0),
        )
    elif mode == "sgd":
        optimizer = jt.optim.SGD(
            params=params,
            lr=initial_lr,
            momentum=optim_cfg.get("momentum", 0),
            weight_decay=optim_cfg.get("weight_decay", 0),
            nesterov=optim_cfg.get("nesterov", False),
        )
    else:
        raise NotImplementedError(mode)
    return optimizer


def lr_groups(optimizer) -> list[float]:
    return [group.get("lr", optimizer.lr) for group in optimizer.param_groups]


def lr_string(optimizer) -> str:
    return ",".join(f"{lr:.3e}" for lr in lr_groups(optimizer))


def maybe_freeze_encoder(model, freeze_encoder: bool):
    if not freeze_encoder:
        return
    for _, param in model.encoder.named_parameters():
        param.requires_grad = False


def maybe_freeze_bn_stats(model, freeze_status: bool, freeze_affine: bool):
    if freeze_status:
        frozen_bn_stats_jt(model.encoder, freeze_affine=freeze_affine)


def save_checkpoint(jt, model, save_dir: Path, step: int):
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"step_{step:06d}.pkl"
    jt.save(model.state_dict(), str(save_path))
    return save_path


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    cfg = load_cfg(args)

    import jittor as jt

    jt.flags.use_cuda = 1 if args.use_cuda else 0

    train_names = args.train_datasets or list(cfg.train.data.names)
    dataset = ImageTrainDatasetJT(
        dataset_infos={data_name: cfg.dataset_infos[data_name] for data_name in train_names},
        shape=cfg.train.data.shape,
    )
    if len(dataset) == 0:
        raise SystemExit("The training dataset is empty. Please check --data-cfg and dataset paths.")

    model = RN50_ZoomNeXt_JT(
        pretrained=args.pretrained,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        siu_groups=4,
        hmu_groups=6,
        weight_path=args.encoder_weight_path,
    )
    if args.resume_from:
        model.load_state_dict(jt.load(args.resume_from))
    model.train()

    maybe_freeze_encoder(model, cfg.train.bn.freeze_encoder)

    optimizer = construct_optimizer_jt(
        jt=jt,
        model=model,
        initial_lr=cfg.train.lr,
        mode=cfg.train.optimizer.mode,
        group_mode=cfg.train.optimizer.group_mode,
        optim_cfg=cfg.train.optimizer.cfg,
    )

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(cfg.train.batch_size)
    num_epochs = int(cfg.train.num_epochs)
    total_iters = args.max_iters or (num_epochs * max(1, len(dataset) // batch_size))
    curr_iter = 0

    start_time = time.perf_counter()
    progress = tqdm(total=total_iters, desc="[JT-TRAIN]", ncols=100)
    for epoch in range(num_epochs):
        model.train()
        maybe_freeze_bn_stats(model, cfg.train.bn.freeze_status, cfg.train.bn.freeze_affine)

        for batch_ids in batch_indices(len(dataset), batch_size, seed=args.seed + epoch, drop_last=True):
            progress.set_description(f"[JT-TRAIN][E{epoch}]")
            progress.set_postfix_str(f"iter={curr_iter}")
            samples = [dataset[idx] for idx in batch_ids]
            np_batch = collate_samples(samples)
            jt_batch = {key: jt.array(value) for key, value in np_batch.items()}

            iter_percentage = curr_iter / max(total_iters - 1, 1)
            outputs = model(data=jt_batch, iter_percentage=iter_percentage)
            loss = outputs["loss"]
            optimizer.step(loss)

            item = {
                "iter": curr_iter,
                "epoch": epoch,
                "lr": lr_groups(optimizer),
                "lr_string": lr_string(optimizer),
                "loss": float(loss.numpy()),
                "loss_str": outputs["loss_str"],
                "shape": list(np_batch["mask"].shape),
            }
            progress.update(1)
            progress.set_postfix_str(f"iter={curr_iter} loss={item['loss']:.5f} lr={item['lr_string']}")
            progress.write(json.dumps(item, ensure_ascii=False))

            curr_iter += 1
            if curr_iter % args.save_every == 0 or curr_iter == total_iters:
                save_path = save_checkpoint(jt, model, checkpoint_dir, curr_iter)
                progress.write(json.dumps({"checkpoint": str(save_path)}, ensure_ascii=False))

            if curr_iter >= total_iters:
                break
        if curr_iter >= total_iters:
            break

    progress.close()
    final_path = save_checkpoint(jt, model, checkpoint_dir, curr_iter)
    elapsed = time.perf_counter() - start_time
    print(json.dumps({"final_checkpoint": str(final_path), "elapsed_sec": elapsed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
