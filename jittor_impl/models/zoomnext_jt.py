"""Jittor implementation of the ResNet50-based ZoomNeXt model."""

from __future__ import annotations

import abc
import logging
import math
from typing import Dict

import jittor as jt
from jittor import nn

from .backbone import build_resnet50
from .layers_jt import MHSIU, RGPU, SimpleASPP
from .ops_jt import ConvBNReLU, PixelNormalizer, resize_to

LOGGER = logging.getLogger("main")


class _Identity(nn.Module):
    def execute(self, x: jt.Var) -> jt.Var:
        return x


class _BilinearUpsample(nn.Module):
    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor

    def execute(self, x: jt.Var) -> jt.Var:
        return nn.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)


def _scalar(var: jt.Var) -> float:
    return float(var.numpy().reshape(-1)[0])


def frozen_bn_stats_jt(model: nn.Module, freeze_affine: bool = False) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            if freeze_affine:
                if hasattr(module, "weight") and module.weight is not None:
                    module.weight.requires_grad = False
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.requires_grad = False


def _is_buffer_like_parameter(name: str) -> bool:
    return name.endswith("running_mean") or name.endswith("running_var") or name.endswith("num_batches_tracked")


class _ZoomNeXt_Base(nn.Module):
    @staticmethod
    def get_coef(iter_percentage=1, method="cos", milestones=(0, 1)):
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = 0, 1

        ual_coef = 1.0
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            if method == "linear":
                ratio = (max_coef - min_coef) / (max_point - min_point)
                ual_coef = ratio * (iter_percentage - min_point)
            elif method == "cos":
                perc = (iter_percentage - min_point) / (max_point - min_point)
                normalized_coef = (1 - math.cos(perc * math.pi)) / 2
                ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
        return ual_coef

    @abc.abstractmethod
    def body(self, data: Dict[str, jt.Var]) -> jt.Var:
        raise NotImplementedError

    def execute(self, data, iter_percentage=1, **kwargs):
        del kwargs
        logits = self.body(data=data)

        if self.is_training():
            mask = data["mask"]
            prob = jt.sigmoid(logits)

            losses = []
            loss_str = []

            sod_loss = nn.BCEWithLogitsLoss()(logits, mask)
            losses.append(sod_loss)
            loss_str.append(f"bce: {_scalar(sod_loss):.5f}")

            ual_coef = self.get_coef(iter_percentage=iter_percentage, method="cos", milestones=(0, 1))
            ual_loss = ual_coef * (1 - jt.abs(2 * prob - 1).pow(2)).mean()
            losses.append(ual_loss)
            loss_str.append(f"powual_{ual_coef:.5f}: {_scalar(ual_loss):.5f}")
            return dict(vis=dict(sal=prob), loss=sum(losses), loss_str=" ".join(loss_str))
        return logits

    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if _is_buffer_like_parameter(name):
                continue
            if name.startswith("encoder.patch_embed1."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                if "clip." in name:
                    param.requires_grad = False
                    param_groups["fixed"].append(param)
                else:
                    param_groups["retrained"].append(param)
        LOGGER.info(
            f"Parameter Groups:{{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups


class RN50_ZoomNeXt_JT(_ZoomNeXt_Base):
    def __init__(
        self,
        pretrained=True,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        siu_groups=4,
        hmu_groups=6,
        weight_path: str | None = None,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        self.encoder = build_resnet50(pretrained=pretrained, weight_path=weight_path)

        self.tra_5 = SimpleASPP(in_dim=2048, out_dim=mid_dim)
        self.siu_5 = MHSIU(mid_dim, siu_groups)
        self.hmu_5 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_4 = ConvBNReLU(1024, mid_dim, 3, 1, 1)
        self.siu_4 = MHSIU(mid_dim, siu_groups)
        self.hmu_4 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_3 = ConvBNReLU(512, mid_dim, 3, 1, 1)
        self.siu_3 = MHSIU(mid_dim, siu_groups)
        self.hmu_3 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_2 = ConvBNReLU(256, mid_dim, 3, 1, 1)
        self.siu_2 = MHSIU(mid_dim, siu_groups)
        self.hmu_2 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_1 = ConvBNReLU(64, mid_dim, 3, 1, 1)
        self.siu_1 = MHSIU(mid_dim, siu_groups)
        self.hmu_1 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.normalizer = PixelNormalizer() if input_norm else _Identity()
        self.predictor = nn.Sequential()
        self.predictor.add_module("0", _BilinearUpsample(scale_factor=2))
        self.predictor.add_module("1", ConvBNReLU(64, 32, 3, 1, 1))
        self.predictor.add_module("2", nn.Conv2d(32, 1, 1))

    def normalize_encoder(self, x: jt.Var):
        x = self.normalizer(x)
        c1, c2, c3, c4, c5 = self.encoder(x)
        return c1, c2, c3, c4, c5

    def body(self, data: Dict[str, jt.Var]) -> jt.Var:
        l_trans_feats = self.normalize_encoder(data["image_l"])
        m_trans_feats = self.normalize_encoder(data["image_m"])
        s_trans_feats = self.normalize_encoder(data["image_s"])

        l, m, s = (
            self.tra_5(l_trans_feats[4]),
            self.tra_5(m_trans_feats[4]),
            self.tra_5(s_trans_feats[4]),
        )
        lms = self.siu_5(l=l, m=m, s=s)
        x = self.hmu_5(lms)

        l, m, s = (
            self.tra_4(l_trans_feats[3]),
            self.tra_4(m_trans_feats[3]),
            self.tra_4(s_trans_feats[3]),
        )
        lms = self.siu_4(l=l, m=m, s=s)
        x = self.hmu_4(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = (
            self.tra_3(l_trans_feats[2]),
            self.tra_3(m_trans_feats[2]),
            self.tra_3(s_trans_feats[2]),
        )
        lms = self.siu_3(l=l, m=m, s=s)
        x = self.hmu_3(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = (
            self.tra_2(l_trans_feats[1]),
            self.tra_2(m_trans_feats[1]),
            self.tra_2(s_trans_feats[1]),
        )
        lms = self.siu_2(l=l, m=m, s=s)
        x = self.hmu_2(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = (
            self.tra_1(l_trans_feats[0]),
            self.tra_1(m_trans_feats[0]),
            self.tra_1(s_trans_feats[0]),
        )
        lms = self.siu_1(l=l, m=m, s=s)
        x = self.hmu_1(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        return self.predictor(x)
