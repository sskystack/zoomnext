"""Jittor ResNet50 backbone migrated from the PyTorch/timm reference.

This module keeps the canonical torchvision-style naming convention so
that the pretrained `resnet50-timm.pth` weights can be loaded directly.
The backbone returns five feature levels that match
`timm.create_model(..., features_only=True, out_indices=range(5))`:

- c1: stem output after `conv1 + bn1 + relu`, before max-pool
- c2: output of `layer1`
- c3: output of `layer2`
- c4: output of `layer3`
- c5: output of `layer4`
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import jittor as jt
from jittor import nn


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


def torch_state_dict_to_numpy(state_dict: Dict[str, object]) -> Dict[str, np.ndarray]:
    converted = {}
    for name, value in state_dict.items():
        if hasattr(value, "detach"):
            converted[name] = value.detach().cpu().numpy()
        else:
            converted[name] = np.asarray(value)
    return converted


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = _conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def execute(self, x: jt.Var) -> jt.Var:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet50Backbone(nn.Module):
    feature_channels = (64, 256, 512, 1024, 2048)

    def __init__(self, pretrained: bool = False, weight_path: Optional[str] = None) -> None:
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(planes=64, blocks=3)
        self.layer2 = self._make_layer(planes=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(planes=256, blocks=6, stride=2)
        self.layer4 = self._make_layer(planes=512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, 1000)

        if pretrained:
            self.load_torch_pretrained(weight_path=weight_path)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        outplanes = planes * Bottleneck.expansion
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential()
            downsample.add_module("0", _conv1x1(self.inplanes, outplanes, stride=stride))
            downsample.add_module("1", nn.BatchNorm2d(outplanes))

        layers = nn.Sequential()
        layers.add_module("0", Bottleneck(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = outplanes
        for block_idx in range(1, blocks):
            layers.add_module(str(block_idx), Bottleneck(self.inplanes, planes))
        return layers

    def forward_features(self, x: jt.Var) -> Tuple[jt.Var, jt.Var, jt.Var, jt.Var, jt.Var]:
        x = self.conv1(x)
        x = self.bn1(x)
        c1 = self.relu(x)

        x = self.maxpool(c1)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c1, c2, c3, c4, c5

    def forward_head(self, c5: jt.Var) -> jt.Var:
        x = self.avgpool(c5)
        x = x.reshape((x.shape[0], -1))
        return self.fc(x)

    def execute(self, x: jt.Var) -> Tuple[jt.Var, jt.Var, jt.Var, jt.Var, jt.Var]:
        return self.forward_features(x)

    def load_torch_pretrained(self, weight_path: Optional[str] = None) -> None:
        if weight_path is None:
            weight_path = str(Path(__file__).resolve().parents[3] / "pretrained_weights" / "resnet50-timm.pth")

        weight_file = Path(weight_path)
        if not weight_file.is_file():
            raise FileNotFoundError(f"Cannot find ResNet50 pretrained weights: {weight_file}")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Loading the PyTorch-format ResNet50 checkpoint requires torch to be installed."
            ) from exc

        state_dict = torch.load(str(weight_file), map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]

        self.load_state_dict(torch_state_dict_to_numpy(state_dict))


def build_resnet50(pretrained: bool = False, weight_path: Optional[str] = None) -> ResNet50Backbone:
    return ResNet50Backbone(pretrained=pretrained, weight_path=weight_path)


def extract_features(model: ResNet50Backbone, x: jt.Var) -> Tuple[jt.Var, jt.Var, jt.Var, jt.Var, jt.Var]:
    return model(x)
