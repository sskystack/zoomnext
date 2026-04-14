"""ResNet50 backbone placeholder for the upcoming Jittor migration step."""

from __future__ import annotations


def build_resnet50(pretrained: bool = False, weight_path: str | None = None):
    del pretrained, weight_path
    raise NotImplementedError("ResNet50 migration is planned for the next step.")


def extract_features(x):
    del x
    raise NotImplementedError("ResNet50 migration is planned for the next step.")
