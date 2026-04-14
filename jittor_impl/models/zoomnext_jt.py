"""ZoomNeXt model placeholder for the upcoming Jittor migration step."""

from __future__ import annotations


class _ZoomNeXt_Base:
    pass


class RN50_ZoomNeXt_JT(_ZoomNeXt_Base):
    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError("RN50_ZoomNeXt_JT will be migrated in the next step.")
