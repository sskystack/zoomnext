"""Jittor counterpart of the PyTorch scaler wrapper used in training."""

from __future__ import annotations

from contextlib import nullcontext


class _JittorAutocast:
    def __init__(self, jt, enabled: bool = False, amp_level: int = 5):
        self.jt = jt
        self.enabled = enabled
        self.amp_level = amp_level
        self.scope = None

    def __enter__(self):
        if not self.enabled:
            self.scope = nullcontext()
        else:
            self.scope = self.jt.flag_scope(
                auto_mixed_precision_level=self.amp_level,
                amp_level=self.amp_level,
            )
        return self.scope.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self.scope.__exit__(exc_type, exc_value, traceback)


def _clip_grad_value_jt(jt, optimizer, clip_value: float):
    for param_group in optimizer.param_groups:
        grads = param_group.get("grads", [])
        for grad in grads:
            clipped = jt.minimum(jt.maximum(grad, -clip_value), clip_value)
            grad.assign(clipped.stop_grad())


class ScalerJT:
    def __init__(
        self,
        jt,
        optimizer,
        use_fp16: bool = False,
        *,
        set_to_none: bool = False,
        clip_grad: bool = False,
        clip_mode=None,
        clip_cfg=None,
        amp_level: int = 5,
    ) -> None:
        self.jt = jt
        self.optimizer = optimizer
        self.use_fp16 = use_fp16
        self.set_to_none = set_to_none
        self.clip_grad = clip_grad
        self.clip_mode = clip_mode
        self.clip_cfg = clip_cfg or {}
        self.amp_level = amp_level

    def autocast(self):
        return _JittorAutocast(self.jt, enabled=self.use_fp16, amp_level=self.amp_level)

    def _apply_grad_clip(self):
        if not self.clip_grad:
            return
        if self.clip_mode == "norm":
            if "max_norm" not in self.clip_cfg:
                raise ValueError("`clip_cfg` must contain `max_norm`.")
            self.optimizer.clip_grad_norm(self.clip_cfg.get("max_norm"), self.clip_cfg.get("norm_type", 2.0))
        elif self.clip_mode == "value":
            if "clip_value" not in self.clip_cfg:
                raise ValueError("`clip_cfg` must contain `clip_value`.")
            _clip_grad_value_jt(self.jt, self.optimizer, clip_value=self.clip_cfg.get("clip_value"))
        else:
            raise NotImplementedError(self.clip_mode)

    def calculate_grad(self, loss):
        self.optimizer.backward(loss)
        self._apply_grad_clip()

    def update_grad(self):
        self.optimizer.step()
        # Jittor does not expose a set_to_none equivalent; keep the interface for parity.
        self.optimizer.zero_grad()

    def state_dict(self):
        return {
            "use_fp16": self.use_fp16,
            "set_to_none": self.set_to_none,
            "clip_grad": self.clip_grad,
            "clip_mode": self.clip_mode,
            "clip_cfg": self.clip_cfg,
            "amp_level": self.amp_level,
        }

    def load_state_dict(self, state_dict):
        self.use_fp16 = state_dict.get("use_fp16", self.use_fp16)
        self.set_to_none = state_dict.get("set_to_none", self.set_to_none)
        self.clip_grad = state_dict.get("clip_grad", self.clip_grad)
        self.clip_mode = state_dict.get("clip_mode", self.clip_mode)
        self.clip_cfg = state_dict.get("clip_cfg", self.clip_cfg)
        self.amp_level = state_dict.get("amp_level", self.amp_level)
