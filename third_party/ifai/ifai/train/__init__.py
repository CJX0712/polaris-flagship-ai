"""L4 训练层 公共入口。"""
from .autodiff import Tensor  # noqa: F401
from .optim import AdamW, cosine_schedule, clip_global_norm, GlobalStep  # noqa: F401
from .trainer import TinyTrainer  # noqa: F401
from .checkpoint import (  # noqa: F401
    checkpoint,
    checkpointed,
    checkpoint_segments,
    grad_diff,
    verify_checkpoint,
)

__all__ = ["Tensor", "AdamW", "cosine_schedule", "clip_global_norm", "GlobalStep",
           "TinyTrainer", "checkpoint", "checkpointed", "checkpoint_segments",
           "grad_diff", "verify_checkpoint"]
