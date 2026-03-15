"""Ensemble voting model that averages predictions from K-fold checkpoints."""

from pathlib import Path
from typing import Any, Type

import lightning as L
import torch
from torch.nn import functional as F


class EnsembleVotingModel(L.LightningModule):
    """Averages logits from multiple model checkpoints for prediction.

    After K-fold training, each fold produces a checkpoint. This model
    loads all of them and averages their outputs at test time.
    """

    def __init__(
        self,
        model_cls: Type[L.LightningModule],
        checkpoint_paths: list[Path],
        loss_fn: Any = F.cross_entropy,
    ) -> None:
        super().__init__()
        self.models = torch.nn.ModuleList(
            [model_cls.load_from_checkpoint(str(p)) for p in checkpoint_paths]
        )
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([m(x) for m in self.models])
        return logits.mean(dim=0)

    def test_step(self, batch: Any, batch_idx: int) -> dict[str, torch.Tensor]:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return {"loss": loss, "logits": logits, "y_true": y}
