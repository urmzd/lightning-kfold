"""High-level K-fold trainer that orchestrates fold iteration and ensemble creation."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Type

import lightning as L
import torch
from torch.nn import functional as F

from lightning_kfold.datamodule import KFoldDataModule
from lightning_kfold.ensemble import EnsembleVotingModel


class KFoldTrainer:
    """Runs stratified K-fold cross-validation with automatic ensemble voting.

    Wraps a standard Lightning ``Trainer`` and manages fold iteration,
    model state isolation between folds, checkpoint saving, and ensemble
    construction.

    Example::

        trainer = KFoldTrainer(num_folds=5, max_epochs=10)
        results = trainer.fit(model, datamodule)
    """

    def __init__(
        self,
        num_folds: int = 5,
        export_path: str | Path = "kfold_checkpoints",
        loss_fn: Any = F.cross_entropy,
        **trainer_kwargs: Any,
    ) -> None:
        self.num_folds = num_folds
        self.export_path = Path(export_path)
        self.loss_fn = loss_fn
        self.trainer_kwargs = trainer_kwargs
        self.fold_results: list[list[dict]] = []

    def fit(
        self,
        model: L.LightningModule,
        datamodule: KFoldDataModule,
        test_ensemble: bool = True,
    ) -> Optional[list[dict]]:
        """Run K-fold training and optionally test the ensemble.

        Returns the ensemble test results if ``test_ensemble`` is True.
        """
        self.export_path.mkdir(parents=True, exist_ok=True)
        initial_state = deepcopy(model.state_dict())

        for fold in range(self.num_folds):
            # Reset model to initial weights for each fold.
            model.load_state_dict(deepcopy(initial_state))

            datamodule.setup_fold(fold)

            trainer = L.Trainer(**self.trainer_kwargs)
            trainer.fit(model, datamodule=datamodule)

            # Test this fold.
            fold_test = trainer.test(model, datamodule=datamodule)
            self.fold_results.append(fold_test)

            # Save checkpoint.
            checkpoint_path = self.export_path / f"fold-{fold}.ckpt"
            trainer.save_checkpoint(str(checkpoint_path))

        if not test_ensemble:
            return None

        return self._test_ensemble(model, datamodule)

    def _test_ensemble(
        self,
        model: L.LightningModule,
        datamodule: KFoldDataModule,
    ) -> list[dict]:
        """Build an ensemble from fold checkpoints and run the test set."""
        checkpoint_paths = [
            self.export_path / f"fold-{fold}.ckpt"
            for fold in range(self.num_folds)
        ]

        ensemble = EnsembleVotingModel(
            model_cls=type(model),
            checkpoint_paths=checkpoint_paths,
            loss_fn=self.loss_fn,
        )

        trainer = L.Trainer(**self.trainer_kwargs)
        return trainer.test(ensemble, datamodule=datamodule)
