"""Abstract K-fold data module for PyTorch Lightning."""

from abc import ABC, abstractmethod
from typing import Optional

import lightning as L
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset


class KFoldDataModule(L.LightningDataModule, ABC):
    """Abstract data module that handles stratified K-fold splitting.

    Subclasses must implement ``setup_datasets`` to provide the full
    train and test datasets along with integer labels for stratification.
    """

    def __init__(
        self,
        num_folds: int = 5,
        batch_size: int = 32,
        num_workers: int = 0,
        train_size: float = 0.8,
    ) -> None:
        super().__init__()
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size

        self._train_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._train_fold: Optional[Dataset] = None
        self._val_fold: Optional[Dataset] = None

    @abstractmethod
    def setup_datasets(self) -> tuple[Dataset, Dataset, np.ndarray]:
        """Return (train_dataset, test_dataset, train_labels).

        ``train_labels`` is a 1-D array of integer class labels aligned
        with ``train_dataset``, used for stratified splitting.
        """

    def setup(self, stage: Optional[str] = None) -> None:
        self._train_dataset, self._test_dataset, labels = self.setup_datasets()

        if self.num_folds >= 2:
            splitter = StratifiedKFold(self.num_folds, shuffle=True)
        else:
            splitter = StratifiedShuffleSplit(1, train_size=self.train_size)

        indices = np.arange(len(labels))
        self._splits = list(splitter.split(indices, labels))

    def setup_fold(self, fold_index: int) -> None:
        """Configure dataloaders for a specific fold."""
        train_idx, val_idx = self._splits[fold_index]
        self._train_fold = Subset(self._train_dataset, train_idx.tolist())
        self._val_fold = Subset(self._train_dataset, val_idx.tolist())

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_fold,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_fold,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
