# lightning-kfold

Drop-in stratified K-fold cross-validation with ensemble voting for PyTorch Lightning.

## Install

```bash
uv add lightning-kfold
```

## Usage

### 1. Define your data module

Subclass `KFoldDataModule` and implement `setup_datasets`:

```python
import numpy as np
from torch.utils.data import TensorDataset
from lightning_kfold import KFoldDataModule


class MyDataModule(KFoldDataModule):
    def setup_datasets(self):
        # Return (train_dataset, test_dataset, train_labels)
        train_ds = TensorDataset(train_x, train_y)
        test_ds = TensorDataset(test_x, test_y)
        labels = train_y.numpy()  # 1-D integer array for stratification
        return train_ds, test_ds, labels
```

### 2. Train with K-fold

```python
from lightning_kfold import KFoldTrainer

model = MyLightningModel()
dm = MyDataModule(num_folds=5, batch_size=32)
dm.setup()

trainer = KFoldTrainer(num_folds=5, max_epochs=10)
ensemble_results = trainer.fit(model, dm)
```

That's it. `KFoldTrainer` will:

1. Reset model weights before each fold
2. Train and test each fold independently
3. Save a checkpoint per fold
4. Build an ensemble that averages logits from all folds
5. Test the ensemble on the held-out test set

### 3. Use the ensemble directly

```python
from lightning_kfold import EnsembleVotingModel

ensemble = EnsembleVotingModel(
    model_cls=MyLightningModel,
    checkpoint_paths=["kfold_checkpoints/fold-0.ckpt", ...],
)

# Use for inference
logits = ensemble(input_tensor)
```

## API

| Class | Purpose |
|---|---|
| `KFoldDataModule` | Abstract data module with stratified splitting. Implement `setup_datasets()`. |
| `KFoldTrainer` | Orchestrates fold training, checkpointing, and ensemble creation. |
| `EnsembleVotingModel` | Loads K checkpoints and averages their logits at test time. |

## License

Apache-2.0
