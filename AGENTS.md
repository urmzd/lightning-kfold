# AGENTS.md

## Identity

**lightning-kfold** -- Drop-in stratified K-fold cross-validation with ensemble voting for PyTorch Lightning.

## Architecture

The library exposes three classes that compose into a single workflow:

| Component | Purpose |
|-----------|---------|
| `KFoldDataModule` | Abstract `LightningDataModule` subclass that performs stratified K-fold splitting. Users implement `setup_datasets()` to provide train/test datasets and labels. |
| `KFoldTrainer` | Orchestrator that iterates over folds, resets model weights, trains, checkpoints, and builds an ensemble. |
| `EnsembleVotingModel` | `LightningModule` that loads K checkpoints and averages logits at inference time. |

The flow is: user subclasses `KFoldDataModule` -> passes it with a model to `KFoldTrainer.fit()` -> trainer iterates folds, saves checkpoints -> builds `EnsembleVotingModel` -> tests ensemble on held-out data.

## Key Files

| File | Description |
|------|-------------|
| `src/lightning_kfold/__init__.py` | Public API; exports `KFoldDataModule`, `KFoldTrainer`, `EnsembleVotingModel`. |
| `src/lightning_kfold/datamodule.py` | `KFoldDataModule` with stratified splitting via scikit-learn. |
| `src/lightning_kfold/trainer.py` | `KFoldTrainer` -- fold iteration, checkpoint saving, ensemble creation. |
| `src/lightning_kfold/ensemble.py` | `EnsembleVotingModel` -- loads checkpoints, averages logits. |
| `pyproject.toml` | Package metadata, dependencies (`lightning`, `torch`, `scikit-learn`), dev deps. |
| `tests/` | Test suite (pytest). |

## Commands

```bash
uv sync               # install dependencies
uv run pytest          # run tests
uv run ruff check .    # lint
uv run ruff format .   # format
```

## Code Style

- **Formatting / Linting**: `ruff`.
- **Testing**: `pytest` with test paths configured in `pyproject.toml`.
- **Type hints**: Used throughout; `py.typed` marker present.
- **Commit convention**: Angular conventional commits (see `sr.yaml`).
