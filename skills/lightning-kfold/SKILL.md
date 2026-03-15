# Skill: lightning-kfold

## Description

Work with lightning-kfold -- a drop-in stratified K-fold cross-validation library with ensemble voting for PyTorch Lightning.

## When to Use

- Adding new data module features or fold-splitting strategies
- Extending `KFoldTrainer` with new checkpointing or evaluation logic
- Modifying `EnsembleVotingModel` inference (e.g. weighted voting, different aggregation)
- Writing tests for K-fold training workflows
- Debugging fold iteration or model state reset between folds

## Context

- **Language**: Python 3.10+
- **Build**: `uv` (package manager), `uv_build` backend
- **Dependencies**: `lightning>=2.0`, `torch>=2.0`, `scikit-learn>=1.0`
- **Public API** (from `__init__.py`):
  - `KFoldDataModule` -- abstract; users implement `setup_datasets() -> (Dataset, Dataset, np.ndarray)`
  - `KFoldTrainer` -- orchestrates fold iteration, model reset, checkpointing, ensemble test
  - `EnsembleVotingModel` -- loads K checkpoints, averages logits
- **Test framework**: pytest (test paths: `tests/`)

## Key Commands

```bash
uv sync               # install all dependencies
uv run pytest          # run tests
uv run ruff check .    # lint
uv run ruff format .   # format
```

## Conventions

- Type hints throughout; `py.typed` marker present
- Fold checkpoints saved as `fold-{i}.ckpt` in the export directory
- Model `state_dict` is deep-copied and restored between folds to ensure isolation
- Stratification uses `StratifiedKFold` (>=2 folds) or `StratifiedShuffleSplit` (1 fold)
- Conventional commits required (see `sr.yaml`)
