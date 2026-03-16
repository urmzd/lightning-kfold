# Contributing

## Prerequisites

- Python 3.10+ (see `pyproject.toml` for version constraint)
- [uv](https://github.com/astral-sh/uv) (package manager)
- [just](https://github.com/casey/just) (task runner)
- A `GH_TOKEN` with repo access (for releases)

## Getting Started

```bash
git clone https://github.com/urmzd/lightning-kfold.git
cd lightning-kfold
just init
```

## Development

```bash
just check    # format, lint, test
just test     # run tests
just fmt      # format code
```

## Commit Convention

We use [Angular Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`, `perf`

Commits are enforced via [gitit](https://github.com/urmzd/gitit).

## Pull Requests

1. Fork the repository
2. Create a feature branch (`feat/my-feature`)
3. Make changes and commit using conventional commits
4. Open a pull request against `main`

## Code Style

- `ruff` for formatting and linting
- `pytest` for testing
