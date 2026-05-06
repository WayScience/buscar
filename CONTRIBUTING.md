# Contributing

Thanks for your interest in improving Buscar. Contributions that fix bugs, improve
documentation, add tests, or make the package easier to install and maintain are
welcome.

## Development Setup

Buscar uses [uv](https://docs.astral.sh/uv/) for environment and package
management.

```bash
git clone https://github.com/WayScience/buscar.git
cd buscar
uv sync --frozen --group dev
```

Install the pre-commit hooks before making changes:

```bash
uv run --frozen pre-commit install
```

## Checks

Run the test suite:

```bash
uv run --frozen pytest
```

Run tests with coverage:

```bash
uv run --frozen pytest --cov=buscar --cov-report=term-missing --cov-report=xml --cov-report=html
```

Run linting and formatting checks:

```bash
uv run --frozen pre-commit run --all-files
```

Build the package:

```bash
uv build
uv run --frozen twine check dist/*
```

## Pull Requests

Before opening a pull request:

- Keep changes focused on one bug fix, feature, or documentation update.
- Add or update tests when behavior changes.
- Update documentation when user-facing behavior changes.
- Run `uv run --frozen pytest` and `uv run --frozen pre-commit run --all-files`.
- Make sure generated files and large local artifacts are not committed.

CI runs pre-commit checks, the test matrix across supported Python versions, and
package build validation.

## Reporting Issues

When reporting a bug, include:

- The Buscar version or commit you are using.
- Your Python version and operating system.
- A minimal example or traceback that reproduces the issue.
- Any relevant input data shape, schema, or metadata column names.

## License

By contributing, you agree that your contribution will be licensed under the
BSD 3-Clause license used by this project.
