# Contributing

Thanks for contributing to `turboquant-kvcache`.

## Scope

This repository prioritizes:

- faithful reference implementations of the TurboQuant paper
- reproducible measurements
- readable code that is easy to compare against the method

Performance-oriented changes are welcome, but they should not obscure the reference path.

## Development Setup

```bash
python3 -m pip install -e .[dev]
```

Run the core checks before opening a pull request:

```bash
PYTHONPATH=src pytest
PYTHONPATH=src python3 demos/reference_demo.py
PYTHONPATH=src python3 benchmarks/synthetic_distortion.py
```

## Pull Requests

Please keep pull requests narrow and explain:

- what changed
- why it is needed
- whether it affects the numerical behavior
- how you validated it

For algorithmic changes, include before-and-after numbers when possible.

## Style

- Prefer simple PyTorch or NumPy code over clever abstractions.
- Keep tensor shapes explicit.
- Preserve deterministic seeds in benchmarks and tests.
- Add comments only when they clarify a non-obvious mathematical or tensor operation.
