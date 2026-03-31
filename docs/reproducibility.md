# Reproducibility

The repository is intentionally small enough to reproduce without custom kernels.

## Baseline Environment

- Python `>=3.9`
- NumPy `>=2.0`
- PyTorch `>=2.0`

## Recommended Commands

```bash
python3 -m pip install -e .[dev]
PYTHONPATH=src pytest
PYTHONPATH=src python3 demos/reference_demo.py
PYTHONPATH=src python3 benchmarks/synthetic_distortion.py --bits-list 2,3,4,5,6,8
PYTHONPATH=src python3 demos/visual_report.py
```

## Notes

- Benchmarks use fixed seeds by default.
- Current numbers are synthetic and meant to validate implementation behavior.
- Full paper reproduction with long-context models is listed as future work.
