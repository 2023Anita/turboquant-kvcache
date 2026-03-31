# Benchmarks

The repository currently provides:

- `benchmarks/synthetic_distortion.py`
  - Sweeps bit-widths and reports KV-cache reconstruction error plus attention-output RMSE.
- `demos/reference_demo.py`
  - Runs a single synthetic KV-cache example and prints a compact summary.
- `demos/visual_report.py`
  - Generates a self-contained HTML report for quick visual inspection.

Current scope is a faithful reference implementation, not a full paper reproduction with external model evaluation.
