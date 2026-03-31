# TurboQuant KV-Cache

Reference implementation of **TurboQuant** for **KV-cache quantization**, based on the paper:

> TurboQuant: An Efficient and Accurate KV Cache Quantization Method  
> https://arxiv.org/html/2504.19874v1

This repository focuses on a clean, reproducible, and extensible PyTorch implementation of the paper's core ideas:

- Random orthogonal rotation before quantization
- Lloyd-Max scalar quantization matched to sphere-coordinate statistics
- A KV-cache oriented codec for tensors shaped like `B x H x T x D`
- An inner-product estimator built from MSE quantization plus a 1-bit residual sketch
- Synthetic benchmarks and a self-contained HTML visual report

## Status

Implemented in `v0.1.0`:

- `TurboQuantMSEQuantizer`
- `TurboQuantInnerProductQuantizer`
- `TurboQuantKVCacheCodec`
- Synthetic KV-cache benchmark
- Browser-friendly HTML report

Not implemented yet:

- Entropy coding for codebook pointers
- Fractional-bit channel split such as `2.5 / 3.5 bits per channel`
- CUDA / Triton kernels
- Hugging Face or vLLM runtime integration
- Full paper-scale long-context evaluation

## Repository Layout

```text
turboquant-kvcache/
  src/turboquant_kvcache/
  benchmarks/
  demos/
  tests/
  docs/
```

## Quickstart

```bash
cd turboquant-kvcache
python3 -m pip install -e .[dev]
PYTHONPATH=src python3 demos/reference_demo.py
```

## Reference Demo

```bash
PYTHONPATH=src python3 demos/reference_demo.py --bits 4 --seq-len 256 --heads 8 --head-dim 128 --batch 2
```

Typical output looks like:

```text
TurboQuant KV Cache Demo
shape                 : (2, 8, 256, 128)
turbo levels          : 16 (~4.000 bits)
estimated compression : 3.76x versus fp16 cache
key mse               : 0.009301
value mse             : 0.009279
uniform key mse       : 0.012294
uniform value mse     : 0.012311
attn output rmse      : 0.013184
uniform output rmse   : 0.015670
ip estimator rmse     : 0.971212
```

## Visual Demo

Generate a self-contained HTML report:

```bash
PYTHONPATH=src python3 demos/visual_report.py
```

This writes `report.html`, which you can open locally in a browser.

## Synthetic Benchmark

```bash
PYTHONPATH=src python3 benchmarks/synthetic_distortion.py
```

## Why This Repo Exists

Most public KV-cache quantization repos optimize quickly for inference integration. This repository takes a different first step:

- make the paper easy to read in code
- make the numerical behavior easy to test
- make future runtime integration easier by keeping a clean reference path

## Design Notes

- The implementation quantizes the last dimension of KV-cache tensors.
- Compression estimates assume `fp16` source cache and account for quantized coordinates plus one stored norm per vector.
- The current code emphasizes clarity and reproducibility over peak performance.

## Development

Run tests:

```bash
PYTHONPATH=src pytest
```

## Roadmap

1. Add entropy coding and packing utilities.
2. Add mixed-bit and outlier-aware channel policies.
3. Integrate with a Hugging Face attention cache path.
4. Add long-context evaluation against real models.
5. Add optimized Triton or CUDA kernels.

## Citation

If you use this repository, please cite the original TurboQuant paper.
