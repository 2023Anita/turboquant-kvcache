from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import torch

from turboquant_kvcache import TurboQuantKVCacheCodec


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
ASSETS = ROOT / "assets"
PYPROJECT = ROOT / "pyproject.toml"


def build_report() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run(
        ["python3", "demos/visual_report.py", "--output", str(SITE / "report.html")],
        cwd=ROOT,
        env=env,
        check=True,
    )


def copy_assets() -> None:
    dest = SITE / "assets"
    dest.mkdir(parents=True, exist_ok=True)
    for path in ASSETS.glob("*"):
        if path.is_file():
            shutil.copy2(path, dest / path.name)


def read_version() -> str:
    match = re.search(r'^version = "([^"]+)"', PYPROJECT.read_text(encoding="utf-8"), flags=re.MULTILINE)
    if match is None:
        raise ValueError("could not parse version from pyproject.toml")
    return match.group(1)


def build_benchmark_rows() -> str:
    torch.manual_seed(0)
    device = torch.device("cpu")
    shape = (1, 4, 64, 64)
    key = torch.randn(shape, device=device, dtype=torch.float32)
    value = torch.randn(shape, device=device, dtype=torch.float32)
    query = torch.randn((1, 4, 1, 64), device=device, dtype=torch.float32)
    rows = []
    for bits in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        codec = TurboQuantKVCacheCodec(64, bits=bits, seed=0, device=device)
        metrics = codec.evaluate(query, key, value)
        rows.append(
            "<tr>"
            f"<td>{bits:.1f}</td>"
            f"<td>{metrics['compression_ratio']:.2f}x</td>"
            f"<td>{metrics['attention_output_rmse']:.6f}</td>"
            f"<td>{metrics['uniform_output_rmse']:.6f}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def build_index() -> None:
    version = read_version()
    benchmark_rows = build_benchmark_rows()
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TurboQuant KV-Cache</title>
  <style>
    :root {
      --bg: #f4efe7;
      --panel: rgba(255, 250, 243, 0.9);
      --ink: #171412;
      --muted: #625b54;
      --line: rgba(23, 20, 18, 0.1);
      --accent: #db563b;
      --accent-2: #2d5cff;
      --shadow: 0 24px 70px rgba(24, 18, 10, 0.14);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "PingFang SC", sans-serif;
      background:
        radial-gradient(circle at 10% 0%, rgba(219, 86, 59, 0.20), transparent 28%),
        radial-gradient(circle at 90% 12%, rgba(45, 92, 255, 0.14), transparent 30%),
        linear-gradient(180deg, #f9f5ed, #ece5db);
    }
    main {
      width: min(1160px, calc(100vw - 32px));
      margin: 24px auto 48px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .hero {
      padding: 26px;
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 20px;
      align-items: center;
    }
    h1 {
      margin: 0 0 12px;
      font-size: clamp(40px, 6vw, 72px);
      line-height: 0.96;
      letter-spacing: -0.04em;
      font-family: "Iowan Old Style", "Times New Roman", serif;
    }
    p {
      margin: 0;
      color: var(--muted);
      line-height: 1.65;
      font-size: 17px;
    }
    .links {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 20px;
    }
    .links a {
      text-decoration: none;
      color: var(--ink);
      padding: 12px 16px;
      border-radius: 999px;
      background: rgba(23, 20, 18, 0.05);
      font-weight: 600;
    }
    .links a.primary {
      background: var(--accent);
      color: white;
    }
    .grid {
      margin-top: 20px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }
    .panel-copy {
      padding: 22px;
    }
    h2 {
      margin: 0 0 10px;
      font-size: 26px;
      letter-spacing: -0.03em;
    }
    ul {
      margin: 12px 0 0;
      padding-left: 20px;
      color: var(--muted);
      line-height: 1.8;
    }
    .visual {
      width: 100%;
      display: block;
      border-top: 1px solid var(--line);
    }
    table {
      width: calc(100% - 28px);
      margin: 0 14px 18px;
      border-collapse: collapse;
      font-size: 15px;
    }
    th, td {
      padding: 12px 10px;
      border-bottom: 1px solid rgba(23, 20, 18, 0.08);
      text-align: left;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
    }
    .release {
      margin-top: 20px;
      padding: 18px 22px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
    }
    .release strong {
      font-size: 28px;
      letter-spacing: -0.03em;
    }
    .release a {
      text-decoration: none;
      color: white;
      background: var(--ink);
      padding: 12px 16px;
      border-radius: 999px;
      font-weight: 600;
      white-space: nowrap;
    }
    .preview {
      padding: 14px;
      background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(246,239,227,0.88));
    }
    .preview img {
      width: 100%;
      border-radius: 20px;
      display: block;
      border: 1px solid rgba(23, 20, 18, 0.08);
    }
    .footer {
      margin-top: 18px;
      text-align: center;
      color: var(--muted);
      font-size: 14px;
    }
    @media (max-width: 920px) {
      .hero, .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <section class="card hero">
      <div>
        <h1>TurboQuant KV-Cache</h1>
        <p>
          A faithful, reproducible, and extensible reference implementation of TurboQuant for KV-cache quantization.
          This site hosts the project overview and a browser-friendly visual report generated from the repository.
        </p>
        <div class="links">
          <a class="primary" href="./report.html">Open Visual Report</a>
          <a href="https://github.com/2023Anita/turboquant-kvcache">GitHub Repository</a>
          <a href="https://github.com/2023Anita/turboquant-kvcache/releases/tag/v0.1.0">v0.1.0 Release</a>
          <a href="https://arxiv.org/html/2504.19874v1">Paper</a>
        </div>
      </div>
      <img class="visual" src="./assets/turboquant-overview.svg" alt="TurboQuant overview" />
    </section>

    <section class="grid">
      <article class="card">
        <div class="panel-copy">
          <h2>What is implemented</h2>
          <p>The current repository focuses on the clean reference path before optimized kernels.</p>
          <ul>
            <li>Random orthogonal rotation before quantization</li>
            <li>Sphere-aware Lloyd-Max scalar quantization</li>
            <li>KV-cache codec for tensors shaped like B x H x T x D</li>
            <li>Inner-product path built with 1-bit residual sketching</li>
            <li>Transformers custom-loop runner for TurboQuant cache storage</li>
          </ul>
        </div>
        <img class="visual" src="./assets/benchmark-lowbit.svg" alt="Benchmark low-bit chart" />
        <table>
          <thead>
            <tr>
              <th>Bits</th>
              <th>Compression</th>
              <th>TurboQuant RMSE</th>
              <th>Uniform RMSE</th>
            </tr>
          </thead>
          <tbody>
            __BENCHMARK_ROWS__
          </tbody>
        </table>
      </article>

      <article class="card">
        <div class="panel-copy">
          <h2>Visual preview</h2>
          <p>
            The report page is generated directly from the repository and summarizes synthetic distortion trends,
            attention maps, token-level fidelity, and codebook usage.
          </p>
        </div>
        <div class="preview">
          <img src="./assets/report-preview.png" alt="TurboQuant visual report preview" />
        </div>
        <div class="release">
          <div>
            <div style="color: var(--muted); text-transform: uppercase; letter-spacing: 0.16em; font-size: 12px;">Current release</div>
            <strong>v__VERSION__</strong>
          </div>
          <a href="https://github.com/2023Anita/turboquant-kvcache/releases/tag/v__VERSION__">Open release</a>
        </div>
      </article>
    </section>

    <div class="footer">
      Built from the <code>2023Anita/turboquant-kvcache</code> repository.
    </div>
  </main>
</body>
</html>
"""
    html = html.replace("__BENCHMARK_ROWS__", benchmark_rows)
    html = html.replace("__VERSION__", version)
    (SITE / "index.html").write_text(html, encoding="utf-8")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")


def main() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True, exist_ok=True)
    copy_assets()
    build_report()
    build_index()
    print(f"wrote {SITE}")


if __name__ == "__main__":
    main()
