from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
ASSETS = ROOT / "assets"


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


def build_index() -> None:
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
      </article>
    </section>

    <div class="footer">
      Built from the <code>2023Anita/turboquant-kvcache</code> repository.
    </div>
  </main>
</body>
</html>
"""
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
