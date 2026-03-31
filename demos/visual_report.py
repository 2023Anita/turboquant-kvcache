from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turboquant_kvcache import TurboQuantMSEQuantizer, UniformAffineQuantizer


def attention_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(query.shape[-1])
    return torch.matmul(query, key.transpose(-1, -2)) * scale


def attention_weights(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    return torch.softmax(attention_scores(query, key), dim=-1)


def attention_output(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    weights = attention_weights(query, key)
    return torch.matmul(weights, value)


def tensor_to_nested_list(tensor: torch.Tensor, digits: int = 6) -> list:
    rounded = torch.round(tensor.detach().cpu() * (10**digits)) / (10**digits)
    return rounded.tolist()


def build_report_data(
    *,
    seed: int,
    batch: int,
    heads: int,
    query_len: int,
    seq_len: int,
    head_dim: int,
    device: torch.device,
    bits_list: list[float],
    focus_bits: float,
) -> dict:
    torch.manual_seed(seed)

    cache_shape = (batch, heads, seq_len, head_dim)
    query_shape = (batch, heads, query_len, head_dim)

    key_cache = torch.randn(cache_shape, device=device, dtype=torch.float32)
    value_cache = torch.randn(cache_shape, device=device, dtype=torch.float32)
    query = torch.randn(query_shape, device=device, dtype=torch.float32)

    ref_weights = attention_weights(query, key_cache)
    ref_output = attention_output(query, key_cache, value_cache)

    metrics = []
    focus_payload = None

    for bits in bits_list:
        turbo = TurboQuantMSEQuantizer(head_dim, bits=bits, seed=seed, device=device)
        uniform = UniformAffineQuantizer(bits=max(2, int(round(bits))))

        key_pack = turbo.quantize(key_cache)
        value_pack = turbo.quantize(value_cache)
        key_turbo = turbo.dequantize(key_pack)
        value_turbo = turbo.dequantize(value_pack)

        key_uniform = uniform.quantize_dequantize(key_cache)
        value_uniform = uniform.quantize_dequantize(value_cache)

        turbo_weights = attention_weights(query, key_turbo)
        uniform_weights = attention_weights(query, key_uniform)
        turbo_output = torch.matmul(turbo_weights, value_turbo)
        uniform_output = torch.matmul(uniform_weights, value_uniform)

        key_mse = torch.mean((key_turbo - key_cache) ** 2).item()
        value_mse = torch.mean((value_turbo - value_cache) ** 2).item()
        uniform_key_mse = torch.mean((key_uniform - key_cache) ** 2).item()
        uniform_value_mse = torch.mean((value_uniform - value_cache) ** 2).item()
        output_rmse = torch.sqrt(torch.mean((turbo_output - ref_output) ** 2)).item()
        uniform_output_rmse = torch.sqrt(torch.mean((uniform_output - ref_output) ** 2)).item()
        weight_l1 = torch.mean(torch.abs(turbo_weights - ref_weights)).item()
        uniform_weight_l1 = torch.mean(torch.abs(uniform_weights - ref_weights)).item()

        raw_bits_per_vector = 16 * head_dim
        turbo_bits_per_vector = bits * head_dim + 32
        compression = raw_bits_per_vector / turbo_bits_per_vector

        metrics.append(
            {
                "bits": bits,
                "levels": turbo.levels,
                "compression": compression,
                "key_mse": key_mse,
                "value_mse": value_mse,
                "output_rmse": output_rmse,
                "weight_l1": weight_l1,
                "uniform_key_mse": uniform_key_mse,
                "uniform_value_mse": uniform_value_mse,
                "uniform_output_rmse": uniform_output_rmse,
                "uniform_weight_l1": uniform_weight_l1,
            }
        )

        if abs(bits - focus_bits) < 1e-6:
            head_index = 0
            ref_head_weights = ref_weights[0, head_index]
            turbo_head_weights = turbo_weights[0, head_index]
            uniform_head_weights = uniform_weights[0, head_index]

            usage = torch.bincount(key_pack.indices.reshape(-1).cpu(), minlength=turbo.levels).float()
            usage = (usage / usage.sum()).tolist()

            token_cos = torch.nn.functional.cosine_similarity(
                key_cache[0, head_index], key_turbo[0, head_index], dim=-1
            )

            focus_payload = {
                "bits": bits,
                "levels": turbo.levels,
                "head_index": head_index,
                "ref_weights": tensor_to_nested_list(ref_head_weights, digits=5),
                "turbo_weights": tensor_to_nested_list(turbo_head_weights, digits=5),
                "uniform_weights": tensor_to_nested_list(uniform_head_weights, digits=5),
                "token_cosine": tensor_to_nested_list(token_cos, digits=5),
                "code_usage": [round(v, 6) for v in usage],
                "codebook": [round(v, 6) for v in turbo.codebook.detach().cpu().tolist()],
            }

    best = min(metrics, key=lambda item: item["output_rmse"])
    best_uniform = min(metrics, key=lambda item: item["uniform_output_rmse"])

    return {
        "meta": {
            "seed": seed,
            "batch": batch,
            "heads": heads,
            "query_len": query_len,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "device": str(device),
            "focus_bits": focus_bits,
        },
        "summary": {
            "best_bits": best["bits"],
            "best_output_rmse": best["output_rmse"],
            "best_uniform_bits": best_uniform["bits"],
            "best_uniform_output_rmse": best_uniform["uniform_output_rmse"],
        },
        "metrics": metrics,
        "focus": focus_payload,
    }


def build_html(report: dict) -> str:
    report_json = json.dumps(report, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TurboQuant Visual Demo</title>
  <style>
    :root {{
      --bg: #f2efe8;
      --panel: rgba(255, 250, 243, 0.92);
      --ink: #161515;
      --muted: #625b53;
      --line: rgba(22, 21, 21, 0.12);
      --turbo: #d84b31;
      --uniform: #2e5bff;
      --accent: #f4bb44;
      --shadow: 0 22px 70px rgba(24, 18, 10, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "PingFang SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(216, 75, 49, 0.18), transparent 28%),
        radial-gradient(circle at 85% 12%, rgba(46, 91, 255, 0.16), transparent 30%),
        linear-gradient(180deg, #f7f3eb 0%, #ece7dd 100%);
      min-height: 100vh;
    }}
    .grain {{
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: 0.12;
      background-image:
        linear-gradient(transparent 0, rgba(0, 0, 0, 0.02) 50%, transparent 100%),
        linear-gradient(90deg, rgba(0, 0, 0, 0.03) 0, transparent 18%, rgba(0, 0, 0, 0.02) 100%);
      background-size: 100% 5px, 7px 100%;
      mix-blend-mode: multiply;
    }}
    main {{
      width: min(1220px, calc(100vw - 32px));
      margin: 28px auto 40px;
      position: relative;
      z-index: 1;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.3fr 0.9fr;
      gap: 18px;
      align-items: stretch;
      margin-bottom: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid rgba(22, 21, 21, 0.08);
      box-shadow: var(--shadow);
      border-radius: 28px;
      overflow: hidden;
      position: relative;
    }}
    .hero-copy {{
      padding: 30px 30px 26px;
      min-height: 280px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.24em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 18px;
    }}
    h1 {{
      font-family: "Iowan Old Style", "Times New Roman", serif;
      font-size: clamp(42px, 6vw, 86px);
      line-height: 0.95;
      margin: 0 0 14px;
      font-weight: 700;
      max-width: 8ch;
    }}
    .lead {{
      font-size: 18px;
      line-height: 1.55;
      color: #312c27;
      max-width: 58ch;
      margin: 0;
    }}
    .hero-note {{
      margin-top: 22px;
      display: inline-flex;
      gap: 12px;
      align-items: center;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(22, 21, 21, 0.04);
      font-size: 13px;
      color: var(--muted);
      width: fit-content;
    }}
    .summary {{
      padding: 0;
      display: grid;
      grid-template-rows: auto 1fr;
    }}
    .summary-top {{
      padding: 22px 22px 12px;
      border-bottom: 1px solid var(--line);
    }}
    .summary-top h2 {{
      margin: 0;
      font-size: 14px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 600;
    }}
    .summary-body {{
      padding: 22px;
      display: grid;
      gap: 12px;
      align-content: start;
    }}
    .stat {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 12px;
      align-items: baseline;
      padding: 12px 0;
      border-bottom: 1px dashed rgba(22, 21, 21, 0.12);
    }}
    .stat:last-child {{ border-bottom: 0; }}
    .stat-value {{
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .stat-label {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .grid {{
      display: grid;
      gap: 18px;
    }}
    .two {{
      grid-template-columns: 1fr 1fr;
    }}
    .section {{
      padding: 22px;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 16px;
    }}
    .section-title {{
      margin: 0;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .section-copy {{
      margin: 4px 0 0;
      color: var(--muted);
      line-height: 1.5;
      font-size: 14px;
      max-width: 50ch;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      font-size: 13px;
      color: var(--muted);
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .legend i {{
      width: 12px;
      height: 12px;
      display: inline-block;
      border-radius: 999px;
    }}
    canvas {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(244,239,231,0.88));
      border: 1px solid rgba(22, 21, 21, 0.06);
    }}
    .meta-strip {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 10px;
      margin-top: 12px;
    }}
    .meta-pill {{
      background: rgba(22, 21, 21, 0.04);
      border-radius: 18px;
      padding: 12px 14px;
    }}
    .meta-pill .k {{
      display: block;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .meta-pill .v {{
      display: block;
      font-size: 16px;
      font-weight: 700;
    }}
    .footer {{
      margin-top: 18px;
      padding: 18px 22px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }}
    @media (max-width: 980px) {{
      .hero, .two, .meta-strip {{
        grid-template-columns: 1fr;
      }}
      h1 {{ max-width: none; }}
    }}
  </style>
</head>
<body>
  <div class="grain"></div>
  <main>
    <section class="hero">
      <article class="card hero-copy">
        <div>
          <div class="eyebrow">TurboQuant / KV Cache / Visual Demo</div>
          <h1>See the Quantization, not just the number.</h1>
          <p class="lead">
            这个报告把 TurboQuant 的 KV cache 量化效果拆成两层：上层看 bit 宽变化带来的整体误差趋势，下层看固定 bit 宽下 attention 分布被量化后发生了什么。
          </p>
        </div>
        <div class="hero-note" id="heroNote"></div>
      </article>
      <aside class="card summary">
        <div class="summary-top"><h2>Snapshot</h2></div>
        <div class="summary-body" id="summaryBody"></div>
      </aside>
    </section>

    <section class="grid two">
      <article class="card section">
        <div class="section-head">
          <div>
            <h3 class="section-title">Quality vs Bits</h3>
            <p class="section-copy">同样的随机 KV cache 和 query，上图比较 TurboQuant 与 uniform baseline 的输出误差。</p>
          </div>
          <div class="legend">
            <span><i style="background: var(--turbo)"></i>TurboQuant</span>
            <span><i style="background: var(--uniform)"></i>Uniform</span>
          </div>
        </div>
        <canvas id="qualityChart" width="560" height="340"></canvas>
      </article>
      <article class="card section">
        <div class="section-head">
          <div>
            <h3 class="section-title">Compression Tradeoff</h3>
            <p class="section-copy">压缩率来自 `fp16` cache 与 `bits * dim + 32bit norm` 的理论存储对比。</p>
          </div>
        </div>
        <canvas id="compressionChart" width="560" height="340"></canvas>
      </article>
    </section>

    <section class="grid two" style="margin-top: 18px;">
      <article class="card section">
        <div class="section-head">
          <div>
            <h3 class="section-title">Attention Map</h3>
            <p class="section-copy">固定 `focus bit` 的一个 head 上，展示原始 attention 权重与量化后的分布。</p>
          </div>
          <div class="legend">
            <span><i style="background: linear-gradient(135deg, #fff1c9, #d84b31)"></i>Heat intensity</span>
          </div>
        </div>
        <canvas id="heatmapChart" width="560" height="360"></canvas>
      </article>
      <article class="card section">
        <div class="section-head">
          <div>
            <h3 class="section-title">Vector Fidelity</h3>
            <p class="section-copy">量化后每个 token 的 key 向量与原向量的 cosine similarity，以及代码本使用分布。</p>
          </div>
        </div>
        <canvas id="tokenChart" width="560" height="170"></canvas>
        <div style="height: 14px;"></div>
        <canvas id="usageChart" width="560" height="170"></canvas>
      </article>
    </section>

    <section class="card section" style="margin-top: 18px;">
      <div class="section-head">
        <div>
          <h3 class="section-title">Run Metadata</h3>
          <p class="section-copy">这份报告是脚本本地生成的，数据已经嵌入 HTML，可以直接离线打开。</p>
        </div>
      </div>
      <div class="meta-strip" id="metaStrip"></div>
    </section>

    <div class="footer">
      文件是静态 HTML，不依赖 matplotlib。重新生成只需要再次运行 `python3 visualize_demo.py`。
    </div>
  </main>

  <script>
    const report = {report_json};

    const turbo = getComputedStyle(document.documentElement).getPropertyValue('--turbo').trim();
    const uniform = getComputedStyle(document.documentElement).getPropertyValue('--uniform').trim();
    const ink = getComputedStyle(document.documentElement).getPropertyValue('--ink').trim();
    const muted = getComputedStyle(document.documentElement).getPropertyValue('--muted').trim();

    function format(v, digits = 4) {{
      return Number(v).toFixed(digits);
    }}

    function cardStat(label, value, suffix = '') {{
      return `<div class="stat"><div class="stat-value">${{value}}${{suffix}}</div><div class="stat-label">${{label}}</div></div>`;
    }}

    function mountSummary() {{
      const s = report.summary;
      const metrics = report.metrics;
      const focus = report.focus;
      const focusMetric = metrics.find(item => Math.abs(item.bits - focus.bits) < 1e-6);
      document.getElementById('heroNote').textContent =
        `Focus bit = ${{focus.bits}}, best TurboQuant output RMSE = ${{format(s.best_output_rmse, 5)}}`;
      document.getElementById('summaryBody').innerHTML =
        cardStat('Best TurboQuant bit width for output RMSE in this run.', format(s.best_bits, 1), ' bit') +
        cardStat('Best TurboQuant output RMSE.', format(s.best_output_rmse, 5)) +
        cardStat('Best uniform output RMSE across the same sweep.', format(s.best_uniform_output_rmse, 5)) +
        cardStat(`Focus bit (${{focus.bits}}) compression ratio.`, format(focusMetric.compression, 2), 'x');
    }}

    function mountMeta() {{
      const meta = report.meta;
      const entries = [
        ['seed', meta.seed],
        ['shape', `${{meta.batch}}×${{meta.heads}}×${{meta.seq_len}}×${{meta.head_dim}}`],
        ['query len', meta.query_len],
        ['device', meta.device],
        ['focus bit', meta.focus_bits]
      ];
      document.getElementById('metaStrip').innerHTML = entries.map(([k, v]) =>
        `<div class="meta-pill"><span class="k">${{k}}</span><span class="v">${{v}}</span></div>`
      ).join('');
    }}

    function drawAxes(ctx, x, y, w, h, yTicks, xLabels) {{
      ctx.strokeStyle = 'rgba(22,21,21,0.16)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x, y + h);
      ctx.lineTo(x + w, y + h);
      ctx.stroke();
      ctx.fillStyle = muted;
      ctx.font = '12px Menlo, monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      yTicks.forEach(tick => {{
        const py = y + h - tick.ratio * h;
        ctx.strokeStyle = 'rgba(22,21,21,0.08)';
        ctx.beginPath();
        ctx.moveTo(x, py);
        ctx.lineTo(x + w, py);
        ctx.stroke();
        ctx.fillText(tick.label, x - 8, py);
      }});
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      xLabels.forEach((label, i) => {{
        const px = x + (i / Math.max(1, xLabels.length - 1)) * w;
        ctx.fillText(label, px, y + h + 8);
      }});
    }}

    function drawLineChart(canvas, seriesA, seriesB, yLabel) {{
      const ctx = canvas.getContext('2d');
      const pad = {{ l: 54, r: 20, t: 20, b: 42 }};
      const w = canvas.width - pad.l - pad.r;
      const h = canvas.height - pad.t - pad.b;
      const xs = report.metrics.map(m => m.bits);
      const all = [...seriesA, ...seriesB];
      const minY = Math.min(...all) * 0.92;
      const maxY = Math.max(...all) * 1.08;
      const yTicks = Array.from({{ length: 4 }}, (_, i) => {{
        const ratio = i / 3;
        const value = minY + (maxY - minY) * ratio;
        return {{ ratio, label: format(value, 3) }};
      }});
      drawAxes(ctx, pad.l, pad.t, w, h, yTicks, xs.map(v => `${{v}}b`));
      ctx.save();
      ctx.translate(pad.l, pad.t);
      function plot(values, color) {{
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        values.forEach((value, i) => {{
          const px = (i / Math.max(1, values.length - 1)) * w;
          const py = h - ((value - minY) / (maxY - minY || 1)) * h;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }});
        ctx.stroke();
        values.forEach((value, i) => {{
          const px = (i / Math.max(1, values.length - 1)) * w;
          const py = h - ((value - minY) / (maxY - minY || 1)) * h;
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(px, py, 4.5, 0, Math.PI * 2);
          ctx.fill();
        }});
      }}
      plot(seriesA, turbo);
      plot(seriesB, uniform);
      ctx.restore();
      ctx.fillStyle = muted;
      ctx.font = '12px Menlo, monospace';
      ctx.textAlign = 'left';
      ctx.fillText(yLabel, pad.l, 14);
    }}

    function drawBarChart(canvas, values, labels, colors, yLabel) {{
      const ctx = canvas.getContext('2d');
      const pad = {{ l: 54, r: 20, t: 20, b: 42 }};
      const w = canvas.width - pad.l - pad.r;
      const h = canvas.height - pad.t - pad.b;
      const maxY = Math.max(...values) * 1.15;
      const yTicks = Array.from({{ length: 4 }}, (_, i) => {{
        const ratio = i / 3;
        return {{ ratio, label: format(maxY * ratio, 2) }};
      }});
      drawAxes(ctx, pad.l, pad.t, w, h, yTicks, labels);
      const barW = w / values.length * 0.58;
      values.forEach((value, i) => {{
        const px = pad.l + (i + 0.5) * (w / values.length) - barW / 2;
        const barH = (value / (maxY || 1)) * h;
        const py = pad.t + h - barH;
        ctx.fillStyle = colors[i];
        ctx.fillRect(px, py, barW, barH);
      }});
      ctx.fillStyle = muted;
      ctx.font = '12px Menlo, monospace';
      ctx.fillText(yLabel, pad.l, 14);
    }}

    function drawHeatmap(canvas, matrices, titles) {{
      const ctx = canvas.getContext('2d');
      const margin = 16;
      const gap = 16;
      const columnWidth = (canvas.width - margin * 2 - gap * 2) / 3;
      const rows = matrices[0].length;
      const cols = matrices[0][0].length;
      const cellW = columnWidth / cols;
      const cellH = (canvas.height - 52) / rows;
      const allValues = matrices.flat(2);
      const maxV = Math.max(...allValues);
      const minV = Math.min(...allValues);
      function colorFor(v) {{
        const t = (v - minV) / (maxV - minV || 1);
        const hue = 36 - t * 28;
        const sat = 92 - t * 26;
        const light = 92 - t * 54;
        return `hsl(${{hue}} ${{sat}}% ${{light}}%)`;
      }}
      titles.forEach((title, block) => {{
        const x0 = margin + block * (columnWidth + gap);
        ctx.fillStyle = ink;
        ctx.font = 'bold 13px Menlo, monospace';
        ctx.fillText(title, x0, 18);
        matrices[block].forEach((row, r) => {{
          row.forEach((value, c) => {{
            ctx.fillStyle = colorFor(value);
            ctx.fillRect(x0 + c * cellW, 30 + r * cellH, Math.ceil(cellW), Math.ceil(cellH));
          }});
        }});
        ctx.strokeStyle = 'rgba(22,21,21,0.08)';
        ctx.strokeRect(x0, 30, columnWidth, rows * cellH);
      }});
    }}

    function drawSparkBars(canvas, values, color, yLabel) {{
      const ctx = canvas.getContext('2d');
      const pad = {{ l: 46, r: 12, t: 16, b: 28 }};
      const w = canvas.width - pad.l - pad.r;
      const h = canvas.height - pad.t - pad.b;
      const maxV = Math.max(...values) * 1.02;
      ctx.strokeStyle = 'rgba(22,21,21,0.12)';
      ctx.beginPath();
      ctx.moveTo(pad.l, pad.t);
      ctx.lineTo(pad.l, pad.t + h);
      ctx.lineTo(pad.l + w, pad.t + h);
      ctx.stroke();
      const barW = w / values.length;
      values.forEach((v, i) => {{
        const bh = (v / (maxV || 1)) * h;
        ctx.fillStyle = color;
        ctx.fillRect(pad.l + i * barW + 1, pad.t + h - bh, Math.max(1, barW - 2), bh);
      }});
      ctx.fillStyle = muted;
      ctx.font = '12px Menlo, monospace';
      ctx.fillText(yLabel, pad.l, 12);
    }}

    function render() {{
      mountSummary();
      mountMeta();
      const metrics = report.metrics;
      drawLineChart(
        document.getElementById('qualityChart'),
        metrics.map(m => m.output_rmse),
        metrics.map(m => m.uniform_output_rmse),
        'output rmse'
      );
      drawBarChart(
        document.getElementById('compressionChart'),
        metrics.map(m => m.compression),
        metrics.map(m => `${{m.bits}}b`),
        metrics.map((_, i) => i === 2 ? turbo : '#23201c'),
        'compression'
      );
      drawHeatmap(
        document.getElementById('heatmapChart'),
        [report.focus.ref_weights, report.focus.turbo_weights, report.focus.uniform_weights],
        ['Reference', 'TurboQuant', 'Uniform']
      );
      drawSparkBars(
        document.getElementById('tokenChart'),
        report.focus.token_cosine,
        turbo,
        'token cosine similarity'
      );
      drawSparkBars(
        document.getElementById('usageChart'),
        report.focus.code_usage,
        '#181818',
        'codebook usage'
      );
    }}

    render();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a self-contained TurboQuant HTML report")
    parser.add_argument("--output", type=str, default="report.html")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--query-len", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--focus-bits", type=float, default=4.0)
    parser.add_argument("--bits-list", type=str, default="2,3,4,5,6,8")
    args = parser.parse_args()

    bits_list = [float(item.strip()) for item in args.bits_list.split(",") if item.strip()]
    if args.focus_bits not in bits_list:
        bits_list.append(args.focus_bits)
        bits_list = sorted(bits_list)

    device = torch.device(args.device)
    report = build_report_data(
        seed=args.seed,
        batch=args.batch,
        heads=args.heads,
        query_len=args.query_len,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        device=device,
        bits_list=bits_list,
        focus_bits=args.focus_bits,
    )

    html = build_html(report)
    output = Path(args.output)
    output.write_text(html, encoding="utf-8")
    print(f"wrote {output.resolve()}")


if __name__ == "__main__":
    main()
