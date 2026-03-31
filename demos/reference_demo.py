from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turboquant_kvcache import (
    TurboQuantInnerProductQuantizer,
    TurboQuantMSEQuantizer,
    UniformAffineQuantizer,
)

from turboquant_kvcache.attention import attention


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant KV cache demo")
    parser.add_argument("--bits", type=float, default=4.0, help="Target TurboQuant bit width")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    shape = (args.batch, args.heads, args.seq_len, args.head_dim)
    key_cache = torch.randn(shape, device=device, dtype=torch.float32)
    value_cache = torch.randn(shape, device=device, dtype=torch.float32)
    query = torch.randn((args.batch, args.heads, 1, args.head_dim), device=device, dtype=torch.float32)

    turbo = TurboQuantMSEQuantizer(args.head_dim, bits=args.bits, seed=args.seed, device=device)
    turbo_ip = TurboQuantInnerProductQuantizer(args.head_dim, bits=args.bits, seed=args.seed, device=device)
    uniform = UniformAffineQuantizer(bits=max(2, int(round(args.bits))))

    qk = turbo.quantize(key_cache)
    qv = turbo.quantize(value_cache)
    key_turbo = turbo.dequantize(qk)
    value_turbo = turbo.dequantize(qv)

    key_uniform = uniform.quantize_dequantize(key_cache)
    value_uniform = uniform.quantize_dequantize(value_cache)

    ref_out = attention(query, key_cache, value_cache)
    turbo_out = attention(query, key_turbo, value_turbo)
    uniform_out = attention(query, key_uniform, value_uniform)

    ip_pack = turbo_ip.quantize(key_cache)
    ip_exact = (query.expand_as(key_cache) * key_cache).sum(dim=-1)
    ip_turbo = turbo_ip.estimate_inner_product(ip_pack, query.expand_as(key_cache))

    mse_key = torch.mean((key_turbo - key_cache) ** 2).item()
    mse_value = torch.mean((value_turbo - value_cache) ** 2).item()
    mse_uniform_k = torch.mean((key_uniform - key_cache) ** 2).item()
    mse_uniform_v = torch.mean((value_uniform - value_cache) ** 2).item()

    out_rmse_turbo = torch.sqrt(torch.mean((turbo_out - ref_out) ** 2)).item()
    out_rmse_uniform = torch.sqrt(torch.mean((uniform_out - ref_out) ** 2)).item()
    ip_rmse = torch.sqrt(torch.mean((ip_turbo - ip_exact) ** 2)).item()

    raw_bits_per_vector = 16 * args.head_dim
    turbo_bits_per_vector = args.bits * args.head_dim + 32
    compression = raw_bits_per_vector / turbo_bits_per_vector

    print("TurboQuant KV Cache Demo")
    print(f"shape                 : {shape}")
    print(f"turbo levels          : {turbo.levels} (~{math.log2(turbo.levels):.3f} bits)")
    print(f"estimated compression : {compression:.2f}x versus fp16 cache")
    print(f"key mse               : {mse_key:.6f}")
    print(f"value mse             : {mse_value:.6f}")
    print(f"uniform key mse       : {mse_uniform_k:.6f}")
    print(f"uniform value mse     : {mse_uniform_v:.6f}")
    print(f"attn output rmse      : {out_rmse_turbo:.6f}")
    print(f"uniform output rmse   : {out_rmse_uniform:.6f}")
    print(f"ip estimator rmse     : {ip_rmse:.6f}")


if __name__ == "__main__":
    main()
