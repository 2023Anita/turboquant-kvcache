from __future__ import annotations

import argparse

import torch

from turboquant_kvcache import TurboQuantKVCacheCodec


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic TurboQuant distortion benchmark")
    parser.add_argument("--bits-list", type=str, default="2,3,4,5,6,8")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    shape = (args.batch, args.heads, args.seq_len, args.head_dim)
    key = torch.randn(shape, device=device, dtype=torch.float32)
    value = torch.randn(shape, device=device, dtype=torch.float32)
    query = torch.randn((args.batch, args.heads, 1, args.head_dim), device=device, dtype=torch.float32)

    bits_list = [float(item.strip()) for item in args.bits_list.split(",") if item.strip()]
    rows = []
    for bits in bits_list:
        codec = TurboQuantKVCacheCodec(args.head_dim, bits=bits, seed=args.seed, device=device)
        metrics = codec.evaluate(query, key, value)
        rows.append((bits, metrics))

    print("| bits | compression | key_mse | value_mse | attn_rmse | uniform_attn_rmse |")
    print("| --- | --- | --- | --- | --- | --- |")
    for bits, metrics in rows:
        print(
            f"| {bits:.1f} | {metrics['compression_ratio']:.2f}x | "
            f"{metrics['key_mse']:.6f} | {metrics['value_mse']:.6f} | "
            f"{metrics['attention_output_rmse']:.6f} | {metrics['uniform_output_rmse']:.6f} |"
        )


if __name__ == "__main__":
    main()
