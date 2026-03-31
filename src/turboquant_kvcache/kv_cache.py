from __future__ import annotations

from dataclasses import dataclass

import torch

from .attention import attention
from .quantizer import QuantizedTensor, TurboQuantMSEQuantizer, UniformAffineQuantizer


@dataclass
class QuantizedKVCache:
    key: QuantizedTensor
    value: QuantizedTensor


class TurboQuantKVCacheCodec:
    def __init__(
        self,
        head_dim: int,
        bits: float = 4.0,
        *,
        seed: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.head_dim = head_dim
        self.bits = bits
        self.device = device or torch.device("cpu")
        self.quantizer = TurboQuantMSEQuantizer(head_dim, bits=bits, seed=seed, device=self.device, dtype=dtype)
        self.uniform = UniformAffineQuantizer(bits=max(2, int(round(bits))))

    def encode(self, key_cache: torch.Tensor, value_cache: torch.Tensor) -> QuantizedKVCache:
        return QuantizedKVCache(
            key=self.quantizer.quantize(key_cache),
            value=self.quantizer.quantize(value_cache),
        )

    def decode(self, pack: QuantizedKVCache) -> tuple[torch.Tensor, torch.Tensor]:
        return self.quantizer.dequantize(pack.key), self.quantizer.dequantize(pack.value)

    def compression_ratio(self) -> float:
        raw_bits_per_vector = 16 * self.head_dim
        turbo_bits_per_vector = self.bits * self.head_dim + 32
        return raw_bits_per_vector / turbo_bits_per_vector

    def evaluate(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> dict[str, float]:
        pack = self.encode(key_cache, value_cache)
        key_hat, value_hat = self.decode(pack)
        key_uniform = self.uniform.quantize_dequantize(key_cache)
        value_uniform = self.uniform.quantize_dequantize(value_cache)

        reference = attention(query, key_cache, value_cache)
        turbo_out = attention(query, key_hat, value_hat)
        uniform_out = attention(query, key_uniform, value_uniform)

        return {
            "compression_ratio": self.compression_ratio(),
            "key_mse": torch.mean((key_hat - key_cache) ** 2).item(),
            "value_mse": torch.mean((value_hat - value_cache) ** 2).item(),
            "uniform_key_mse": torch.mean((key_uniform - key_cache) ** 2).item(),
            "uniform_value_mse": torch.mean((value_uniform - value_cache) ** 2).item(),
            "attention_output_rmse": torch.sqrt(torch.mean((turbo_out - reference) ** 2)).item(),
            "uniform_output_rmse": torch.sqrt(torch.mean((uniform_out - reference) ** 2)).item(),
        }
