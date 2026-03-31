from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from ..kv_cache import QuantizedKVCache, TurboQuantKVCacheCodec

LegacyLayerCache = tuple[torch.Tensor, torch.Tensor]
LegacyPastKeyValues = tuple[LegacyLayerCache, ...]


@dataclass
class TurboQuantTransformersState:
    layers: tuple[QuantizedKVCache, ...]
    cache_cls: Optional[type[Any]] = None

    def num_layers(self) -> int:
        return len(self.layers)


class TurboQuantTransformersRunner:
    """
    A minimal Hugging Face integration path built around the documented
    `past_key_values` contract and the legacy cache format.

    It is intended for custom generation loops where you want to store the cache
    in TurboQuant form between decoding steps, then dequantize it immediately
    before the next model forward pass.
    """

    def __init__(
        self,
        head_dim: int,
        bits: float = 4.0,
        *,
        seed: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.codec = TurboQuantKVCacheCodec(head_dim, bits=bits, seed=seed, device=device, dtype=dtype)
        self.head_dim = head_dim
        self.bits = bits
        self.seed = seed
        self.device = device or torch.device("cpu")
        self.dtype = dtype

    @classmethod
    def from_model_config(
        cls,
        config: Any,
        bits: float = 4.0,
        *,
        seed: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "TurboQuantTransformersRunner":
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(config, "hidden_size", None)
            num_heads = getattr(config, "num_key_value_heads", None) or getattr(config, "num_attention_heads", None)
            if hidden_size is None or num_heads is None:
                raise ValueError("could not infer head_dim from config; provide it explicitly")
            head_dim = hidden_size // num_heads
        return cls(head_dim, bits=bits, seed=seed, device=device, dtype=dtype)

    def compression_ratio(self) -> float:
        return self.codec.compression_ratio()

    def _to_legacy(self, past_key_values: Any) -> tuple[LegacyPastKeyValues, Optional[type[Any]]]:
        if isinstance(past_key_values, tuple):
            return past_key_values, None
        if hasattr(past_key_values, "to_legacy_cache"):
            return past_key_values.to_legacy_cache(), past_key_values.__class__
        raise TypeError(
            "Unsupported cache format. Expected a legacy tuple cache or a cache object "
            "implementing to_legacy_cache()."
        )

    def _from_legacy(self, legacy_cache: LegacyPastKeyValues, cache_cls: Optional[type[Any]]) -> Any:
        if cache_cls is not None and hasattr(cache_cls, "from_legacy_cache"):
            return cache_cls.from_legacy_cache(legacy_cache)
        return legacy_cache

    def quantize_past_key_values(self, past_key_values: Any) -> TurboQuantTransformersState:
        legacy_cache, cache_cls = self._to_legacy(past_key_values)
        layers = tuple(self.codec.encode(key_states, value_states) for key_states, value_states in legacy_cache)
        return TurboQuantTransformersState(layers=layers, cache_cls=cache_cls)

    def dequantize_past_key_values(self, state: TurboQuantTransformersState) -> Any:
        legacy_cache = tuple(self.codec.decode(layer) for layer in state.layers)
        return self._from_legacy(legacy_cache, state.cache_cls)

    def step(
        self,
        model: Any,
        model_inputs: dict[str, Any],
        *,
        state: TurboQuantTransformersState | None = None,
        use_cache: bool = True,
        **forward_kwargs: Any,
    ) -> tuple[Any, TurboQuantTransformersState | None]:
        inputs = dict(model_inputs)
        if state is not None:
            inputs["past_key_values"] = self.dequantize_past_key_values(state)
        outputs = model(**inputs, use_cache=use_cache, **forward_kwargs)
        past_key_values = getattr(outputs, "past_key_values", None)
        next_state = self.quantize_past_key_values(past_key_values) if past_key_values is not None else None
        return outputs, next_state

    def quantize_model_cache(
        self,
        outputs: Any,
    ) -> tuple[Any, TurboQuantTransformersState | None]:
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None:
            return outputs, None
        return outputs, self.quantize_past_key_values(past_key_values)
