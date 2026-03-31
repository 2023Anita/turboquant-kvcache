from __future__ import annotations

from dataclasses import dataclass

import torch

from turboquant_kvcache import TurboQuantTransformersRunner


@dataclass
class FakeOutputs:
    logits: torch.Tensor
    past_key_values: object


class FakeCache:
    def __init__(self, legacy_cache):
        self._legacy_cache = legacy_cache

    def to_legacy_cache(self):
        return self._legacy_cache

    @classmethod
    def from_legacy_cache(cls, legacy_cache):
        return cls(legacy_cache)


class FakeModel:
    def __init__(self, layer_count: int = 2, heads: int = 4, head_dim: int = 16):
        self.layer_count = layer_count
        self.heads = heads
        self.head_dim = head_dim

    def __call__(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=True):
        batch = input_ids.shape[0]
        seq = input_ids.shape[1]
        if past_key_values is None:
            past = tuple(
                (
                    torch.randn(batch, self.heads, seq, self.head_dim),
                    torch.randn(batch, self.heads, seq, self.head_dim),
                )
                for _ in range(self.layer_count)
            )
        else:
            if hasattr(past_key_values, "to_legacy_cache"):
                past = past_key_values.to_legacy_cache()
            else:
                past = past_key_values
        logits = torch.randn(batch, seq, 32)
        return FakeOutputs(logits=logits, past_key_values=past)


def test_runner_quantizes_legacy_cache() -> None:
    torch.manual_seed(0)
    runner = TurboQuantTransformersRunner(head_dim=16, bits=4.0, seed=0)
    legacy = tuple((torch.randn(1, 4, 3, 16), torch.randn(1, 4, 3, 16)) for _ in range(2))
    state = runner.quantize_past_key_values(legacy)
    restored = runner.dequantize_past_key_values(state)
    assert len(restored) == len(legacy)
    assert restored[0][0].shape == legacy[0][0].shape


def test_runner_roundtrips_cache_object() -> None:
    torch.manual_seed(0)
    runner = TurboQuantTransformersRunner(head_dim=16, bits=4.0, seed=0)
    legacy = tuple((torch.randn(1, 4, 3, 16), torch.randn(1, 4, 3, 16)) for _ in range(2))
    state = runner.quantize_past_key_values(FakeCache(legacy))
    restored = runner.dequantize_past_key_values(state)
    assert isinstance(restored, FakeCache)
    assert len(restored.to_legacy_cache()) == 2


def test_runner_step_returns_quantized_state() -> None:
    torch.manual_seed(0)
    model = FakeModel()
    runner = TurboQuantTransformersRunner(head_dim=16, bits=4.0, seed=0)
    outputs, state = runner.step(
        model,
        {"input_ids": torch.ones(1, 3, dtype=torch.long), "attention_mask": torch.ones(1, 3, dtype=torch.long)},
    )
    assert outputs.logits.shape[:2] == (1, 3)
    assert state is not None
    assert state.num_layers() == 2
