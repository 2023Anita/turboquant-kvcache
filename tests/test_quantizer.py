from __future__ import annotations

import torch

from turboquant_kvcache import (
    TurboQuantInnerProductQuantizer,
    TurboQuantKVCacheCodec,
    TurboQuantMSEQuantizer,
    build_lloyd_max_codebook,
)


def test_lloyd_max_codebook_is_sorted() -> None:
    codebook = build_lloyd_max_codebook(dim=64, levels=16)
    assert len(codebook) == 16
    assert all(codebook[i] < codebook[i + 1] for i in range(len(codebook) - 1))


def test_mse_quantizer_roundtrip_has_expected_shape() -> None:
    torch.manual_seed(0)
    tensor = torch.randn(2, 4, 32, 64)
    quantizer = TurboQuantMSEQuantizer(64, bits=4.0, seed=0)
    pack = quantizer.quantize(tensor)
    restored = quantizer.dequantize(pack)
    assert restored.shape == tensor.shape
    assert torch.isfinite(restored).all()


def test_inner_product_estimator_is_finite() -> None:
    torch.manual_seed(0)
    key = torch.randn(1, 2, 16, 32)
    query = torch.randn(1, 2, 16, 32)
    quantizer = TurboQuantInnerProductQuantizer(32, bits=4.0, seed=0)
    pack = quantizer.quantize(key)
    estimate = quantizer.estimate_inner_product(pack, query)
    assert estimate.shape == key.shape[:-1]
    assert torch.isfinite(estimate).all()


def test_kv_cache_codec_beats_uniform_on_reference_seed() -> None:
    torch.manual_seed(0)
    key = torch.randn(1, 4, 64, 64)
    value = torch.randn(1, 4, 64, 64)
    query = torch.randn(1, 4, 1, 64)
    codec = TurboQuantKVCacheCodec(64, bits=4.0, seed=0)
    metrics = codec.evaluate(query, key, value)
    assert metrics["key_mse"] < metrics["uniform_key_mse"]
    assert metrics["value_mse"] < metrics["uniform_value_mse"]
