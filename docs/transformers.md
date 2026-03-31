# Transformers Integration

This repository now includes a minimal Hugging Face integration helper in `turboquant_kvcache.integrations.transformers`.

## Scope

The integration is intentionally narrow:

- it targets custom generation loops
- it stores `past_key_values` in TurboQuant form between decoding steps
- it dequantizes immediately before the next model forward pass
- it keeps `transformers` as an optional dependency

This keeps the implementation aligned with the documented cache API while avoiding premature commitment to a fast-path kernel design.

## Why This Shape

According to the Hugging Face caching documentation, a cache is updated through the `update(key_states, value_states, layer_idx)` flow, and cache objects can be converted to and from the legacy tuple format with `to_legacy_cache()` and `from_legacy_cache()`.

Source:

- <https://huggingface.co/docs/transformers/main/cache_explanation>
- <https://huggingface.co/docs/transformers/v4.53.3/cache_explanation>

This repository uses that documented legacy-cache conversion path as the most stable minimal integration point.

## Example

```python
from turboquant_kvcache import TurboQuantTransformersRunner

runner = TurboQuantTransformersRunner.from_model_config(model.config, bits=4.0, seed=0, device=model.device)

state = None
inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
for _ in range(8):
    outputs, state = runner.step(model, inputs, state=state)
    next_token = outputs.logits[:, -1:].argmax(-1)
    attention_mask = inputs["attention_mask"]
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
    inputs = {"input_ids": next_token, "attention_mask": attention_mask}
```

## Limitations

- This is not yet a true in-model `Cache` subclass.
- It is designed for clarity, not for peak decode throughput.
- Beam-search cache reordering and model-specific cache subclasses are not implemented yet.
