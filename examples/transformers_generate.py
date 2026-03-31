from __future__ import annotations

import argparse

import torch

from turboquant_kvcache import TurboQuantTransformersRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Hugging Face generation example with TurboQuant KV-cache storage")
    parser.add_argument("--model-id", type=str, required=True, help="A decoder-only causal LM on Hugging Face")
    parser.add_argument("--prompt", type=str, default="Explain KV-cache quantization in one paragraph.")
    parser.add_argument("--bits", type=float, default=4.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is not installed. Install it first, for example:\n"
            "python3 -m pip install transformers"
        ) from exc

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    runner = TurboQuantTransformersRunner.from_model_config(
        model.config,
        bits=args.bits,
        seed=args.seed,
        device=device,
        dtype=torch.float32,
    )

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids, device=device))

    state = None
    generated_ids = input_ids
    next_input_ids = input_ids

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            model_inputs = {"input_ids": next_input_ids, "attention_mask": attention_mask}
            outputs, state = runner.step(model, model_inputs, state=state)
            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if tokenizer.eos_token_id is not None and bool((next_token == tokenizer.eos_token_id).all()):
                break
            next_input_ids = next_token
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )

    print(f"model_id            : {args.model_id}")
    print(f"turboquant bits     : {args.bits}")
    print(f"cache compression   : {runner.compression_ratio():.2f}x versus fp16")
    print("---")
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
