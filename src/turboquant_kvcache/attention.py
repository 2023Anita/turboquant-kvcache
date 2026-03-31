from __future__ import annotations

import math

import torch


def attention_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(query.shape[-1])
    return torch.matmul(query, key.transpose(-1, -2)) * scale


def attention_weights(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    return torch.softmax(attention_scores(query, key), dim=-1)


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return torch.matmul(attention_weights(query, key), value)
