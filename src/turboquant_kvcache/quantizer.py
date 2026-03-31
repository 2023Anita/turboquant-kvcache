from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def _haar_rotation(dim: int, *, device: torch.device, dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    gaussian = torch.randn((dim, dim), generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(gaussian)
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    q = q * signs.unsqueeze(0)
    return q.to(device=device, dtype=dtype)


def _sphere_coordinate_pdf(grid: np.ndarray, dim: int) -> np.ndarray:
    if dim < 2:
        raise ValueError("dim must be >= 2")
    log_c = math.lgamma(dim / 2.0) - 0.5 * math.log(math.pi) - math.lgamma((dim - 1.0) / 2.0)
    base = np.clip(1.0 - grid**2, 0.0, None)
    with np.errstate(divide="ignore"):
        log_pdf = log_c + ((dim - 3.0) / 2.0) * np.log(base)
    pdf = np.exp(log_pdf)
    pdf[~np.isfinite(pdf)] = 0.0
    return pdf


def _cumtrapz(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    diffs = np.diff(grid)
    avg = 0.5 * (values[1:] + values[:-1])
    return np.concatenate(([0.0], np.cumsum(diffs * avg)))


def build_lloyd_max_codebook(
    dim: int,
    levels: int,
    *,
    max_iters: int = 128,
    tol: float = 1e-8,
    grid_size: int = 20001,
) -> np.ndarray:
    if levels < 2:
        raise ValueError("levels must be >= 2")

    grid = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    pdf = _sphere_coordinate_pdf(grid, dim)
    mass_cdf = _cumtrapz(pdf, grid)
    moment_cdf = _cumtrapz(pdf * grid, grid)
    total_mass = mass_cdf[-1]
    mass_cdf /= total_mass
    moment_cdf /= total_mass

    quantiles = np.linspace(0.0, 1.0, levels + 1)
    boundaries = np.interp(quantiles, mass_cdf, grid)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0

    def integral(cdf: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_vals = np.interp(left, grid, cdf)
        right_vals = np.interp(right, grid, cdf)
        return right_vals - left_vals

    centroids = np.zeros(levels, dtype=np.float64)
    for _ in range(max_iters):
        prev = centroids.copy()
        left = boundaries[:-1]
        right = boundaries[1:]
        masses = integral(mass_cdf, left, right)
        moments = integral(moment_cdf, left, right)
        masses = np.maximum(masses, 1e-12)
        centroids = moments / masses
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        if np.max(np.abs(centroids - prev)) < tol:
            break
    return centroids.astype(np.float32)


@dataclass
class QuantizedTensor:
    indices: torch.Tensor
    norms: torch.Tensor
    shape: torch.Size
    rotation: torch.Tensor
    codebook: torch.Tensor
    residual_signs: Optional[torch.Tensor] = None
    residual_norms: Optional[torch.Tensor] = None
    residual_projection: Optional[torch.Tensor] = None

    @property
    def last_dim(self) -> int:
        return self.shape[-1]


class TurboQuantMSEQuantizer:
    def __init__(
        self,
        dim: int,
        bits: float = 4.0,
        *,
        seed: int = 0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.dim = dim
        self.bits = bits
        self.levels = max(2, int(round(2**bits)))
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.rotation = _haar_rotation(dim, device=self.device, dtype=dtype, seed=seed)
        codebook = build_lloyd_max_codebook(dim, self.levels)
        self.codebook = torch.tensor(codebook, device=self.device, dtype=dtype)
        self.boundaries = 0.5 * (self.codebook[:-1] + self.codebook[1:])

    def quantize(self, tensor: torch.Tensor) -> QuantizedTensor:
        if tensor.shape[-1] != self.dim:
            raise ValueError(f"expected last dim {self.dim}, got {tensor.shape[-1]}")
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        flat = tensor.reshape(-1, self.dim)
        norms = flat.norm(dim=-1, keepdim=True)
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        unit = flat / safe_norms
        rotated = unit @ self.rotation.T
        rotated = torch.clamp(rotated, -1.0, 1.0)
        indices = torch.bucketize(rotated, self.boundaries)
        return QuantizedTensor(
            indices=indices.reshape(tensor.shape),
            norms=norms.reshape(*tensor.shape[:-1], 1),
            shape=tensor.shape,
            rotation=self.rotation,
            codebook=self.codebook,
        )

    def dequantize(self, qt: QuantizedTensor) -> torch.Tensor:
        centers = qt.codebook[qt.indices.long()]
        flat = centers.reshape(-1, self.dim)
        restored = flat @ qt.rotation
        norms = qt.norms.reshape(-1, 1).to(restored.dtype)
        restored = restored * norms
        return restored.reshape(qt.shape)


class TurboQuantInnerProductQuantizer(TurboQuantMSEQuantizer):
    def __init__(
        self,
        dim: int,
        bits: float = 4.0,
        *,
        residual_dim: Optional[int] = None,
        seed: int = 0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(dim, bits, seed=seed, device=device, dtype=dtype)
        residual_dim = residual_dim or dim
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 1)
        gaussian = torch.randn((residual_dim, dim), generator=generator, dtype=torch.float64)
        gaussian /= math.sqrt(residual_dim)
        self.residual_projection = gaussian.to(device=self.device, dtype=self.dtype)

    def quantize(self, tensor: torch.Tensor) -> QuantizedTensor:
        qt = super().quantize(tensor)
        restored = super().dequantize(qt)
        residual = tensor.to(device=self.device, dtype=self.dtype) - restored
        flat = residual.reshape(-1, self.dim)
        residual_norms = flat.norm(dim=-1, keepdim=True)
        safe_norms = torch.where(residual_norms > 0, residual_norms, torch.ones_like(residual_norms))
        unit = flat / safe_norms
        signs = torch.sign(unit @ self.residual_projection.T)
        signs[signs == 0] = 1
        qt.residual_signs = signs.reshape(*tensor.shape[:-1], -1).to(torch.int8)
        qt.residual_norms = residual_norms.reshape(*tensor.shape[:-1], 1)
        qt.residual_projection = self.residual_projection
        return qt

    def estimate_inner_product(self, qt: QuantizedTensor, other: torch.Tensor) -> torch.Tensor:
        if qt.residual_signs is None or qt.residual_norms is None or qt.residual_projection is None:
            raise ValueError("quantized tensor does not contain residual state")
        approx = self.dequantize(qt)
        base = (approx * other.to(device=self.device, dtype=self.dtype)).sum(dim=-1)
        other_flat = other.to(device=self.device, dtype=self.dtype).reshape(-1, self.dim)
        projected = other_flat @ qt.residual_projection.T
        signed = qt.residual_signs.reshape(other_flat.shape[0], -1).to(projected.dtype)
        correction = math.sqrt(math.pi / 2.0) * (signed * projected).mean(dim=-1, keepdim=True)
        correction = correction * qt.residual_norms.reshape(-1, 1).to(projected.dtype)
        return (base.reshape(-1, 1) + correction).reshape(base.shape)


class UniformAffineQuantizer:
    def __init__(self, bits: int = 4) -> None:
        self.bits = bits
        self.levels = 2**bits

    def quantize_dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        max_abs = tensor.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(max_abs > 0, max_abs / ((self.levels - 1) / 2.0), torch.ones_like(max_abs))
        q = torch.round(tensor / scale).clamp(-(self.levels // 2), self.levels // 2 - 1)
        return q * scale
