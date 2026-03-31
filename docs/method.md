# Method

This repository implements the two algorithmic paths highlighted in the TurboQuant paper:

1. `MSE` path
   - Normalize a vector into norm and direction.
   - Apply a random orthogonal rotation.
   - Quantize each rotated coordinate with a Lloyd-Max scalar quantizer tuned to the sphere-coordinate distribution.
2. `Inner-product` path
   - Reuse the MSE quantizer as the base approximation.
   - Sketch the residual with a 1-bit random projection in the style of QJL.

For KV-cache tensors, the implementation operates on the last dimension, i.e. each per-token per-head vector in a tensor of shape `B x H x T x D`.
