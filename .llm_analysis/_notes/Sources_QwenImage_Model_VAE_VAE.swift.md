# Sources/QwenImage/Model/VAE/VAE.swift Analysis

## Purpose
- The Variational Autoencoder (VAE) wrapper.
- Compresses images into latents and decodes latents back to images.

## Key Observations
- **3D Nature**: Wraps `QwenImageEncoder3D` and `QwenImageDecoder3D`. Even 2D images are treated as 3D (Time=1).
- **Normalization**: Hardcoded Mean and Std for latent normalization.
- **Legacy Support**: `decodeWithDenormalization` vs `decode`.

## Quality Assessment
- Clean wrapper.
- Explicit handling of normalization is good.
