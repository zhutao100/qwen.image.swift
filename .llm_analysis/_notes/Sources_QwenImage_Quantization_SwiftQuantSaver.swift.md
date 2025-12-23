# Sources/QwenImage/Quantization/SwiftQuantSaver.swift Analysis

## Purpose
- Utility to quantize models offline and save them to disk.
- Generates `quantization.json` manifest.

## Key Observations
- **Sharding**: Handles splitting large weight files.
- **Filtering**: Can filter layers based on `allowedLayerMap` from `LinearLayerRegistry`.

## Quality Assessment
- Useful tool for creating optimized models for distribution.
