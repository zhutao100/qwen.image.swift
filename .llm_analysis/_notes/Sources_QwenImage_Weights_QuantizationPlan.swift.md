# Sources/QwenImage/Weights/QuantizationPlan.swift Analysis

## Purpose
- Data structures for defining quantization configurations.
- `QwenQuantizationPlan` manages per-layer or global quantization settings.

## Quality Assessment
- Well-structured. Supports `affine` and `mxfp4` modes.
- Capable of loading config from JSON.
