# Sources/QwenImage/Util/EditSizing.swift Analysis

## Purpose
- Helper for calculating image dimensions ensuring they align with model requirements (multiples of 16 or 32).

## Key Functions
- `computeDimensions`: Generic dimension scaling.
- `computeVAEDimensions`: Scaling for VAE (multiple of 32).
- `computeVisionConditionDimensions`: Scaling for Vision Tower (multiple of 32).

## Quality Assessment
- Simple, pure functions.
- Good use of `roundToNearestMultiple`.
