# Sources/QwenImage/Pipeline/Scheduler.swift Analysis

## Purpose
- Implements the Flow Match Scheduler.

## Key Observations
- Handles noise scheduling and sigma calculations.
- Supports **Sigma Shifting** (Static, Dynamic, Linear).
- Implements the `step` function for the ODE solver.

## Recent Updates
- **Aspect Ratio Logic**: Updated `QwenSchedulerFactory` to use `EditSizing.computeDimensions`. This ensures that when an `editResolution` is provided, the output dimensions preserve the original image's aspect ratio while targeting the requested pixel area.

## Quality Assessment
- The logic seems consistent with modern Flow Matching implementations (like Stable Diffusion 3 / Qwen).