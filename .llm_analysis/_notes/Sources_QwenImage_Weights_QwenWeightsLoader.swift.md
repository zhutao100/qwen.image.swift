# Sources/QwenImage/Weights/QwenWeightsLoader.swift Analysis

## Purpose
- Orchestrates the loading of weights from disk into the model objects.
- Handles TextEncoder, Transformer, UNet, VAE, and VisionTower.

## Key Observations
- **Integration**: Bridges `HubSnapshot` (download), `SafeTensorsReader` (parse), and `WeightsMapping` (assign).
- **Quantization**: Applies quantization plans during loading.

## Quality Assessment
- Solid orchestration class.
- Depends heavily on the correctness of `WeightsMapping`.
