# Sources/QwenImage/Weights/WeightsMapping.swift Analysis

## Purpose
- Maps flat dictionary keys (from SafeTensors) to the hierarchical module structure of the Swift code.
- "The Rosetta Stone" of the project.

## Key Observations
- **High Complexity**: Contains massive mapping functions (`transformerParameters`, `vaeParameters`, etc.) with hardcoded string replacements.
- **Fragility**: Tightly coupled to specific naming conventions of the upstream PyTorch model.
- **Duplication**: `transformerParameters` and `layeredTransformerParameters` share similar logic but are separate functions.

## Quality Assessment
- Necessary evil for porting weights, but a nightmare to maintain.
- Uses `LinearLayerRegistry` to track which layers are linear (for quantization purposes).
