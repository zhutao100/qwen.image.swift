# Module: QwenImageWeights

## Purpose
Handles the loading, mapping, and conversion of model weights from Hugging Face format (SafeTensors) to the internal `MLX` module structures.

## Key Files
- `QwenWeightsLoader.swift`: High-level loader API for each component.
- `WeightsMapping.swift`: Dictionary logic mapping HF tensor keys to internal module paths.
- `SafeTensorsReader.swift`: Parser for the SafeTensors format.

## Functionality
- **SafeTensors Parsing:** Reads header metadata and memory-maps the data buffer.
    - **Robustness:** Handles various JSON numeric types (`Int`, `Int64`, `Double`) for broader platform compatibility.
- **Key Mapping:**
    - Maps flat keys (e.g., `model.layers.0.attn.q_proj.weight`) to hierarchical module paths.
    - Handles structural differences (e.g., renaming `mlp.gate` to `mlp.gate_proj`).
- **Tensor Conversion:**
    - **Convolution Layout:** Transposes weights from NCHW (PyTorch) to NHWC (MLX).
    - **DType Casting:** Option to cast weights to a specific precision during load.
- **Quantization Loading:**
    - Detects quantization metadata (scales/biases) in the source files.
    - Automatically loads these into `QuantizedLinear` layers if the quantization plan is active.

## Design Patterns
- **Visitor/Builder:** `WeightsMapping` acts as a builder that populates a `ModuleParameters` dictionary.
- **Registry:** `LinearLayerRegistry` is used to track specific layer types during mapping.