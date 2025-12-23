# Module: QwenImageData (Tokenizer & Processor)

## Sub-Module: Tokenizer (`Sources/QwenImage/Tokenizer/`)
### Purpose
Handles text tokenization using the Qwen vocabulary and formats prompts with the specific chat template required by the model.

### Key Features
- **Wrapper:** Wraps `AutoTokenizer` from `swift-transformers`.
- **Prompt Templating:** Manually constructs the chat prompt:
    ```
    <|im_start|>system ... <|im_end|>
    <|im_start|>user [Vision Tokens] [Text] <|im_end|>
    <|im_start|>assistant
    ```
- **Special Tokens:** Explicitly manages `<|image_pad|>`, `<|vision_start|>`, `<|vision_end|>` for multimodal inputs.

## Sub-Module: Processor (`Sources/QwenImage/Processor/`)
### Purpose
Preprocesses input images (for editing) and reference images (for generation) into the patch format expected by the `VisionTower`.

### Key Logic
- **Smart Resizing:** Adjusts image dimensions to be multiples of `patchSize * mergeSize` (typically 14 * 2 = 28) to ensure perfect patching.
- **Patching:** Converts 3D image data `[Time, Height, Width, Channels]` into flattened patch tokens.
- **Grid Calculation:** Computes the `(Temporal, Height, Width)` grid dimensions corresponding to the patches, which is passed to the TextEncoder for 3D positional embedding generation.
- **Normalization:** Standard ImageNet-style normalization (Mean/Std).
    - **Optimized:** Uses vectorized MLX operations for normalization, replacing previous CPU-based loops.