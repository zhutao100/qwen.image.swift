# Sources/QwenImage/Tokenizer/Tokenizer.swift Analysis

## Purpose
- Wraps the HF Tokenizer for the Qwen model.
- Handles chat template formatting.

## Key Observations
- **Hardcoded Prompts**: `promptPrefix` and `promptSuffix` are hardcoded strings.
    - *Issue*: This makes the tokenizer coupled to a specific version/instruction-tuning of the model.
- **Loading Logic**: Robust loading from directory (handling `tokenizer.json` vs `vocab.json` + `merges.txt`).
- **Special Tokens**: Explicit handling of `<|image_pad|>`, `<|vision_start|>`, etc.

## Quality Assessment
- Good abstraction over `swift-transformers`.
- Hardcoded prompt templates are a minor inflexibility.
