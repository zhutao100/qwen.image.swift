# Sources/QwenImage/Util/EditTokenUtils.swift Analysis

## Purpose
- Manages special tokens for vision-language tasks.
- specifically `expandVisionPlaceholders` which injects image tokens between vision start/end tags.

## Key Observations
- **Complexity**: The logic in `expandVisionPlaceholders` is quite dense with manual array manipulation and index tracking.
- **Error Handling**: Uses a specific `EditTokenUtilsError` enum, which is good.
- **Logging**: Uses specific logger `QwenLogger.editTokens`.

## Quality Assessment
- Functional but complex. This kind of token manipulation is often a source of bugs (off-by-one errors).
