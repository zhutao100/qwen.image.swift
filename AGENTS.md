# Project context
- For the project context and instructions, use the same as Claude Code from the `CLAUDE.md` in the project root dir
- Some core libraries in this project were implemented referring to the `Diffusers` project, which
  - is accessible locally at `~/workspace/custom-builds/diffusers` 
  - is hosted at `https://github.com/huggingface/diffusers`
  - refer to it when analyzing the core library implementations in this project.
- This project leverages the model `Qwen/Qwen-Image` and its variants; read the `docs/Qwen-Image.md` and `docs/Qwen-Image-Model-Structures.md` to understand the models and their structures.
- Checkout the markdown documents in the project for more context; when making changes, keep them synchronized with the latest codebase for accurate analyses.

# Useful resources
- For analyzing `.safetensors` file structure, you can use the python script `~/bin/stls.py`
  - use `--format toon` to ouput in the LLM friendly format "TOON"
  - if the script is not present, it's downloadable via `curl https://gist.githubusercontent.com/zhutao100/cc481d2cd248aa8769e1abb3887facc8/raw/89d644c490bcf5386cb81ebcc36c92471f578c60/stls.py > ~/bin/stls.py`
- If need to inpsect the models used by the project, they are typically available in the default huggingface cache location `~/.cache/huggingface/hub/`.
