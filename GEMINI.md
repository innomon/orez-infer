# orez-infer: Native GoMLX Inference Engine

## Project Mandates

- **Native over Wrapper:** Implement the computational graph directly in GoMLX. Avoid `llama.cpp` or C++ wrappers.
- **Minimalist CLI:** Use the standard `flag` package. Do **NOT** use `spf13/cobra` or `spf13/viper`.
- **Hand-Crafted Registry:** Use a registry to map model architectures (Llama, Gemma, Graphite) to optimized GoMLX graph templates.
- **Backend Targeting:**
    - **Mac M4:** Use `go-darwinml` for Metal-accelerated unified memory operations.
    - **Raspberry Pi 5:** Use the `go` (CPU) backend.
- **Memory Strategy:** Use `syscall.Mmap` for weight loading and pre-allocate KV-Cache in GoMLX `Context`.
- **References:** Refer to the following local projects for idiomatic GoMLX patterns and GGUF/Safetensors handling:
    - `/home/innomon/orez/llm/dyna-slm`
    - `/home/innomon/orez/llm/go-LiteRT-LM`
    - `/home/innomon/orez/llm/go-turboquant`

- **Unified Downloader:** Use `pkg/downloader` for all weight fetching. Support GGUF, Safetensors, and LiteRT.

do not search/look/grep inside .venv dirs.

## Coding Standards

- Follow idiomatic Go patterns.
- Ensure type safety in tensor operations.
- Document JIT-compiled graph assumptions (sequence length, batch size).
