# Product Definition: orez-infer

A high-performance, native Go inference engine built on GoMLX, targeting Apple Silicon (M4) and ARM64 (Raspberry Pi 5).

## Key Features
- **Zero-Dependency GGUF/Safetensors Parser:** Native Go implementation for reading model weights.
- **Unified Transformer Template:** A single, configurable GoMLX graph template supporting Llama, Gemma, and Graphite (Mamba-2).
- **Metal Acceleration:** Deep integration with `go-darwinml` for Mac M4 performance.
- **Efficient Memory Management:** Static KV-Cache allocation and Mmap weight loading.
