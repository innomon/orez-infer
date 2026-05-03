# Technical Stack: orez-infer

## Language
- **Go:** Primary implementation language.

## Machine Learning Framework
- **GoMLX:** For computational graph construction and JIT compilation (XLA).

## Backends
- **go-darwinml:** Metal acceleration for macOS.
- **go (XLA CPU):** For Raspberry Pi 5 and general ARM64/x86_64.

## Data Formats
- **GGUF:** Primary quantized weight format.
- **Safetensors:** For high-precision research models.
- **Flatbuffers (LiteRT):** For interoperability.

## Low-Level Operations
- **syscall.Mmap:** Efficient weight loading.
- **standard flag package:** Minimalist CLI interface.
