# Plan: Track-003 - Mac M4 (Metal) Optimization

Optimize the inference engine for Apple Silicon M4 using `go-darwinml`.

## Objectives
- Integrate `go-darwinml` as a GoMLX backend.
- Ensure zero-copy tensor operations using Metal's unified memory.
- Optimize de-quantization kernels for Metal.

## Tasks
1. **Backend Integration:** Wire up `go-darwinml` to the CLI's `-backend metal` flag.
2. **Metal Kernels:** Research or implement Metal-specific de-quantization kernels for GGUF blocks (Q4_K, etc.).
3. **Memory Alignment:** Ensure mmapped weights are properly aligned for Metal access.
4. **Performance Tuning:** Profile inference on M4 and identify bottlenecks in graph execution.

## Verification
- Run inference on an M4 Mac and verify Metal backend usage.
- Compare performance between CPU and Metal backends.
- Verify memory footprint stays within unified memory limits.
