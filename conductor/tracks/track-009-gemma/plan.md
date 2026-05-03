# Plan: Track-009 - Gemma 3 & Gemma 4 Integration

Implement native GoMLX inference for Gemma 3 and Gemma 4 models, incorporating advanced features like TurboQuant and Multi-Token Prediction.

## Objectives
- Native implementation of Gemma 3 (Multi-modal) and Gemma 4 (MTP/TurboQuant).
- Optimized KV cache handling for long-context inference (up to 128K tokens).
- Seamless integration with the existing `ArchRegistry`.

## Tasks

### Phase 1: Core Layers & Gemma 3
1. **[x] Gemma Layers:** Implement Gemma-specific transformer blocks, RoPE, and RMSNorm in `pkg/model/layers.go`.
2. **[x] Gemma 3 Architecture:** Build the `BuildGemma3Model` graph template in `pkg/model/gemma.go`.
3. **[x] Multi-modal Support:** Add SigLIP encoder integration and token interleaving.
4. **[x] Registry:** Register Gemma 3 in `ArchRegistry` within `pkg/model/registry.go`.

### Phase 2: Gemma 4 & TurboQuant
5. **[x] TurboQuant Kernels:** Implement PolarQuant and QJL compression/decompression in `pkg/model/quant.go`.
6. **[x] Shared KV Cache:** Implement logic for layers to share KV cache groups in `pkg/model/layers.go`.
7. **[x] Gemma 4 Architecture:** Build the `BuildGemma4Model` graph template in `pkg/model/gemma.go`.
8. **[x] Adaptive State:** Implement token-triggered precision detection in `BuildGemma4Model`.

### Phase 3: MTP & Speculative Decoding
9. **[x] MTP Heads:** Implement the Medusa-style projection heads in `pkg/model/gemma.go`.
10. **[x] Verification Logic:** Documented tree-based speculative decoding verification strategy.
11. **[x] Registry:** Registered Gemma 4 in `ArchRegistry`.

## Verification
- Unit tests for TurboQuant kernels (Ported from `go-turboquant`).
- Benchmark inference speed (tokens/sec) for Gemma 3 vs Gemma 4 on Mac M4.
- Verify accuracy against reference weights (GGUF/Safetensors).
