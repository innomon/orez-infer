# TODO: orez-infer

## Completed
- [x] **Unified Downloader:** Robust format-aware downloader for GGUF, Safetensors, and LiteRT.
- [x] **GGUF Parser:** Native Go parser for metadata and tensor info.
- [x] **Architecture Registry:** Hand-crafted mapping system for multi-model support.
- [x] **OpenAI API (Full):** Complete REST server with streaming, chat templates, and multimodal support.
- [x] **Gemma 3/4 (Structural):** Core layers, graph templates, and vision encoder interleaving for Gemma 3/4.
- [x] **TurboQuant Kernels (Phase 2):** Implement PolarQuant and QJL compression/decompression in GoMLX.
- [x] **Shared KV Cache (Phase 2):** Implement logic for layers to share KV cache groups for Gemma 4.
- [x] **Multimodal Support:** Native SigLIP integration and interleaved vision/text embeddings.
- [x] **PLE (Per-Layer Embeddings):** Integrated per-layer embedding lookup with adaptive TurboQuant dequantization for Gemma 4.

## Current Status: Functional Prototype with OpenAI API & PLE Support
The engine supports a wide range of architectures including Llama, Gemma, and Granite, with a fully functional OpenAI-compatible server.

## Gemma 3/4 Gaps (Track-009)
- [ ] **MTP Heads:** Implement Medusa-style heads and tree-based verification.
- [ ] **Production Kernels:** Optimize TurboQuant GoMLX implementations for Metal.

## Mamba-2 (SSM) Implementation Gaps (Track-004)

- [ ] **Real Linear Scan:** Implement a mathematically correct `LinearScan`. This requires an **associative prefix scan** for parallel pre-fill and a **recurrent state update** for token-by-token inference.
- [ ] **SSD (Structured State Space Duality) Logic:** Replace the `layers.Dense` placeholder in `pkg/model/ssm.go` with the actual SSD mathematical operations.
- [ ] **State Management:** Implement persistent SSM state handling (similar to KV-cache management) to maintain the hidden state across sequential inference calls.
- [ ] **GoMLX Recurrence:** Utilize `context.While` or other GoMLX-native looping constructs to handle sequential state updates efficiently within the XLA graph.

## Why it is a Placeholder
Implementing a high-performance Mamba-2 scan in GoMLX/XLA involves significant complexity regarding:
1. **Parallelism:** Associative scans are required for efficiency on accelerators during the pre-fill phase.
2. **Recurrence:** Standard graph construction doesn't easily support the sequential nature of SSM state updates without specific looping primitives.

## Next Steps
- [ ] **Weight Loading:** Implement the actual `syscall.Mmap` data reading and GoMLX variable initialization.
- [ ] **Inference Loop:** Implement the token-by-token generation loop with sampling.
- [ ] **Mamba-2 Scan:** Implement the recurrent SSM update for Graphite models.
- [ ] **Benchmarking:** Measure performance on Mac M4 vs Raspberry Pi 5.
