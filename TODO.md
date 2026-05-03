# TODO: orez-infer

## Completed
- [x] **Unified Downloader:** Robust format-aware downloader for GGUF, Safetensors, and LiteRT.
- [x] **GGUF Parser:** Native Go parser for metadata and tensor info.
- [x] **Architecture Registry:** Hand-crafted mapping system for multi-model support.
- [x] **OpenAI API (Phase 1 & 2):** Core API structure and basic inference integration.
- [x] **Gemma 3/4 (Phase 1):** Core layers (RMSNorm, SwiGLU) and graph templates for Gemma 3 (multi-modal) and Gemma 4.
- [x] **TurboQuant Kernels (Phase 2):** Implement PolarQuant and QJL compression/decompression in GoMLX.
- [x] **Shared KV Cache (Phase 2):** Implement logic for layers to share KV cache groups for Gemma 4.

## Current Status: Structural Implementation
The engine architecture is fully established, including the GGUF parser, the architecture registry, and the universal transformer template.

## Gemma 3/4 Implementation Gaps (Track-009)

- [ ] **TurboQuant Kernels:** Implement PolarQuant and QJL compression/decompression in GoMLX.
- [ ] **MTP Heads:** Implement Medusa-style heads and tree-based verification.
- [ ] **Shared KV Cache:** Implement group-based KV cache sharing for Gemma 4.
- [ ] **SigLIP Interleaving:** Finalize vision token insertion logic for Gemma 3.

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
