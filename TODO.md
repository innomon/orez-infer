# TODO: orez-infer

## Completed
- [x] **Unified Downloader:** Robust format-aware downloader for GGUF, Safetensors, and LiteRT.
- [x] **GGUF Parser:** Native Go parser for metadata and tensor info.
- [x] **Architecture Registry:** Hand-crafted mapping system for multi-model support.

## Current Status: Structural Implementation
The engine architecture is fully established, including the GGUF parser, the architecture registry, and the universal transformer template. However, the Mamba-2 (SSM) logic is currently a **structural placeholder** (dummy logic).

## Mamba-2 (SSM) Implementation Gaps

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
