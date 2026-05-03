# TODO: orez-infer

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
- Implement a recurrent version of the Mamba-2 layer suitable for single-token inference.
- Benchmark and optimize the scan operation for various hardware backends (CPU vs. Metal).
