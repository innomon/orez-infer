# Plan: Track-010 - BitNet b1.58 Implementation

Implement the BitNet b1.58 architecture natively in GoMLX to enable high-efficiency LLM inference on edge devices like Raspberry Pi 5 and Mac M4.

## Objectives
- Implement `BitLinear` layer with ternary weights (-1, 0, 1) and 8-bit activations.
- Support Straight-Through Estimator (STE) for quantization-aware training/fine-tuning.
- Optimize for ARM64 and Apple Silicon backends.
- Integrate BitNet-specific models into the `ArchRegistry`.

## Tasks

### Phase 1: Core BitNet Components
1. **[ ] BitLinear Implementation:** Implement the `BitLinear` function in `pkg/model/layers.go` (or a new `pkg/model/bitnet.go`).
2. **[ ] Quantization Kernels:** Implement weight ($\gamma$-scaling) and activation (AbsMax) quantization logic.
3. **[ ] STE Support:** Implement `STERound` for gradient flow during training.

### Phase 2: Architecture & Registry
4. **[ ] BitNet Model Template:** Create a `BuildBitNetModel` graph template in `pkg/model/bitnet.go`.
5. **[ ] ArchRegistry Integration:** Register `bitnet-b1.58` in `pkg/model/registry.go`.
6. **[ ] Backend Optimization:** Ensure XLA graph fusion for `LayerNorm` + `BitLinear` sequences.

### Phase 3: Validation & CLI
7. **[ ] Unit Tests:** Add tests for `BitLinear` quantization accuracy and STE gradient passing.
8. **[ ] Inference Integration:** Update `cmd/orez-infer` to support loading BitNet weights.
9. **[ ] Performance Benchmarking:** Measure inference speed on RPi5 and Mac M4.

## Verification
- Verify that `BitLinear` produces identical outputs to the reference Python implementation for a given set of weights and inputs.
- Confirm ternary weight distribution in the compiled GoMLX graph.
- Benchmark memory usage reduction compared to standard FP16/BF16 models.
