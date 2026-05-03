# Specification: Track-009 - Gemma 3 & Gemma 4 Integration

Integration of Google's Gemma 3 (Unified Multi-modal) and Gemma 4 (MTP & TurboQuant) model families into `orez-infer`.

## Architecture Overview

### Gemma 3
- **Unified Multi-modal:** Processes text and vision tokens in the same embedding space.
- **SigLIP Vision Encoder:** (Referenced from `dyna-slm`).
- **Transformer Block:** Standard Gemma architecture with RoPE and RMSNorm.

### Gemma 4 & MedGemma 1.5
- **TurboQuant (Polar KV Cache):** Extreme compression of KV cache using Polar coordinates (Radius/Angle) + QJL (Quantized Johnson-Lindenstrauss) 1-bit residual.
- **Multi-Token Prediction (MTP):** Medusa-style heads sharing the unembedding matrix for speculative decoding.
- **Adaptive Precision:** Toggles precision/quantization modes based on token triggers:
    - `<|think|>` (5001): Reasoning mode.
    - `<|medical|>` (5003): Medical mode (switches to `MedicalRadiusLevels` codebook).
    - `<|audio|>` (5004): Audio mode (switches to `AudioRadiusLevels` codebook).
    - `<|image|>` (5005): Image/Vision mode.
- **Shared KV Cache:** Groups of layers (e.g., 8) sharing the same KV cache to save memory.

## GoMLX Implementation Strategy

### 1. Model Registry
- Add `Gemma3Builder` and `Gemma4Builder` to `pkg/model/registry.go`.
- Define configuration structures for both models in `pkg/model/gemma.go`.

### 2. Graph Templates
- Implement `BuildGemma3Graph` and `BuildGemma4Graph`.
- Modularize `GemmaAttention` and `GemmaBlock` to handle standard, shared, and TurboQuant modes.

### 3. TurboQuant Kernels
- Port `TurboQuantize` and `TurboDequantize` logic from `go-turboquant`.
- Implement `ApplyRoPEWithOffset` for KV cache indexing during incremental inference.

### 4. MTP Heads
- Implement speculative decoding verification logic in `pkg/model/speculative.go`.
- Support multiple output logits for MTP heads.

## Memory Management
- Use `KVCache` persistent variables in GoMLX `context.Context`.
- Support `Uint8` packed storage for TurboQuant to minimize memory footprint.
- Pre-allocate KV-Cache based on `max_seq_len` and `batch_size`.
