# Track Plan: IBM Granite-4.0-H-Micro (Graphite Hybrid) Implementation

Implementation of the IBM Granite-4.0-H-Micro architecture in GoMLX, featuring a hybrid structure of 4 Transformer blocks and 36 Mamba-2 (SSD) blocks with a "Chunked Parallel SSD" builder.

## 1. SSD (Mamba-2) Core Implementation
- [ ] **SSD Chunked Parallel Algorithm**:
  - [ ] Implement sequence chunking (default size $Q=64$).
  - [ ] Implement intra-chunk causal output via masked matrix multiplication.
  - [ ] Implement inter-chunk state passing logic ($O(L/Q \cdot N)$).
- [ ] **Einsum Optimization**:
  - [ ] Implement `SSDMatrixMultiply` using GoMLX `Einsum` to ensure XLA fusion on Mac M4 (Metal).
- [ ] **1D Depthwise Convolution**:
  - [ ] Implement the Mamba-style 1D convolution pre-processing.

## 2. Hybrid Architecture Orchestration
- [ ] **Interleaved Registry Pattern**:
  - [ ] Configure the registry to map 4 Attention layers and 36 SSD layers for the 40-layer `h-micro` variant.
  - [ ] Implement the "NoPE" (No Positional Embeddings) strategy for SSD layers while maintaining positional support for Transformer layers.
- [ ] **MLP (SwiGLU)**:
  - [ ] Ensure the MLP blocks use the 8192 hidden size specified for Granite-4.0-H-Micro.

## 3. SSD Builder Component (`pkg/model/ssd.go`)
- [ ] **BuildSSDLayer**:
  - [ ] Implement Grouped-Value structure in Mamba heads as per spec.
  - [ ] Implement the gated activation output: $y = \text{Dense}(y_{\text{ssd}} \cdot \text{SiLU}(z))$.

## 4. Hardware Optimization & Memory Management
- [ ] **XLA Fusion Verification**:
  - [ ] Verify that `SSDMatrixMultiply` contracts into a single Metal kernel on M4.
- [ ] **Linear Scaling Memory Strategy**:
  - [ ] Ensure the chunked scan maintains $O(L)$ scaling for long-context (128K) support.

## 5. Verification & Benchmarking
- [ ] Create `pkg/model/granite_test.go` to verify the 40-layer hybrid graph construction.
- [ ] Benchmark token throughput on 16GB targets (Rpi5/M4) to verify linear scaling.
- [ ] Validate against GGUF metadata for `h-micro`.
