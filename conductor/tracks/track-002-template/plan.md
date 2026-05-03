# Plan: Track-002 - Universal Transformer Template

Implement a configurable GoMLX graph template capable of representing various Transformer architectures.

## Objectives
- Implement a core `Transformer` struct that can be configured for Llama, Gemma, etc.
- Support Grouped-Query Attention (GQA).
- Implement RMSNorm (standard and scaled).
- Integrate RoPE (Rotary Positional Embeddings) with configurable theta.
- Support SwiGLU and GeLU activations.

## Tasks
1. **Registry Setup:** Implement the `ArchRegistry` and `GraphBuilder` interface.
2. **Attention Module:** Create a GoMLX module for Multi-Head and Grouped-Query Attention.
3. **Normalization:** Implement standard and scaled RMSNorm.
4. **MLP Module:** Implement the Feed-Forward Network with support for various activations.
5. **Template Assembly:** Combine modules into a `BuildGraph` function that uses the registry for architecture-specific logic.

## Verification
- Unit test each module (Attention, Norm, MLP) with synthetic data.
- Verify graph construction for Llama-3 and Gemma configurations.
- Benchmark graph compilation time.
