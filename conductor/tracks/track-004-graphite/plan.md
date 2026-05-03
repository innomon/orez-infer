# Plan: Track-004 - Graphite (SSM) Integration

Add support for the Graphite (Mamba-2) architecture to the inference engine.

## Objectives
- Implement Mamba-2 Linear Scan as a GoMLX operation.
- Integrate Graphite-specific metadata and tensor mapping into the registry.
- Support hybrid SSM/Transformer models (like IBM Granite).

## Tasks
1. **SSM Core:** Implement the Linear Scan/Recurrence logic for Mamba-2 in GoMLX.
2. **Registry Extension:** Add a `GraphiteBuilder` to the `ArchRegistry`.
3. **Hybrid Logic:** Support switching between Attention and SSM layers within a single model template.
4. **State Management:** Implement efficient SSM state handling (similar to KV-Cache but for SSM).

## Verification
- Unit test the SSM Linear Scan logic.
- Verify graph construction for a Graphite-style model.
- Compare inference results against reference implementations.
