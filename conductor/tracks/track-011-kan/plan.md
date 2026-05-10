# Implementation Plan: Kolmogorov-Arnold Networks (KAN)

## Phase 1: Core Primitives (B-Splines)
- [ ] Implement `BSplineBasis` function in GoMLX. This should handle arbitrary order $k$ (default 3) and grid.
- [ ] Implement `EvaluateSpline` which takes input $x$ and coefficients $c$ and returns the spline value.
- [ ] Unit tests for `BSplineBasis` against known values (or cross-checked with `pykan` values).

## Phase 2: KAN Layer & Model
- [ ] Define `KANLayer` struct and `NewKANLayer` constructor.
- [ ] Implement `KANLayer.Forward(ctx *context.Context, x *graph.Node)` in GoMLX.
- [ ] Define `KANModel` and its forward pass.
- [ ] Support `SiLU` as the default base function.

## Phase 3: Training Pathway
- [ ] Implement a simple training loop using `train.Trainer`.
- [ ] Implement MSE loss and optional L1 regularization for coefficients.
- [ ] Create a "refinement" utility to increase grid density (mimicking `pykan`'s `update_grid`).

## Phase 4: Inference & Optimization
- [ ] Ensure the KAN model can be exported/saved for inference.
- [ ] Optimize the B-spline evaluation for inference (e.g., using a lookup table or optimized polynomial evaluation).

## Phase 5: Examples & Documentation
- [ ] Create `examples/kan_fitting/main.go` for basic function approximation.
- [ ] Create a more complex example (e.g., symbolic discovery from a KAN).
- [ ] Write `docs/KAN_SUPPORT.md`.

## Validation
- [ ] Verify that a 1-layer KAN can fit $\sin(x)$ with low error.
- [ ] Verify that a multi-layer KAN can fit $x \cdot y$ or similar.
