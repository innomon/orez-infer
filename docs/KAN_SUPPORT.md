# Kolmogorov-Arnold Networks (KAN) in orez-infer

This document describes the implementation of Kolmogorov-Arnold Networks (KAN) within the `orez-infer` engine using GoMLX.

## Introduction

KANs are a new type of neural network architecture where learnable activation functions (splines) are placed on edges instead of nodes. This provides several advantages over traditional MLPs:
- **Better Accuracy:** Often requires fewer parameters for the same accuracy.
- **Interpretability:** Individual edge functions can be visualized and often mapped to symbolic mathematical expressions.
- **Continual Learning:** Splines are local, making them less prone to catastrophic forgetting.

## Implementation Details

### B-Splines
We implement B-splines using the Cox-de Boor recursion formula directly in GoMLX graphs. This allows for full JIT compilation and hardware acceleration (CPU/Metal).

### KAN Layer
A KAN layer in `orez-infer` is defined by:
- `InDim`, `OutDim`: Input and output dimensions.
- `Grid`: The knots for the splines (usually uniform).
- `Coefficients`: Learnable spline weights.
- `BaseWeight`: Weight for the residual base function (SiLU).

### Training Pathway
The training pathway uses GoMLX's `train.Trainer` and supports:
- MSE Loss.
- L1 Regularization on coefficients for sparsity.
- Grid refinement (increasing spline resolution during training).

## Usage Example (Go)

```go
import (
    "github.com/gomlx/gomlx/graph"
    "github.com/gomlx/gomlx/ml/context"
    "github.com/gomlx/orez-infer/pkg/model"
)

func BuildModel(ctx *context.Context, x *graph.Node) *graph.Node {
    // 2 inputs, 5 hidden, 1 output
    l1 := model.NewKANLayer(ctx, "kan_1", 2, 5, 3) // 3 is grid size
    x = l1.Forward(x)
    
    l2 := model.NewKANLayer(ctx, "kan_2", 5, 1, 3)
    x = l2.Forward(x)
    
    return x
}
```

## Comparison with pykan
- **Native GoMLX:** Our implementation is written from scratch in GoMLX, avoiding any Python dependency.
- **Metal Acceleration:** Support for Mac M4 via the `go-darwinml` backend.
- **Unified Interface:** Integrated into the `orez-infer` registry.

## Future Work
- [ ] Multiplication nodes (MultKAN).
- [ ] Symbolic regression utilities for discovery.
- [ ] Specialized dequantization for KAN splines.
