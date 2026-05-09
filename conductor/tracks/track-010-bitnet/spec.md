# Specification: Track-010 - BitNet b1.58 Implementation

This specification defines the implementation of BitNet b1.58 ("The 1.58-bit LLM") using the GoMLX framework, targeting high-efficiency inference and training on Raspberry Pi 5 (ARM64) and Mac M4 (Apple Silicon).

## 1. Hardware & Backend Configuration

| Target | Architecture | GoMLX Backend | Acceleration |
| --- | --- | --- | --- |
| **Raspberry Pi 5** | ARM64 (Cortex-A76) | `go` (CPU) | CPU |
| **Mac M4** | Apple Silicon | `go_darwinml` | Metal Performance Shaders (MPS) |

### Context Initialization

```go
// Example initialization for backend switching
import (
    "github.com/gomlx/gomlx/ml/context"
    "github.com/gomlx/gomlx/backends"
    _ "github.com/gomlx/gomlx/backends/pjrt/cpu" // For RPi5
    // _ "github.com/gomlx/go-darwinml/backend"  // For Mac M4
)
```

## 2. Mathematical Foundation (BitNet b1.58)

The core of BitNet is the replacement of standard `Linear` layers with `BitLinear` layers using ternary weights $W \in \{-1, 0, 1\}$.

### 2.1 Weight Quantization

Weights are scaled by their average absolute value and rounded:

$$\gamma = \frac{1}{nm} \sum |W_{ij}|$$

$$\tilde{W} = \text{Round}\left(\text{Clip}\left(\frac{W}{\gamma + \epsilon}, -1, 1\right)\right)$$

### 2.2 Activation Quantization

Activations are quantized to 8-bit for efficient integer matrix multiplication:

$$\tilde{x} = \text{Clip}\left(x \cdot \frac{127.0}{\max|x| + \epsilon}, -128, 127\right)$$

## 3. GoMLX Implementation Components

### 3.1 Straight-Through Estimator (STE)

To train the model, we use STE to pass gradients through the non-differentiable `Round` function.

```go
func STERound(x *ml.Node) *ml.Node {
    // Forward pass uses Round, Backward pass acts as Identity
    return ml.Add(x.StopGradient(), ml.Round(x)).Sub(ml.Round(x).StopGradient())
}
```

### 3.2 The BitLinear Layer

This layer encapsulates the quantization of both weights and inputs before performing a `Dot` product.

```go
func BitLinear(ctx *context.Context, x *ml.Node, outChannels int) *ml.Node {
    g := x.Graph()
    dtype := x.DType()
    
    // 1. Weight Quantization logic
    wVar := ctx.VariableWithShape("weights", x.Shape().LastDim(), outChannels)
    gamma := ml.ReduceMean(ml.Abs(wVar.Value()))
    wQuant := STERound(ml.Div(wVar.Value(), ml.Add(gamma, ml.Scalar(g, dtype, 1e-5))))

    // 2. Activation Quantization (AbsMax)
    absMax := ml.ReduceMax(ml.Abs(x))
    xQuant := ml.Mul(x, ml.Div(ml.Scalar(g, dtype, 127.0), ml.Add(absMax, ml.Scalar(g, dtype, 1e-5))))
    xQuant = STERound(ml.Clip(xQuant, -128.0, 127.0))

    // 3. Bitwise-Equivalent Dot Product
    // XLA will optimize this dot product based on the backend
    return ml.Dot(xQuant, wQuant)
}
```

## 4. Optimization Strategy for Edge Hardware

1. **Memory Mapping:** On RPi5, use `gomlx` memory-mapped tensors if weights exceed available RAM to prevent OOM.
2. **XLA Fusion:** Ensure `LayerNorm` and quantization steps are in the same computation graph to allow XLA to fuse them into a single memory pass, critical for the bandwidth-limited RPi5.
3. **Inference:** For pure inference, the floating-point master weights can be discarded and only the `int8` or ternary representations should be stored.

## 5. Deployment for Gemini CLI

- Use a Go wrapper to handle tokenization (e.g., using a library like `sentencepiece-go`).
- Implement an iterative generation loop where the XLA graph is compiled once and executed repeatedly.
- Provide a CLI flag to switch between `go` and `darwinml` backends.

## 6. References

- [github source code](https://github.com/microsoft/BitNet)
- [BitNet CPU Optimization](https://github.com/microsoft/BitNet/blob/main/src/README.md)
