# Specification: Kolmogorov-Arnold Networks (KAN) in GoMLX

## Overview
Kolmogorov-Arnold Networks (KAN) are an alternative to Multi-Layer Perceptrons (MLPs). While MLPs have fixed activation functions on nodes and learnable weights on edges, KANs have learnable activation functions (splines) on edges and no fixed weights on nodes (just summation).

This track implements KAN in GoMLX, targeting both training (research/optimization) and inference (production) pathways.

## Mathematical Formulation

### 1. KAN Layer
A KAN layer with $n_{in}$ inputs and $n_{out}$ outputs consists of $n_{in} \times n_{out}$ 1D functions $\phi_{i,j}$.
The $j$-th output $y_j$ is:
$$y_j = \sum_{i=1}^{n_{in}} \phi_{i,j}(x_i)$$

### 2. Activation Function $\phi(x)$
In `pykan`, $\phi(x)$ is a combination of a base function (typically SiLU) and a B-spline:
$$\phi(x) = w_{base} \cdot \text{SiLU}(x) + w_{spline} \cdot \text{spline}(x)$$
Where:
$$\text{spline}(x) = \sum_{i} c_i B_i(x)$$
$B_i(x)$ are B-spline basis functions, and $c_i$ are learnable coefficients.

### 3. B-Splines
- **Grid:** A set of knots $t_0, t_1, \dots, t_G$.
- **Order $k$:** Typically cubic ($k=3$).
- **Basis Functions:** Defined recursively using the Cox-de Boor formula.

## Implementation Details (GoMLX)

### 1. Primitives
- **B-Spline Evaluation:** Implement `B_batch` as a GoMLX graph operation. This will involve recursive calculation or a pre-computed matrix multiplication if the grid is fixed.
- **Dequantization:** (Optional for this track, but relevant for `orez-infer`) Support for quantized coefficients.

### 2. Layers & Model
- `KANLayer`: Struct holding `Grid`, `Coefficients`, `BaseWeight`, `SplineWeight`.
- `KANModel`: Sequence of `KANLayer`.

### 3. Training
- Use `train.Trainer` with GoMLX.
- Support `L1` regularization on spline coefficients to encourage sparsity.
- Grid extension/refinement (from `pykan`) should be implemented or at least considered for future expansion.

### 4. Examples
- **Example 1:** Fitting $f(x) = \sin(x) \cdot \exp(x)$.
- **Example 2:** Fitting $f(x, y) = \exp(\sin(x)^2 + \cos(y)^2)$.

## Deliverables
- `pkg/model/kan.go`: Core KAN implementation.
- `pkg/model/kan_test.go`: Unit tests for B-spline math and layer forward pass.
- `examples/kan_fitting/main.go`: 1D/2D function fitting example.
- `docs/KAN_SUPPORT.md`: Documentation on how to use KAN in `orez-infer`.
