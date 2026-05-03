# Track Plan: Recurrent Block (RDT) Architecture

Implementation of the Recurrent-Depth Transformer (RDT) / Recurrent Block architecture as defined in OpenMythos, utilizing weight-sharing across depth and stable LTI-based input injection.

## 1. Research & Mathematical Modeling
- [ ] Define the stable discretization for the linear injection matrix $A$.
  - $A_{\text{cont}} = -\exp(\text{log\_A})$
  - $A_{\text{disc}} = \exp(\Delta t \cdot A_{\text{cont}})$
- [ ] Model the input injection update rule: $h_{t+1} = A_{\text{disc}} \cdot h_t + B_{\text{disc}} \cdot e + \text{Transformer}(h_t, e)$.
- [ ] Research MLA (Multi-Latent Attention) implementation if required for memory efficiency.

## 2. Layer Implementation (`pkg/model/recurrent.go`)
- [ ] **LTI Injection Module**: Implement the stable linear recurrence with learned $\Delta t$ and diagonal $A$.
- [ ] **Shared Transformer Block**: Modify standard Transformer block to accept an "Injection Signal" $e$.
- [ ] **Loop Embedding**: Add a positional embedding for the loop index $t$ to allow depth-specific processing.

## 3. Graph Builder Integration
- [ ] Implement `RecurrentBuilder` in `pkg/model/registry.go`.
- [ ] Define `TensorMap` for weight-shared parameters (mapping multiple logical loops to one set of physical tensors).
- [ ] Update `UniversalTransformer` or create `RecurrentTransformer` to handle the iterative refinement loop.

## 4. Stability & Optimization
- [ ] Implement "Adaptive Computation Time" (ACT) placeholder for early exit.
- [ ] Ensure spectral radius stability ($\rho(A) < 1$) via GoMLX constraints.
- [ ] Optimize the iterative loop for XLA/Metal backends.

## 5. Verification & Testing
- [ ] Create `pkg/model/recurrent_test.go` to verify state convergence and stability.
- [ ] Test hybrid configurations (Prelude -> Recurrent Block -> Coda).
