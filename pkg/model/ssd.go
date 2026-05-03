package model

import (
	"fmt"
	"github.com/gomlx/gomlx/pkg/ml/context"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
)

// SSDMatrixMultiply implements the Structured State Space Duality core via Einsum.
// It handles the causal recurrence within chunks.
// x: [batch, seq, heads, head_dim]
// a: [batch, seq, heads] (discretized A)
func SSDMatrixMultiply(ctx *context.Context, x, a *Node, config ModelConfig) *Node {
	// Equation for intra-chunk causal scan:
	// y_{i} = sum_{j=0}^{i} (prod_{k=j+1}^{i} a_k) * x_j
	
	// In Mamba-2 SSD, this is implemented as a block-diagonal matrix multiplication.
	// For small chunks (e.g. 64), we can precompute the semi-separable matrix.
	
	// For simplicity in the template, we'll use a batched MatMul approach 
	// that represents the semi-separable structure.
	// In a full implementation, we'd use Einsum to contract the causal mask.
	
	// y = Einsum("bshd,bs->bshd", x, a) // Placeholder for actual SSD contraction
	y := Mul(x, Reshape(a, -1, a.Shape().Dimensions[1], a.Shape().Dimensions[2], 1))
	
	return y
}

// BuildSSDLayer implements a Mamba-2 block for Granite.
func BuildSSDLayer(ctx *context.Context, x *Node, config ModelConfig) *Node {
	dModel := config.HiddenSize
	dInner := dModel * 2 // Expansion factor
	nHeads := 64         // From spec
	headDim := dInner / nHeads
	
	// 1. Input Projection
	// We need Z (gate) and X (input)
	// totalDim = 2 * dInner + heads (for delta) + heads * stateDim * 2 (for B, C)
	// For H-Micro, we'll simplify the projection to match the spec's intent.
	proj := layers.Dense(ctx.In("in_proj"), x, false, 2*dInner + nHeads)
	
	z := Slice(proj, AxisRange(), AxisRange(), AxisRange(0, dInner))
	xSSM := Slice(proj, AxisRange(), AxisRange(), AxisRange(dInner, 2*dInner))
	delta := Slice(proj, AxisRange(), AxisRange(), AxisRange(2*dInner, 2*dInner + nHeads))

	// 2. 1D Depthwise Convolution
	// GoMLX Convolution1D expects [batch, seq, depth]
	xSSM = Reshape(xSSM, -1, xSSM.Shape().Dimensions[1], nHeads, headDim)
	// Placeholder for Conv1D (usually 4-tap)
	xConv := xSSM 

	// 3. SSD Core Math
	// Discretize delta (A is usually -1.0 in SSD)
	dt := Softplus(delta)
	aDisc := Exp(Neg(dt)) // exp(-dt)
	
	y := SSDMatrixMultiply(ctx, xConv, aDisc, config)
	
	// 4. Final Projection
	y = Reshape(y, -1, y.Shape().Dimensions[1], dInner)
	y = Mul(y, activations.Swish(z))
	
	return layers.Dense(ctx.In("out_proj"), y, false, dModel)
}

// GraniteBuilder implements the hybrid architecture.
type GraniteBuilder struct{}

func (b *GraniteBuilder) Build(ctx *context.Context, config ModelConfig, x *Node, pos *Node, image *Node) *Node {
	g := x.Graph()
	dtype := x.Shape().DType

	// 1. Embeddings
	embd := ctx.In("token_embd").VariableWithShape("weight", shapes.Make(dtype, config.VocabSize, config.HiddenSize)).SetTrainable(false).ValueGraph(g)
	h := Gather(embd, x)

	// 2. Vision Encoder (Optional)
	if image != nil {
		visualTokens := SigLIPVisionEncoder(ctx, image, config)
		h = InterleaveTokens(ctx, h, visualTokens)
	}

	// 3. Hybrid Layers (40 total: 4 Transformer, 36 Mamba-2)
	for i := 0; i < 40; i++ {
		layerCtx := ctx.In(fmt.Sprintf("layer_%d", i))
		residual := h
		h = RMSNorm(layerCtx.In("pre_norm"), h, config.RMSNormEPS)

		// Interleaving: 1 Transformer layer every 10 layers (at 0, 10, 20, 30)
		if i%10 == 0 {
			h = Attention(layerCtx.In("attention"), h, config, pos)
		} else {
			// SSD / Mamba-2 (NoPE: skip pos)
			h = BuildSSDLayer(layerCtx.In("ssd"), h, config)
		}
		h = Add(h, residual)

		// MLP (SwiGLU)
		residual = h
		h = RMSNorm(layerCtx.In("post_norm"), h, config.RMSNormEPS)
		h = MLP(layerCtx.In("mlp"), h, 8192, "swiglu")
		h = Add(h, residual)
	}

	// 4. Final Norm & Head
	h = RMSNorm(ctx.In("final_norm"), h, config.RMSNormEPS)
	return Dot(h, Transpose(embd, 0, 1)).MatMul()
}
func (b *GraniteBuilder) TensorMap(config ModelConfig) map[string]string {
	m := make(map[string]string)
	m["token_embd.weight"] = "token_embd/weight"
	m["output_norm.weight"] = "final_norm/weight"
	
	for i := 0; i < 40; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)
		internal := fmt.Sprintf("layer_%d.", i)
		if i%10 == 0 {
			m[prefix+"attn_norm.weight"] = internal + "pre_norm/weight"
			m[prefix+"ffn_norm.weight"] = internal + "post_norm/weight"
		} else {
			m[prefix+"ssm_norm.weight"] = internal + "pre_norm/weight"
			m[prefix+"ffn_norm.weight"] = internal + "post_norm/weight"
			m[prefix+"ssm_in_proj.weight"] = internal + "ssd/in_proj/weight"
			m[prefix+"ssm_out_proj.weight"] = internal + "ssd/out_proj/weight"
		}
	}
	return m
}
