package model

import (
	"github.com/gomlx/gomlx/pkg/ml/context"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
)

// RMSNorm implements Root Mean Square Layer Normalization.
func RMSNorm(ctx *context.Context, x *Node, epsilon float64) *Node {
	g := x.Graph()
	dtype := x.Shape().DType
	
	// x: [batch, seq, hidden]
	// Mean of squares over hidden dimension
	ms := ReduceMean(Square(x), -1)
	ms = ExpandDims(ms, -1) // Keep rank for broadcasting
	normed := Mul(x, Rsqrt(Add(ms, Scalar(g, dtype, epsilon))))
	
	// Weight (Scale)
	hiddenSize := x.Shape().Dimensions[x.Rank()-1]
	scale := ctx.VariableWithShape("weight", shapes.Make(dtype, hiddenSize)).SetTrainable(false).ValueGraph(g)
	
	// Reshape scale for broadcasting if x rank > 2
	if x.Rank() > 2 {
		scaleDims := make([]int, x.Rank())
		for i := 0; i < len(scaleDims)-1; i++ {
			scaleDims[i] = 1
		}
		scaleDims[len(scaleDims)-1] = hiddenSize
		scale = Reshape(scale, scaleDims...)
	}
	
	return Mul(normed, scale)
}

// MLP implements the Feed-Forward Network.
func MLP(ctx *context.Context, x *Node, intermediateSize int, activation string) *Node {
	g := x.Graph()
	dtype := x.Shape().DType
	hiddenSize := x.Shape().Dimensions[x.Rank()-1]
	
	// W1, W3 (for SwiGLU)
	w1 := ctx.In("w1").VariableWithShape("weight", shapes.Make(dtype, hiddenSize, intermediateSize)).SetTrainable(false).ValueGraph(g)
	w3 := ctx.In("w3").VariableWithShape("weight", shapes.Make(dtype, hiddenSize, intermediateSize)).SetTrainable(false).ValueGraph(g)
	
	gate := Dot(x, w1).MatMul()
	up := Dot(x, w3).MatMul()
	
	// Activation
	var activatedGate *Node
	switch activation {
	case "swiglu":
		activatedGate = Mul(gate, Sigmoid(gate)) // Silu/Swish
	case "gelu":
		activatedGate = activations.Gelu(gate)
	default:
		activatedGate = gate
	}
	
	intermediate := Mul(activatedGate, up)
	
	// W2
	w2 := ctx.In("w2").VariableWithShape("weight", shapes.Make(dtype, intermediateSize, hiddenSize)).SetTrainable(false).ValueGraph(g)
	return Dot(intermediate, w2).MatMul()
}

// RoPE applies Rotary Positional Embeddings.
func RoPE(x *Node, offset *Node, base float64) *Node {
	g := x.Graph()
	shape := x.Shape()
	// Expecting [batch, seq, num_heads, head_dim] or [batch, seq, head_dim]
	rank := x.Rank()
	seqLen := shape.Dimensions[1]
	headDim := shape.Dimensions[rank-1]
	
	halfDim := headDim / 2
	indices := Iota(g, shapes.Make(dtypes.Int32, halfDim), 0)
	indicesF := ConvertType(indices, shape.DType)
	exponent := Mul(indicesF, Scalar(g, shape.DType, -2.0/float64(headDim)))
	freqs := Exp(Mul(Log(Scalar(g, shape.DType, base)), exponent))
	
	t := Iota(g, shapes.Make(dtypes.Int32, seqLen), 0)
	if offset != nil {
		// Ensure offset is rank 1 or 0 for addition to Iota(seqLen)
		if offset.Rank() > 1 {
			offset = Reshape(offset) // Flatten to rank 1 if it has 1 element, or similar
		}
		t = Add(t, ConvertType(offset, dtypes.Int32))
	}
	tF := ConvertType(t, shape.DType)
	
	phases := Mul(Reshape(tF, seqLen, 1), Reshape(freqs, 1, halfDim))
	cos := Cos(phases)
	sin := Sin(phases)
	
	// Broadcast shapes based on rank
	if rank == 4 {
		// [batch, seq, num_heads, head_dim]
		cos = Reshape(cos, 1, seqLen, 1, halfDim)
		sin = Reshape(sin, 1, seqLen, 1, halfDim)
	} else {
		// [batch, seq, head_dim]
		cos = Reshape(cos, 1, seqLen, halfDim)
		sin = Reshape(sin, 1, seqLen, halfDim)
	}
	
	x1 := Slice(x, AxisRange(), AxisRange(), AxisRange(), AxisRange(0, halfDim))
	x2 := Slice(x, AxisRange(), AxisRange(), AxisRange(), AxisRange(halfDim, headDim))
	
	x_rope1 := Sub(Mul(x1, cos), Mul(x2, sin))
	x_rope2 := Add(Mul(x1, sin), Mul(x2, cos))
	
	return Concatenate([]*Node{x_rope1, x_rope2}, rank-1)
}

// Attention implements Multi-Head or Grouped-Query Attention.
func Attention(ctx *context.Context, x *Node, config ModelConfig, pos *Node, kvCtx *context.Context) *Node {
	g := x.Graph()
	dtype := x.Shape().DType
	batchSize := x.Shape().Dimensions[0]
	seqLen := x.Shape().Dimensions[1]
	hiddenSize := config.HiddenSize
	numHeads := config.NumHeads
	numKVHeads := config.NumKVHeads
	headDim := config.HeadDim
	
	// Q Projection (always local to the layer)
	wq := ctx.In("wq").VariableWithShape("weight", shapes.Make(dtype, hiddenSize, numHeads*headDim)).SetTrainable(false).ValueGraph(g)
	
	// K, V Projections (can be shared across layers)
	kProjCtx := ctx
	vProjCtx := ctx
	if kvCtx != nil {
		kProjCtx = kvCtx
		vProjCtx = kvCtx
	}
	
	wk := kProjCtx.In("wk").VariableWithShape("weight", shapes.Make(dtype, hiddenSize, numKVHeads*headDim)).SetTrainable(false).ValueGraph(g)
	wv := vProjCtx.In("wv").VariableWithShape("weight", shapes.Make(dtype, hiddenSize, numKVHeads*headDim)).SetTrainable(false).ValueGraph(g)
	
	q := Dot(x, wq).MatMul()
	k := Dot(x, wk).MatMul()
	v := Dot(x, wv).MatMul()
	
	q = Reshape(q, batchSize, seqLen, numHeads, headDim)
	k = Reshape(k, batchSize, seqLen, numKVHeads, headDim)
	v = Reshape(v, batchSize, seqLen, numKVHeads, headDim)
	
	// Apply RoPE
	if pos != nil {
		q = RoPE(q, pos, config.RoPEBase)
		k = RoPE(k, pos, config.RoPEBase)
	}
	
	// KV Cache handling would go here (pre-allocated in context)
	
	// Scaled Dot-Product Attention
	// If numKVHeads < numHeads, we need to repeat K and V (GQA)
	if numKVHeads != numHeads {
		// Repeat K, V
		k = Repeat(k, numHeads/numKVHeads, 2)
		v = Repeat(v, numHeads/numKVHeads, 2)
	}
	
	// [batch, heads, seq, head_dim]
	// Swapping seq (axis 1) and heads (axis 2)
	q = Transpose(q, 1, 2)
	k = Transpose(k, 1, 2)
	v = Transpose(v, 1, 2)
	
	// Attention scores
	// q: [batch, heads, q_seq, head_dim]
	// k: [batch, heads, k_seq, head_dim]
	// Contraction on head_dim (axis 3)
	scores := Dot(q, Transpose(k, 2, 3)).MatMul()
	scores = Div(scores, Sqrt(Scalar(g, dtype, float64(headDim))))
	
	// Causal Mask
	mask := CausalMask(g, seqLen)
	// Reshape mask for broadcasting: [1, 1, seqLen, seqLen]
	mask = Reshape(mask, 1, 1, seqLen, seqLen)
	scores = Add(scores, Mul(mask, Scalar(g, dtype, -1e10)))
	
	probs := Softmax(scores, -1)
	out := Dot(probs, v).MatMul()
	
	// [batch, seq, hidden]
	// Swap heads (axis 1) and seq (axis 2) back
	out = Transpose(out, 1, 2)
	out = Reshape(out, batchSize, seqLen, hiddenSize)
	
	// WO
	wo := ctx.In("wo").VariableWithShape("weight", shapes.Make(dtype, hiddenSize, hiddenSize)).SetTrainable(false).ValueGraph(g)
	return Dot(out, wo).MatMul()
}

func CausalMask(g *Graph, seqLen int) *Node {
	i := Iota(g, shapes.Make(dtypes.Int32, seqLen, seqLen), 0)
	j := Iota(g, shapes.Make(dtypes.Int32, seqLen, seqLen), 1)
	mask := GreaterThan(i, j)
	return ConvertType(mask, dtypes.Float32)
}

func Repeat(x *Node, n int, axis int) *Node {
	parts := make([]*Node, n)
	for i := 0; i < n; i++ {
		parts[i] = x
	}
	return Concatenate(parts, axis)
}

// GemmaRMSNorm implements Root Mean Square Layer Normalization for Gemma.
func GemmaRMSNorm(ctx *context.Context, x *Node, epsilon float64) *Node {
	g := x.Graph()
	dtype := x.Shape().DType
	
	// x: [batch, seq, hidden]
	ms := ReduceMean(Square(x), -1)
	ms = ExpandDims(ms, -1)
	normed := Mul(x, Rsqrt(Add(ms, Scalar(g, dtype, epsilon))))
	
	// Gemma adds 1 to the weight
	hiddenSize := x.Shape().Dimensions[x.Rank()-1]
	weight := ctx.VariableWithShape("weight", shapes.Make(dtype, hiddenSize)).SetTrainable(false).ValueGraph(g)
	
	// Reshape weight for broadcasting
	weightDims := make([]int, x.Rank())
	for i := 0; i < len(weightDims)-1; i++ {
		weightDims[i] = 1
	}
	weightDims[len(weightDims)-1] = hiddenSize
	weight = Reshape(weight, weightDims...)
	
	// Gemma RMSNorm: x * (1 + weight)
	return Mul(normed, Add(weight, Scalar(g, dtype, 1.0)))
}

// GemmaMLP implements the Feed-Forward Network for Gemma architectures.
func GemmaMLP(ctx *context.Context, x *Node, intermediateSize int) *Node {
	g := x.Graph()
	dtype := x.Shape().DType
	hiddenSize := x.Shape().Dimensions[x.Rank()-1]
	
	// Gemma uses SwiGLU: (Silu(xW_gate) * xW_up) * W_down
	wGate := ctx.In("gate_proj").VariableWithShape("weight", shapes.Make(dtype, hiddenSize, intermediateSize)).SetTrainable(false).ValueGraph(g)
	wUp := ctx.In("up_proj").VariableWithShape("weight", shapes.Make(dtype, hiddenSize, intermediateSize)).SetTrainable(false).ValueGraph(g)
	wDown := ctx.In("down_proj").VariableWithShape("weight", shapes.Make(dtype, intermediateSize, hiddenSize)).SetTrainable(false).ValueGraph(g)
	
	gate := Dot(x, wGate).MatMul()
	up := Dot(x, wUp).MatMul()
	
	// Silu(gate) * up
	activatedGate := Mul(gate, Sigmoid(gate))
	intermediate := Mul(activatedGate, up)
	
	return Dot(intermediate, wDown).MatMul()
}

// LayerNorm implements Layer Normalization.
func LayerNorm(ctx *context.Context, x *Node, epsilon float64) *Node {
	g := x.Graph()
	dtype := x.Shape().DType
	
	mean := ReduceMean(x, -1)
	mean = ExpandDims(mean, -1)
	variance := ReduceMean(Square(Sub(x, mean)), -1)
	variance = ExpandDims(variance, -1)
	
	normed := Div(Sub(x, mean), Sqrt(Add(variance, Scalar(g, dtype, epsilon))))
	
	hiddenSize := x.Shape().Dimensions[x.Rank()-1]
	scale := ctx.VariableWithShape("weight", shapes.Make(dtype, hiddenSize)).SetTrainable(false).ValueGraph(g)
	bias := ctx.VariableWithShape("bias", shapes.Make(dtype, hiddenSize)).SetTrainable(false).ValueGraph(g)
	
	// Reshape for broadcasting
	dims := make([]int, x.Rank())
	for i := 0; i < len(dims)-1; i++ {
		dims[i] = 1
	}
	dims[len(dims)-1] = hiddenSize
	scale = Reshape(scale, dims...)
	bias = Reshape(bias, dims...)
	
	return Add(Mul(normed, scale), bias)
}

// SigLIPBlock represents a single SigLIP vision transformer layer.
func SigLIPBlock(ctx *context.Context, x *Node, config ModelConfig) *Node {
	// 1. Pre-Attention Norm
	normX := LayerNorm(ctx.In("pre_attention_norm"), x, 1e-6)
	
	// 2. Attention (Non-causal)
	// We need a non-causal version of Attention
	attn := NonCausalAttention(ctx.In("attention"), normX, config)
	x = Add(x, attn)
	
	// 3. Pre-MLP Norm
	normX = LayerNorm(ctx.In("pre_mlp_norm"), x, 1e-6)
	
	// 4. MLP
	// SigLIP uses Gelu and a simpler MLP than SwiGLU
	hiddenSize := x.Shape().Dimensions[x.Rank()-1]
	intermediate := layers.Dense(ctx.In("mlp/gate"), normX, true, config.IntermediateSize)
	intermediate = activations.Gelu(intermediate)
	mlpOut := layers.Dense(ctx.In("mlp/down"), intermediate, true, hiddenSize)
	x = Add(x, mlpOut)
	
	return x
}

// NonCausalAttention implements Multi-Head Attention without a causal mask.
func NonCausalAttention(ctx *context.Context, x *Node, config ModelConfig) *Node {
	g := x.Graph()
	dtype := x.Shape().DType
	batchSize := x.Shape().Dimensions[0]
	seqLen := x.Shape().Dimensions[1]
	hiddenSize := x.Shape().Dimensions[2]
	numHeads := config.NumHeads
	headDim := hiddenSize / numHeads
	
	q := layers.Dense(ctx.In("wq"), x, true, numHeads*headDim)
	k := layers.Dense(ctx.In("wk"), x, true, numHeads*headDim)
	v := layers.Dense(ctx.In("wv"), x, true, numHeads*headDim)
	
	q = Reshape(q, batchSize, seqLen, numHeads, headDim)
	k = Reshape(k, batchSize, seqLen, numHeads, headDim)
	v = Reshape(v, batchSize, seqLen, numHeads, headDim)
	
	q = Transpose(q, 1, 2)
	k = Transpose(k, 1, 2)
	v = Transpose(v, 1, 2)
	
	scores := Dot(q, Transpose(k, 2, 3)).MatMul()
	scores = Div(scores, Sqrt(Scalar(g, dtype, float64(headDim))))
	
	probs := Softmax(scores, -1)
	out := Dot(probs, v).MatMul()
	
	out = Transpose(out, 1, 2)
	out = Reshape(out, batchSize, seqLen, hiddenSize)
	
	return layers.Dense(ctx.In("wo"), out, true, hiddenSize)
}

// ApplyPLE applies Per-Layer Embedding lookup, optionally using TurboQuant dequantization.
func ApplyPLE(ctx *context.Context, tokens *Node, config ModelConfig, isAudio, isMedical *Node) *Node {
	g := tokens.Graph()
	pleCtx := ctx.In("ple")
	
	if config.TurboQuantPLE {
		// 8-bit packed TurboQuant PLE weights
		packed := pleCtx.VariableWithShape("weight", shapes.Make(dtypes.Uint8, config.VocabSize, config.HiddenSize)).SetTrainable(false).ValueGraph(g)
		xPacked := Gather(packed, tokens)
		// Dequantize on-the-fly using the adaptive state.
		return DequantizeTurboQuant(xPacked, isAudio, isMedical)
	}
	
	// Fallback to standard float weights
	dtype := tokens.Shape().DType
	weight := pleCtx.VariableWithShape("weight", shapes.Make(dtype, config.VocabSize, config.HiddenSize)).SetTrainable(false).ValueGraph(g)
	return Gather(weight, tokens)
}

// GemmaBlock represents a single Gemma transformer layer.
func GemmaBlock(ctx *context.Context, x *Node, tokens *Node, config ModelConfig, pos *Node, kvCtx *context.Context, isAudio, isMedical *Node) *Node {
	// 1. Pre-Attention Norm
	normX := GemmaRMSNorm(ctx.In("pre_attention_norm"), x, config.RMSNormEPS)
	
	// 2. Attention
	attn := Attention(ctx.In("attention"), normX, config, pos, kvCtx)
	x = Add(x, attn)
	
	// 3. PLE (Per-Layer Embedding) for Gemma 4
	if config.UsePLE && tokens != nil {
		pleEmb := ApplyPLE(ctx, tokens, config, isAudio, isMedical)
		x = Add(x, pleEmb)
	}

	// 4. Pre-MLP Norm
	normX = GemmaRMSNorm(ctx.In("pre_mlp_norm"), x, config.RMSNormEPS)
	
	// 5. MLP
	mlpOut := GemmaMLP(ctx.In("mlp"), normX, config.IntermediateSize)
	x = Add(x, mlpOut)
	
	return x
}
