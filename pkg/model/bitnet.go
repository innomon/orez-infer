package model

import (
	"fmt"
	"strconv"

	"github.com/gomlx/gomlx/pkg/ml/context"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

// BitNetBuilder implements GraphBuilder for BitNet architectures.
type BitNetBuilder struct{}

func (b *BitNetBuilder) Build(ctx *context.Context, config ModelConfig, x *Node, pos *Node, image *Node) *Node {
	g := x.Graph()
	dtype := dtypes.Float32 // Use Float32 for hidden states and weights

	// 1. Token Embedding
	// In BitNet, the embedding layer is usually standard floating point or high precision.
	emb := ctx.In("token_embd").VariableWithShape("weight", shapes.Make(dtype, config.VocabSize, config.HiddenSize)).SetTrainable(true).ValueGraph(g)
	x = Gather(emb, ExpandDims(x, -1))

	// 2. Transformer Layers
	for i := 0; i < config.NumLayers; i++ {
		layerCtx := ctx.In("blk").In(strconv.Itoa(i))
		x = BitNetBlock(layerCtx, x, config, pos)
	}

	// 3. Final RMSNorm
	x = RMSNorm(ctx.In("final_norm"), x, config.RMSNormEPS)

	// 4. Output Projection
	// Note: The output projection can also be a BitLinear, but often it's standard 
	// or tied to the embedding matrix. For b1.58, it's often BitLinear.
	logits := BitLinear(ctx.In("output"), x, config.VocabSize)

	return logits
}

func (b *BitNetBuilder) TensorMap(config ModelConfig) map[string]string {
	m := make(map[string]string)
	m["token_embd.weight"] = "token_embd/weight"
	m["output_norm.weight"] = "final_norm/weight"
	m["output.weight"] = "output/weight"

	for i := 0; i < config.NumLayers; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)
		ctxPrefix := fmt.Sprintf("blk/%d/", i)
		m[prefix+"attn_norm.weight"] = ctxPrefix + "pre_attention_norm/weight"
		m[prefix+"attn_q.weight"] = ctxPrefix + "attention/wq/weight"
		m[prefix+"attn_k.weight"] = ctxPrefix + "attention/wk/weight"
		m[prefix+"attn_v.weight"] = ctxPrefix + "attention/wv/weight"
		m[prefix+"attn_output.weight"] = ctxPrefix + "attention/wo/weight"
		m[prefix+"ffn_norm.weight"] = ctxPrefix + "pre_mlp_norm/weight"
		m[prefix+"ffn_gate.weight"] = ctxPrefix + "mlp/gate_proj/weight"
		m[prefix+"ffn_up.weight"] = ctxPrefix + "mlp/up_proj/weight"
		m[prefix+"ffn_down.weight"] = ctxPrefix + "mlp/down_proj/weight"
	}
	return m
}

// STERound implements a Straight-Through Estimator for the Round function.
// Forward pass: Round(x)
// Backward pass: Identity (gradient passes through unchanged)
func STERound(x *Node) *Node {
	// STE trick: Round(x) + StopGradient(x - Round(x))
	// This makes the forward pass Round(x) and the backward pass 1.0 (Identity).
	return Add(StopGradient(Sub(Round(x), x)), x)
}

// BitLinear implements the BitLinear layer from BitNet b1.58.
// It uses ternary weights {-1, 0, 1} and 8-bit quantized activations.
func BitLinear(ctx *context.Context, x *Node, outChannels int) *Node {
	g := x.Graph()
	dtype := x.Shape().DType
	inputChannels := x.Shape().Dimensions[x.Rank()-1]

	// 1. Weight Quantization
	// We use a variable for weights. During inference, these might be pre-quantized.
	wVar := ctx.VariableWithShape("weight", shapes.Make(dtype, inputChannels, outChannels)).SetTrainable(true)
	w := wVar.ValueGraph(g)

	// gamma = 1/nm * sum(|W|)
	gamma := ReduceMean(Abs(w))
	
	// W_tilde = Round(Clip(W / (gamma + eps), -1, 1))
	eps := Scalar(g, dtype, 1e-5)
	wScaled := Div(w, Add(gamma, eps))
	wQuant := STERound(Clip(wScaled, Scalar(g, dtype, -1.0), Scalar(g, dtype, 1.0)))

	// 2. Activation Quantization (AbsMax)
	// x_tilde = Clip(x * 127 / (max(|x|) + eps), -128, 127)
	absMax := ReduceMax(Abs(x))
	xScaled := Mul(x, Div(Scalar(g, dtype, 127.0), Add(absMax, eps)))
	xQuant := STERound(Clip(xScaled, Scalar(g, dtype, -128.0), Scalar(g, dtype, 127.0)))

	// 3. Dot Product
	// The result is in the range of quantized values, usually needs scaling back if 
	// integrated into a float32 network, but BitNet often keeps it quantized or 
	// scales by (gamma * absMax / 127) at the end.
	// For b1.58, we often scale the output back:
	// y = (x_quant @ w_quant) * (gamma * absMax / 127)
	
	y := Dot(xQuant, wQuant).MatMul()
	
	// Scale back to float range
	scaleBack := Mul(gamma, Div(absMax, Scalar(g, dtype, 127.0)))
	return Mul(y, scaleBack)
}

// BitNetMLP implements the Feed-Forward Network using BitLinear layers.
func BitNetMLP(ctx *context.Context, x *Node, intermediateSize int) *Node {
	hiddenSize := x.Shape().Dimensions[x.Rank()-1]
	
	// In BitNet b1.58, the MLP usually uses BitLinear
	gate := BitLinear(ctx.In("gate_proj"), x, intermediateSize)
	up := BitLinear(ctx.In("up_proj"), x, intermediateSize)
	
	// SwiGLU or similar activation
	activatedGate := Mul(gate, Sigmoid(gate))
	intermediate := Mul(activatedGate, up)
	
	return BitLinear(ctx.In("down_proj"), intermediate, hiddenSize)
}

// BitNetBlock represents a single BitNet transformer layer.
func BitNetBlock(ctx *context.Context, x *Node, config ModelConfig, pos *Node) *Node {
	// 1. RMSNorm (BitNet uses RMSNorm before BitLinear)
	normX := RMSNorm(ctx.In("pre_attention_norm"), x, config.RMSNormEPS)
	
	// 2. Attention with BitLinear projections
	attn := BitNetAttention(ctx.In("attention"), normX, config, pos)
	x = Add(x, attn)
	
	// 3. Pre-MLP Norm
	normX = RMSNorm(ctx.In("pre_mlp_norm"), x, config.RMSNormEPS)
	
	// 4. MLP with BitLinear
	mlpOut := BitNetMLP(ctx.In("mlp"), normX, config.IntermediateSize)
	x = Add(x, mlpOut)
	
	return x
}

// BitNetAttention implements Multi-Head Attention using BitLinear for projections.
func BitNetAttention(ctx *context.Context, x *Node, config ModelConfig, pos *Node) *Node {
	g := x.Graph()
	dtype := x.Shape().DType
	batchSize := x.Shape().Dimensions[0]
	seqLen := x.Shape().Dimensions[1]
	hiddenSize := config.HiddenSize
	numHeads := config.NumHeads
	headDim := config.HeadDim

	// Projections using BitLinear
	q := BitLinear(ctx.In("wq"), x, numHeads*headDim)
	k := BitLinear(ctx.In("wk"), x, config.NumKVHeads*headDim)
	v := BitLinear(ctx.In("wv"), x, config.NumKVHeads*headDim)

	q = Reshape(q, batchSize, seqLen, numHeads, headDim)
	k = Reshape(k, batchSize, seqLen, config.NumKVHeads, headDim)
	v = Reshape(v, batchSize, seqLen, config.NumKVHeads, headDim)

	// Apply RoPE
	if pos != nil {
		q = RoPE(q, pos, config.RoPEBase)
		k = RoPE(k, pos, config.RoPEBase)
	}

	// GQA / MQA repeat
	if config.NumKVHeads != numHeads {
		k = Repeat(k, numHeads/config.NumKVHeads, 2)
		v = Repeat(v, numHeads/config.NumKVHeads, 2)
	}

	q = Transpose(q, 1, 2)
	k = Transpose(k, 1, 2)
	v = Transpose(v, 1, 2)

	scores := Dot(q, Transpose(k, 2, 3)).MatMul()
	scores = Div(scores, Sqrt(Scalar(g, dtype, float64(headDim))))

	mask := CausalMask(g, seqLen)
	mask = Reshape(mask, 1, 1, seqLen, seqLen)
	scores = Add(scores, Mul(mask, Scalar(g, dtype, -1e10)))

	probs := Softmax(scores, -1)
	out := Dot(probs, v).MatMul()

	out = Transpose(out, 1, 2)
	out = Reshape(out, batchSize, seqLen, hiddenSize)

	return BitLinear(ctx.In("wo"), out, hiddenSize)
}
