package model

import (
	"github.com/gomlx/gomlx/pkg/ml/context"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// StableLTIModule implements the stable discretization of a linear system.
// h_{t+1} = A_disc * h_t + B_disc * e
func StableLTIModule(ctx *context.Context, h, e *Node) *Node {
	g := h.Graph()
	dtype := h.Shape().DType
	dModel := h.Shape().Dimensions[len(h.Shape().Dimensions)-1]

	// 1. Log-A parameterization for stability: A_cont = -exp(log_A)
	logA := ctx.In("lti").VariableWithShape("log_A", shapes.Make(dtype, dModel)).SetTrainable(false).ValueGraph(g)
	aCont := Neg(Exp(logA))

	// 2. Learned step size delta_t
	dt := ctx.In("lti").VariableWithShape("delta_t", shapes.Make(dtype, dModel)).SetTrainable(false).ValueGraph(g)
	
	// 3. Discretization: A_disc = exp(dt * aCont)
	aDisc := Exp(Mul(dt, aCont))

	// 4. B injection (simplified as a learned diagonal or dense)
	bWeight := ctx.In("lti").VariableWithShape("b_weight", shapes.Make(dtype, dModel)).SetTrainable(false).ValueGraph(g)
	
	return Add(Mul(aDisc, h), Mul(bWeight, e))
}

// RecurrentBlock implements a single iteration of the RDT loop.
func RecurrentBlock(ctx *context.Context, h, e *Node, config ModelConfig, loopIdx int) *Node {
	// 1. Stable LTI Injection
	h = StableLTIModule(ctx.In("injection"), h, e)

	// 2. Loop Embedding (RoPE or learned additive)
	// For now, we'll assume the weights can differentiate depth via some internal mechanism 
	// or we can add a loop-index bias.
	
	// 3. Shared Transformer Block
	// Note: In RDT, the Transformer part often takes (h_t, e) or just h_t
	// OpenMythos uses Transformer(h_t, e)
	// We'll pass both by concatenating or using e as a 'context' for cross-attention.
	// Simplified: Residual(h_t + TransformerBlock(h_t))
	
	// We reuse the existing Attention and MLP logic
	residual := h
	h = RMSNorm(ctx.In("input_norm"), h, config.RMSNormEPS)
	h = Attention(ctx.In("attention"), h, config, nil, nil) // Positional embedding handled inside
	h = Add(h, residual)

	residual = h
	h = RMSNorm(ctx.In("post_norm"), h, config.RMSNormEPS)
	h = MLP(ctx.In("mlp"), h, config.IntermediateSize, config.Activation)
	h = Add(h, residual)

	return h
}

// RecurrentTransformer handles the iterative refinement loop.
type RecurrentTransformer struct {
	Config ModelConfig
	Loops  int
}

func (t *RecurrentTransformer) BuildGraph(ctx *context.Context, x *Node) *Node {
	g := x.Graph()
	dtype := x.Shape().DType
	
	// 1. Prelude (Encoding)
	embd := ctx.In("token_embd").VariableWithShape("weight", shapes.Make(dtype, t.Config.VocabSize, t.Config.HiddenSize)).SetTrainable(false).ValueGraph(g)
	e := Gather(embd, ExpandDims(x, -1)) // Injection Signal
	
	// Initial hidden state
	h := e

	// 2. Recurrent Block Loop (Shared Weights)
	loopCtx := ctx.In("recurrent_block")
	for i := 0; i < t.Loops; i++ {
		// All iterations use the SAME loopCtx to share weights
		h = RecurrentBlock(loopCtx, h, e, t.Config, i)
	}

	// 3. Coda (Decoding)
	h = RMSNorm(ctx.In("final_norm"), h, t.Config.RMSNormEPS)
	output := Dot(h, Transpose(embd, 0, 1)).MatMul()
	
	return output
}
