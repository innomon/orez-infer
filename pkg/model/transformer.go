package model

import (
	"fmt"
	"github.com/gomlx/gomlx/pkg/ml/context"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// UniversalTransformer implements a generic Transformer architecture.
type UniversalTransformer struct {
	Config ModelConfig
}

func (t *UniversalTransformer) BuildGraph(ctx *context.Context, x *Node, pos *Node) *Node {
	return t.buildCore(ctx, x, pos, false)
}

func (t *UniversalTransformer) BuildHybridGraph(ctx *context.Context, x *Node, pos *Node) *Node {
	return t.buildCore(ctx, x, pos, true)
}

func (t *UniversalTransformer) buildCore(ctx *context.Context, x *Node, pos *Node, hybrid bool) *Node {
	g := x.Graph()
	dtype := x.Shape().DType
	
	// Token Embeddings
	embd := ctx.In("token_embd").VariableWithShape("weight", shapes.Make(dtype, t.Config.VocabSize, t.Config.HiddenSize)).SetTrainable(false).ValueGraph(g)
	h := Gather(embd, x)
	
	// Layers
	for i := 0; i < t.Config.NumLayers; i++ {
		layerCtx := ctx.In(fmt.Sprintf("layer_%d", i))
		
		residual := h
		h = RMSNorm(layerCtx.In("input_norm"), h, t.Config.RMSNormEPS)
		
		// Layer Logic: Switch between Attention and SSM for hybrid models
		if hybrid && i%4 != 0 { // Example: Every 4th layer is Attention, others are SSM
			h = Mamba2(layerCtx.In("ssm"), h, t.Config)
		} else {
			h = Attention(layerCtx.In("attention"), h, t.Config, pos, nil)
		}
		h = Add(h, residual)
		
		// Post Attention/SSM Norm
		residual = h
		h = RMSNorm(layerCtx.In("post_norm"), h, t.Config.RMSNormEPS)
		
		// MLP
		h = MLP(layerCtx.In("mlp"), h, t.Config.IntermediateSize, t.Config.Activation)
		h = Add(h, residual)
	}
	
	// Final Norm
	h = RMSNorm(ctx.In("final_norm"), h, t.Config.RMSNormEPS)
	
	// Output Head (shared with embeddings usually)
	output := Dot(h, Transpose(embd, 0, 1)).MatMul()
	
	return output
}
