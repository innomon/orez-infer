package model

import (
	"fmt"
	"github.com/gomlx/gomlx/pkg/ml/context"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// GraphiteBuilder implements GraphBuilder for SSM-based architectures.
type GraphiteBuilder struct{}

func (b *GraphiteBuilder) Build(ctx *context.Context, config ModelConfig, x *Node, pos *Node) *Node {
	// Hybrid architecture: some layers are Attention, some are SSM.
	// For simplicity in this implementation, we'll assume a configurable layer type.
	
	// We'll reuse UniversalTransformer but with a custom layer loop.
	t := &UniversalTransformer{Config: config}
	return t.BuildHybridGraph(ctx, x, pos)
}

func (b *GraphiteBuilder) TensorMap(config ModelConfig) map[string]string {
	m := make(map[string]string)
	m["token_embd.weight"] = "token_embd/weight"
	m["output_norm.weight"] = "final_norm/weight"
	
	for i := 0; i < config.NumLayers; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)
		internal := fmt.Sprintf("layer_%d.", i)
		
		// SSM layers have different tensor names than Attention layers
		m[prefix+"ssm_in_proj.weight"] = internal + "ssm/in_proj/weight"
		m[prefix+"ssm_out_proj.weight"] = internal + "ssm/out_proj/weight"
		
		// Common norm
		m[prefix+"attn_norm.weight"] = internal + "input_norm/weight"
	}
	return m
}
