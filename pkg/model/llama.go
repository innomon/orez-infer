package model

import (
	"fmt"
	"github.com/gomlx/gomlx/pkg/ml/context"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// LlamaBuilder implements GraphBuilder for Llama architectures.
type LlamaBuilder struct{}

func (b *LlamaBuilder) Build(ctx *context.Context, config ModelConfig, x *Node, pos *Node, image *Node) *Node {
	t := &UniversalTransformer{Config: config}
	return t.BuildGraph(ctx, x, pos)
}

func (b *LlamaBuilder) TensorMap(config ModelConfig) map[string]string {
	m := make(map[string]string)
	m["token_embd.weight"] = "token_embd/weight"
	m["output_norm.weight"] = "final_norm/weight"
	m["output.weight"] = "output/weight" // If not shared
	
	for i := 0; i < config.NumLayers; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)
		internal := fmt.Sprintf("layer_%d.", i)
		
		m[prefix+"attn_q.weight"] = internal + "attention/wq/weight"
		m[prefix+"attn_k.weight"] = internal + "attention/wk/weight"
		m[prefix+"attn_v.weight"] = internal + "attention/wv/weight"
		m[prefix+"attn_output.weight"] = internal + "attention/wo/weight"
		
		m[prefix+"attn_norm.weight"] = internal + "input_norm/weight"
		m[prefix+"ffn_norm.weight"] = internal + "post_attention_norm/weight"
		
		m[prefix+"ffn_gate.weight"] = internal + "mlp/w1/weight"
		m[prefix+"ffn_down.weight"] = internal + "mlp/w2/weight"
		m[prefix+"ffn_up.weight"] = internal + "mlp/w3/weight"
	}
	return m
}

// Re-defining GraphBuilder in registry.go might be needed.
