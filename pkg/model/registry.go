package model

import (
	"fmt"
	"strconv"

	"github.com/gomlx/gomlx/pkg/ml/context"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// ModelConfig contains the parameters for a specific model architecture.
type ModelConfig struct {
	Name             string
	VocabSize        int
	HiddenSize       int
	NumHeads         int
	NumKVHeads       int
	NumLayers        int
	IntermediateSize int
	MaxSeqLen        int
	HeadDim          int
	RoPEBase         float64
	RMSNormEPS       float64
	Activation       string // "swiglu", "gelu", etc.

	// Vision Config (for Multi-modal models like Gemma 3)
	ImageSize        int
	PatchSize        int
	VisionHiddenSize int
	VisionLayers     int
	VisionHeads      int

	// Gemma 4 specific
	NumMTPHeads      int
	KVSharingRange   int // e.g., 8
}

// GraphBuilder defines the interface for constructing architecture-specific graphs.
type GraphBuilder interface {
	Build(ctx *context.Context, config ModelConfig, x *Node, pos *Node, image *Node) *Node
	TensorMap(config ModelConfig) map[string]string
}

// ArchRegistry holds the mapping of architecture names to their builders.
type ArchRegistry struct {
	builders map[string]GraphBuilder
}

func NewArchRegistry() *ArchRegistry {
	r := &ArchRegistry{
		builders: make(map[string]GraphBuilder),
	}
	r.Register("recurrent", &RecurrentBuilder{})
	r.Register("gemma-3", &Gemma3Builder{})
	r.Register("gemma-4", &Gemma4Builder{})
	r.Register("llama", &LlamaBuilder{})
	r.Register("graphite", &GraphiteBuilder{})
	r.Register("granite", &GraniteBuilder{})
	return r
}

func (r *ArchRegistry) Register(name string, builder GraphBuilder) {
	r.builders[name] = builder
}

func (r *ArchRegistry) Get(name string) (GraphBuilder, bool) {
	b, ok := r.builders[name]
	return b, ok
}

// RecurrentBuilder implements GraphBuilder for RDT architectures.
type RecurrentBuilder struct{}

func (b *RecurrentBuilder) Build(ctx *context.Context, config ModelConfig, x *Node, pos *Node, image *Node) *Node {
	// For RDT, we use a fixed number of loops from config or default.
	numLoops := 8 // Default to 8 loops if not specified
	t := &RecurrentTransformer{Config: config, Loops: numLoops}
	return t.BuildGraph(ctx, x)
}

func (b *RecurrentBuilder) TensorMap(config ModelConfig) map[string]string {
	m := make(map[string]string)
	m["token_embd.weight"] = "token_embd/weight"
	m["output_norm.weight"] = "final_norm/weight"
	
	// Shared block parameters
	prefix := "recurrent_block."
	m["blk.0.attn_norm.weight"] = prefix + "input_norm/weight"
	m["blk.0.attn_q.weight"] = prefix + "attention/wq/weight"
	m["blk.0.attn_k.weight"] = prefix + "attention/wk/weight"
	m["blk.0.attn_v.weight"] = prefix + "attention/wv/weight"
	m["blk.0.attn_output.weight"] = prefix + "attention/wo/weight"
	m["blk.0.ffn_norm.weight"] = prefix + "post_norm/weight"
	m["blk.0.ffn_gate.weight"] = prefix + "mlp/gate_proj/weight"
	m["blk.0.ffn_up.weight"] = prefix + "mlp/up_proj/weight"
	m["blk.0.ffn_down.weight"] = prefix + "mlp/down_proj/weight"
	
	// LTI parameters
	m["blk.0.lti_log_a.weight"] = prefix + "injection/lti/log_A"
	m["blk.0.lti_dt.weight"] = prefix + "injection/lti/delta_t"
	m["blk.0.lti_b.weight"] = prefix + "injection/lti/b_weight"

	return m
}
