package model

import (
	"fmt"
	"os"
	"testing"
	"github.com/gomlx/gomlx/pkg/ml/context"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/simplego" // Use SimpleGo for testing
)

func TestLlamaGraphConstruction(t *testing.T) {
	os.Setenv("GOMLX_BACKEND", "go")
	backend, err := backends.New()
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	ctx := context.New()
	
	config := ModelConfig{
		Name:             "llama-test",
		VocabSize:        1000,
		HiddenSize:       128,
		NumHeads:         4,
		NumKVHeads:       2,
		NumLayers:        2,
		IntermediateSize: 256,
		MaxSeqLen:        128,
		HeadDim:          32,
		RoPEBase:         10000.0,
		RMSNormEPS:       1e-6,
		Activation:       "swiglu",
	}

	builder := &LlamaBuilder{}
	
	// Create a computation to build the graph
	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, x *Node, pos *Node) *Node {
		return builder.Build(ctx, config, x, pos)
	})
	if err != nil {
		t.Fatalf("Failed to create Exec: %v", err)
	}

	fmt.Println("Llama graph constructed successfully")
	_ = exec
}

func TestGraphiteGraphConstruction(t *testing.T) {
	os.Setenv("GOMLX_BACKEND", "go")
	backend, err := backends.New()
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	ctx := context.New()
	
	config := ModelConfig{
		Name:             "graphite-test",
		VocabSize:        1000,
		HiddenSize:       128,
		NumHeads:         4,
		NumLayers:        4,
		IntermediateSize: 256,
		MaxSeqLen:        128,
		HeadDim:          32,
		RoPEBase:         10000.0,
		RMSNormEPS:       1e-6,
		Activation:       "swiglu",
	}

	builder := &GraphiteBuilder{}
	
	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, x *Node, pos *Node) *Node {
		return builder.Build(ctx, config, x, pos)
	})
	if err != nil {
		t.Fatalf("Failed to create Exec: %v", err)
	}

	fmt.Println("Graphite hybrid graph constructed successfully")
	_ = exec
}
