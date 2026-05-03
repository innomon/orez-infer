package model

import (
	"fmt"
	"os"
	"testing"
	"github.com/gomlx/gomlx/pkg/ml/context"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/simplego"
)

func TestGraniteGraphConstruction(t *testing.T) {
	os.Setenv("GOMLX_BACKEND", "go")
	backend, err := backends.New()
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	ctx := context.New()
	
	config := ModelConfig{
		Name:             "granite-h-micro-test",
		VocabSize:        1000,
		HiddenSize:       2048,
		NumHeads:         32, // Transformer heads
		NumLayers:        40,
		IntermediateSize: 8192,
		MaxSeqLen:        128,
		HeadDim:          64,
		RoPEBase:         10000.0,
		RMSNormEPS:       1e-6,
		Activation:       "swiglu",
	}

	builder := &GraniteBuilder{}
	
	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, x *Node, pos *Node) *Node {
		return builder.Build(ctx, config, x, pos)
	})
	if err != nil {
		t.Fatalf("Failed to create Exec: %v", err)
	}

	fmt.Println("Granite-4.0-H-Micro graph constructed successfully")
	_ = exec
}
