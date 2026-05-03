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

func TestRecurrentGraphConstruction(t *testing.T) {
	os.Setenv("GOMLX_BACKEND", "go")
	backend, err := backends.New()
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	ctx := context.New()
	
	config := ModelConfig{
		Name:             "rdt-test",
		VocabSize:        1000,
		HiddenSize:       128,
		NumHeads:         4,
		NumLayers:        1, // Shared layers in RDT
		IntermediateSize: 256,
		MaxSeqLen:        128,
		HeadDim:          32,
		RoPEBase:         10000.0,
		RMSNormEPS:       1e-6,
		Activation:       "swiglu",
	}

	builder := &RecurrentBuilder{}
	
	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, x *Node, pos *Node) *Node {
		// x: [batch, seq], pos: [batch, seq]
		return builder.Build(ctx, config, x, pos)
	})
	if err != nil {
		t.Fatalf("Failed to create Exec: %v", err)
	}

	fmt.Println("Recurrent (RDT) graph constructed successfully")
	_ = exec
}
