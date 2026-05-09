package model

import (
	"fmt"
	"os"
	"testing"
	"github.com/gomlx/gomlx/pkg/ml/context"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	_ "github.com/gomlx/gomlx/backends/simplego"
)

func TestSTERound(t *testing.T) {
	os.Setenv("GOMLX_BACKEND", "go")
	backend, err := backends.New()
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	ctx := context.New()

	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
		return STERound(x)
	})
	if err != nil {
		t.Fatalf("Failed to create Exec: %v", err)
	}

	// Test forward pass: rounding
	input := []float32{0.4, 0.6, 1.4, -0.4, -0.6, -1.4}
	expected := []float32{0.0, 1.0, 1.0, 0.0, -1.0, -1.0}
	
	inputTensor := tensors.FromFlatDataAndDimensions(input, len(input))
	results, err := exec.Exec(inputTensor)
	if err != nil {
		t.Fatalf("Failed to execute: %v", err)
	}
	output := results[0].Value().([]float32)

	for i, val := range output {
		if val != expected[i] {
			t.Errorf("At index %d: expected %f, got %f", i, expected[i], val)
		}
	}
}

func TestBitLinearConstruction(t *testing.T) {
	os.Setenv("GOMLX_BACKEND", "go")
	backend, err := backends.New()
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	ctx := context.New()

	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
		return BitLinear(ctx, x, 128)
	})
	if err != nil {
		t.Fatalf("Failed to create Exec: %v", err)
	}

	fmt.Println("BitLinear graph constructed successfully")
	
	// Try executing with dummy input
	input := make([]float32, 128)
	inputTensor := tensors.FromFlatDataAndDimensions(input, 1, 128)
	_, err = exec.Exec(inputTensor)
	if err != nil {
		t.Fatalf("Failed to execute BitLinear: %v", err)
	}
}

func TestBitNetGraphConstruction(t *testing.T) {
	os.Setenv("GOMLX_BACKEND", "go")
	backend, err := backends.New()
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	ctx := context.New()

	config := ModelConfig{
		Name:             "bitnet-test",
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

	builder := &BitNetBuilder{}

	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, x *Node, pos *Node) *Node {
		return builder.Build(ctx, config, x, pos, nil)
	})
	if err != nil {
		t.Fatalf("Failed to create Exec: %v", err)
	}

	fmt.Println("BitNet graph constructed successfully")

	// Execute with dummy input
	tokens := []int32{1, 2, 3, 4}
	tokensTensor := tensors.FromFlatDataAndDimensions(tokens, 1, 4)
	posTensor := tensors.FromScalarAndDimensions(int32(0), 1, 1) // Scalar-like [1, 1]
	
	_, err = exec.Exec(tokensTensor, posTensor)
	if err != nil {
		t.Fatalf("Failed to execute BitNet: %v", err)
	}
}
