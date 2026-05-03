package model

import (
	"fmt"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/ml/context"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/innomon/orez-infer/pkg/tokenizer"
)

// Runner handles the execution of a model graph for inference.
type Runner struct {
	Backend backends.Backend
	Ctx     *context.Context
	Builder GraphBuilder
	Config  ModelConfig
	Exec    *context.Exec
}

// NewRunner initializes a new Runner with the given architecture and config.
func NewRunner(backend backends.Backend, ctx *context.Context, builder GraphBuilder, config ModelConfig) (*Runner, error) {
	r := &Runner{
		Backend: backend,
		Ctx:     ctx,
		Builder: builder,
		Config:  config,
	}

	// JIT compile the graph
	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, x *Node, pos *Node) *Node {
		return builder.Build(ctx, config, x, pos)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to compile graph: %w", err)
	}
	r.Exec = exec

	return r, nil
}

// Generate produces a sequence of tokens starting from the given prompt.
func (r *Runner) Generate(prompt string, t tokenizer.Tokenizer, maxTokens int, callback func(string)) (string, error) {
	tokens := t.Encode(prompt)
	currentIds := make([]int32, len(tokens))
	for i, id := range tokens {
		currentIds[i] = int32(id)
	}

	var generated []int32
	for i := 0; i < maxTokens; i++ {
		// 1. Prepare Tensors
		// For now, we pass the full sequence (stateless prototype)
		// TODO: Implement stateful KV cache with pos indexing
		inputTensor := tensors.FromFlatDataAndDimensions(currentIds, 1, len(currentIds))
		posTensor := tensors.FromScalarAndDimensions(int32(0), 1, len(currentIds))

		// 2. Execute
		results, err := r.Exec.Exec(inputTensor, posTensor)
		if err != nil {
			return "", fmt.Errorf("inference error at step %d: %w", i, err)
		}

		logits := results[0]
		
		// 3. Sample (Greedy for now)
		// logits: [batch, seq, vocab]
		// We want the last token's logits
		nextId := r.greedySample(logits)
		
		if nextId == 1 { // EOS placeholder
			break
		}

		generated = append(generated, nextId)
		currentIds = append(currentIds, nextId)

		// 4. Callback for streaming
		if callback != nil {
			callback(t.Decode([]int32{nextId}))
		}
	}

	return t.Decode(generated), nil
}

func (r *Runner) greedySample(logitsTensor *tensors.Tensor) int32 {
	// logits: [1, seq, vocab]
	shape := logitsTensor.Shape()
	seqLen := shape.Dimensions[1]
	vocabSize := shape.Dimensions[2]

	var maxIdx int32 = 0
	logitsTensor.MustConstFlatData(func(flat any) {
		data := flat.([]float32)
		
		// Only look at the last token
		lastTokenOffset := (seqLen - 1) * vocabSize
		lastLogits := data[lastTokenOffset : lastTokenOffset+vocabSize]

		var maxVal float32 = -1e10
		for i, val := range lastLogits {
			if val > maxVal {
				maxVal = val
				maxIdx = int32(i)
			}
		}
	})

	return maxIdx
}
