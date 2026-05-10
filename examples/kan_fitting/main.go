package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/innomon/orez-infer/pkg/model"
)

func main() {
	backend := backends.New("go") // Use simple CPU backend

	// 1. Create context
	ctx := context.NewContext(backend)

	// 2. Define KAN Model
	// 1 input, 5 hidden, 1 output
	// Grid size 5, Spline order 3
	kan1 := model.NewKANLayer(ctx, "layer1", 1, 5, 5, 3)
	kan2 := model.NewKANLayer(ctx, "layer2", 5, 1, 5, 3)

	modelFn := func(ctx *context.Context, x *graph.Node) *graph.Node {
		x = kan1.Forward(x)
		x = kan2.Forward(x)
		return x
	}

	// 3. Generate synthetic data: y = sin(pi * x)
	numSamples := 1000
	xData := make([]float32, numSamples)
	yData := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		x := rand.Float32()*2.0 - 1.0 // x in [-1, 1]
		xData[i] = x
		yData[i] = float32(math.Sin(float64(math.Pi * x)))
	}

	xTensor := backends.NewTensor(backend, shapes.Make(shapes.F32, numSamples, 1))
	xTensor.SetFromSlice(xData)
	yTensor := backends.NewTensor(backend, shapes.Make(shapes.F32, numSamples, 1))
	yTensor.SetFromSlice(yData)

	// 4. Training loop
	trainer := train.NewTrainer(backend, ctx, modelFn,
		losses.MeanSquaredError,
		optimizers.Adam().LearningRate(0.01).Done(),
		nil, nil)

	fmt.Println("Training KAN to fit sin(pi * x)...")
	for epoch := 0; epoch < 100; epoch++ {
		_, metrics, _ := trainer.TrainStep([]backends.Tensor{xTensor}, []backends.Tensor{yTensor})
		if epoch%10 == 0 {
			fmt.Printf("Epoch %d: Loss = %v\n", epoch, metrics[0])
		}
	}

	// 5. Evaluation
	evalFn := graph.New(backend, func(x *graph.Node) *graph.Node {
		return modelFn(ctx, x)
	})
	
	testInput := backends.NewTensor(backend, shapes.Make(shapes.F32, 1, 1))
	testInput.SetFromSlice([]float32{0.5})
	prediction := evalFn.Run(testInput)[0]
	fmt.Printf("Prediction for sin(pi * 0.5): %v (Expected: 1.0)\n", prediction.Value())
}
