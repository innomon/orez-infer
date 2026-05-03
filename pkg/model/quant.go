package model

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// DequantizeQ4_0 converts GGUF Q4_0 blocks to Float32.
// Each block has 32 elements.
// input: [num_blocks, 18] (2 bytes scale, 16 bytes weights)
// output: [num_blocks, 32]
func DequantizeQ4_0(ctx *Node) *Node {
	g := ctx.Graph()
	// num_blocks := ctx.Shape().Dimensions[0]
	
	// 1. Extract scales (first 2 bytes of each block)
	// GGUF uses Float16 for scales in Q4_0.
	scalesRaw := Slice(ctx, AxisRange(), AxisRange(0, 2))
	// Convert bytes to F16. This is tricky in pure GoMLX without bitcast.
	// For now, let's assume we can handle the conversion or use a placeholder.
	// In a real implementation, we'd use a custom op or Bitcast if supported.
	scales := ConvertType(scalesRaw, dtypes.Float32) // Placeholder for actual F16 conversion
	
	// 2. Extract weights (remaining 16 bytes)
	weightsRaw := Slice(ctx, AxisRange(), AxisRange(2, 18)) // [num_blocks, 16]
	
	// 3. Unpack nibbles
	// Low nibbles: [num_blocks, 16]
	lowNibbles := Mod(weightsRaw, Scalar(g, dtypes.Uint8, 16))
	// High nibbles: [num_blocks, 16]
	highNibbles := Div(weightsRaw, Scalar(g, dtypes.Uint8, 16))
	
	// 4. Combine and shift
	// We want [num_blocks, 32]
	// We can reshape/transpose to interleave them if needed, or just concatenate.
	lowF := Sub(ConvertType(lowNibbles, dtypes.Float32), Scalar(g, dtypes.Float32, 8.0))
	highF := Sub(ConvertType(highNibbles, dtypes.Float32), Scalar(g, dtypes.Float32, 8.0))
	
	// Interleave: [num_blocks, 16, 2] -> [num_blocks, 32]
	combined := Concatenate([]*Node{
		Reshape(lowF, -1, 16, 1),
		Reshape(highF, -1, 16, 1),
	}, 2)
	
	unscaled := Reshape(combined, -1, 32)
	
	// 5. Apply scales
	return Mul(unscaled, Reshape(scales, -1, 1))
}

// XLAFusion: Example of fusing dequant with MatMul
func FusedDequantMatMul(x, weightsPacked *Node) *Node {
	weights := DequantizeQ4_0(weightsPacked)
	return Dot(x, Transpose(weights, 0, 1)).MatMul()
}
