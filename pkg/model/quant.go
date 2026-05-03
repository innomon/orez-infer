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

// TurboQuant Radius levels (4-bit Lloyd-Max)
var (
	DefaultRadiusLevels = []float64{
		0.0, 0.1, 0.25, 0.4, 0.6, 0.85, 1.15, 1.5, 1.9, 2.4, 3.0, 3.7, 4.6, 5.7, 7.1, 9.0,
	}
	AudioRadiusLevels = []float64{
		0.0, 0.2, 0.5, 0.9, 1.4, 2.0, 2.8, 3.8, 5.0, 6.5, 8.5, 11.0, 14.5, 19.0, 25.0, 35.0,
	}
	MedicalRadiusLevels = []float64{
		0.0, 0.05, 0.12, 0.22, 0.35, 0.52, 0.75, 1.05, 1.45, 2.0, 2.7, 3.6, 4.8, 6.4, 8.5, 12.0,
	}
)

// DequantizeTurboQuant performs polar dequantization with QJL correction.
// input: packed [num_elements] (8-bit: 4b Radius, 3b Angle, 1b QJL)
func DequantizeTurboQuant(packed *Node, isAudio, isMedical *Node) *Node {
	g := packed.Graph()
	dtype := dtypes.Float32

	// 1. Unpack bits
	// radiusIdx: bits [0:4]
	radiusIdx := Mod(packed, Scalar(g, dtypes.Uint8, 16))
	// angleIdx: bits [4:7]
	angleIdx := Mod(Div(packed, Scalar(g, dtypes.Uint8, 16)), Scalar(g, dtypes.Uint8, 8))
	// qjlBit: bit [7]
	qjlBit := Div(packed, Scalar(g, dtypes.Uint8, 128))

	// 2. Dequantize Radius
	r := DequantizeRadius(radiusIdx, isAudio, isMedical)

	// 3. Dequantize Angle (3-bit circular grid)
	theta := DequantizeAngle(angleIdx)

	// 4. Polar to Cartesian
	x := Mul(r, Cos(theta))
	y := Mul(r, Sin(theta))

	// 5. QJL Correction (1-bit residual)
	// TurboQuant uses a fixed scale for QJL residual
	qjlScale := Scalar(g, dtype, 0.05)
	qjlCorr := Sub(Mul(ConvertType(qjlBit, dtype), Scalar(g, dtype, 2.0)), Scalar(g, dtype, 1.0))
	x = Add(x, Mul(qjlScale, qjlCorr))
	
	// Interleave x and y if they represent complex or 2D manifolds
	// For simplicity, let's return x (assuming real-valued weights)
	// In some variants, x and y are two separate weights.
	return x
}

func DequantizeRadius(indices *Node, isAudio, isMedical *Node) *Node {
	g := indices.Graph()
	
	codebookStd := Const(g, DefaultRadiusLevels)
	codebookAudio := Const(g, AudioRadiusLevels)
	codebookMed := Const(g, MedicalRadiusLevels)
	
	intIndices := ConvertType(indices, dtypes.Int64)
	shape := intIndices.Shape()
	flatIndices := Reshape(intIndices, -1)
	expanded := ExpandDims(flatIndices, -1)
	
	resStd := Reshape(Gather(codebookStd, expanded), shape.Dimensions...)
	resAudio := Reshape(Gather(codebookAudio, expanded), shape.Dimensions...)
	resMed := Reshape(Gather(codebookMed, expanded), shape.Dimensions...)
	
	res := resStd
	if isAudio != nil {
		res = Where(isAudio, resAudio, res)
	}
	if isMedical != nil {
		res = Where(isMedical, resMed, res)
	}
	return ConvertType(res, dtypes.Float32)
}

func DequantizeAngle(indices *Node) *Node {
	g := indices.Graph()
	pi := Scalar(g, dtypes.Float32, 3.1415926535)
	twoPi := Scalar(g, dtypes.Float32, 2*3.1415926535)
	sectors := Scalar(g, dtypes.Float32, 8.0)
	
	idxF := ConvertType(indices, dtypes.Float32)
	centerOffset := Scalar(g, dtypes.Float32, 0.5)
	normalized := Div(Add(idxF, centerOffset), sectors)
	theta := Sub(Mul(normalized, twoPi), pi)
	return theta
}

// XLAFusion: Example of fusing dequant with MatMul
func FusedDequantMatMul(x, weightsPacked *Node) *Node {
	weights := DequantizeQ4_0(weightsPacked)
	return Dot(x, Transpose(weights, 0, 1)).MatMul()
}
