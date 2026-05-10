package model

import (
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// KANLayer implements a Kolmogorov-Arnold Network layer.
type KANLayer struct {
	InDim  int
	OutDim int
	Num    int // Number of grid intervals
	K      int // Spline order (typically 3)
}

// NewKANLayer creates a new KANLayer and initializes its parameters in the context.
func NewKANLayer(ctx *context.Context, name string, inDim, outDim, num, k int) *KANLayer {
	l := &KANLayer{
		InDim:  inDim,
		OutDim: outDim,
		Num:    num,
		K:      k,
	}

	// Initialize coefficients: (inDim, outDim, num+k)
	ctx.WithScope(name).Variable("coef", shapes.Make(shapes.F32, inDim, outDim, num+k)).SetRandomNormal(0, 0.1)
	
	// Initialize base function weight: (inDim, outDim)
	ctx.WithScope(name).Variable("base_weight", shapes.Make(shapes.F32, inDim, outDim)).SetRandomNormal(1.0, 0.1)

	// Initialize spline function weight: (inDim, outDim)
	ctx.WithScope(name).Variable("spline_weight", shapes.Make(shapes.F32, inDim, outDim)).SetRandomNormal(1.0, 0.1)

	// Initialize grid (knots): (inDim, num + 2*k + 1)
	// We'll use a uniform grid in [-1, 1] extended by k on each side.
	numKnots := num + 2*k + 1
	gridData := make([]float32, inDim*numKnots)
	h := 2.0 / float32(num)
	for i := 0; i < inDim; i++ {
		for j := 0; j < numKnots; j++ {
			gridData[i*numKnots+j] = -1.0 - float32(k)*h + float32(j)*h
		}
	}
	ctx.WithScope(name).Variable("grid", gridData).SetShape(shapes.Make(shapes.F32, inDim, numKnots)).SetTrainable(false)

	return l
}

// Forward performs the forward pass of the KAN layer.
func (l *KANLayer) Forward(x *graph.Node) *graph.Node {
	g := x.Graph()
	ctx := context.GetContextToGraph(g)
	
	// x shape: (batch, inDim)

	// 1. Base function: phi_base(x) = base_weight * SiLU(x)
	silu := graph.Mul(x, graph.Sigmoid(x))
	baseWeight := ctx.VariableValue(g, "base_weight")
	baseOut := graph.Dot(silu, baseWeight)

	// 2. Spline function: phi_spline(x) = spline_weight * sum(c_i * B_i(x))
	grid := ctx.VariableValue(g, "grid")
	bases := BSplineBasis(x, grid, l.K) // (batch, inDim, num+k)
	
	coef := ctx.VariableValue(g, "coef") // (inDim, outDim, num+k)
	
	// We want to compute: sum_i (c_{i,j,k} * B_{i,k}(x))
	// bases: (batch, inDim, basis_idx)
	// coef: (inDim, outDim, basis_idx)
	// result_ij = sum_k (bases_{batch, i, k} * coef_{i, j, k})
	
	basesExp := graph.ExpandDims(bases, 2) // (batch, inDim, 1, num+k)
	coefExp := graph.ExpandDims(coef, 0)   // (1, inDim, outDim, num+k)
	
	splineValues := graph.ReduceSum(graph.Mul(basesExp, coefExp), -1) // (batch, inDim, outDim)
	
	// Now multiply by spline_weight and sum over inDim.
	splineWeight := ctx.VariableValue(g, "spline_weight") // (inDim, outDim)
	splineOut := graph.ReduceSum(graph.Mul(splineValues, graph.ExpandDims(splineWeight, 0)), 1) // (batch, outDim)

	return graph.Add(baseOut, splineOut)
}

// BSplineBasis implements the Cox-de Boor recursion formula.
// x: (batch, inDim)
// knots: (inDim, numKnots)
// k: spline order
// Returns: (batch, inDim, numKnots-k-1)
func BSplineBasis(x *graph.Node, knots *graph.Node, k int) *graph.Node {
	g := x.Graph()
	// x is (batch, inDim) -> reshape to (batch, inDim, 1)
	x = graph.ExpandDims(x, -1)
	// knots is (inDim, numKnots) -> reshape to (1, inDim, numKnots)
	knots = graph.ExpandDims(knots, 0)

	// k=0 case: step functions
	// B_{i,0}(x) = 1 if knots[i] <= x < knots[i+1] else 0
	k0_left := graph.GreaterThanOrEqual(x, graph.Slice(knots, nil, nil, []int{0}, []int{-1}))
	k0_right := graph.LessThan(x, graph.Slice(knots, nil, nil, []int{1}, nil))
	bases := graph.ConvertType(graph.And(k0_left, k0_right), shapes.F32)

	// Recursive step
	for j := 1; j <= k; j++ {
		// B_{i,j}(x) = (x - t_i)/(t_{i+j} - t_i) * B_{i,j-1}(x) + (t_{i+j+1} - x)/(t_{i+j+1} - t_{i+1}) * B_{i+1,j-1}(x)
		
		t_i := graph.Slice(knots, nil, nil, []int{0}, []int{-j - 1})
		t_ij := graph.Slice(knots, nil, nil, []int{j}, []int{-1})
		t_i1 := graph.Slice(knots, nil, nil, []int{1}, []int{-j})
		t_ij1 := graph.Slice(knots, nil, nil, []int{j + 1}, nil)

		denom1 := graph.Sub(t_ij, t_i)
		denom2 := graph.Sub(t_ij1, t_i1)
		
		// Avoid division by zero
		eps := graph.Scalar(g, shapes.F32, 1e-8)
		denom1 = graph.Where(graph.Equal(denom1, graph.Scalar(g, shapes.F32, 0)), eps, denom1)
		denom2 = graph.Where(graph.Equal(denom2, graph.Scalar(g, shapes.F32, 0)), eps, denom2)

		term1 := graph.Mul(graph.Div(graph.Sub(x, t_i), denom1), graph.Slice(bases, nil, nil, []int{0}, []int{-1}))
		term2 := graph.Mul(graph.Div(graph.Sub(t_ij1, x), denom2), graph.Slice(bases, nil, nil, []int{1}, nil))
		
		bases = graph.Add(term1, term2)
	}

	return bases
}
