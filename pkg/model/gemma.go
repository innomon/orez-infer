package model

import (
	"fmt"
	"strconv"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
)

// Gemma3Builder implements GraphBuilder for Gemma 3.
type Gemma3Builder struct{}

func (b *Gemma3Builder) Build(ctx *context.Context, config ModelConfig, x *Node, pos *Node, image *Node) *Node {
	return BuildGemma3Model(ctx, x, config, pos, image)
}

func (b *Gemma3Builder) TensorMap(config ModelConfig) map[string]string {
	m := make(map[string]string)
	m["token_embd.weight"] = "token_embd/weight"
	m["output_norm.weight"] = "output_norm/weight"
	
	// Optional: add vision encoder weights mapping
	
	for i := 0; i < config.NumLayers; i++ {
		p := fmt.Sprintf("blk.%d.", i)
		m[p+"attn_norm.weight"] = p + "pre_attention_norm/weight"
		m[p+"attn_q.weight"] = p + "attention/wq/weight"
		m[p+"attn_k.weight"] = p + "attention/wk/weight"
		m[p+"attn_v.weight"] = p + "attention/wv/weight"
		m[p+"attn_output.weight"] = p + "attention/wo/weight"
		m[p+"ffn_norm.weight"] = p + "pre_mlp_norm/weight"
		m[p+"ffn_gate.weight"] = p + "mlp/gate_proj/weight"
		m[p+"ffn_up.weight"] = p + "mlp/up_proj/weight"
		m[p+"ffn_down.weight"] = p + "mlp/down_proj/weight"
	}
	return m
}

// BuildGemma3Model builds the full Gemma 3 model graph.
func BuildGemma3Model(ctx *context.Context, tokens *Node, config ModelConfig, pos *Node, image *Node) *Node {
	g := tokens.Graph()
	dtype := tokens.Shape().DType

	// 1. Embedding
	embWeight := ctx.In("token_embd").VariableWithShape("weight", shapes.Make(dtype, config.VocabSize, config.HiddenSize)).SetTrainable(false).ValueGraph(g)
	x := Gather(embWeight, tokens)

	// 2. Vision Encoder (Optional)
	if image != nil {
		visualTokens := SigLIPVisionEncoder(ctx, image, config)
		x = InterleaveTokens(ctx, x, visualTokens)
	}

	// 3. Transformer Blocks
	for i := 0; i < config.NumLayers; i++ {
		layerCtx := ctx.In("blk").In(strconv.Itoa(i))
		x = GemmaBlock(layerCtx, x, tokens, config, pos, nil, nil, nil) // Gemma 3 does not use shared KV
	}

	// 4. Final Norm
	x = GemmaRMSNorm(ctx.In("output_norm"), x, config.RMSNormEPS)

	// 5. Output Head
	return Dot(x, Transpose(embWeight, 0, 1)).MatMul()
}

// SigLIPVisionEncoder implements the vision encoder for Gemma 3.
func SigLIPVisionEncoder(ctx *context.Context, image *Node, config ModelConfig) *Node {
	ctx = ctx.In("siglip_vision")
	g := image.Graph()
	dtype := image.Shape().DType
	
	// 1. Patch Embedding
	x := layers.Convolution(ctx.In("patch_embed"), image).
		Filters(config.VisionHiddenSize).
		KernelSize(config.PatchSize).
		Strides(config.PatchSize).
		Done()
	
	// 2. Flatten and Position Embeddings
	batchSize := x.Shape().Dimensions[0]
	numPatches := (config.ImageSize / config.PatchSize) * (config.ImageSize / config.PatchSize)
	x = Reshape(x, batchSize, numPatches, config.VisionHiddenSize)
	
	posEmbed := ctx.VariableWithShape("position_embeddings", shapes.Make(dtype, numPatches, config.VisionHiddenSize)).SetTrainable(false).ValueGraph(g)
	x = Add(x, posEmbed)
	
	// 3. Transformer Blocks
	for i := 0; i < config.VisionLayers; i++ {
		layerCtx := ctx.In(strconv.Itoa(i))
		x = SigLIPBlock(layerCtx, x, config)
	}
	
	x = LayerNorm(ctx.In("final_norm"), x, 1e-6)

	// 4. Projection Layer (maps VisionHiddenSize to HiddenSize)
	// We keep this in high precision (Float32/FP16) as suggested.
	return layers.Dense(ctx.In("projection"), x, true, config.HiddenSize)
}

// InterleaveTokens combines text and visual embeddings.
func InterleaveTokens(ctx *context.Context, textX, visionX *Node) *Node {
	// Simple strategy: prepend vision tokens to text tokens.
	return Concatenate([]*Node{visionX, textX}, 1)
}

// Gemma4Builder implements GraphBuilder for Gemma 4.
type Gemma4Builder struct{}

func (b *Gemma4Builder) Build(ctx *context.Context, config ModelConfig, x *Node, pos *Node, image *Node) *Node {
	return BuildGemma4Model(ctx, x, config, pos, image)
}

func (b *Gemma4Builder) TensorMap(config ModelConfig) map[string]string {
	m := make(map[string]string)
	m["token_embd.weight"] = "token_embd/weight"
	m["output_norm.weight"] = "output_norm/weight"

	for i := 0; i < config.NumLayers; i++ {
		p := fmt.Sprintf("blk.%d.", i)
		m[p+"attn_norm.weight"] = p + "pre_attention_norm/weight"
		m[p+"attn_q.weight"] = p + "attention/wq/weight"
		m[p+"ffn_norm.weight"] = p + "pre_mlp_norm/weight"
		m[p+"ffn_gate.weight"] = p + "mlp/gate_proj/weight"
		m[p+"ffn_up.weight"] = p + "mlp/up_proj/weight"
		m[p+"ffn_down.weight"] = p + "mlp/down_proj/weight"
		m[p+"attn_output.weight"] = p + "attention/wo/weight"

		// Shared KV mapping
		kvIdx := i
		if config.KVSharingRange > 1 {
			kvIdx = (i / config.KVSharingRange) * config.KVSharingRange
		}
		kvP := fmt.Sprintf("blk.%d.", kvIdx)
		m[p+"attn_k.weight"] = kvP + "attention/wk/weight"
		m[p+"attn_v.weight"] = kvP + "attention/wv/weight"

		// PLE mapping
		if config.UsePLE {
			m[p+"ple.weight"] = p + "ple/weight"
		}
	}

	// Add MTP head mappings
	for i := 0; i < config.NumMTPHeads; i++ {
		p := fmt.Sprintf("mtp_head.%d.", i)
		m[p+"weight"] = p + "weight"
	}
	return m
}

// BuildGemma4Model builds the full Gemma 4 model graph.
func BuildGemma4Model(ctx *context.Context, tokens *Node, config ModelConfig, pos *Node, image *Node) *Node {
	g := tokens.Graph()
	dtype := tokens.Shape().DType

	// 1. Adaptive State Detection (Trigger Tokens)
	// <|think|> (5001), <|audio|> (5004), <|image|> (5005)
	// We use ReduceMax on Int32 as a proxy for ReduceAny on Bool.
	isAudio := GreaterThan(ReduceMax(ConvertType(Equal(tokens, Scalar(g, dtypes.Int32, 5004)), dtypes.Int32), -1), Scalar(g, dtypes.Int32, 0))
	isMedical := GreaterThan(ReduceMax(ConvertType(Equal(tokens, Scalar(g, dtypes.Int32, 5003)), dtypes.Int32), -1), Scalar(g, dtypes.Int32, 0))

	// 2. Embedding
	embWeight := ctx.In("token_embd").VariableWithShape("weight", shapes.Make(dtype, config.VocabSize, config.HiddenSize)).SetTrainable(false).ValueGraph(g)
	x := Gather(embWeight, tokens)

	// 3. Vision Encoder (Optional)
	if image != nil {
		visualTokens := SigLIPVisionEncoder(ctx, image, config)
		x = InterleaveTokens(ctx, x, visualTokens)
	}

	// 4. Transformer Blocks with Shared KV Cache and Adaptive Precision
	for i := 0; i < config.NumLayers; i++ {
		layerCtx := ctx.In("blk").In(strconv.Itoa(i))

		var kvCtx *context.Context
		if config.KVSharingRange > 1 {
			sharedIdx := (i / config.KVSharingRange) * config.KVSharingRange
			kvCtx = ctx.In("blk").In(strconv.Itoa(sharedIdx)).In("attention")
		}

		// In a full implementation, we would pass isAudio/isMedical to dequant kernels
		// For now, they establish the conditional graph structure.
		x = GemmaBlock(layerCtx, x, tokens, config, pos, kvCtx, isAudio, isMedical)
	}

	// 5. Final Norm
	x = GemmaRMSNorm(ctx.In("output_norm"), x, config.RMSNormEPS)

	// 6. Output Head
	logits := Dot(x, Transpose(embWeight, 0, 1)).MatMul()

	// 7. MTP Heads (Phase 3)
	var mtpLogits []*Node
	if config.NumMTPHeads > 0 {
		mtpLogits = BuildMTPHeads(ctx, x, config, embWeight)
	}

	// For standard inference, we return the primary logits.
	// In speculative mode, the runner would access mtpLogits.
	_ = mtpLogits
	return logits
}

// BuildMTPHeads implements Medusa-style Multi-Token Prediction heads.
func BuildMTPHeads(ctx *context.Context, x *Node, config ModelConfig, embWeight *Node) []*Node {
	heads := make([]*Node, config.NumMTPHeads)

	for i := 0; i < config.NumMTPHeads; i++ {
		headCtx := ctx.In("mtp_head").In(strconv.Itoa(i))
		// Each head predicts the token at t + i + 1
		
		// 1. Non-linear projection
		h := layers.Dense(headCtx.In("proj"), x, false, config.HiddenSize)
		h = activations.Swish(h)
		h = Add(h, x) // Residual connection
		
		// 2. Head Norm
		h = GemmaRMSNorm(headCtx.In("norm"), h, config.RMSNormEPS)
		
		// 3. Output Projection (tied to embedding)
		logits := Dot(h, Transpose(embWeight, 0, 1)).MatMul()
		heads[i] = logits
	}
	return heads
}
