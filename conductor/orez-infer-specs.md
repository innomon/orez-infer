This specification outlines a native GoMLX implementation of a high-performance LLM inference engine. By avoiding heavy external CLI frameworks and C++ wrappers, 
we prioritize a "Go-first" philosophy that leverages backend capabilities for the Mac M4 and Raspberry Pi 5.

---

# Specification: `orez-infer` Inference Engine

## 1. Project Intent & Philosophy
The goal is to build a **native Go** inference engine using **GoMLX** that targets edge hardware (Apple Silicon and ARM64). 

* **Native over Wrapper:** Unlike Ollama (which wraps `llama.cpp`), this project implements the computational graph directly in GoMLX to allow for better integration with Go's concurrency model with `go` backend.
* **Minimalist Core:** No heavy CLI libraries (Cobra/Viper). We use the standard `flag` package to keep the binary footprint small and the execution path transparent.
* **Architecture-Aware:** Recognizes that "Generic" is a misnomer; instead, we use a **Hand-Crafted Registry** to map specific model dialects to optimized GoMLX graph templates.

---

## 2. Technical Stack & Hardware Targeting

### Backends
* **Mac M4 (16GB):** Targeted via the `go-darwinml` backend. Intent: Use Metal's unified memory for zero-copy tensor operations.
* **Raspberry Pi 5 (16GB):** Targeted via `go` backend. 

### Formats
* **GGUF:** Primary format for quantized weights.
* **Safetensors:** Used for full-precision (FP16/BF16) research models.
* **LiteRT:** Integration of pre-existing GoMLX implementations for flatbuffer-based graphs.

---

## 3. Architecture: The Hand-Crafted Registry

To handle the structural differences between **Llama-3**, **Gemma**, and **Graphite (Mamba-2)**, the engine uses an internal registry.

### Why a Hand-Crafted Registry?
1.  **Memory Layout:** Different models name their tensors differently. A registry allows us to map `model.layers.0.input_layernorm.weight` (Llama) and `backbone.layers.0.pre_attention_norm.scale` (Gemma) to a single internal `InputNorm` variable.
2.  **Logic Injection:** Architecture-specific quirks (like Gemma's logit capping) can be injected into the graph construction phase as functional closures.

### Registry Structure
```go
type ArchRegistry struct {
    builders map[string]GraphBuilder
}

type GraphBuilder interface {
    // BuildGraph constructs the XLA computation graph
    BuildGraph(ctx *context.Context, config ModelConfig) *mlx.Graph
    // TensorMap returns a dictionary mapping GGUF keys to internal names
    TensorMap() map[string]string
}
```

---

## 4. The "Universal" Transformer Template

For Transformer-based models (Llama, Gemma), we implement a highly configurable base template.



| Feature | Llama-3 Mode | Gemma-3/4 Mode | Graphite (Mamba-2) Mode |
| :--- | :--- | :--- | :--- |
| **Norm** | RMSNorm | RMSNorm (Scaled) | N/A (SSM State) |
| **Attention** | GQA (Grouped Query) | Sliding Window / Capped | **Linear Scan** (Custom Op) |
| **Activation** | SwiGLU | GeLU | SiLU |
| **Positional** | RoPE | RoPE (Different Theta) | N/A |

---

## 5. Memory & Performance Strategies

### KV-Cache Management
For 16GB devices, managing the context window is critical. 
* **Static Allocation:** The GoMLX `Context` will pre-allocate the KV-Cache for a fixed sequence length (e.g., 4096) to prevent memory fragmentation during inference.
* **Mmap Loading:** Use `syscall.Mmap` to map model weights directly into memory. This allows the OS to handle swapping and keeps the Go heap clean.

### Quantization (GGUF)
Since GoMLX does not natively support all GGML block types, the CLI will implement:
1.  **De-quantization Kernels:** Small GoMLX graphs that convert `Q4_K` blocks to `FP16` on-the-fly during the forward pass.
2.  **XLA Fusion:** Using GoMLX to fuse the de-quantization step with the Matrix Multiplication step to minimize memory bandwidth bottlenecks.

---

## 6. CLI Specification (`orez-infer`)

We avoid `Cobra/spf13` to maintain a "no-magic" codebase. Logic is handled by a simple `switch` on `os.Args`.

### Flags and Intent
* `-model`: Path to `.gguf` or `.safetensors`.
* `-backend`: `metal`
 or `cpu` (explicit control for the developer).
* `-temp`: Sampling temperature.
* `-max-tokens`: Hard limit on the JIT-compiled graph sequence length.

#
## Execution Flow
1.  **Parse Flags:** Determine model path and hardware backend.
2.  **Probe Metadata:** Read the GGUF header to identify `general.architecture`.
3.  **Registry Lookup:** Fetch the corresponding `GraphBuilder`.
4.  **JIT Compile:** Use `GoMLX` to compile the graph for the specific sequence length.
5.  **Loop:** Run the `Predict` graph until the `<|end_of_text|>` token is sampled.

---

## 7. Implementation Milestones

1.  **Phase 1: The Parser.** Build a zero-dependency GGUF metadata reader in Go.
2.  **Phase 2: The Template.** Implement a `BaseTransformer` in GoMLX with toggleable RMSNorm and GQA.
3.  **Phase 3: The M4 Optimization.** Implement the `go-darwinml` backend integration to ensure weights reside in Metal-accessible memory.
4.  **Phase 4: The Graphite Branch.** Add a specialized SSM builder to the registry for Mamba-style models.

## 8. Local Projects

Use the following projects for go goMLX codes as references and examples :
- /home/innomon/orez/llm/dyna-slm
- /home/innomon/orez/llm/go-LiteRT-LM
- /home/innomon/orez/llm/go-turboquant














