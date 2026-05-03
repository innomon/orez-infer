# orez-infer: Native GoMLX Inference Engine

`orez-infer` is a high-performance, native Go inference engine for Large Language Models (LLMs). It leverages **GoMLX** for computational graph construction and JIT compilation via XLA, targeting edge hardware like Apple Silicon (Mac M4) and Raspberry Pi 5.

## Features

- **Native Implementation:** Built directly in Go and GoMLX. No heavy C++ wrappers or external CLI frameworks.
- **Multi-Format Support:** Robust downloader and parser for **GGUF**, **Safetensors**, and **LiteRT** formats.
- **OpenAI Compatible API:** Built-in REST server supporting `/v1/chat/completions` with streaming (SSE).
- **Hardware Optimized:**
    - **Mac M4:** Metal-accelerated operations via `go-darwinml`.
    - **Raspberry Pi 5:** Optimized CPU backend.
- **Architecture Registry:** Hand-crafted mappings for Llama, Gemma, Granite, and Graphite (Mamba-2) architectures.
- **Advanced Gemma Support:** Native integration for Gemma 3 (Multi-modal) and Gemma 4 (MTP/TurboQuant).
- **Per-Layer Embeddings (PLE):** Memory-efficient, adaptive embedding integration for extreme context long-inference.
- **Memory Efficient:** Uses `syscall.Mmap` for zero-copy weight loading and static KV-cache allocation.

## Installation

```bash
go build -o orez-infer ./cmd/orez-infer/main.go
```

## Usage

### 1. Download Model Weights
The unified downloader handles different formats and ensures all required files (config, tokenizers) are fetched.

```bash
# Download a GGUF model
./orez-infer download --repo orez-sh/gemma-4-E2B --quant Q4_K_M

# Download Safetensors (automatically fetches config and all shards)
./orez-infer download --repo google/gemma-2b --format safetensors
```

### 2. Run Inference
```bash
./orez-infer infer --model models/gemma-4-E2B.gguf --backend metal --temp 0.8
```

### 3. Start OpenAI API Server
Launch a local, OpenAI-compatible server to use `orez-infer` with standard clients like AnythingLLM or Python `openai` library.

For detailed information on multimodal support, see **[MedGemma 1.5 Support](./docs/MEDGEMMA_SUPPORT.md)**.

```bash
./orez-infer serve --model models/gemma-4-E2B.gguf --tokenizer models/tokenizer.model --port 8080 --backend metal
```

**Endpoint Support:**
- `GET /v1/models`: List loaded models.
- `POST /v1/chat/completions`: Chat with support for `stream: true`.

## Architecture

`orez-infer` uses a **Hand-Crafted Registry** to bridge the gap between model-specific tensor names and internal graph variables.

| Component | Description |
| :--- | :--- |
| **`pkg/gguf`** | Native GGUF parser for metadata and tensor info. |
| **`pkg/model`** | Universal Transformer templates and architecture-specific builders. |
| **`pkg/backend`** | Unified interface for CPU, Metal, and XLA backends. |
| **`pkg/downloader`** | Format-aware Hugging Face Hub client. |
| **`pkg/server`** | OpenAI-compatible REST API server. |
| **`pkg/tokenizer`** | SentencePiece and fallback tokenizers. |

## Supported Models
- **Llama 3**
- **Gemma 3 (Unified Multi-modal)**
- **Gemma 4 (TurboQuant & MTP)**
- **MedGemma 1.5 (Adaptive Medical Precision)**
- **Granite 4.0**
- **Graphite (Mamba-2 / SSM)**

## License
Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
