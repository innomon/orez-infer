# Track 007: Unified Weight Downloader

Implement a robust, format-aware downloader for model weights from Hugging Face, supporting GGUF, Safetensors, and LiteRT formats.

## 1. Goal
Provide a seamless CLI experience to fetch model weights and associated configuration files, ensuring the local environment is ready for inference.

## 2. Brainstorming: The "Most Effective Option"
Given the three target formats, the downloader must handle different file structures:

| Format | Structure | Logic |
| :--- | :--- | :--- |
| **GGUF** | Single `.gguf` file | Often many quants in one repo. User should select one. |
| **Safetensors** | `*.safetensors` + `config.json` + `tokenizer.json` | Requires parallel download of a file set. |
| **LiteRT** | `.tflite` / `.litert` + `tokenizer.model` | Hybrid of single weight file and specific tokenizer. |

**The Winning Strategy:** A `pkg/downloader` that implements a **Format-Aware Hugging Face Client**. 
- It uses the HF API to list files and identify "Format Clusters".
- It provides a `Download` method that takes a `FormatSpec` and handles the cluster-specific logic.

## 3. Implementation Plan

### Phase 1: Core Downloader Infrastructure
- [ ] Implement `pkg/downloader/http.go` for robust HTTP requests with progress reporting (using `github.com/schollz/progressbar/v3` or similar minimal approach).
- [ ] Implement `pkg/downloader/hf.go` to interact with Hugging Face Model Hub API.

### Phase 2: Format-Specific Logic
- [ ] **GGUF Handler:** Logic to list available quants and download a specific one.
- [ ] **Safetensors Handler:** Logic to fetch the `config.json` and all shards.
- [ ] **LiteRT Handler:** Logic to fetch the flatbuffer and the associated tokenizer.

### Phase 3: CLI Integration
- [ ] Add `download` command to `cmd/orez-infer/main.go`.
- [ ] Implement flags: `--repo`, `--format`, `--dest`, `--quant` (for GGUF).

### Phase 4: Validation & Cleanup
- [ ] Verify checksums (SHA256) provided by Hugging Face.
- [ ] Test with `smollm` (small footprint) across all three formats.

## 4. References
- `../go-LiteRT-LM/cmd/litert-lm/main.go` (Subcommand pattern)
- `../dyna-slm/download_weights.py` (Python equivalent)
