# Track 008: OpenAI API Implementation

Implement an OpenAI-compatible REST API for `orez-infer` to enable seamless integration with existing LLM clients and tools.

## 1. Goal
Provide a production-ready, OpenAI-compatible server that leverages `orez-infer`'s native GoMLX backends for inference, supporting both text and multimodal (vision) models.

## 2. Brainstorming: The "Most Effective Option"
The API server should be integrated into the main `orez-infer` CLI but also exist as a reusable package in `pkg/server`.

| Feature | Requirement | Reference Implementation |
| :--- | :--- | :--- |
| **Compatibility** | `/v1/models`, `/v1/chat/completions` | `go-turboquant/internal/api` |
| **Inference Engine** | Use `pkg/model` registry and GoMLX backends | `orez-infer` Core |
| **Multimodal** | Support for image inputs (SigLIP/Vision Encoders) | `go-turboquant` (MedGemma integration) |
| **Streaming** | Server-Sent Events (SSE) for real-time responses | OpenAI Standard |

**The Winning Strategy:** A `pkg/server` that abstracts the HTTP handling and maps OpenAI requests to `orez-infer` inference calls.
- Use `net/http` (standard library) for the server to maintain minimalism.
- Leverage the existing `pkg/model` registry to dynamically load the requested model.
- Implement a `Generator` interface that handles the token-by-token generation loop, supporting both blocking and streaming responses.

## 3. Implementation Plan

### Phase 1: API Structure & Schema
- [ ] Define OpenAI-compatible structs in `pkg/server/types.go` (Requests, Responses, Message types).
- [ ] Implement the base `Server` struct and handler registry in `pkg/server/server.go`.

### Phase 2: Inference Integration
- [ ] Implement `handleListModels` using `pkg/model/registry.go`.
- [ ] Implement `handleChatCompletions` bridging to `pkg/model`'s `Generate` functions.
- [ ] Support for GGUF metadata extraction to populate model information.

### Phase 3: Advanced Features
- [ ] **Multimodal Support:** Integrate vision encoder handling for models like MedGemma/Gemma 3.
- [ ] **Streaming (SSE):** Implement chunked responses for `/v1/chat/completions`.
- [ ] **MTP/Speculative Decoding:** (Optional) Support for accelerated generation if available in the model architecture.

### Phase 4: CLI & Validation
- [ ] Add `serve` subcommand to `cmd/orez-infer/main.go`.
- [ ] Implement flags: `--port`, `--model-path`, `--backend`, `--device`.
- [ ] Validate with standard tools like `curl`, `openai-python` client, and `AnythingLLM`.

## 4. References
- `../go-turboquant/cmd/api/main.go` (Entry point)
- `../go-turboquant/internal/api/server.go` (Server logic)
- `pkg/model/registry.go` (Local model factory)
