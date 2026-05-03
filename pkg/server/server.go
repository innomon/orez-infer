package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/innomon/orez-infer/pkg/model"
	"github.com/innomon/orez-infer/pkg/tokenizer"
)

// Server handles OpenAI-compatible requests.
type Server struct {
	Backend   backends.Backend
	Context   *context.Context
	Port      int
	Runner    *model.Runner
	Tokenizer tokenizer.Tokenizer
}

// New creates a new Server instance.
func New(backend backends.Backend, ctx *context.Context, port int, runner *model.Runner, t tokenizer.Tokenizer) *Server {
	return &Server{
		Backend:   backend,
		Context:   ctx,
		Port:      port,
		Runner:    runner,
		Tokenizer: t,
	}
}

// Start launches the HTTP server.
func (s *Server) Start() error {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", s.handleListModels)
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)

	fmt.Printf("🚀 orez-infer API server listening on :%d\n", s.Port)
	return http.ListenAndServe(fmt.Sprintf(":%d", s.Port), mux)
}

func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	modelID := "orez-infer-model"
	if s.Runner != nil && s.Runner.Config.Name != "" {
		modelID = s.Runner.Config.Name
	}

	models := []Model{
		{
			ID:      modelID,
			Object:  "model",
			Created: time.Now().Unix(),
			OwnedBy: "orez",
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ListModelsResponse{
		Object: "list",
		Data:   models,
	})
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if len(req.Messages) == 0 {
		http.Error(w, "no messages provided", http.StatusBadRequest)
		return
	}

	if req.Stream {
		s.handleStreamingChat(w, req)
	} else {
		s.handleBlockingChat(w, req)
	}
}

func (s *Server) handleBlockingChat(w http.ResponseWriter, req ChatCompletionRequest) {
	prompt := s.extractPrompt(req)
	imageTensor := s.extractImage(req)
	
	if s.Runner == nil {
		http.Error(w, "model not loaded", http.StatusInternalServerError)
		return
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 128 // Default
	}

	content, err := s.Runner.GenerateMultimodal(prompt, imageTensor, s.Tokenizer, maxTokens, nil)
	if err != nil {
		http.Error(w, fmt.Sprintf("inference failed: %v", err), http.StatusInternalServerError)
		return
	}

	resp := ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []Choice{
			{
				Index: 0,
				Message: Message{
					Role:    "assistant",
					Content: content,
				},
				FinishReason: "stop",
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleStreamingChat(w http.ResponseWriter, req ChatCompletionRequest) {
	prompt := s.extractPrompt(req)
	imageTensor := s.extractImage(req)

	if s.Runner == nil {
		http.Error(w, "model not loaded", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 128 // Default
	}

	id := fmt.Sprintf("chatcmpl-%d", time.Now().Unix())
	
	_, err := s.Runner.GenerateMultimodal(prompt, imageTensor, s.Tokenizer, maxTokens, func(token string) {
		chunk := ChatCompletionChunk{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   req.Model,
			Choices: []ChunkChoice{
				{
					Index: 0,
					Delta: Delta{
						Content: token,
					},
				},
			},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	})

	if err != nil {
		fmt.Fprintf(w, "data: {\"error\": \"%v\"}\n\n", err)
	}

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (s *Server) extractImage(req ChatCompletionRequest) *tensors.Tensor {
	// Search for image_url in messages
	for _, msg := range req.Messages {
		if parts, ok := msg.Content.([]any); ok {
			for _, p := range parts {
				if part, ok := p.(map[string]any); ok {
					if t, ok := part["type"].(string); ok && t == "image_url" {
						if imgURL, ok := part["image_url"].(map[string]any); ok {
							url := imgURL["url"].(string)
							fmt.Printf("🖼  Detected image URL: %s (Downloading placeholder)\n", url)
							
							// Placeholder: Return a zero tensor of the required size
							if s.Runner != nil && s.Runner.Config.ImageSize > 0 {
								size := s.Runner.Config.ImageSize
								return tensors.FromScalarAndDimensions(float32(0), 1, 3, size, size)
							}
						}
					}
				}
			}
		}
	}
	return nil
}

func (s *Server) extractPrompt(req ChatCompletionRequest) string {
	arch := "gemma" // Default
	if s.Runner != nil {
		arch = s.Runner.Config.Name
	}

	var sb strings.Builder
	switch {
	case strings.Contains(arch, "gemma"):
		// Gemma Chat Template: <start_of_turn>user\nPROMPT<end_of_turn>\n<start_of_turn>model\n
		for _, msg := range req.Messages {
			content := ""
			if c, ok := msg.Content.(string); ok {
				content = c
			} else if parts, ok := msg.Content.([]any); ok {
				// Basic multimodal part extraction
				for _, p := range parts {
					if part, ok := p.(map[string]any); ok {
						if t, ok := part["type"].(string); ok && t == "text" {
							content += part["text"].(string)
						}
					}
				}
			}
			
			sb.WriteString(fmt.Sprintf("<start_of_turn>%s\n%s<end_of_turn>\n", msg.Role, content))
		}
		sb.WriteString("<start_of_turn>model\n")
	
	case strings.Contains(arch, "llama"):
		// Llama 3 Template: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nPROMPT<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
		sb.WriteString("<|begin_of_text|>")
		for _, msg := range req.Messages {
			content := ""
			if c, ok := msg.Content.(string); ok {
				content = c
			}
			sb.WriteString(fmt.Sprintf("<|start_header_id|>%s<|end_header_id|>\n\n%s<|eot_id|>", msg.Role, content))
		}
		sb.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")

	default:
		// Fallback to simple concatenation
		for _, msg := range req.Messages {
			if content, ok := msg.Content.(string); ok {
				sb.WriteString(fmt.Sprintf("%s: %s\n", msg.Role, content))
			}
		}
		sb.WriteString("assistant: ")
	}
	return sb.String()
}
