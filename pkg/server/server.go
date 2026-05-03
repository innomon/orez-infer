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
	// For now, return a static list or query the registry if possible.
	models := []Model{
		{
			ID:      "orez-recurrent-v1",
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

	// TODO: Implement inference bridge
	// This will involve:
	// 1. Tokenizing the input messages.
	// 2. Running the GoMLX graph.
	// 3. Handling streaming vs non-streaming responses.

	if req.Stream {
		s.handleStreamingChat(w, req)
	} else {
		s.handleBlockingChat(w, req)
	}
}

func (s *Server) handleBlockingChat(w http.ResponseWriter, req ChatCompletionRequest) {
	prompt := s.extractPrompt(req)
	
	if s.Runner == nil {
		http.Error(w, "model not loaded", http.StatusInternalServerError)
		return
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 50 // Default
	}

	content, err := s.Runner.Generate(prompt, s.Tokenizer, maxTokens, nil)
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
		maxTokens = 50 // Default
	}

	id := fmt.Sprintf("chatcmpl-%d", time.Now().Unix())
	
	_, err := s.Runner.Generate(prompt, s.Tokenizer, maxTokens, func(token string) {
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
		// Can't set error header if we already started streaming
		fmt.Fprintf(w, "data: {\"error\": \"%v\"}\n\n", err)
	}

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (s *Server) extractPrompt(req ChatCompletionRequest) string {
	// Simple concat for now. In production, use chat templates.
	var sb strings.Builder
	for _, msg := range req.Messages {
		if content, ok := msg.Content.(string); ok {
			sb.WriteString(content)
			sb.WriteString("\n")
		}
	}
	return sb.String()
}
