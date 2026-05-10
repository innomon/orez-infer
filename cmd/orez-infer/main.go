package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/innomon/orez-infer/pkg/backend"
	"github.com/innomon/orez-infer/pkg/downloader"
	"github.com/innomon/orez-infer/pkg/gguf"
	"github.com/innomon/orez-infer/pkg/model"
	"github.com/innomon/orez-infer/pkg/server"
	"github.com/innomon/orez-infer/pkg/tokenizer"
)

type Command struct {
	Name        string
	Description string
	Exec        func(args []string) error
}

var commands = []Command{
	{
		Name:        "infer",
		Description: "Run inference on a model",
		Exec:        runInfer,
	},
	{
		Name:        "download",
		Description: "Download model weights from Hugging Face",
		Exec:        runDownload,
	},
	{
		Name:        "serve",
		Description: "Start an OpenAI-compatible API server",
		Exec:        runServe,
	},
}

func main() {
	if len(os.Args) < 2 {
		printUsage()
		return
	}

	cmdName := os.Args[1]
	for _, cmd := range commands {
		if cmd.Name == cmdName {
			if err := cmd.Exec(os.Args[2:]); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			return
		}
	}

	// Fallback to legacy behavior if first arg is a flag (starts with -)
	if len(os.Args[1]) > 0 && os.Args[1][0] == '-' {
		if err := runInfer(os.Args[1:]); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	fmt.Fprintf(os.Stderr, "Unknown command: %s\n", cmdName)
	printUsage()
	os.Exit(1)
}

func printUsage() {
	fmt.Println("Usage: orez-infer <command> [arguments]")
	fmt.Println("\nAvailable commands:")
	for _, cmd := range commands {
		fmt.Printf("  %-10s %s\n", cmd.Name, cmd.Description)
	}
}

func runDownload(args []string) error {
	fs := flag.NewFlagSet("download", flag.ExitOnError)
	repoID := fs.String("repo", "", "Hugging Face repository ID (e.g., orez-sh/gemma-4-E2B)")
	format := fs.String("format", "gguf", "Model format: gguf, safetensors, litert")
	dest := fs.String("dest", "models", "Destination directory")
	quant := fs.String("quant", "", "Quantization level for GGUF (e.g., Q4_K_M)")
	token := fs.String("token", "", "Hugging Face API token")

	if err := fs.Parse(args); err != nil {
		return err
	}

	if *repoID == "" {
		return fmt.Errorf("--repo is required")
	}

	mgr := downloader.NewManager(*token)
	opts := downloader.DownloadOptions{
		RepoID:  *repoID,
		Format:  downloader.Format(*format),
		DestDir: *dest,
		Quant:   *quant,
	}

	return mgr.Download(opts)
}

func runInfer(args []string) error {
	fs := flag.NewFlagSet("infer", flag.ExitOnError)
	modelPath := fs.String("model", "", "Path to the .gguf model file")
	backendFlag := fs.String("backend", "cpu", "Backend to use: cpu, metal, xla")
	temp := fs.Float64("temp", 0.7, "Sampling temperature")
	maxTokens := fs.Int("max-tokens", 128, "Maximum tokens to generate")

	if err := fs.Parse(args); err != nil {
		return err
	}

	if *modelPath == "" {
		return fmt.Errorf("--model is required")
	}

	// 1. Initialize Backend
	be, err := backend.Init(*backendFlag)
	if err != nil {
		return fmt.Errorf("error initializing backend %s: %v", *backendFlag, err)
	}
	fmt.Printf("🚀 Running on backend: %s\n", be.Name())

	// 2. Parse GGUF
	fmt.Printf("📂 Loading model: %s\n", *modelPath)
	reader, err := gguf.NewReader(*modelPath)
	if err != nil {
		return fmt.Errorf("error opening model: %v", err)
	}
	defer reader.Close()

	ggufFile, err := reader.Parse()
	if err != nil {
		return fmt.Errorf("error parsing GGUF: %v", err)
	}

	// 3. Registry Lookup & Config
	archEntry, ok := ggufFile.Metadata["general.architecture"]
	if !ok {
		return fmt.Errorf("general.architecture not found in metadata")
	}
	archName := archEntry.Value.(string)
	fmt.Printf("🏗  Architecture: %s\n", archName)

	registry := model.NewArchRegistry()
	builder, ok := registry.Get(archName)
	if !ok {
		// Try fallback or exact match check
		fmt.Printf("⚠️  Exact match for %s not found, trying normalized search...\n", archName)
		for _, name := range []string{"llama", "gemma-3", "gemma-4", "recurrent", "graphite", "granite", "bitnet-b1.58"} {
			if strings.Contains(strings.ToLower(archName), name) {
				builder, _ = registry.Get(name)
				ok = true
				break
			}
		}
	}

	if !ok {
		return fmt.Errorf("unsupported architecture: %s", archName)
	}

	// 4. Extract Model Config from GGUF
	config := extractConfig(ggufFile, archName)
	config.MaxSeqLen = *maxTokens

	fmt.Println("✅ Model loaded and weights mapped (Inference loop placeholder)")
	_ = temp
	_ = builder
	return nil
}

func extractConfig(f *gguf.File, arch string) model.ModelConfig {
	cfg := model.ModelConfig{Name: arch}

	getMeta := func(key string, fallback any) any {
		if entry, ok := f.Metadata[key]; ok {
			return entry.Value
		}
		return fallback
	}

	prefix := arch
	cfg.HiddenSize = int(getMeta(prefix+".embedding_length", uint64(0)).(uint64))
	cfg.NumHeads = int(getMeta(prefix+".attention.head_count", uint64(0)).(uint64))
	cfg.NumKVHeads = int(getMeta(prefix+".attention.head_count_kv", uint64(0)).(uint64))
	cfg.NumLayers = int(getMeta(prefix+".block_count", uint64(0)).(uint64))
	cfg.IntermediateSize = int(getMeta(prefix+".feed_forward_length", uint64(0)).(uint64))
	cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	cfg.RoPEBase = float64(getMeta(prefix+".attention.rope.freq_base", float32(10000.0)).(float32))
	cfg.RMSNormEPS = float64(getMeta(prefix+".attention.layer_norm_rms_epsilon", float32(1e-6)).(float32))

	// PLE & TurboQuant detection
	cfg.UsePLE = getMeta(prefix+".use_ple", false).(bool)
	cfg.TurboQuantPLE = getMeta(prefix+".turboquant_ple", false).(bool)
	cfg.TurboQuantKV = getMeta(prefix+".turboquant_kv", false).(bool)

	return cfg
	}
func runServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	port := fs.Int("port", 8080, "Port to listen on")
	backendFlag := fs.String("backend", "cpu", "Backend to use: cpu, metal, xla")
	modelPath := fs.String("model", "", "Path to the .gguf model file")
	tokenizerPath := fs.String("tokenizer", "", "Path to the .model tokenizer file (optional)")

	if err := fs.Parse(args); err != nil {
		return err
	}

	// 1. Initialize Backend
	be, err := backend.Init(*backendFlag)
	if err != nil {
		return fmt.Errorf("error initializing backend %s: %v", *backendFlag, err)
	}

	// 2. Initialize ML Context
	ctx := context.New()

	var runner *model.Runner
	var t tokenizer.Tokenizer = &tokenizer.SimpleTokenizer{}

	// 3. Load Model if provided
	if *modelPath != "" {
		fmt.Printf("📂 Loading model: %s\n", *modelPath)
		reader, err := gguf.NewReader(*modelPath)
		if err != nil {
			return fmt.Errorf("error opening model: %v", err)
		}
		defer reader.Close()

		ggufFile, err := reader.Parse()
		if err != nil {
			return fmt.Errorf("error parsing GGUF: %v", err)
		}

		archEntry, _ := ggufFile.Metadata["general.architecture"]
		arch := archEntry.Value.(string)
		
		registry := model.NewArchRegistry()
		builder, ok := registry.Get(arch)
		if !ok {
			// Try fallback
			for _, name := range []string{"llama", "gemma-3", "gemma-4", "recurrent", "graphite", "granite", "bitnet-b1.58"} {
				if strings.Contains(strings.ToLower(arch), name) {
					builder, _ = registry.Get(name)
					ok = true
					break
				}
			}
		}

		if !ok {
			return fmt.Errorf("unsupported architecture: %s", arch)
		}

		config := extractConfig(ggufFile, arch)
		runner, err = model.NewRunner(be, ctx, builder, config)
		if err != nil {
			return fmt.Errorf("failed to create runner: %v", err)
		}

		// Initialize variables
		if err := ctx.InitializeVariables(be, nil); err != nil {
			return fmt.Errorf("failed to initialize variables: %v", err)
		}

		if *tokenizerPath != "" {
			t, err = tokenizer.NewSentencePiece(*tokenizerPath)
			if err != nil {
				return fmt.Errorf("failed to load tokenizer: %v", err)
			}
		}
	}

	// 4. Setup and Start API Server
	srv := server.New(be, ctx, *port, runner, t)
	return srv.Start()
}
