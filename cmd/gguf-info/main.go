package main

import (
	"flag"
	"fmt"
	"os"
	"sort"

	"github.com/innomon/orez-infer/pkg/gguf"
)

func main() {
	modelPath := flag.String("model", "", "Path to the GGUF model file")
	showTensors := flag.Bool("tensors", false, "Show tensor information")
	flag.Parse()

	if *modelPath == "" {
		fmt.Println("Usage: gguf-info -model <path> [-tensors]")
		os.Exit(1)
	}

	reader, err := gguf.NewReader(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening file: %v\n", err)
		os.Exit(1)
	}
	defer reader.Close()

	file, err := reader.Parse()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing GGUF: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("GGUF File: %s\n", *modelPath)
	fmt.Printf("Version:  %d\n", file.Header.Version)
	fmt.Printf("Tensors:  %d\n", file.Header.TensorCount)
	fmt.Printf("Metadata: %d\n", file.Header.MetadataCount)

	fmt.Println("\n--- Metadata ---")
	keys := make([]string, 0, len(file.Metadata))
	for k := range file.Metadata {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		entry := file.Metadata[k]
		fmt.Printf("%-40s [%-8s] %v\n", k, entry.Type.String(), entry.Value)
	}

	if *showTensors {
		fmt.Println("\n--- Tensors ---")
		for i, t := range file.Tensors {
			fmt.Printf("%3d: %-50s %v (Type: %d, Offset: %d)\n", i, t.Name, t.Dimensions, t.Type, t.Offset)
		}
	}
}
