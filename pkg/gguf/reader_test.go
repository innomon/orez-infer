package gguf

import (
	"bytes"
	"encoding/binary"
	"os"
	"testing"
)

func TestParseGGUF(t *testing.T) {
	// Create a mock GGUF v3 file
	buf := new(bytes.Buffer)

	// Header
	buf.Write([]byte(Magic))
	binary.Write(buf, binary.LittleEndian, uint32(3))      // Version
	binary.Write(buf, binary.LittleEndian, uint64(1))      // Tensor Count
	binary.Write(buf, binary.LittleEndian, uint64(2))      // Metadata Count

	// Metadata 1: general.architecture (string)
	writeGGUFString(buf, "general.architecture")
	binary.Write(buf, binary.LittleEndian, uint32(TypeString))
	writeGGUFString(buf, "llama")

	// Metadata 2: general.alignment (uint32)
	writeGGUFString(buf, "general.alignment")
	binary.Write(buf, binary.LittleEndian, uint32(TypeUint32))
	binary.Write(buf, binary.LittleEndian, uint32(32))

	// Tensor 1
	writeGGUFString(buf, "token_embd.weight")
	binary.Write(buf, binary.LittleEndian, uint32(2)) // n_dims
	binary.Write(buf, binary.LittleEndian, uint64(128)) // dim 0
	binary.Write(buf, binary.LittleEndian, uint64(64))  // dim 1
	binary.Write(buf, binary.LittleEndian, uint32(0))   // type F32
	binary.Write(buf, binary.LittleEndian, uint64(0))   // offset (relative to data start)

	// Write to a temporary file
	tmpFile, err := os.CreateTemp("", "test.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	
	if _, err := tmpFile.Write(buf.Bytes()); err != nil {
		t.Fatal(err)
	}
	// Add some padding to simulate the start of data
	padding := make([]byte, 32)
	if _, err := tmpFile.Write(padding); err != nil {
		t.Fatal(err)
	}
	tmpFile.Close()

	// Parse
	reader, err := NewReader(tmpFile.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()

	file, err := reader.Parse()
	if err != nil {
		t.Fatal(err)
	}

	// Verify
	if file.Header.Version != 3 {
		t.Errorf("expected version 3, got %d", file.Header.Version)
	}
	if file.Header.TensorCount != 1 {
		t.Errorf("expected 1 tensor, got %d", file.Header.TensorCount)
	}
	
	arch, ok := file.Metadata["general.architecture"]
	if !ok || arch.Value.(string) != "llama" {
		t.Errorf("expected architecture llama, got %v", arch.Value)
	}

	if len(file.Tensors) != 1 || file.Tensors[0].Name != "token_embd.weight" {
		t.Errorf("tensor mismatch")
	}
}

func writeGGUFString(b *bytes.Buffer, s string) {
	binary.Write(b, binary.LittleEndian, uint64(len(s)))
	b.Write([]byte(s))
}
