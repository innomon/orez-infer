package gguf

import (
	"fmt"
	"unsafe"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// GGUF Constants
const (
	Magic = "GGUF"
)

// ValueType represents the type of a GGUF metadata value.
type ValueType uint32

const (
	TypeUint8   ValueType = 0
	TypeInt8    ValueType = 1
	TypeUint16  ValueType = 2
	TypeInt16   ValueType = 3
	TypeUint32  ValueType = 4
	TypeInt32   ValueType = 5
	TypeFloat32 ValueType = 6
	TypeBool    ValueType = 7
	TypeString  ValueType = 8
	TypeArray   ValueType = 9
	TypeUint64  ValueType = 10
	TypeInt64   ValueType = 11
	TypeFloat64 ValueType = 12
)

// Header represents the GGUF file header.
type Header struct {
	Magic           [4]byte
	Version         uint32
	TensorCount     uint64
	MetadataCount   uint64
}

// MetadataEntry represents a single key-value metadata pair.
type MetadataEntry struct {
	Key   string
	Type  ValueType
	Value any
}

// TensorInfo represents metadata about a tensor in the GGUF file.
type TensorInfo struct {
	Name       string
	Dimensions []uint64
	Type       uint32 // GGML type
	Offset     uint64
}

// File represents a parsed GGUF file.
type File struct {
	Header   Header
	Metadata map[string]MetadataEntry
	Tensors  []TensorInfo
	Data     []byte // Mmapped data
}

// TensorData returns the raw bytes for a given tensor.
func (f *File) TensorData(t TensorInfo) ([]byte, error) {
	size := f.calculateTensorSize(t)
	if t.Offset+size > uint64(len(f.Data)) {
		return nil, fmt.Errorf("tensor %s out of bounds", t.Name)
	}
	return f.Data[t.Offset : t.Offset+size], nil
}

func (f *File) calculateTensorSize(t TensorInfo) uint64 {
	numElements := uint64(1)
	for _, d := range t.Dimensions {
		numElements *= d
	}

	switch t.Type {
	case 0: // F32
		return numElements * 4
	case 1: // F16
		return numElements * 2
	default:
		return 0 
	}
}

// ToGoMLXTensor creates a GoMLX tensor from a TensorInfo using the mmapped data.
func (f *File) ToGoMLXTensor(t TensorInfo) (*tensors.Tensor, error) {
	size := f.calculateTensorSize(t)
	if t.Offset+size > uint64(len(f.Data)) {
		return nil, fmt.Errorf("tensor %s out of bounds", t.Name)
	}

	data := f.Data[t.Offset : t.Offset+size]
	
	dims := make([]int, len(t.Dimensions))
	for i, d := range t.Dimensions {
		dims[i] = int(d)
	}

	switch t.Type {
	case 0: // F32
		slice := (*[1 << 30]float32)(unsafe.Pointer(&data[0]))[:len(data)/4 : len(data)/4]
		return tensors.FromFlatDataAndDimensions(slice, dims...), nil
	default:
		// Return as raw bytes for quantized types
		return tensors.FromFlatDataAndDimensions(data, dims...), nil
	}
}

func (v ValueType) String() string {
	types := map[ValueType]string{
		TypeUint8:   "uint8",
		TypeInt8:    "int8",
		TypeUint16:  "uint16",
		TypeInt16:   "int16",
		TypeUint32:  "uint32",
		TypeInt32:   "int32",
		TypeFloat32: "float32",
		TypeBool:    "bool",
		TypeString:  "string",
		TypeArray:   "array",
		TypeUint64:  "uint64",
		TypeInt64:   "int64",
		TypeFloat64: "float64",
	}
	if s, ok := types[v]; ok {
		return s
	}
	return fmt.Sprintf("unknown(%d)", v)
}
