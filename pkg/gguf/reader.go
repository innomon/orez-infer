package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"syscall"
)

// Reader provides methods to parse a GGUF file.
type Reader struct {
	file *os.File
	pos  int64
}

// NewReader creates a new GGUF reader for the given file path.
func NewReader(path string) (*Reader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	return &Reader{file: f}, nil
}

// Close closes the underlying file.
func (r *Reader) Close() error {
	return r.file.Close()
}

// ReadHeader parses the GGUF header.
func (r *Reader) ReadHeader() (Header, error) {
	var h Header
	if err := binary.Read(r.file, binary.LittleEndian, &h.Magic); err != nil {
		return h, err
	}
	if string(h.Magic[:]) != Magic {
		return h, fmt.Errorf("invalid magic: expected %s, got %s", Magic, string(h.Magic[:]))
	}

	if err := binary.Read(r.file, binary.LittleEndian, &h.Version); err != nil {
		return h, err
	}
	if h.Version < 2 {
		return h, fmt.Errorf("unsupported GGUF version: %d", h.Version)
	}

	if err := binary.Read(r.file, binary.LittleEndian, &h.TensorCount); err != nil {
		return h, err
	}
	if err := binary.Read(r.file, binary.LittleEndian, &h.MetadataCount); err != nil {
		return h, err
	}

	return h, nil
}

// ReadString reads a GGUF string.
func (r *Reader) ReadString() (string, error) {
	// GGUF v2+ uses uint64 for string length.
	var length uint64
	if err := binary.Read(r.file, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r.file, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

// ReadValue reads a metadata value based on its type.
func (r *Reader) ReadValue(vt ValueType) (any, error) {
	switch vt {
	case TypeUint8:
		var v uint8
		err := binary.Read(r.file, binary.LittleEndian, &v)
		return v, err
	case TypeInt8:
		var v int8
		err := binary.Read(r.file, binary.LittleEndian, &v)
		return v, err
	case TypeUint16:
		var v uint16
		err := binary.Read(r.file, binary.LittleEndian, &v)
		return v, err
	case TypeInt16:
		var v int16
		err := binary.Read(r.file, binary.LittleEndian, &v)
		return v, err
	case TypeUint32:
		var v uint32
		err := binary.Read(r.file, binary.LittleEndian, &v)
		return v, err
	case TypeInt32:
		var v int32
		err := binary.Read(r.file, binary.LittleEndian, &v)
		return v, err
	case TypeFloat32:
		var v float32
		err := binary.Read(r.file, binary.LittleEndian, &v)
		return v, err
	case TypeBool:
		var v bool
		err := binary.Read(r.file, binary.LittleEndian, &v)
		return v, err
	case TypeString:
		return r.ReadString()
	case TypeArray:
		innerType, err := r.ReadValueType()
		if err != nil {
			return nil, err
		}
		var length uint64
		if err := binary.Read(r.file, binary.LittleEndian, &length); err != nil {
			return nil, err
		}
		arr := make([]any, length)
		for i := uint64(0); i < length; i++ {
			val, err := r.ReadValue(innerType)
			if err != nil {
				return nil, err
			}
			arr[i] = val
		}
		return arr, nil
	case TypeUint64:
		var v uint64
		err := binary.Read(r.file, binary.LittleEndian, &v)
		return v, err
	case TypeInt64:
		var v int64
		err := binary.Read(r.file, binary.LittleEndian, &v)
		return v, err
	case TypeFloat64:
		var v float64
		err := binary.Read(r.file, binary.LittleEndian, &v)
		return v, err
	default:
		return nil, fmt.Errorf("unsupported value type: %v", vt)
	}
}

// ReadValueType reads the metadata value type.
func (r *Reader) ReadValueType() (ValueType, error) {
	var vt uint32
	if err := binary.Read(r.file, binary.LittleEndian, &vt); err != nil {
		return 0, err
	}
	return ValueType(vt), nil
}

// Mmap maps the file into memory.
func (r *Reader) Mmap() ([]byte, error) {
	info, err := r.file.Stat()
	if err != nil {
		return nil, err
	}
	return syscall.Mmap(int(r.file.Fd()), 0, int(info.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
}

// ReadMetadata parses the metadata section.
func (r *Reader) ReadMetadata(count uint64) (map[string]MetadataEntry, error) {
	metadata := make(map[string]MetadataEntry)
	for i := uint64(0); i < count; i++ {
		key, err := r.ReadString()
		if err != nil {
			return nil, fmt.Errorf("failed to read metadata key at index %d: %v", i, err)
		}
		vt, err := r.ReadValueType()
		if err != nil {
			return nil, fmt.Errorf("failed to read metadata type for key %s: %v", key, err)
		}
		val, err := r.ReadValue(vt)
		if err != nil {
			return nil, fmt.Errorf("failed to read metadata value for key %s: %v", key, err)
		}
		metadata[key] = MetadataEntry{Key: key, Type: vt, Value: val}
	}
	return metadata, nil
}

// ReadTensors parses the tensor information section.
func (r *Reader) ReadTensors(count uint64) ([]TensorInfo, error) {
	tensors := make([]TensorInfo, count)
	for i := uint64(0); i < count; i++ {
		name, err := r.ReadString()
		if err != nil {
			return nil, fmt.Errorf("failed to read tensor name at index %d: %v", i, err)
		}
		var nDims uint32
		if err := binary.Read(r.file, binary.LittleEndian, &nDims); err != nil {
			return nil, fmt.Errorf("failed to read tensor n_dims for %s: %v", name, err)
		}
		dims := make([]uint64, nDims)
		for j := uint32(0); j < nDims; j++ {
			if err := binary.Read(r.file, binary.LittleEndian, &dims[j]); err != nil {
				return nil, fmt.Errorf("failed to read tensor dim %d for %s: %v", j, name, err)
			}
		}
		var ggmlType uint32
		if err := binary.Read(r.file, binary.LittleEndian, &ggmlType); err != nil {
			return nil, fmt.Errorf("failed to read tensor type for %s: %v", name, err)
		}
		var offset uint64
		if err := binary.Read(r.file, binary.LittleEndian, &offset); err != nil {
			return nil, fmt.Errorf("failed to read tensor offset for %s: %v", name, err)
		}
		tensors[i] = TensorInfo{
			Name:       name,
			Dimensions: dims,
			Type:       ggmlType,
			Offset:     offset,
		}
	}
	return tensors, nil
}

// Parse orchestrates the full parsing of the GGUF file.
func (r *Reader) Parse() (*File, error) {
	header, err := r.ReadHeader()
	if err != nil {
		return nil, err
	}

	metadata, err := r.ReadMetadata(header.MetadataCount)
	if err != nil {
		return nil, err
	}

	tensors, err := r.ReadTensors(header.TensorCount)
	if err != nil {
		return nil, err
	}

	data, err := r.Mmap()
	if err != nil {
		return nil, fmt.Errorf("mmap failed: %v", err)
	}

	// Calculate alignment padding after metadata/tensor info
	// GGUF alignment is usually 32 bytes by default.
	// The tensor data starts at an offset relative to the end of the header/metadata/tensor info block.
	// Actually, the 'offset' in TensorInfo is relative to the start of the data block.
	// We need to find where the data block starts.
	
	dataOffset, err := r.file.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}
	
	// Alignment
	alignment := uint64(32)
	if val, ok := metadata["general.alignment"]; ok {
		if a, ok := val.Value.(uint32); ok {
			alignment = uint64(a)
		} else if a, ok := val.Value.(uint64); ok {
			alignment = a
		}
	}
	
	padding := (uint64(dataOffset) + alignment - 1) / alignment * alignment - uint64(dataOffset)
	actualDataStart := uint64(dataOffset) + padding

	// Adjust tensor offsets to be absolute within the mmapped data
	for i := range tensors {
		tensors[i].Offset += actualDataStart
	}

	return &File{
		Header:   header,
		Metadata: metadata,
		Tensors:  tensors,
		Data:     data,
	}, nil
}
