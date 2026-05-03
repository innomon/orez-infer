# Plan: Track-001 - GGUF Parser & Metadata Reader

Implement a native Go parser for GGUF files to extract metadata and tensor information without external dependencies.

## Objectives
- Read GGUF header and identify file version.
- Parse metadata key-value pairs (e.g., `general.architecture`, `llama.attention.head_count`).
- Map tensor info (name, shape, type, offset).
- Implement `syscall.Mmap` for efficient tensor data access.

## Tasks
1. **Define GGUF Structures:** Create Go structs for GGUF header, metadata entries, and tensor info.
2. **Implement Reader:** Build a sequential reader for the GGUF header and metadata block.
3. **Handle Data Types:** Support GGUF's diverse metadata value types (strings, ints, floats, arrays).
4. **Mmap Integration:** Use `syscall.Mmap` to map the weight data into memory.
5. **Validation Tool:** Create a small utility to print GGUF metadata to verify the parser.

## Verification
- Run the validation tool against a known GGUF file (e.g., Llama-3-8B).
- Verify metadata matches expected values from `gguf-py` or similar tools.
- Ensure mmap offsets correctly point to tensor data starts.
