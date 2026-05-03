package tokenizer

import (
	"strings"
	"github.com/eliben/go-sentencepiece"
)

// Tokenizer handles encoding and decoding of text.
type Tokenizer interface {
	Encode(text string) []int32
	Decode(ids []int32) string
}

// SentencePieceTokenizer implements Tokenizer using a .model file.
type SentencePieceTokenizer struct {
	proc *sentencepiece.Processor
}

func NewSentencePiece(modelPath string) (*SentencePieceTokenizer, error) {
	proc, err := sentencepiece.NewProcessorFromPath(modelPath)
	if err != nil {
		return nil, err
	}
	return &SentencePieceTokenizer{proc: proc}, nil
}

func (t *SentencePieceTokenizer) Encode(text string) []int32 {
	tokens := t.proc.Encode(text)
	ids := make([]int32, len(tokens))
	for i, token := range tokens {
		ids[i] = int32(token.ID)
	}
	return ids
}

func (t *SentencePieceTokenizer) Decode(ids []int32) string {
	intIds := make([]int, len(ids))
	for i, id := range ids {
		intIds[i] = int(id)
	}
	return t.proc.Decode(intIds)
}

// SimpleTokenizer is a fallback that uses space-based splitting.
type SimpleTokenizer struct{}

func (t *SimpleTokenizer) Encode(text string) []int32 {
	// Very naive: just use ASCII values as a placeholder
	var ids []int32
	for _, r := range text {
		ids = append(ids, int32(r))
	}
	return ids
}

func (t *SimpleTokenizer) Decode(ids []int32) string {
	var sb strings.Builder
	for _, id := range ids {
		sb.WriteRune(rune(id))
	}
	return sb.String()
}
