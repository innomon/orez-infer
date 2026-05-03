package downloader

import (
	"fmt"
	"path/filepath"
	"strings"
)

// Format represents the model weight format.
type Format string

const (
	FormatGGUF        Format = "gguf"
	FormatSafetensors Format = "safetensors"
	FormatLiteRT      Format = "litert"
	FormatUnknown     Format = "unknown"
)

// DownloadOptions configures the download process.
type DownloadOptions struct {
	RepoID   string
	Format   Format
	DestDir  string
	Quant    string // For GGUF, e.g., "Q4_K_M"
	Token    string
}

// Manager orchestrates the download process across formats.
type Manager struct {
	client *HFClient
}

func NewManager(token string) *Manager {
	return &Manager{client: NewHFClient(token)}
}

// Download performs the download based on options.
func (m *Manager) Download(opts DownloadOptions) error {
	info, err := m.client.GetModelInfo(opts.RepoID)
	if err != nil {
		return err
	}

	var files []string
	switch opts.Format {
	case FormatGGUF:
		files, err = m.resolveGGUF(info, opts.Quant)
	case FormatSafetensors:
		files, err = m.resolveSafetensors(info)
	case FormatLiteRT:
		files, err = m.resolveLiteRT(info)
	default:
		// Attempt to auto-detect if format is unknown
		return fmt.Errorf("unsupported or unknown format: %s", opts.Format)
	}

	if err != nil {
		return err
	}

	fmt.Printf("Downloading %d files to %s...\n", len(files), opts.DestDir)
	for _, rpath := range files {
		url := m.client.ResolveURL(opts.RepoID, rpath)
		dest := filepath.Join(opts.DestDir, rpath)
		fmt.Printf("-> %s\n", rpath)
		if err := DownloadFile(url, dest); err != nil {
			return err
		}
	}

	return nil
}

func (m *Manager) resolveGGUF(info *HFModelInfo, quant string) ([]string, error) {
	var candidates []string
	for _, f := range info.Siblings {
		if strings.HasSuffix(f.Rpath, ".gguf") {
			candidates = append(candidates, f.Rpath)
		}
	}

	if len(candidates) == 0 {
		return nil, fmt.Errorf("no GGUF files found in repo")
	}

	if quant != "" {
		for _, f := range candidates {
			if strings.Contains(strings.ToUpper(f), strings.ToUpper(quant)) {
				return []string{f}, nil
			}
		}
		return nil, fmt.Errorf("no GGUF file matching quant '%s' found", quant)
	}

	// Default to first one if only one, or return error if multiple
	if len(candidates) == 1 {
		return candidates, nil
	}

	return nil, fmt.Errorf("multiple GGUF files found, please specify --quant. Available: %v", candidates)
}

func (m *Manager) resolveSafetensors(info *HFModelInfo) ([]string, error) {
	var files []string
	required := []string{"config.json", "tokenizer.json", "tokenizer_config.json"}
	
	foundRequired := make(map[string]bool)
	for _, f := range info.Siblings {
		if strings.HasSuffix(f.Rpath, ".safetensors") {
			files = append(files, f.Rpath)
		}
		for _, r := range required {
			if f.Rpath == r {
				foundRequired[r] = true
				files = append(files, f.Rpath)
			}
		}
	}

	if len(files) == 0 || !strings.Contains(fmt.Sprintf("%v", files), ".safetensors") {
		return nil, fmt.Errorf("no safetensors found in repo")
	}

	return files, nil
}

func (m *Manager) resolveLiteRT(info *HFModelInfo) ([]string, error) {
	var files []string
	for _, f := range info.Siblings {
		if strings.HasSuffix(f.Rpath, ".litert") || strings.HasSuffix(f.Rpath, ".tflite") || f.Rpath == "tokenizer.model" {
			files = append(files, f.Rpath)
		}
	}

	if len(files) == 0 {
		return nil, fmt.Errorf("no LiteRT files found in repo")
	}

	return files, nil
}
