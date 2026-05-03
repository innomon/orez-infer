package downloader

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
)

// HFClient handles interaction with the Hugging Face Hub.
type HFClient struct {
	Token string
}

// NewHFClient creates a new client. It looks for HF_TOKEN env var if token is empty.
func NewHFClient(token string) *HFClient {
	if token == "" {
		token = os.Getenv("HF_TOKEN")
	}
	return &HFClient{Token: token}
}

// HFModelInfo contains basic information about a model from HF API.
type HFModelInfo struct {
	ID        string   `json:"id"`
	Siblings  []HFFile `json:"siblings"`
	Config    any      `json:"config"`
}

// HFFile represents a file in a repository.
type HFFile struct {
	Rpath string `json:"rpath"`
}

// GetModelInfo fetches metadata for a repository.
func (c *HFClient) GetModelInfo(repoID string) (*HFModelInfo, error) {
	url := fmt.Sprintf("https://huggingface.co/api/models/%s", repoID)
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}

	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get model info: %s", resp.Status)
	}

	var info HFModelInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, err
	}

	return &info, nil
}

// ResolveURL returns the download URL for a file in a repo.
func (c *HFClient) ResolveURL(repoID, rpath string) string {
	return fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repoID, rpath)
}
