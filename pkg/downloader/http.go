package downloader

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"golang.org/x/term"
)

// Progress holds the current state of a download.
type Progress struct {
	Total      int64
	Current    int64
	StartTime  time.Time
	LastUpdate time.Time
}

// Write implements the io.Writer interface for progress tracking.
func (p *Progress) Write(b []byte) (int, error) {
	n := len(b)
	p.Current += int64(n)
	p.Report()
	return n, nil
}

// Report prints the progress to the console.
func (p *Progress) Report() {
	now := time.Now()
	if now.Sub(p.LastUpdate) < 100*time.Millisecond && p.Current < p.Total {
		return
	}
	p.LastUpdate = now

	width, _, err := term.GetSize(int(os.Stdout.Fd()))
	if err != nil {
		width = 80
	}

	percent := float64(p.Current) / float64(p.Total) * 100
	speed := float64(p.Current) / now.Sub(p.StartTime).Seconds() / 1024 / 1024 // MB/s
	
	barWidth := width - 40
	if barWidth < 10 {
		barWidth = 10
	}
	
	filled := int(float64(barWidth) * percent / 100)
	bar := strings.Repeat("=", filled) + strings.Repeat(" ", barWidth-filled)

	fmt.Printf("\r[%s] %5.1f%% %6.1f MB/s", bar, percent, speed)
	if p.Current >= p.Total {
		fmt.Println()
	}
}

// DownloadFile downloads a file from url to dest.
func DownloadFile(url, dest string) error {
	if err := os.MkdirAll(filepath.Dir(dest), 0755); err != nil {
		return err
	}

	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()

	progress := &Progress{
		Total:      resp.ContentLength,
		StartTime:  time.Now(),
		LastUpdate: time.Now(),
	}

	_, err = io.Copy(f, io.TeeReader(resp.Body, progress))
	return err
}
