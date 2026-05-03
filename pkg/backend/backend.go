package backend

import (
	"fmt"
	"os"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/simplego"
	_ "github.com/gomlx/gomlx/backends/xla"
)

// Init initializes the GoMLX backend based on the provided name.
func Init(name string) (backends.Backend, error) {
	backendName := "go"
	switch name {
	case "metal":
		backendName = "go-darwinml"
	case "xla":
		backendName = "xla"
	case "cpu":
		backendName = "go"
	default:
		backendName = name
	}

	os.Setenv("GOMLX_BACKEND", backendName)
	fmt.Printf("Initializing GoMLX backend: %s\n", backendName)
	return backends.New()
}
