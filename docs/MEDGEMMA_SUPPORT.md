# MedGemma 1.5 Support in orez-infer

`orez-infer` provides native support for **MedGemma 1.5**, a specialized multimodal model optimized for medical precision. This document outlines how to use MedGemma 1.5 within `orez-infer`.

## Key Features

- **Multimodal Inference:** Seamlessly process both medical text and imaging data.
- **SigLIP Vision Encoder:** Native GoMLX implementation of the SigLIP architecture for visual feature extraction.
- **OpenAI Compatibility:** Full support for multimodal chat completions via the `/v1/chat/completions` endpoint.
- **Hardware Acceleration:** Metal-accelerated vision and transformer layers on Mac M4.

## Getting Started

### 1. Download MedGemma 1.5 Weights
Use the unified downloader to fetch the GGUF model and the SentencePiece tokenizer.

```bash
./orez-infer download --repo orez-sh/medgemma-1.5-4b-it --quant Q4_K_M
```

### 2. Start the API Server
Launch the server with the MedGemma model. The server will automatically detect the architecture and initialize the SigLIP vision encoder.

```bash
./orez-infer serve \
  --model models/medgemma-1.5-4b-it.gguf \
  --tokenizer models/tokenizer.model \
  --backend metal \
  --port 8080
```

### 3. Perform Multimodal Inference
You can interact with the model using any OpenAI-compatible client. Here is an example using `curl`:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "medgemma-1.5-4b-it",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Analyze this chest X-ray for potential abnormalities."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/chest-xray.jpg"
            }
          }
        ]
      }
    ]
  }'
```

## Architecture Details

MedGemma 1.5 integration in `orez-infer` leverages the following components:

- **`Gemma3Builder`**: Handles the hybrid graph construction for vision and text tokens.
- **`SigLIPVisionEncoder`**: A GoMLX-native implementation of the vision backbone.
- **`InterleaveTokens`**: Combines visual embeddings with text embeddings in the unified manifold.
- **`Runner.GenerateMultimodal`**: The core inference loop that handles simultaneous text and image tensor execution.

## Limitations

- **Image Preprocessing:** Currently, the API server uses a placeholder for image downloading. Real-world usage requires local preprocessing of images into normalized tensors [1, 3, H, W].
- **State Management:** KV-cache for multimodal tokens is still being optimized for very long clinical contexts.
