# Local Model

Demonstrates fully offline inference using a local GGUF model via llama.cpp. No API key or network connection is required. The model file path is passed as a command-line argument.

## Build & Run

```bash
make

# Download a GGUF model (e.g. from Hugging Face)
./local-model ../../models/your-model.gguf
```

## What It Demonstrates

- Configuring local inference with `adam_settings_set_local`
- Running a prompt entirely offline via llama.cpp
- Adjusting generation parameters (`temperature`, `max_tokens`)
