# Local Vision

Demonstrates multimodal image understanding using a local vision model (e.g. Gemma-3, LLaVA). The program loads a GGUF language model and its companion multimodal projector, reads an image file from disk, attaches it to the conversation, and asks the model to describe what it sees. Everything runs offline.

## Build & Run

```bash
make

# Requires a vision-capable GGUF model and its mmproj file
./local-vision model.gguf mmproj.gguf photo.jpg
```

If no image path is given, it defaults to `image.jpg` in the current directory.

## What It Demonstrates

- Configuring a multimodal local model with `adam_settings_set_local` and `adam_settings_set_mmproj`
- Loading and attaching image data with `adam_history_attach`
- Auto-detecting image media type from file extension
- Reading binary files and passing raw bytes to the library
