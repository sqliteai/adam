# Local Vision

Demonstrates multimodal image understanding using a local vision model. The program loads a GGUF language model and its companion multimodal projector (mmproj), reads an image file from disk, attaches it to the conversation, and asks the model to describe what it sees. Everything runs offline.

## Models

You need two GGUF files: the language model and its vision projector (mmproj).

**Gemma 3** (recommended) — [ggml-org/gemma-3](https://huggingface.co/collections/ggml-org/gemma-3-6804993feeaa7e12026ee4f9)

| Model | Size | Links |
|-------|------|-------|
| Gemma 3 4B | ~2.4 GB (Q4_K_S) | [model](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF) + [mmproj](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/blob/main/mmproj-gemma-3-4b-it-F16.gguf) |
| Gemma 3 12B | ~7.1 GB (Q4_K_M) | [model](https://huggingface.co/ggml-org/gemma-3-12b-it-GGUF) + [mmproj](https://huggingface.co/ggml-org/gemma-3-12b-it-GGUF/blob/main/mmproj-gemma-3-12b-it-F16.gguf) |

**Other vision models** — [ggml-org/multimodal GGUFs](https://huggingface.co/collections/ggml-org/multimodal-ggufs-68262e2bf5b6b66ebf28f77a)

Any model supported by llama.cpp's mtmd library works: LLaVA, MiniCPM-V, Qwen-VL, InternVL, Pixtral, etc.

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
