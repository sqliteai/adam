# Image Generation

Demonstrates generating images using Gemini's image generation model. The program sends a descriptive prompt and receives a response that includes image data as a base64-encoded data URI.

## Build & Run

```bash
echo "GEMINI_API_KEY=AI..." > .env

make
./image-generation
```

## What It Demonstrates

- Using Gemini's image generation model (`gemini-2.0-flash-preview-image-generation`)
- Sending a creative prompt and receiving image data in the response
- Detecting and handling data URI responses
