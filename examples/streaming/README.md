# Streaming

Demonstrates real-time token delivery using Adam's streaming callback. Instead of waiting for the full response, the `on_stream` callback fires for each token as the LLM generates it, allowing immediate display. The program asks for a short story and prints each token to stdout as it arrives, producing a typewriter-like effect.

## Build & Run

```bash
# Create a .env file with your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

make
./streaming
```

## What It Demonstrates

- Setting a stream callback with `adam_settings_set_stream`
- Real-time token-by-token output via `adam_stream_fn`
- The `is_done` flag to detect the final chunk
- `adam_run` still returns the complete response even when streaming
