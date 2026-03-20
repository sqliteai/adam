# WASM Chat

Browser-based chat UI powered by Adam compiled to WebAssembly. The full agent loop runs in the browser — LLM API calls go through JavaScript `fetch()` bridged to Adam's HTTP layer via Emscripten Asyncify. No server needed.

## Build & Run

```bash
make         # builds adam.js + adam.wasm (requires Emscripten)
make serve   # starts local server on http://localhost:8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

## Usage

1. Select a provider (Anthropic, OpenAI, or Gemini)
2. Enter your API key
3. Optionally set a model name
4. Click **Connect** and start chatting

Your API key stays in the browser — it's only sent to the LLM provider's API, never to any other server.

## How It Works

```
Browser                          LLM Provider
───────                          ────────────
User types message
  → adam_run() in WASM
    → adam_net_post_json()
      → EM_ASYNC_JS → fetch()  ──→  POST /v1/messages
      ← response JSON          ←──  200 OK + JSON
    → adam_json_parse_response()
  ← result.final_response
Display in chat UI
```

The key piece is the Emscripten net implementation (`adam.c`): `adam_net_post_json` calls an `EM_ASYNC_JS` function that does `await fetch()`. Emscripten's Asyncify allows the synchronous C code to suspend while JavaScript completes the async HTTP request, then resume with the response.

Streaming is not supported in the WASM build — the streaming path returns an error and the agent loop falls back to non-streaming HTTP automatically.

## Requirements

- [Emscripten](https://emscripten.org/docs/getting_started/downloads.html) (`emcc` in PATH)
- A modern browser (Chrome, Firefox, Safari, Edge)
- An API key for any supported provider
