# Adam

Embeddable AI agent library in C.

Adam gives you a complete agent loop: tool calling, memory, sessions, voice, streaming, structured output, in one `#include`. Works with cloud APIs (Anthropic, OpenAI, Google Gemini, Groq, Together, xAI) and local models (llama.cpp) through the same interface. Compiles on macOS, Linux, Windows, iOS, Android, and WASM.

## Quick Start

```c
#include "adam.h"

int main(void) {
    adam_init();

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_provider(s, ADAM_API_ANTHROPIC,
                               getenv("ANTHROPIC_API_KEY"),
                               "claude-sonnet-4-20250514");

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "What is the capital of France?");

    printf("%s\n", r.final_response);  // "The capital of France is Paris."

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
    adam_cleanup();
}
```

```bash
make deps    # build llama.cpp + whisper.cpp
make all     # build libadam.a
make test    # run 161 tests (ASan + UBSan)
```

## Features

| Feature | Description |
|---------|-------------|
| **Agent loop** | Tool calling with automatic iteration until final response |
| **Three providers** | Anthropic, OpenAI, Google Gemini + any compatible API + local GGUF via llama.cpp |
| **Local vision** | Multimodal image understanding via llama.cpp + mmproj (Gemma 3, LLaVA, etc.) |
| **Image generation** | Native image output via Gemini image models (gemini-3.1-flash-image-preview) |
| **13 built-in tools** | File I/O, shell, calculator, SQL, web fetch/search, HTTP POST, memory, research, multi-agent |
| **Long-term memory** | Hybrid BM25 + vector search via SQLite (sqlite-memory + sqlite-vector) |
| **Session persistence** | Save/load conversations with UUIDv7 keys |
| **Voice** | STT (Whisper cloud/local) + TTS (cloud/system) + full audio pipeline |
| **Streaming** | Real-time token delivery via callback |
| **Structured output** | `adam_run_json()` with validation and retry |
| **Evolution loop** | Self-improving agent: iterate, score, refine strategy |
| **Research mode** | Autonomous multi-iteration information gathering with report synthesis |
| **Multi-agent** | Agent A invokes Agent B as a tool, with independent settings/tools |
| **Guardrails** | Pre-send and post-receive validation callbacks |
| **Response cache** | LRU hash table keyed on model + message history |
| **History management** | Clone, summarize (LLM-based compression), token estimation |
| **Thread pool** | Concurrent agent execution with job queue |
| **Filesystem sandbox** | Tools restricted to explicitly allowed directories |
| **Cross-platform** | macOS, Linux, Windows, iOS, Android, WASM (Emscripten) |
| **Arena allocator** | Zero-leak per-iteration memory with automatic cleanup |

## Build

```bash
make deps          # Build llama.cpp, whisper.cpp (+ mbedtls/curl on Linux)
make all           # Build libadam.a
make test          # Build & run unit tests (ASan + UBSan)
make chat          # Interactive text chat (cloud API)
make chat GGUF=models/model.gguf  # Interactive text chat (local)
make vision GGUF=models/model.gguf MMPROJ=models/mmproj.gguf  # Local vision test
make talk          # Voice agent (cloud)
make talk LOCAL=1  # Voice agent (fully local)
make memory        # Memory system tests
make clean         # Remove all build artifacts
```

## API Reference

Full API documentation with every function, type, and callback: **[API.md](API.md)**

## Examples

```c
adam_settings_t *s = adam_create_settings();
adam_settings_set_provider(s, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");

adam_history_t *h = adam_history_create();
adam_run_result_t r = adam_run(s, h, "Explain quantum entanglement simply.");
printf("%s\n", r.final_response);
adam_run_result_free(&r);

// Continue the conversation — history carries forward
r = adam_run(s, h, "Can you give an analogy?");
printf("%s\n", r.final_response);
adam_run_result_free(&r);

adam_history_destroy(h);
adam_settings_destroy(s);
```

More examples are available in the **[examples/](examples/)** directory:

| Example | Description |
|---------|-------------|
| [simple-conversation](examples/simple-conversation/) | Multi-turn chat with a cloud API |
| [tool-calling](examples/tool-calling/) | Register a custom tool and let the agent call it |
| [local-model](examples/local-model/) | Run a local GGUF model via llama.cpp |
| [local-vision](examples/local-vision/) | Image understanding with a local vision model + mmproj |
| [google-gemini](examples/google-gemini/) | Use Google Gemini models |
| [image-generation](examples/image-generation/) | Generate images with Gemini |
| [structured-json](examples/structured-json/) | Get validated JSON output with retry |
| [memory](examples/memory/) | Long-term memory with hybrid BM25 + vector search |
| [sessions](examples/sessions/) | Save and restore conversations |
| [streaming](examples/streaming/) | Real-time token streaming via callback |
| [voice](examples/voice/) | Speech-to-text + agent + text-to-speech pipeline |
| [multi-agent](examples/multi-agent/) | Agent A delegates to Agent B as a tool |
| [evolution](examples/evolution/) | Self-improving agent with scoring loop |
| [research](examples/research/) | Autonomous multi-iteration research with report |
| [filesystem-sandbox](examples/filesystem-sandbox/) | File and shell tools restricted to allowed directories |
| [guardrails](examples/guardrails/) | Pre-send and post-receive validation |
| [response-cache](examples/response-cache/) | LRU cache for repeated queries |
| [thread-pool](examples/thread-pool/) | Concurrent agent execution |
| [full-agent](examples/full-agent/) | All features combined |

## Architecture

```
adam_run() loop:
  build system prompt (identity + instructions + bootstrap files + memory + datetime)
  -> check guardrails (on_before_send)
  -> check cache
  -> dispatch LLM (mock | local/llama.cpp | remote/HTTP)
  -> check guardrails (on_after_receive)
  -> if tool_calls: execute tools -> append results -> loop
  -> if text: return final response -> auto-save session -> extract memory
```

**Platform abstraction**: macOS uses NSURLSession, Linux uses libcurl+mbedtls, WASM uses embedder-provided `http_fn` callback.

**Memory management**: Arena allocators for per-iteration zero-copy work. malloc/free for long-lived structures. Arena-allocated strings are only valid within the current iteration.

## Feature Gates

Define before `#include "adam.h"` to disable features:

| Gate | Effect |
|------|--------|
| `ADAM_NO_CURL` | No libcurl (must provide `http_fn` callback) |
| `ADAM_NO_LOCAL` | No llama.cpp (no local inference) |
| `ADAM_NO_PTHREADS` | No thread pool, no voice thread |
| `ADAM_NO_SQLITE` | No SQLite (no memory, sessions, or SQL tool) |
| `ADAM_NO_VOICE` | No voice subsystem |
| `ADAM_NO_FILESYSTEM` | No file_read/file_write/list_directory tools |
| `ADAM_NO_SHELL` | No shell_exec tool |

## Dependencies

All vendored as git submodules in `modules/`:

| Module | Purpose |
|--------|---------|
| llama.cpp | Local LLM inference + GGML compute |
| whisper.cpp | Local speech-to-text (shares ggml via symlink) |
| miniaudio | Cross-platform audio I/O |
| sqlite | Amalgamation build |
| sqlite-memory | Hybrid BM25 + vector knowledge store |
| sqlite-vector | Vector similarity search |
| mbedtls | TLS (Linux only) |
| curl | HTTP (Linux only) |

macOS uses system frameworks (Foundation, Security, Metal, AVFoundation, Accelerate) instead of curl/mbedtls.

## License

MIT
