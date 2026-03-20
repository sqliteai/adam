# Adam API Reference

Complete API documentation for the Adam embeddable AI agent library.

## Table of Contents

- [Global Lifecycle](#global-lifecycle)
- [Settings](#settings)
- [History](#history)
- [Agent Run](#agent-run)
- [Structured JSON Output](#structured-json-output)
- [Cancellation](#cancellation)
- [Thread Pool](#thread-pool)
- [Voice](#voice)
- [Memory](#memory)
- [Sessions](#sessions)
- [Model Registry](#model-registry)
- [Response Cache](#response-cache)
- [Token Estimation](#token-estimation)
- [Evolution Loop](#evolution-loop)
- [Research Mode](#research-mode)
- [Built-in Tools](#built-in-tools)
- [Status Codes](#status-codes)
- [Enums](#enums)
- [Callbacks](#callbacks)
- [Feature Gates](#feature-gates)

---

## Global Lifecycle

```c
adam_status_t adam_init(void);
```
Initialize global state (libcurl, SQLite, llama.cpp backend). Call once at program start. Returns `ADAM_OK` on success.

```c
void adam_cleanup(void);
```
Clean up all global state. Call once at program exit.

---

## Settings

All settings have sensible defaults. Create with `adam_create_settings()`, modify what you need, pass to `adam_run()`.

**Thread safety:** A settings struct is **not thread-safe**. It contains mutable internal state (llama.cpp context, curl handles, rate limiter counters) that is modified during `adam_run()`. Each concurrent agent must use its own settings instance. The thread pool (`adam_pool_t`) enforces this by requiring a separate settings per job.

```c
adam_settings_t *adam_create_settings(void);
```
Create a settings struct with all defaults filled in. Must be freed with `adam_settings_destroy()`.

```c
void adam_settings_destroy(adam_settings_t *settings);
```
Destroy a settings struct and all owned resources (tools, bootstrap files, allowed dirs, curl handles).

### Provider Configuration

```c
adam_status_t adam_settings_set_provider(adam_settings_t *s,
    adam_api_format_t format, const char *api_key, const char *model);
```
Configure a remote LLM API provider. `format` is `ADAM_API_ANTHROPIC`, `ADAM_API_OPENAI`, or `ADAM_API_GEMINI`.

```c
adam_status_t adam_settings_set_base_url(adam_settings_t *s, const char *url);
```
Override the default API endpoint. Use for Groq, Together, Ollama, vLLM, LM Studio, etc.

```c
adam_status_t adam_settings_set_local(adam_settings_t *s,
    const char *gguf_path, int gpu_layers, int ctx_size);
```
Configure local GGUF inference via llama.cpp. Overrides remote API when set. `gpu_layers=-1` for all layers on GPU. `ctx_size=0` for model default. Requires `!ADAM_NO_LOCAL`.

```c
adam_status_t adam_settings_set_mmproj(adam_settings_t *s,
    const char *mmproj_path);
```
Set the multimodal projector GGUF for local vision models. When set, image attachments on user messages are processed through the vision encoder (via llama.cpp's mtmd library) before being fed to the LLM. The vision path activates automatically when `mmproj_path` is set and the conversation contains image attachments; text-only messages use the standard text path. Compatible with any vision model supported by llama.cpp (Gemma 3, LLaVA, MiniCPM-V, Qwen-VL, etc.). Requires `!ADAM_NO_LOCAL`.

### System Prompt

```c
adam_status_t adam_settings_set_identity(adam_settings_t *s, const char *text);
```
Set the agent's identity (first line of system prompt). Default: "You are a helpful assistant."

```c
adam_status_t adam_settings_set_instructions(adam_settings_t *s, const char *text);
```
Set additional instructions appended to the system prompt.

```c
adam_status_t adam_settings_add_bootstrap_file(adam_settings_t *s, const char *path);
```
Add a file whose contents are injected into the system prompt. Can be called multiple times.

### Tools

```c
adam_status_t adam_settings_add_tool(adam_settings_t *s, adam_tool_def_t tool);
```
Register a tool. The `tool.name` must be unique. `tool.execute` is the callback.

```c
adam_status_t adam_settings_remove_tool(adam_settings_t *s, const char *name);
```
Unregister a tool by name. Returns `ADAM_ERR_TOOL_NOT_FOUND` if not found.

### Filesystem Sandbox

```c
adam_status_t adam_settings_allow_dir(adam_settings_t *s, const char *dir_path);
```
Grant filesystem access to a directory for file/shell tools. Path is resolved via `realpath()` to prevent symlink escapes. Can be called multiple times. No allowed dirs = no filesystem access.

### Streaming

```c
adam_status_t adam_settings_set_stream(adam_settings_t *s,
    adam_stream_fn fn, void *ctx);
```
Enable streaming responses. `fn` is called for each token as it arrives.

### Logging

```c
adam_status_t adam_settings_set_logger(adam_settings_t *s,
    adam_log_fn fn, void *ctx, adam_log_level_t level);
```
Set logging callback and minimum level. The library never writes to stdout/stderr directly.

### Callbacks

```c
adam_status_t adam_settings_set_callbacks(adam_settings_t *s,
    void (*on_response)(void *, const char *, size_t),
    void (*on_tool_call)(void *, const char *, const char *),
    void (*on_error)(void *, adam_status_t, const char *),
    void *ctx);
```
Set response, tool call, and error notification callbacks.

### Custom Backends

```c
adam_status_t adam_settings_set_llm_callback(adam_settings_t *s,
    adam_llm_fn fn, void *ctx);
```
Replace the built-in LLM dispatch with a custom function. Primary testing injection point.

```c
adam_status_t adam_settings_set_http_callback(adam_settings_t *s,
    adam_http_fn fn, void *ctx);
```
Replace libcurl/NSURLSession with a custom HTTP transport (for WASM or custom networking).

### Memory Configuration

```c
adam_status_t adam_settings_set_memory(adam_settings_t *s,
    const char *embedding_provider, const char *embedding_model,
    const char *api_key);
```
Configure memory embedding provider. `embedding_provider`: "local", "openai", "voyage", etc.

### Voice Configuration

Requires `!ADAM_NO_VOICE && !ADAM_NO_PTHREADS`.

```c
adam_status_t adam_settings_set_stt(adam_settings_t *s,
    adam_stt_backend_t backend, const char *api_url,
    const char *api_key, const char *model);
```
Configure speech-to-text backend. `backend`: `ADAM_STT_CLOUD` or `ADAM_STT_LOCAL`.

```c
adam_status_t adam_settings_set_tts(adam_settings_t *s,
    adam_tts_backend_t backend, const char *api_url,
    const char *api_key, const char *model, const char *voice);
```
Configure text-to-speech backend. `backend`: `ADAM_TTS_CLOUD`, `ADAM_TTS_LOCAL`, or `ADAM_TTS_SYSTEM`.

```c
adam_status_t adam_settings_set_stt_callback(adam_settings_t *s,
    adam_stt_fn fn, void *ctx);
adam_status_t adam_settings_set_tts_callback(adam_settings_t *s,
    adam_tts_fn fn, void *ctx);
adam_status_t adam_settings_set_voice_callback(adam_settings_t *s,
    adam_voice_event_fn fn, void *ctx);
```
Set custom STT/TTS/voice event callbacks.

### Settings Fields

Key fields you can set directly on `adam_settings_t`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gguf_path` | `const char *` | NULL | Local GGUF model path (enables local inference) |
| `mmproj_path` | `const char *` | NULL | Multimodal projector GGUF for vision |
| `local_gpu_layers` | `int` | -1 | GPU layers (-1 = all) |
| `local_ctx_size` | `int` | 0 | Context window (0 = model default) |
| `local_batch_size` | `int` | 512 | Prompt batch size |
| `temperature` | `float` | 0.7 | Generation temperature |
| `max_tokens` | `int` | 4096 | Max output tokens |
| `response_format` | `const char *` | NULL | "json" for JSON mode |
| `top_p` | `float` | 1.0 | Nucleus sampling |
| `max_iterations` | `int` | 25 | Max tool-call iterations |
| `max_history` | `int` | 100 | Max messages before compression |
| `timeout_ms` | `int` | 0 | Wall-clock timeout (0 = none) |
| `retry_max` | `int` | 3 | Max retry attempts |
| `inject_memory` | `int` | 0 | 1 = enrich system prompt from memory |
| `inject_datetime` | `int` | 1 | 1 = inject current date/time |
| `session_id` | `const char *` | NULL | Session ID for auto-save |
| `auto_save` | `int` | 1 | Auto-save session after each run |
| `memory_extract` | `int` | 0 | 1 = extract facts after conversation |
| `on_before_send` | `adam_guardrail_fn` | NULL | Pre-send guardrail |
| `on_after_receive` | `adam_guardrail_response_fn` | NULL | Post-receive guardrail |
| `cache` | `adam_cache_t *` | NULL | Response cache |
| `abort_flag` | `volatile int` | 0 | Set to 1 to stop execution |

---

## History

```c
adam_history_t *adam_history_create(void);
```
Create an empty conversation history.

```c
void adam_history_destroy(adam_history_t *h);
```
Free history and all owned messages.

```c
void adam_history_clear(adam_history_t *h);
```
Clear all messages without destroying the history object.

```c
size_t adam_history_count(const adam_history_t *h);
```
Get the number of messages.

```c
adam_history_t *adam_history_clone(const adam_history_t *h);
```
Deep-copy an entire conversation history. All messages, tool calls, and attachments are independently allocated. Returns NULL on allocation failure.

```c
adam_status_t adam_history_append_user(adam_history_t *h, const char *content);
adam_status_t adam_history_append_assistant(adam_history_t *h,
    const char *content, const adam_tool_call_t *tool_calls,
    size_t tool_call_count);
adam_status_t adam_history_append_tool(adam_history_t *h,
    const char *content, const char *tool_call_id);
```
Append messages. All strings are copied via `strdup`.

```c
adam_status_t adam_history_attach(adam_history_t *h,
    adam_media_type_t type, const uint8_t *data, size_t len,
    const char *filename);
```
Attach multimodal media (image, audio, PDF) to the last message.

```c
size_t adam_history_estimate_tokens(const adam_history_t *h);
```
Estimate total token count (~4 chars per token heuristic).

```c
adam_status_t adam_history_summarize(adam_settings_t *s, adam_history_t *h,
    size_t target_token_count);
```
Use the LLM to compress old messages into a summary. Keeps system message and recent messages, replaces old ones with a single summary. If `target_token_count` is 0, uses 75% of model context window.

---

## Agent Run

```c
adam_run_result_t adam_run(adam_settings_t *settings,
    adam_history_t *history, const char *user_message);
```
Run the agent loop. Appends `user_message` to history, calls the LLM, executes tool calls iteratively until a final text response is produced. History is updated in-place. Returns a result struct — caller must call `adam_run_result_free()`.

```c
void adam_run_result_free(adam_run_result_t *result);
```
Free the `final_response` string in the result.

### Result Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | `adam_status_t` | ADAM_OK on success |
| `final_response` | `char *` | The agent's text response (malloc'd, caller frees) |
| `total_iterations` | `int` | Tool loop iterations executed |
| `input_tokens` | `int` | Total input tokens consumed |
| `output_tokens` | `int` | Total output tokens generated |
| `cost_usd` | `float` | Estimated cost in USD |
| `elapsed_ms` | `double` | Wall-clock time in milliseconds |
| `http_status` | `int` | HTTP status code on failure (0 for non-HTTP errors) |

The `http_status` field is populated on failure. For rate limit errors it is 429, for auth errors 401 or 403, for other provider errors the raw HTTP status code.

---

## Structured JSON Output

```c
adam_json_result_t adam_run_json(adam_settings_t *s, adam_history_t *h,
    const char *user_message, const char *json_hint, int max_retries);
```
Run the agent in JSON mode. Validates the response is valid JSON. If invalid, retries up to `max_retries` times with a correction prompt. `json_hint` (optional) describes the expected JSON structure.

```c
void adam_json_result_free(adam_json_result_t *r);
```
Free the JSON result.

```c
char *adam_json_extract(const char *json, const char *key);
```
Extract a top-level string/number value from JSON by key. Returns a `malloc`'d string or NULL. Caller must `free()`.

### JSON Result Fields

| Field | Type | Description |
|-------|------|-------------|
| `base` | `adam_run_result_t` | Embedded run result |
| `json_valid` | `int` | 1 if response was valid JSON |
| `retries_used` | `int` | Retries needed for valid JSON |

---

## Cancellation

```c
void adam_abort(adam_settings_t *s);
```
Thread-safe: set the abort flag to stop execution. Can be called from any thread or signal handler.

```c
void adam_abort_reset(adam_settings_t *s);
```
Reset the abort flag.

---

## Thread Pool

Requires `!ADAM_NO_PTHREADS`.

```c
adam_pool_t *adam_pool_create(int num_workers);
```
Create a thread pool. `num_workers=0` defaults to CPU core count.

```c
adam_status_t adam_pool_submit(adam_pool_t *pool, adam_job_t job);
```
Queue a job for execution. The `adam_job_t` struct is copied internally.

```c
void adam_pool_destroy(adam_pool_t *pool);
```
Wait for all jobs to complete, then destroy the pool.

```c
int adam_pool_pending(const adam_pool_t *pool);
int adam_pool_active(const adam_pool_t *pool);
```
Query pool status.

---

## Voice

Requires `!ADAM_NO_VOICE && !ADAM_NO_PTHREADS`.

```c
adam_status_t adam_voice_start(adam_settings_t *s, adam_history_t *h);
```
Start the voice thread. Listens for audio, transcribes, feeds into agent loop.

```c
void adam_voice_stop(adam_settings_t *s);
```
Stop the voice thread and wait for it to finish.

```c
int adam_voice_is_running(const adam_settings_t *s);
```
Check if the voice thread is active.

```c
adam_status_t adam_stt_transcribe(adam_settings_t *s, arena_t *arena,
    const uint8_t *audio, size_t audio_len,
    adam_audio_format_t format, const char **out_text);
```
One-shot speech-to-text. Result is arena-allocated.

```c
adam_status_t adam_tts_synthesize(adam_settings_t *s, arena_t *arena,
    const char *text, uint8_t **out_audio, size_t *out_len);
```
One-shot text-to-speech. Result is arena-allocated.

```c
adam_status_t adam_tts_speak(adam_settings_t *s, const char *text);
```
Synthesize text and play it aloud.

```c
adam_run_result_t adam_voice_run(adam_settings_t *s, adam_history_t *h,
    const uint8_t *audio, size_t audio_len, adam_audio_format_t format);
```
Full voice turn: transcribe audio -> run agent -> synthesize response -> play.

---

## Memory

Requires `!ADAM_NO_SQLITE`.

```c
adam_memory_t *adam_memory_open(const char *db_path);
void adam_memory_close(adam_memory_t *mem);
```
Open/close a memory database. Creates tables if they don't exist.

```c
adam_status_t adam_memory_configure(adam_memory_t *mem, const adam_settings_t *s);
```
Apply embedding settings. Call after open, before add/search.

```c
adam_status_t adam_memory_add(adam_memory_t *mem, const char *text,
    const char *context, const char *source);
adam_status_t adam_memory_add_file(adam_memory_t *mem, const char *path,
    const char *context);
adam_status_t adam_memory_add_directory(adam_memory_t *mem,
    const char *dir_path, const char *context);
```
Ingest knowledge. `context` is an optional grouping label. `source` is an optional citation.

```c
adam_status_t adam_memory_search(adam_memory_t *mem, arena_t *arena,
    const char *query, const char *context, int limit,
    adam_memory_result_t **results, size_t *count);
```
Hybrid search (BM25 + vector similarity). `context=NULL` searches all. Results are arena-allocated.

```c
adam_status_t adam_memory_delete_context(adam_memory_t *mem, const char *ctx);
adam_status_t adam_memory_clear(adam_memory_t *mem);
```
Delete by context or clear all.

```c
adam_status_t adam_memory_sync(adam_memory_t *mem,
    const char *alias, const char *context);
```
Multi-agent memory sync (stub — will use sqlite-sync CRDT extension).

---

## Sessions

Requires `!ADAM_NO_SQLITE`.

```c
adam_status_t adam_session_create(adam_memory_t *mem,
    char *out_id, size_t out_id_size);
```
Create a new session with a UUIDv7 primary key. `out_id` must be at least 37 bytes.

```c
adam_status_t adam_session_save(adam_memory_t *mem,
    const char *session_id, const adam_history_t *h);
adam_status_t adam_session_load(adam_memory_t *mem,
    const char *session_id, adam_history_t *h);
```
Save/load conversation history.

```c
adam_status_t adam_session_list(adam_memory_t *mem,
    char ***ids, size_t *count);
adam_status_t adam_session_delete(adam_memory_t *mem,
    const char *session_id);
```
List all sessions or delete one.

```c
adam_status_t adam_session_set_sync(adam_memory_t *mem,
    const char *session_id, int sync);
```
Mark/unmark a session for sync.

---

## Model Registry

```c
adam_status_t adam_model_register(const adam_model_info_t *info);
```
Register a custom model (extends the built-in table).

```c
const adam_model_info_t *adam_model_lookup(const char *model);
```
Lookup by longest prefix match. Returns NULL if not found.

```c
int adam_model_context_window(const char *model);
```
Get context window size for a model.

```c
float adam_model_estimate_cost(const char *model,
    int input_tokens, int output_tokens);
```
Estimate cost in USD.

### Built-in Models

| Prefix | Context | Input $/Mtok | Output $/Mtok |
|--------|---------|-------------|--------------|
| `claude-opus-4` | 200K | 15.0 | 75.0 |
| `claude-sonnet-4` | 200K | 3.0 | 15.0 |
| `claude-haiku-4` | 200K | 0.8 | 4.0 |
| `gpt-4o` | 128K | 2.5 | 10.0 |
| `gpt-4o-mini` | 128K | 0.15 | 0.60 |
| `o3` | 200K | 10.0 | 40.0 |
| `o4-mini` | 200K | 1.1 | 4.4 |
| `gemini-2.5-pro` | 1M | 1.25 | 10.0 |
| `gemini-2.5-flash` | 1M | 0.15 | 0.60 |
| `gemini-3.1-flash` | 1M | 0.15 | 0.60 |
| `gemini-3-pro` | 1M | 1.25 | 10.0 |
| `gemini-2.0-flash` | 1M | 0.10 | 0.40 |
| `gemini-1.5-pro` | 2M | 1.25 | 5.0 |
| `gemini-1.5-flash` | 1M | 0.075 | 0.30 |

---

## Response Cache

```c
adam_cache_t *adam_cache_create(size_t max_entries);
```
Create an LRU response cache. `max_entries=0` defaults to 256.

```c
void adam_cache_destroy(adam_cache_t *c);
void adam_cache_clear(adam_cache_t *c);
```
Destroy or clear the cache.

```c
size_t adam_cache_count(const adam_cache_t *c);
size_t adam_cache_hits(const adam_cache_t *c);
size_t adam_cache_misses(const adam_cache_t *c);
```
Cache statistics.

Assign `s->cache = adam_cache_create(256)` to enable caching for an agent. The cache key is a FNV-1a hash of the model name and full message history. Only final text responses (no tool calls) are cached.

---

## Token Estimation

```c
size_t adam_estimate_tokens(const char *text, size_t len);
```
Rough heuristic: ~4 characters per token. Good enough for budgeting.

---

## Evolution Loop

Self-improving agent that iterates: attempt -> evaluate -> refine strategy -> accumulate insights.

```c
adam_evolve_config_t adam_evolve_config_defaults(void);
```
Create a config with defaults (`max_iterations=10`, `target_score=95`, `plateau_iters=3`). Caller must set `task` and `eval_fn`.

```c
adam_evolve_result_t adam_evolve(adam_settings_t *settings,
    const adam_evolve_config_t *config);
```
Run the evolution loop. Blocks until a stop condition is reached (target score, plateau, max iterations, abort, or error).

```c
void adam_evolve_result_free(adam_evolve_result_t *result);
```
Free all strings in the result.

### Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `task` | `const char *` | required | The immutable goal |
| `initial_strategy` | `const char *` | NULL | Starting approach |
| `metrics` | `const char *` | NULL | Scoring criteria description |
| `max_iterations` | `int` | 10 | Maximum attempts |
| `target_score` | `int` | 95 | Stop when score >= this |
| `plateau_iters` | `int` | 3 | Stop after N iterations without improvement |
| `eval_fn` | callback | required | Scores output 0-100 (or -1 for error) |
| `progress_fn` | callback | NULL | Called after each iteration |

---

## Research Mode

Autonomous multi-iteration information gathering with tool use and report synthesis.

```c
adam_research_config_t adam_research_config_defaults(void);
```
Create a config with defaults (`max_iterations=5`). Caller must set `question`.

```c
adam_research_result_t adam_research(adam_settings_t *settings,
    const adam_research_config_t *config);
```
Run the research loop. The agent uses tools to gather information across iterations, accumulates findings (tagged with `FINDING:` / `SOURCE:`), and produces a synthesized report.

```c
void adam_research_result_free(adam_research_result_t *result);
```
Free all strings and findings in the result.

### Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | `const char *` | required | The research question |
| `instructions` | `const char *` | NULL | Additional guidance |
| `max_iterations` | `int` | 5 | Maximum research iterations |
| `is_complete` | callback | NULL | Return 1 to stop early |
| `on_progress` | callback | NULL | Called after each iteration |

---

## Built-in Tools

All tools match the `adam_tool_fn` signature and can be passed to `adam_settings_add_tool()`.

| Tool | Function | ctx | Args |
|------|----------|-----|------|
| Web fetch | `adam_tool_web_fetch` | — | `{"url":"...","method":"GET"}` |
| Memory search | `adam_tool_memory_search` | `adam_memory_t*` | `{"query":"...","limit":5}` |
| Memory add | `adam_tool_memory_add` | `adam_memory_t*` | `{"text":"...","context":"...","source":"..."}` |
| Research | `adam_tool_research` | `adam_settings_t*` | `{"question":"...","instructions":"...","max_iterations":5}` |
| Sub-agent | `adam_tool_agent` | `adam_agent_tool_ctx_t*` | `{"message":"..."}` |
| File read | `adam_tool_file_read` | `adam_settings_t*` | `{"path":"...","max_bytes":65536}` |
| File write | `adam_tool_file_write` | `adam_settings_t*` | `{"path":"...","content":"...","append":false}` |
| List directory | `adam_tool_list_directory` | `adam_settings_t*` | `{"path":"..."}` |
| Shell exec | `adam_tool_shell_exec` | `adam_settings_t*` | `{"command":"...","timeout":30}` |
| Calculator | `adam_tool_calculator` | — | `{"expression":"..."}` |
| Web search | `adam_tool_web_search` | `adam_settings_t*` | `{"query":"...","count":5}` |
| HTTP POST | `adam_tool_http_post` | `adam_settings_t*` | `{"url":"...","body":"..."}` |
| SQL query | `adam_tool_sql_query` | `adam_memory_t*` | `{"sql":"..."}` |

File, directory, and shell tools require `adam_settings_allow_dir()` for sandbox access.

---

## Status Codes

| Code | Value | Description |
|------|-------|-------------|
| `ADAM_OK` | 0 | Success |
| `ADAM_ERR_ALLOC` | 1 | Memory allocation failed |
| `ADAM_ERR_PROVIDER` | 2 | LLM provider error |
| `ADAM_ERR_RATE_LIMIT` | 3 | HTTP 429 rate limited |
| `ADAM_ERR_AUTH` | 4 | HTTP 401/403 authentication error |
| `ADAM_ERR_CONTEXT_OVERFLOW` | 5 | Context window exceeded |
| `ADAM_ERR_MAX_ITERATIONS` | 6 | Tool loop hit max_iterations |
| `ADAM_ERR_ABORTED` | 7 | Cancelled via abort_flag |
| `ADAM_ERR_CURL` | 8 | libcurl error |
| `ADAM_ERR_LOCAL` | 9 | llama.cpp error |
| `ADAM_ERR_SQLITE` | 10 | SQLite error |
| `ADAM_ERR_JSON` | 11 | JSON parse or build error |
| `ADAM_ERR_TOOL_NOT_FOUND` | 12 | Tool name not in registry |
| `ADAM_ERR_INVALID_PARAM` | 13 | Invalid parameter |
| `ADAM_ERR_NO_PROVIDER` | 14 | No provider configured |
| `ADAM_ERR_VOICE` | 15 | Voice subsystem error |
| `ADAM_ERR_NOT_IMPLEMENTED` | 16 | Feature stub |
| `ADAM_ERR_TIMEOUT` | 17 | Wall-clock timeout exceeded |
| `ADAM_ERR_GUARDRAIL` | 18 | Blocked by guardrail |

---

## Enums

### `adam_api_format_t`
`ADAM_API_NONE` (0), `ADAM_API_ANTHROPIC` (1), `ADAM_API_OPENAI` (2), `ADAM_API_GEMINI` (3)

### `adam_role_t`
`ADAM_ROLE_SYSTEM` (0), `ADAM_ROLE_USER` (1), `ADAM_ROLE_ASSISTANT` (2), `ADAM_ROLE_TOOL` (3)

### `adam_media_type_t`
`ADAM_MEDIA_IMAGE_PNG` (0), `ADAM_MEDIA_IMAGE_JPEG` (1), `ADAM_MEDIA_IMAGE_GIF` (2), `ADAM_MEDIA_IMAGE_WEBP` (3), `ADAM_MEDIA_AUDIO_WAV` (4), `ADAM_MEDIA_AUDIO_MP3` (5), `ADAM_MEDIA_PDF` (6)

### `adam_log_level_t`
`ADAM_LOG_ERROR` (0), `ADAM_LOG_WARN` (1), `ADAM_LOG_INFO` (2), `ADAM_LOG_DEBUG` (3), `ADAM_LOG_TRACE` (4)

### `adam_stt_backend_t`
`ADAM_STT_NONE` (0), `ADAM_STT_CLOUD` (1), `ADAM_STT_LOCAL` (2)

### `adam_tts_backend_t`
`ADAM_TTS_NONE` (0), `ADAM_TTS_CLOUD` (1), `ADAM_TTS_LOCAL` (2), `ADAM_TTS_SYSTEM` (3)

### `adam_audio_format_t`
`ADAM_AUDIO_PCM16` (0), `ADAM_AUDIO_WAV` (1), `ADAM_AUDIO_MP3` (2), `ADAM_AUDIO_OGG_OPUS` (3), `ADAM_AUDIO_FLAC` (4)

### `adam_evolve_stop_t`
`ADAM_EVOLVE_STOP_MAX_ITERS` (0), `ADAM_EVOLVE_STOP_SCORE` (1), `ADAM_EVOLVE_STOP_PLATEAU` (2), `ADAM_EVOLVE_STOP_ABORTED` (3), `ADAM_EVOLVE_STOP_ERROR` (4)

### `adam_research_stop_t`
`ADAM_RESEARCH_STOP_COMPLETE` (0), `ADAM_RESEARCH_STOP_MAX_ITERS` (1), `ADAM_RESEARCH_STOP_ABORTED` (2), `ADAM_RESEARCH_STOP_ERROR` (3), `ADAM_RESEARCH_STOP_CALLBACK` (4)

---

## Callbacks

### `adam_log_fn`
```c
void (*)(void *ctx, adam_log_level_t level, const char *msg, size_t len)
```

### `adam_stream_fn`
```c
void (*)(void *ctx, const char *chunk, size_t len, int is_done)
```

### `adam_tool_fn`
```c
adam_tool_result_t (*)(arena_t *arena, void *ctx, const char *args_json, size_t args_len)
```

### `adam_llm_fn`
```c
adam_llm_response_t (*)(void *ctx, arena_t *arena, const adam_message_t *msgs,
    size_t msg_count, const adam_tool_def_t *tools, size_t tool_count)
```

### `adam_guardrail_fn`
```c
int (*)(void *ctx, const adam_message_t *msgs, size_t msg_count)
```
Return 0 to allow, non-zero to deny.

### `adam_guardrail_response_fn`
```c
int (*)(void *ctx, const char *content, const adam_tool_call_t *tool_calls,
    size_t tool_call_count)
```
Return 0 to allow, non-zero to deny.

### `adam_evolve_eval_fn`
```c
int (*)(void *ctx, const char *output, int iteration)
```
Return 0-100 for score, or -1 for error.

### `adam_research_complete_fn`
```c
int (*)(void *ctx, const char *report, int iteration)
```
Return 1 if research is complete, 0 to continue.

---

## Feature Gates

Define before `#include "adam.h"`:

| Gate | Disables |
|------|----------|
| `ADAM_NO_CURL` | libcurl HTTP (must provide `http_fn`) |
| `ADAM_NO_LOCAL` | llama.cpp local inference |
| `ADAM_NO_PTHREADS` | Thread pool, voice thread |
| `ADAM_NO_SQLITE` | Memory, sessions, SQL tool |
| `ADAM_NO_VOICE` | Voice subsystem (STT/TTS) |
| `ADAM_NO_FILESYSTEM` | file_read, file_write, list_directory tools |
| `ADAM_NO_SHELL` | shell_exec tool |

WASM (Emscripten) automatically defines `ADAM_NO_FILESYSTEM` and `ADAM_NO_SHELL`.
