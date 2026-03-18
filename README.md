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
make talk          # Voice agent (cloud)
make talk LOCAL=1  # Voice agent (fully local)
make memory        # Memory system tests
make clean         # Remove all build artifacts
```

## API Reference

Full API documentation with every function, type, and callback: **[API.md](API.md)**

## Examples

### 1. Simple Conversation

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

### 2. Tool Calling

```c
// Define a weather tool
adam_tool_result_t get_weather(arena_t *arena, void *ctx,
                                const char *args, size_t len) {
    // Parse {"city":"..."} from args, call weather API
    return (adam_tool_result_t){
        .for_llm = arena_strdup(arena, "72F, sunny in San Francisco"),
        .success = 1,
    };
}

adam_settings_t *s = adam_create_settings();
adam_settings_set_provider(s, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");
adam_settings_add_tool(s, (adam_tool_def_t){
    .name = "get_weather",
    .description = "Get current weather for a city",
    .parameters_json = "{\"type\":\"object\",\"properties\":"
                       "{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}",
    .execute = get_weather,
});

adam_history_t *h = adam_history_create();
// The agent will call get_weather, then use the result to answer
adam_run_result_t r = adam_run(s, h, "What's the weather in SF?");
printf("%s\n", r.final_response);
// "It's currently 72F and sunny in San Francisco."
```

### 3. Local Model (llama.cpp)

```c
adam_settings_t *s = adam_create_settings();
adam_settings_set_local(s, "models/llama-3.2-3b.gguf", -1, 4096);

adam_history_t *h = adam_history_create();
adam_run_result_t r = adam_run(s, h, "Write a haiku about programming.");
printf("%s\n", r.final_response);
```

### 4. Google Gemini

```c
adam_settings_t *s = adam_create_settings();
adam_settings_set_provider(s, ADAM_API_GEMINI,
                           getenv("GEMINI_API_KEY"),
                           "gemini-2.5-flash");

adam_history_t *h = adam_history_create();
adam_run_result_t r = adam_run(s, h, "Explain how transformers work.");
printf("%s\n", r.final_response);
// Works with all Gemini models: 2.5-pro, 2.5-flash, 2.0-flash, 1.5-pro, etc.
```

### 5. Image Generation (Gemini)

```c
adam_settings_t *s = adam_create_settings();
adam_settings_set_provider(s, ADAM_API_GEMINI,
                           getenv("GEMINI_API_KEY"),
                           "gemini-3.1-flash-image-preview");

adam_history_t *h = adam_history_create();
adam_run_result_t r = adam_run(s, h, "Draw a cute cat wearing a tiny hat");

// r.final_response contains text + embedded image as data URI:
//   "Here is your cat:\n![image](data:image/png;base64,iVBORw0KGgo...)\n"
//
// Extract the base64 data to save as PNG, or in React Native/Expo:
//   <Image source={{uri: dataUri}} />
printf("%s\n", r.final_response);
```

### 6. Structured JSON Output

```c
adam_settings_t *s = adam_create_settings();
adam_settings_set_provider(s, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");

adam_history_t *h = adam_history_create();
adam_json_result_t r = adam_run_json(s, h,
    "List 3 programming languages with their year of creation",
    "{\"languages\":[{\"name\":\"...\",\"year\":...}]}",
    3);  // max 3 retries

if (r.json_valid) {
    printf("Valid JSON: %s\n", r.base.final_response);
    char *first = adam_json_extract(r.base.final_response, "languages");
    if (first) { printf("Languages: %s\n", first); free(first); }
}
adam_json_result_free(&r);
```

### 7. Long-Term Memory

```c
// Open memory database
adam_memory_t *mem = adam_memory_open("agent_memory.db");
adam_memory_configure(mem, s);

// Ingest knowledge
adam_memory_add(mem, "The user prefers dark mode.", "preferences", "onboarding");
adam_memory_add_file(mem, "docs/architecture.md", "project");
adam_memory_add_directory(mem, "docs/", "project");

// Agent with memory enrichment
s->memory = mem;
s->inject_memory = 1;  // auto-inject relevant memories into system prompt
adam_history_t *h = adam_history_create();
adam_run_result_t r = adam_run(s, h, "What does the user prefer for UI theme?");
// Agent sees relevant memories and answers: "The user prefers dark mode."

adam_memory_close(mem);
```

### 8. Session Persistence

```c
adam_memory_t *mem = adam_memory_open("sessions.db");

// Create a session
char session_id[37];
adam_session_create(mem, session_id, sizeof(session_id));

// Chat and auto-save
s->memory = mem;
s->session_id = session_id;
s->auto_save = 1;  // saves after each adam_run

adam_history_t *h = adam_history_create();
adam_run(s, h, "Remember that my name is Marco.");
adam_run(s, h, "What's my name?");
// Session is automatically saved to SQLite

// Later: restore the conversation
adam_history_t *h2 = adam_history_create();
adam_session_load(mem, session_id, h2);
// h2 now contains the full conversation history
```

### 9. Streaming Responses

```c
void on_token(void *ctx, const char *chunk, size_t len, int is_done) {
    fwrite(chunk, 1, len, stdout);
    if (is_done) printf("\n");
}

adam_settings_t *s = adam_create_settings();
adam_settings_set_provider(s, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");
adam_settings_set_stream(s, on_token, NULL);

adam_history_t *h = adam_history_create();
adam_run(s, h, "Write a short story about a robot.");
// Tokens stream to stdout in real-time
```

### 10. Voice Agent

```c
adam_settings_t *s = adam_create_settings();
adam_settings_set_provider(s, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");
adam_settings_set_stt(s, ADAM_STT_CLOUD, NULL, openai_key, "gpt-4o-mini-transcribe");
adam_settings_set_tts(s, ADAM_TTS_SYSTEM, NULL, NULL, NULL, NULL);

adam_history_t *h = adam_history_create();

// Full voice turn: audio in -> transcribe -> agent -> speak
uint8_t *audio = /* recorded audio */;
adam_run_result_t r = adam_voice_run(s, h, audio, audio_len, ADAM_AUDIO_WAV);
// Plays the response aloud via system TTS
```

### 11. Multi-Agent Orchestration

```c
// Create a specialized researcher agent
adam_settings_t *researcher = adam_create_settings();
adam_settings_set_provider(researcher, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");
adam_settings_set_identity(researcher, "You are a research specialist.");

adam_agent_tool_ctx_t researcher_ctx = {
    .settings = researcher,
    .history = NULL,  // fresh conversation each call
};

// Main agent can delegate to the researcher
adam_settings_t *s = adam_create_settings();
adam_settings_set_provider(s, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");
adam_settings_add_tool(s, (adam_tool_def_t){
    .name = "ask_researcher",
    .description = "Delegate a research question to the specialist",
    .parameters_json = "{\"type\":\"object\",\"properties\":"
                       "{\"message\":{\"type\":\"string\"}},\"required\":[\"message\"]}",
    .execute = adam_tool_agent,
    .ctx = &researcher_ctx,
});

adam_history_t *h = adam_history_create();
adam_run_result_t r = adam_run(s, h,
    "I need a summary of recent advances in quantum computing.");
// Main agent delegates to researcher, gets back findings, synthesizes response
```

### 12. Evolution Loop (Self-Improving Agent)

```c
// Evaluation function: scores each attempt
int eval(void *ctx, const char *output, int iteration) {
    // Score based on your criteria (0-100)
    return strlen(output) > 100 ? 70 + iteration * 5 : 30;
}

adam_settings_t *s = adam_create_settings();
adam_settings_set_provider(s, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");

adam_evolve_config_t cfg = adam_evolve_config_defaults();
cfg.task = "Write a compelling product description for a smart water bottle.";
cfg.initial_strategy = "Focus on health benefits and smart features.";
cfg.metrics = "Score based on persuasiveness, clarity, and call to action.";
cfg.eval_fn = eval;
cfg.target_score = 90;
cfg.max_iterations = 10;

adam_evolve_result_t r = adam_evolve(s, &cfg);
printf("Best (score %d):\n%s\n", r.best_score, r.best_output);
printf("Strategy evolved to:\n%s\n", r.strategy);
printf("Insights:\n%s\n", r.insights);
adam_evolve_result_free(&r);
```

### 13. Research Mode

```c
adam_settings_t *s = adam_create_settings();
adam_settings_set_provider(s, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");

// Add tools for the research agent to use
adam_settings_add_tool(s, (adam_tool_def_t){
    .name = "web_fetch", .execute = adam_tool_web_fetch });
adam_settings_add_tool(s, (adam_tool_def_t){
    .name = "memory_search", .execute = adam_tool_memory_search, .ctx = mem });

adam_research_config_t cfg = adam_research_config_defaults();
cfg.question = "What are the latest breakthroughs in solid-state batteries?";
cfg.instructions = "Focus on papers from 2025. Include specific energy densities.";
cfg.max_iterations = 5;

adam_research_result_t r = adam_research(s, &cfg);
printf("Report:\n%s\n", r.report);
printf("Found %zu findings across %d iterations\n",
       r.finding_count, r.total_iterations);
adam_research_result_free(&r);
```

### 14. Built-in Tools with Filesystem Sandbox

```c
adam_settings_t *s = adam_create_settings();
adam_settings_set_provider(s, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");

// Grant access to specific directories only
adam_settings_allow_dir(s, "/home/user/project");
adam_settings_allow_dir(s, "/tmp");

// Register filesystem + shell tools
adam_settings_add_tool(s, (adam_tool_def_t){
    .name = "read_file", .description = "Read a file",
    .execute = adam_tool_file_read, .ctx = s });
adam_settings_add_tool(s, (adam_tool_def_t){
    .name = "write_file", .description = "Write a file",
    .execute = adam_tool_file_write, .ctx = s });
adam_settings_add_tool(s, (adam_tool_def_t){
    .name = "list_dir", .description = "List directory contents",
    .execute = adam_tool_list_directory, .ctx = s });
adam_settings_add_tool(s, (adam_tool_def_t){
    .name = "shell", .description = "Run a shell command",
    .execute = adam_tool_shell_exec, .ctx = s });
adam_settings_add_tool(s, (adam_tool_def_t){
    .name = "calc", .description = "Evaluate a math expression",
    .execute = adam_tool_calculator });

adam_history_t *h = adam_history_create();
adam_run_result_t r = adam_run(s, h,
    "Read the README.md in /home/user/project and summarize it. "
    "Then count the lines of code in all .c files.");
// Agent reads files, runs shell commands — all sandboxed to allowed dirs
// Attempts to access /etc or ~ will be denied
```

### 15. Guardrails

```c
// Block requests containing sensitive topics
int check_input(void *ctx, const adam_message_t *msgs, size_t count) {
    for (size_t i = 0; i < count; i++)
        if (msgs[i].content && strstr(msgs[i].content, "password"))
            return 1;  // deny
    return 0;  // allow
}

// Block responses containing PII
int check_output(void *ctx, const char *content,
                  const adam_tool_call_t *tc, size_t tc_count) {
    if (content && strstr(content, "SSN"))
        return 1;  // deny
    return 0;  // allow
}

adam_settings_t *s = adam_create_settings();
s->on_before_send = check_input;
s->on_after_receive = check_output;
// If either guardrail denies, adam_run returns ADAM_ERR_GUARDRAIL
```

### 16. Response Caching

```c
adam_cache_t *cache = adam_cache_create(256);  // LRU, max 256 entries

adam_settings_t *s = adam_create_settings();
adam_settings_set_provider(s, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");
s->cache = cache;

adam_history_t *h1 = adam_history_create();
adam_run(s, h1, "What is 2+2?");  // cache miss — calls LLM

adam_history_t *h2 = adam_history_create();
adam_run(s, h2, "What is 2+2?");  // cache hit — instant, no LLM call

printf("Hits: %zu, Misses: %zu\n",
       adam_cache_hits(cache), adam_cache_misses(cache));

adam_cache_destroy(cache);
```

### 17. Thread Pool (Concurrent Agents)

```c
void on_done(void *ctx, adam_run_result_t result) {
    printf("Agent %d: %s\n", *(int *)ctx, result.final_response);
    adam_run_result_free(&result);
}

adam_pool_t *pool = adam_pool_create(4);  // 4 worker threads

for (int i = 0; i < 10; i++) {
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_provider(s, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");
    adam_history_t *h = adam_history_create();
    int *idx = malloc(sizeof(int)); *idx = i;

    adam_pool_submit(pool, (adam_job_t){
        .settings = s,
        .history = h,
        .user_message = "Generate a random fun fact.",
        .on_done = on_done,
        .on_done_ctx = idx,
    });
}

adam_pool_destroy(pool);  // waits for all jobs to complete
```

### 18. Full Agent with Everything

```c
adam_init();

// Memory
adam_memory_t *mem = adam_memory_open("agent.db");
adam_memory_configure(mem, s);
adam_memory_add_directory(mem, "docs/", "knowledge");

// Settings
adam_settings_t *s = adam_create_settings();
adam_settings_set_provider(s, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");
adam_settings_set_identity(s, "You are Adam, an AI research assistant.");
adam_settings_set_instructions(s, "Be concise. Cite sources when possible.");
s->memory = mem;
s->inject_memory = 1;
s->memory_extract = 1;  // learn from conversations
s->timeout_ms = 60000;  // 60s timeout

// Session
char session_id[37];
adam_session_create(mem, session_id, sizeof(session_id));
s->session_id = session_id;
s->auto_save = 1;

// Cache
s->cache = adam_cache_create(128);

// Guardrails
s->on_before_send = check_input;
s->on_after_receive = check_output;

// Sandbox + tools
adam_settings_allow_dir(s, "/home/user/workspace");
adam_settings_add_tool(s, (adam_tool_def_t){
    .name = "read_file", .execute = adam_tool_file_read, .ctx = s });
adam_settings_add_tool(s, (adam_tool_def_t){
    .name = "write_file", .execute = adam_tool_file_write, .ctx = s });
adam_settings_add_tool(s, (adam_tool_def_t){
    .name = "shell", .execute = adam_tool_shell_exec, .ctx = s });
adam_settings_add_tool(s, (adam_tool_def_t){
    .name = "calc", .execute = adam_tool_calculator });
adam_settings_add_tool(s, (adam_tool_def_t){
    .name = "memory_search", .execute = adam_tool_memory_search, .ctx = mem });
adam_settings_add_tool(s, (adam_tool_def_t){
    .name = "sql", .execute = adam_tool_sql_query, .ctx = mem });

// Streaming
adam_settings_set_stream(s, on_token, NULL);

// Logging
adam_settings_set_logger(s, my_logger, NULL, ADAM_LOG_INFO);

// Run
adam_history_t *h = adam_history_create();
adam_run(s, h, "Analyze the codebase in my workspace and write a summary.");

// Cleanup
adam_cache_destroy(s->cache);
adam_history_destroy(h);
adam_settings_destroy(s);
adam_memory_close(mem);
adam_cleanup();
```

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
