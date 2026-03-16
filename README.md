# Adam — A Self-Evolving Agent in C

## Deep Architecture Analysis: NullClaw, Nanobot, OpenClaw, PicoClaw

---

## 1. Overview Table

| Aspect | **NullClaw** | **Nanobot** | **OpenClaw** | **PicoClaw** |
|--------|-------------|------------|-------------|-------------|
| **Language** | Zig 0.15.2 | Python 3.11+ | TypeScript/ESM | Go 1.25.7 |
| **Binary Size** | ~678 KB | N/A (interpreted) | N/A (Node 22+) | <10 MB |
| **RAM Usage** | ~1 MB | Higher (Python GC) | Higher (V8 GC) | <10 MB |
| **Startup Time** | <2ms | ~1-2s | ~1-2s | <1s |
| **Tools** | 35+ | 11+ (+ MCP) | 40+ | 26+ (+ MCP) |
| **Channels** | 19 | 8+ | 40+ | 10+ |
| **Providers** | 50+ (OpenAI-compat) | Multi (via LiteLLM) | 20+ (native) | 7+ (native) |
| **Max Tool Iterations** | 25 | 40 | Streaming-driven | Per-loop |
| **Memory Backend** | SQLite FTS5 + Vector | MEMORY.md + HISTORY.md | Vector DB | MEMORY.md + Daily Notes |

---

## 2. Shared Architecture: The Universal Agent Loop

All four agents converge on the same fundamental loop pattern:

```
┌─────────────────────────────────────────────┐
│  1. RECEIVE message from channel/bus        │
│  2. RESOLVE session + route + agent         │
│  3. LOAD history from session store         │
│  4. BUILD system prompt + context           │
│  5. ┌──────── TOOL ITERATION LOOP ────────┐ │
│     │  a. Call LLM (messages + tools)      │ │
│     │  b. If text-only response → BREAK    │ │
│     │  c. Parse tool calls                 │ │
│     │  d. Execute tools (batch/parallel)   │ │
│     │  e. Append results to messages       │ │
│     │  f. Repeat until done or max iters   │ │
│     └──────────────────────────────────────┘ │
│  6. SAVE response to session                │
│  7. COMPRESS/SUMMARIZE if threshold met     │
│  8. PUBLISH response to channel             │
└─────────────────────────────────────────────┘
```

This is the **universal agentic loop** — every agent framework converges on this pattern.

---

## 3. Individual Agent Architectures

### 3.1 NullClaw (Zig)

**Technology**: Pure Zig (0.15.2) — statically compiled binary (678 KB, ~1 MB RAM). No external runtime dependencies (except vendored SQLite). Multi-platform: ARM, x86, RISC-V.

**Project Structure**:
```
src/
├── main.zig                    # CLI entry point (router for 18+ commands)
├── agent/                      # Core agent loop (root.zig, dispatcher.zig)
├── providers/                  # LLM provider adapters (50+ OpenAI-compatible)
├── channels/                   # Messaging platforms (19 integrations)
├── tools/                      # Agent tools (35+ including shell, files, git, memory)
├── memory/                     # Persistent knowledge storage (multi-backend)
├── security/                   # Sandbox & autonomy policies
├── gateway.zig                 # HTTP/WebSocket server
└── config*.zig                 # Configuration system
```

**Agent Loop** (`src/agent/root.zig::Agent::turn()`):

1. **Input Processing** — Slash command handling, memory enrichment with hybrid search, response cache lookup
2. **System Prompt Injection** — Detects workspace file changes (fingerprinting), builds dynamic system prompt from bootstrap files (SOUL.md, TOOLS.md, AGENTS.md), capabilities section, conversation context, profile-specific prompts. Cached in `history[0]`.
3. **Tool Iteration Loop** (max 25 iterations):
   - Build provider messages (filtered history)
   - Calculate effective max_tokens (context budget)
   - Call LLM (streaming or blocking path, up to 3 retry attempts)
   - Parse tool calls (structured JSON first, XML `<tool_call>` fallback)
   - Execute all tools in batch
   - Format results + reflection prompt
   - Add tool results to history
4. **Error Recovery**: Vision error → auto-disable + retry. Context exhaustion → force-compress + retry (3x). Rate-limiting → mark route degraded. Other → 500ms backoff + 1 retry.
5. **Graceful Degradation**: If tool iterations exhausted, request LLM to summarize accomplishments without tool calls.

**Memory System** — Multi-layered architecture:
- **Layer A** (Primary Storage): SQLite FTS5, Markdown, Redis, PostgreSQL, ClickHouse, LanceDB
- **Layer B** (Retrieval): Hybrid search — keyword BM25 + vector cosine, RRF merge, temporal decay, MMR, query expansion, LLM reranking
- **Layer C** (Vectors): OpenAI, Gemini, Voyage, Ollama embeddings → SQLite, Qdrant, PgVector stores. Durable outbox for async sync.
- **Layer D** (Lifecycle): Response cache, semantic cache, auto-hygiene, summarization, snapshots

**Tool Dispatch** (`src/agent/dispatcher.zig`):
- OpenAI native JSON tool_calls parsed first
- Falls back to XML `<tool_call>` tags
- Structured recovery for malformed payloads (bare JSON, unclosed tags)
- Security: Only extracts JSON from explicit tags (prevents prompt injection)
- Credential scrubbing before display

**Provider Interface**: Vtable-based polymorphism (`ptr: *anyopaque + vtable: *const VTable`). Methods: `chat()`, `chatWithSystem()`, `streamChat()`, `supportsNativeTools()`, `supportsVision()`, `supportsStreaming()`.

**Unique Patterns**:
- Vtable-based polymorphism for all subsystems (zero-overhead dispatch)
- Arena allocators per iteration (freed at iteration end)
- Build-time feature gates (`-Dchannels=telegram,cli`)
- Fingerprinting for cache invalidation
- Adaptive model routing: keyword hints (fast/balanced/deep/reasoning/vision) with score-based promotion

---

### 3.2 Nanobot (Python)

**Technology**: Python 3.11+, async-first, minimal dependencies. Core agent loop ~1,800 lines.

**Project Structure**:
```
nanobot/
├── agent/
│   ├── loop.py         # Main agent loop (processing engine)
│   ├── context.py      # Context/prompt building
│   ├── memory.py       # Persistent memory system
│   ├── skills.py       # Skill discovery & loading
│   ├── subagent.py     # Background task subagents
│   └── tools/          # Tool implementations (6 types + extensible)
├── providers/          # Multi-provider LLM abstraction (via LiteLLM)
├── bus/                # Async message queue (channels ↔ agent)
├── config/             # Configuration management (Pydantic)
├── session/            # Conversation history persistence (JSONL)
├── cli/                # Interactive CLI with prompt_toolkit
└── templates/          # Bootstrap files (SOUL, TOOLS, etc.)
```

**Agent Loop** (`agent/loop.py`):

1. `await run()` — Main event loop listening on message bus
2. Receive `InboundMessage` from channel
3. `_dispatch(msg)` — Acquire async lock, call `_process_message()`
4. `_process_message()` — Load session (JSONL), check commands (`/new`, `/stop`), build context, call `_run_agent_loop()`
5. `_run_agent_loop()` — Iterate tool calls until LLM returns final answer (max 40 iterations):
   - `provider.chat_with_retry(messages, tools, model)`
   - If `has_tool_calls`: extract, validate against JSON schema, `await tools.execute()`, add result
   - If `finish_reason == "error"`: break
   - Otherwise: final response, break
6. Save turn to session, consolidate memory if needed

**Memory System** — Two-layer architecture:
- **MEMORY.md** (Long-term facts): User-edited markdown, injected into every system prompt
- **HISTORY.md** (Grep-searchable log): Timestamped entries `[YYYY-MM-DD HH:MM] SUMMARY`
- **LLM Consolidation**: When token threshold (50% of context) exceeded, LLM decides what's important via forced `save_memory` tool call. Falls back to raw archive if LLM fails 3x.

**Tools**: read_file, write_file, edit_file, list_dir, exec (shell), web_search (multi-provider), web_fetch, message, spawn (subagents), cron, MCP tools.

**Provider**: LiteLLM wrapper for unified multi-provider support. Provider auto-detection via model name keywords. Anthropic cache_control injection. Retry: 3 attempts with 1s/2s/4s backoff.

**Unique Patterns**:
- Append-only session storage (LLM cache efficiency)
- LLM-driven memory consolidation (AI decides importance)
- Progressive skill loading (listed in prompt, read on demand)
- Subagent pattern (background tasks via spawn tool)
- `json_repair` library for malformed JSON recovery

---

### 3.3 OpenClaw (TypeScript)

**Technology**: TypeScript/ESM, Node 22+, strict typing. ~116,413 lines in `/src/agents`.

**Project Structure**:
```
src/
├── agents/
│   ├── pi-embedded-runner/     # Main agent execution loop
│   ├── pi-embedded-subscribe.ts # Event streaming & response handling
│   ├── tools/                  # Built-in tool implementations
│   ├── pi-extensions/          # Extensions (compaction, context pruning)
│   ├── auth-profiles/          # Model auth/fallback logic
│   └── sandbox/                # Execution sandboxing
├── commands/, channels/, gateway/
├── plugins/                    # Plugin SDK and loading system
└── memory/                     # Vector search memory backend
```

**Agent Loop** (`runEmbeddedPiAgent()` in `run.ts`):

1. **Initialization**: Load session (JSONL transcript), resolve model/provider with fallback, set up auth profiles (OAuth/API keys with rotation), build system prompt and tool set, apply bootstrap files
2. **API Call & Streaming**: Call LLM via `@mariozechner/pi-ai`, stream response events asynchronously
3. **Event Subscription** (`subscribeEmbeddedPiSession()`):
   - `message_start` → Initialize response buffer
   - `message_update` → Accumulate text, emit partial replies
   - `tool_execution_start/update/end` → Execute tools locally, handle errors, emit metrics
   - `message_end` → Finalize, save to session, handle compaction
4. **Compaction**: Monitor token usage (70-85% trigger), multi-chunk summarization, preserve task state, re-inject summary
5. **Error Handling & Failover**: Classify errors (auth/billing/rate_limit/timeout/format/overflow), auth profile cooldown, fall back to next model, exponential backoff (250ms initial, 1.5s cap, 2x factor, ±20% jitter)

**Error Classification**:
| Error Type | Code | Recovery |
|---|---|---|
| auth | 401 | Profile cooldown, try next |
| billing | 402 | Permanent failure |
| rate_limit | 429 | Backoff, try next profile |
| timeout | 408 | Backoff, try next model |
| format | 400 | Tool schema fix, retry |
| context_overflow | — | Trigger compaction, retry |

**Memory**: Vector DB backend (Qmd default), tool interface (`memory_search`, `memory_get`), session-bound indexing.

**Tool System**: 40+ tools with granular policy pipeline (global allow/deny → agent-specific → owner-only → channel-specific → model provider filters → plugin expansion). Tool mutation tracking prevents duplicate execution.

**Unique Patterns**:
- Streaming-first architecture with real-time event emission
- Auth profile rotation with per-profile cooldown (transient: 30s, permanent: forever)
- Session write locks prevent concurrent corruption
- Bootstrap budget analysis (warns when approaching char limits)
- Compaction token margin (1.2x safety factor)
- Plugin/extension lifecycle hooks
- Message deduplication (normalized comparison)
- Global singleton state via `Symbol.for()` (survives bundler splits)

---

### 3.4 PicoClaw (Go)

**Technology**: Go 1.25.7, ultra-lightweight (<10 MB RAM, <1s startup), multiplatform (x86_64, ARM64, MIPS, RISC-V).

**Project Structure**:
```
cmd/                # CLI applications (picoclaw binary, launcher TUI)
pkg/
├── agent/          # Core agent loop and instance management
├── providers/      # LLM provider abstractions
├── tools/          # Tool implementations (26 types)
├── session/        # Conversation history & persistence
├── memory/         # Long-term and daily note storage
├── routing/        # Multi-agent message routing
├── channels/       # Channel integrations (10+)
├── config/         # Configuration management
├── mcp/            # Model Context Protocol integration
└── commands/       # Built-in command handling
```

**Agent Loop** (`loop.go`, `instance.go` — 8 phases):

1. **History Loading** — Retrieve conversation history + existing summary from session store
2. **Message Building** — `ContextBuilder.BuildMessages()`: static cached system prompt (identity, bootstrap, skills, memory) + dynamic context (time, runtime, session) + summary + history + current message with media
3. **Session Persistence** — Save user message immediately
4. **LLM Iteration Loop** (`runLLMIteration`):
   - Model selection: `selectCandidates()` via Router (primary vs light model based on complexity scoring)
   - Build tool definitions from registry
   - LLM request with fallback chain + retry logic for timeouts and context errors
   - **Parallel tool execution**: All tool calls run concurrently via `sync.WaitGroup`
   - Async tools fire callbacks that publish results back to message bus
   - TTL decrement for discovered tools
   - Loop until LLM returns no tool calls
5. **Response Handling** — Apply default response if empty
6. **Session Save** — Persist to disk
7. **Summarization** (background goroutine): Triggered when messages > 20 or tokens > 75% of context window. Multi-part summarization, merges parts, truncates to last 4 messages + summary.
8. **Response Publishing** — Publish via message bus

**Memory**:
- **MEMORY.md**: Persistent file, included in system prompt
- **Daily Notes** (`YYYYMM/YYYYMMDD.md`): Last 3 days included in prompt
- **Session**: JSONL backend with auto-migration from JSON. Emergency compression: drop oldest 50% on overflow.

**Tool System**: Registry with core (always visible) and hidden (searchable via BM25/Regex) tools. TTL-based promotion/demotion. `AsyncExecutor` interface for long-running operations. 26 tool categories: filesystem, exec, web, communication, media, hardware (i2c, spi), scheduling, skills, spawn, MCP.

**Routing**: 7-level priority cascade: peer → parent_peer → guild → team → account → channel → default. Complexity-based model routing (light model for simple queries, saves cost).

**Unique Patterns**:
- Parallel tool execution via goroutines (natural concurrency)
- Tool TTL system (dynamically promotes/demotes tools per conversation)
- Complexity-based model routing (message scoring)
- Pure Go SQLite (no CGo) — truly portable
- Mtime-based prompt cache invalidation with recursive directory watching
- Request-scoped tool context via `context.Value` (thread-safe)
- Atomic file writes with explicit sync (flash storage reliability)

---

## 4. Key Architectural Concepts Comparison

### 4.1 Tool Dispatch

| Feature | NullClaw | Nanobot | OpenClaw | PicoClaw |
|---------|----------|---------|----------|----------|
| **Parsing** | Native JSON → XML fallback | Native JSON only | Streaming events | Native JSON |
| **Execution** | Batch sequential | Sequential | Event-driven | Parallel (goroutines) |
| **Async tools** | No | Subagent spawn | Yes (callbacks) | Yes (AsyncExecutor) |
| **Discovery** | Keyword filtering | Always visible | Policy-filtered | BM25/Regex + TTL |
| **Sandboxing** | landlock/firejail/bubblewrap/Docker | Workspace restriction | Policy pipeline | Context-scoped |

### 4.2 Memory & Context Management

| Feature | NullClaw | Nanobot | OpenClaw | PicoClaw |
|---------|----------|---------|----------|----------|
| **Long-term** | SQLite FTS5 + Vector hybrid | MEMORY.md file | Vector DB | MEMORY.md file |
| **Short-term** | History array (50 max) | Session JSONL | Session JSONL | Session JSONL/JSON |
| **Compression** | Force-compress on overflow | LLM consolidation | Multi-chunk summarization | LLM summarization |
| **Trigger** | Context exhaustion | Token threshold (50%) | 70-85% context | 75% or 20 messages |
| **Retrieval** | Hybrid BM25 + cosine | File read | Vector search tool | File inclusion |
| **Vector search** | Yes (6 backends) | No | Yes | No |

### 4.3 LLM Communication

| Feature | NullClaw | Nanobot | OpenClaw | PicoClaw |
|---------|----------|---------|----------|----------|
| **Abstraction** | Vtable polymorphism | LiteLLM wrapper | pi-ai library | Interface + factories |
| **Streaming** | Callback-based (optional) | No | Event-driven (primary) | No |
| **Retry** | 3 retries + backoff | 3 retries (1s,2s,4s) | Exponential + jitter | 3 attempts (5s,10s,15s) |
| **Fallback** | Route degradation | No chain | Auth profile rotation | Candidate chain |
| **Prompt caching** | Fingerprint-based | Anthropic cache_control | Anthropic + OpenAI | Mtime-based invalidation |
| **Model routing** | Keyword → hint (fast/deep/vision) | Single model | Configured fallbacks | Complexity scoring |

### 4.4 Error Recovery

| Strategy | NullClaw | Nanobot | OpenClaw | PicoClaw |
|----------|----------|---------|----------|----------|
| **Vision fail** | Auto-disable + retry | N/A | N/A | N/A |
| **Context overflow** | Force-compress (3x) | Consolidation | Compaction + retry | forceCompression (2x) |
| **Rate limit** | Blacklist 5 min | Backoff + retry | Profile cooldown | Provider cooldown |
| **Auth fail** | Fail | Fail | Profile rotation | Provider cooldown |
| **Tool error** | Formatted in results | Hint appended | In session | Error result |

---

## 5. Pros & Cons

### NullClaw (Zig)

**Pros**:
- Smallest binary (678 KB) and memory footprint (~1 MB) — runs on embedded devices
- Zero external runtime dependencies — fully self-contained
- Most sophisticated memory system (6 vector backends, hybrid search, RRF, MMR)
- Best security model (landlock, firejail, bubblewrap, Docker sandboxing)
- Intelligent model routing with keyword-based hints
- Compile-time feature gates — only compile what you need
- Cross-platform including RISC-V

**Cons**:
- Zig is niche — smaller contributor pool
- No streaming tool execution (batch only)
- More complex codebase to maintain
- Arena allocator patterns require careful lifetime management
- XML fallback parsing adds complexity

### Nanobot (Python)

**Pros**:
- Simplest codebase (~1,800 lines for core loop) — easiest to understand and modify
- Async-first with Python asyncio — natural for I/O-bound work
- LiteLLM abstraction gives broadest provider compatibility for free
- LLM-driven memory consolidation — AI decides what's important
- Minimal dependencies, no heavyweight frameworks
- Progressive skill loading reduces prompt size
- Append-only session design optimizes LLM cache hit rates

**Cons**:
- Highest runtime overhead (Python interpreter, GC)
- No vector search — memory is purely file-based
- No parallel tool execution
- Single async lock → serial message processing
- No model routing intelligence
- No sandboxing beyond workspace restriction

### OpenClaw (TypeScript)

**Pros**:
- Most production-hardened — extensive error classification and failover
- Streaming-first architecture with real-time event emission
- Auth profile rotation with per-profile cooldown tracking
- Most tools (40+) and channels (40+)
- Plugin/extension system with lifecycle hooks
- Compaction with quality safeguards and token margin safety
- Session write locks prevent corruption
- Tool mutation tracking prevents duplicate execution

**Cons**:
- Largest codebase (~116K lines in agents) — most complex
- Depends on pi-agent-core family — tight coupling to custom framework
- Node.js runtime overhead (V8, event loop)
- TypeScript compilation step adds friction
- Heavy dependency tree
- More complex deployment

### PicoClaw (Go)

**Pros**:
- Best balance of performance vs simplicity — small binary, fast startup, low RAM
- Parallel tool execution via goroutines — natural concurrency
- 7-level routing priority cascade — sophisticated multi-agent support
- Tool TTL system — dynamically promotes/demotes tools per conversation
- Complexity-based model routing — saves cost on simple queries
- Clean Go interfaces — easy to extend
- Pure Go SQLite (no CGo) — truly portable
- Mtime-based prompt cache invalidation — efficient and correct

**Cons**:
- Fewer providers supported natively (7 vs 50+)
- No vector search — memory is file-based
- No streaming support
- Less sophisticated error recovery than OpenClaw
- No plugin/extension system
- Context window compression is simpler (drop 50% on overflow)

---

## 6. The Agent Loop in C — Architecture Design

Based on the patterns across all four agents, here's the architecture for Adam: a self-contained agent library in C using **arena allocation**, **jsmn** for JSON parsing, and **libcurl + mbedtls** for networking.

### 6.0 Dependencies & Repository Layout

```
Adam/
├── src/
│   ├── adam.h               # Public API header
│   ├── adam.c               # Core agent loop + message management
│   ├── adam_http.c          # LLM provider (Anthropic/OpenAI) via libcurl
│   ├── adam_local.c         # Local LLM inference via llama.cpp
│   ├── adam_json.c          # JSON building & tool-call parsing via jsmn
│   ├── adam_memory.c        # Persistent memory (SQLite + sqlite-memory + sqlite-vector)
│   ├── adam_session.c       # Session persistence (SQLite-backed)
│   ├── adam_stream.c        # Streaming responses (SSE + llama.cpp tokens)
│   ├── adam_context.c       # System prompt builder + memory enrichment
│   ├── adam_evolution.c     # Evolution loop + parallel workers
│   ├── adam_tools.c         # Built-in tool implementations
│   ├── adam_log.c           # Logging subsystem
│   ├── arena.h              # Arena allocator (existing)
│   ├── arena.c              # Arena allocator (existing)
│   └── jsmn.h               # JSMN JSON parser (existing, header-only)
├── deps/
│   ├── curl/                # libcurl (git submodule)
│   ├── mbedtls/             # mbedTLS (git submodule)
│   ├── llama.cpp/           # llama.cpp (git submodule) — native only
│   ├── sqlite/              # SQLite amalgamation
│   ├── sqlite-memory/       # sqlite-memory extension (git submodule)
│   └── sqlite-vector/       # sqlite-vector extension (git submodule)
├── evolution/               # Runtime evolution state (markdown files)
├── Makefile
└── README.md
```

**Dependency rationale**:

| Dependency | Role | Why |
|-----------|------|-----|
| `arena.h/c` | All per-turn memory | Arena reset per iteration — zero individual frees, no leaks, ~200 lines |
| `jsmn.h` | Parse LLM JSON responses & tool call arguments | Zero-alloc, ~450 lines, token-based — perfect for arena pairing |
| `libcurl` | HTTP/HTTPS client for LLM APIs, web_fetch tool, SMTP, SCP | Battle-tested, every protocol Adam could need |
| `mbedtls` | TLS backend for libcurl | Small footprint, embeddable, no OpenSSL dependency |
| `llama.cpp` | Local GGUF model inference | Run models offline, no API costs, full privacy. Native only (not WASM) |
| `SQLite` | Persistent storage for sessions, memory, state | Single-file database, zero-config, battle-tested |
| `sqlite-memory` | Long-term agent knowledge with hybrid search | Intelligent chunking, local embeddings, BM25 + vector search |
| `sqlite-vector` | Fast vector similarity search | 6 vector formats, 6 distance metrics, quantization support |

**Build configurations**:

| Config | Components | Use case |
|--------|-----------|----------|
| `ADAM_FULL` | All deps | Native CLI, server, embedded device |
| `ADAM_NO_LOCAL` | No llama.cpp | API-only mode (smaller binary) |
| `ADAM_WASM` | No curl, no llama.cpp, no pthreads, no SQLite filesystem | Browser/WASI embedding |
| `ADAM_MINIMAL` | arena + jsmn + core loop only | Maximum portability, embedder provides everything |

### 6.1 Core Data Structures

All strings in the hot path are **arena-owned** — allocated from the per-turn arena and freed in bulk via `arena_reset()`. No individual `free()` calls anywhere in the hot path. Strings that must survive across turns (history messages) are `strdup`'d into malloc.

```c
/* === adam.h === */
#pragma once

#include "arena.h"
#include "jsmn.h"
#include <stddef.h>

/* ===================================================================
 * LOGGING
 * =================================================================== */

typedef enum {
    ADAM_LOG_ERROR,
    ADAM_LOG_WARN,
    ADAM_LOG_INFO,
    ADAM_LOG_DEBUG,
    ADAM_LOG_TRACE
} AdamLogLevel;

typedef void (*AdamLogFn)(void *ctx, AdamLogLevel level,
                           const char *msg, size_t len);

/* ===================================================================
 * MESSAGE TYPES
 * =================================================================== */

typedef enum {
    ADAM_ROLE_SYSTEM,
    ADAM_ROLE_USER,
    ADAM_ROLE_ASSISTANT,
    ADAM_ROLE_TOOL
} AdamRole;

/* A tool call attached to an assistant message.
 * Stored in malloc'd memory (survives across turns). */
typedef struct {
    char        *id;                /* malloc'd: unique call ID */
    char        *name;              /* malloc'd: tool name */
    char        *arguments_json;    /* malloc'd: raw JSON string */
} AdamToolCallEntry;

/* Multimodal attachment (image, audio, etc.) */
typedef enum {
    ADAM_MEDIA_IMAGE_PNG,
    ADAM_MEDIA_IMAGE_JPEG,
    ADAM_MEDIA_IMAGE_GIF,
    ADAM_MEDIA_IMAGE_WEBP,
    ADAM_MEDIA_AUDIO_WAV,
    ADAM_MEDIA_AUDIO_MP3,
    ADAM_MEDIA_PDF,
} AdamMediaType;

typedef struct {
    AdamMediaType  type;
    unsigned char *data;             /* malloc'd: raw bytes */
    size_t         data_len;
    char          *filename;         /* malloc'd: optional display name */
} AdamAttachment;

typedef struct {
    AdamRole             role;
    char                *content;            /* malloc'd, null-terminated */
    size_t               content_len;
    /* For ROLE_ASSISTANT: outbound tool calls */
    AdamToolCallEntry   *tool_calls;         /* malloc'd array (or NULL) */
    size_t               tool_call_count;
    /* For ROLE_TOOL: which tool call this result belongs to */
    char                *tool_call_id;       /* malloc'd (or NULL) */
    /* Multimodal: image/audio attachments (for ROLE_USER primarily) */
    AdamAttachment      *attachments;        /* malloc'd array (or NULL) */
    size_t               attachment_count;
} AdamMessage;

/* Conversation history: malloc-backed, survives across turns. */
typedef struct {
    AdamMessage *items;             /* malloc'd growable array */
    size_t       count;
    size_t       capacity;
} AdamHistory;

/* ===================================================================
 * TOOL SYSTEM
 * =================================================================== */

/* Result returned by a tool execution (arena-owned strings). */
typedef struct {
    const char  *for_llm;           /* arena-owned: text sent back to LLM */
    const char  *for_user;          /* arena-owned: text shown to user (optional) */
    int          success;
} AdamToolResult;

/* A parsed tool call from the LLM response (arena-owned strings). */
typedef struct {
    const char  *id;                /* arena-owned */
    const char  *name;              /* arena-owned */
    const char  *arguments_json;    /* arena-owned: raw JSON string */
} AdamToolCall;

typedef AdamToolResult (*AdamToolFn)(
    arena_t     *arena,             /* allocate results from this arena */
    void        *ctx,               /* user-provided context */
    const char  *args_json,         /* raw JSON arguments */
    size_t       args_len
);

typedef struct {
    const char  *name;
    const char  *description;
    const char  *parameters_json;   /* JSON Schema string */
    AdamToolFn   execute;
    void        *ctx;
} AdamToolDef;

typedef struct {
    AdamToolDef *tools;             /* malloc'd growable array */
    size_t       count;
    size_t       capacity;
} AdamToolRegistry;

/* ===================================================================
 * LLM PROVIDER
 * =================================================================== */

typedef enum {
    ADAM_PROVIDER_ANTHROPIC,
    ADAM_PROVIDER_OPENAI,
    ADAM_PROVIDER_OPENAI_COMPAT,     /* any OpenAI-compatible endpoint */
#ifndef ADAM_NO_LOCAL
    ADAM_PROVIDER_LOCAL,              /* llama.cpp GGUF inference */
#endif
} AdamProviderType;

typedef struct {
    AdamProviderType type;
    const char      *api_key;        /* for remote providers */
    const char      *base_url;       /* override for OpenAI-compat */
    const char      *model;          /* e.g. "claude-sonnet-4-20250514" or path.gguf */
    float            temperature;
    int              max_tokens;
    const char      *response_format; /* NULL, "json", or a JSON schema string */

#ifndef ADAM_NO_LOCAL
    /* llama.cpp settings (only for ADAM_PROVIDER_LOCAL) */
    const char      *gguf_path;      /* path to .gguf model file */
    int              n_gpu_layers;    /* GPU offload (-1 = all, 0 = CPU only) */
    int              n_ctx;           /* context window size (0 = model default) */
    int              n_batch;         /* batch size for prompt processing */
#endif
} AdamProvider;

/* ===================================================================
 * STREAMING
 * =================================================================== */

/* Called for each chunk of a streaming response.
 * For HTTP: each SSE data line. For llama.cpp: each token.
 * is_done=1 on the final chunk. */
typedef void (*AdamStreamFn)(void *ctx, const char *chunk, size_t len,
                              int is_done);

/* ===================================================================
 * LLM RESPONSE
 * =================================================================== */

/* All strings arena-owned (freed on arena_reset). */
typedef struct {
    const char    *content;          /* assistant text (may be NULL if tool_calls) */
    AdamToolCall  *tool_calls;       /* arena-allocated array */
    size_t         tool_call_count;
    int            input_tokens;
    int            output_tokens;
    int            error;            /* 0=ok, 1=rate_limit, 2=ctx_overflow, 3=auth */
    const char    *error_msg;        /* arena-owned */
} AdamLLMResponse;

/* ===================================================================
 * RETRY POLICY
 * =================================================================== */

typedef struct {
    int    max_retries;              /* default: 3 */
    int    backoff_ms[4];            /* default: {1000, 2000, 5000, 10000} */
    int    rate_limit_cooldown_ms;   /* default: 30000 (30s on 429) */
} AdamRetryPolicy;

/* ===================================================================
 * MEMORY SYSTEM (SQLite + sqlite-memory + sqlite-vector)
 * =================================================================== */

typedef struct adam_memory_t adam_memory_t;

/* A single search result (arena-owned strings). */
typedef struct {
    const char *content;             /* matched text chunk */
    const char *source;              /* origin: file path, URL, or label */
    const char *context;             /* context/namespace it was stored under */
    float       score;               /* relevance score (0.0 - 1.0) */
} AdamMemoryResult;

/* ===================================================================
 * SESSION PERSISTENCE (SQLite-backed)
 * =================================================================== */

typedef struct adam_session_t adam_session_t;

/* ===================================================================
 * SYSTEM PROMPT BUILDER
 * =================================================================== */

typedef struct {
    const char  *identity;           /* who the agent is */
    const char  *instructions;       /* behavioral rules */
    const char **bootstrap_files;    /* paths to SOUL.md, TOOLS.md, etc. */
    size_t       bootstrap_count;
    int          inject_memory;      /* 1 = auto-enrich from sqlite-memory */
    int          inject_datetime;    /* 1 = inject current date/time */
    const char  *session_info;       /* channel, peer, etc. (optional) */
} AdamSystemPrompt;

/* ===================================================================
 * RATE LIMITER (proactive — prevents 429s before they happen)
 * =================================================================== */

typedef struct {
    int    requests_per_minute;      /* 0 = unlimited */
    int    tokens_per_minute;        /* 0 = unlimited (input+output combined) */
} AdamRateLimit;

/* ===================================================================
 * MODEL CONTEXT WINDOW REGISTRY
 * =================================================================== */

typedef struct {
    const char *model_prefix;        /* e.g. "claude-sonnet-4" */
    int         context_window;      /* e.g. 200000 */
    float       input_cost_per_mtok; /* $ per million input tokens */
    float       output_cost_per_mtok;/* $ per million output tokens */
} AdamModelInfo;

/* ===================================================================
 * AGENT CONFIGURATION
 * =================================================================== */

typedef struct {
    AdamProvider      provider;
    AdamToolRegistry  tools;
    AdamRetryPolicy   retry;
    AdamRateLimit     rate_limit;
    AdamSystemPrompt  system_prompt;
    int               max_iterations;      /* default: 25 */
    int               max_history;         /* default: 50 */
    size_t            arena_block_size;    /* default: 256 * 1024 */

    /* Memory & sessions (NULL = disabled) */
    adam_memory_t    *memory;              /* long-term knowledge store */
    adam_session_t   *sessions;            /* conversation persistence */
    const char       *memory_context;      /* namespace for memory ops */

    /* Streaming (NULL = blocking mode) */
    AdamStreamFn      on_stream;
    void             *stream_ctx;

    /* Callbacks */
    void (*on_response)(void *ctx, const char *text, size_t len);
    void (*on_tool_call)(void *ctx, const char *name, const char *args);
    void (*on_error)(void *ctx, int code, const char *msg);
    void *callback_ctx;

    /* Logging (NULL = no logging) */
    AdamLogFn         log_fn;
    void             *log_ctx;
    AdamLogLevel      log_level;           /* minimum level to emit */

    /* Cancellation: set to 1 from any thread to abort adam_run() */
    volatile int      abort_flag;

    /* WASM: embedder-provided HTTP (overrides libcurl when non-NULL) */
    AdamLLMResponse (*http_fn)(void *ctx, const char *url, const char *method,
                               const char *headers_json, const char *body,
                               size_t body_len, arena_t *arena);
    void             *http_ctx;

    /* Internal (managed by the library) */
    void             *_local_ctx;          /* AdamLocalCtx* for llama.cpp */
    int               _rate_request_count; /* requests in current window */
    int               _rate_token_count;   /* tokens in current window */
    long              _rate_window_start;  /* timestamp of window start */
} AdamConfig;

/* ===================================================================
 * RUN RESULT
 * =================================================================== */

typedef enum {
    ADAM_OK = 0,
    ADAM_ERR_PROVIDER = 1,
    ADAM_ERR_MAX_ITERATIONS = 2,
    ADAM_ERR_ALLOC = 3,
    ADAM_ERR_CONTEXT_OVERFLOW = 4,
    ADAM_ERR_CURL = 5,
    ADAM_ERR_LOCAL = 6,               /* llama.cpp error */
    ADAM_ERR_SQLITE = 7,              /* SQLite / memory / session error */
} AdamStatus;

typedef struct {
    AdamStatus  status;
    char       *final_response;     /* malloc'd — caller must free */
    int         total_iterations;
    int         input_tokens_total;
    int         output_tokens_total;
    float       cost_usd;           /* estimated cost ($0.0 for local models) */
} AdamRunResult;
```

### 6.2 JSON Handling with jsmn + Arena

jsmn produces token offsets into the raw JSON string — no allocation. We pair it with the arena for any strings we need to extract:

```c
/* === adam_json.c === */

#define JSMN_HEADER          /* use jsmn as header-only declarations */
#include "jsmn.h"
#include "arena.h"
#include <string.h>

/* Helper: compare a jsmn token against a string key */
static int json_tok_eq(const char *json, const jsmntok_t *tok, const char *s) {
    size_t slen = strlen(s);
    return (tok->type == JSMN_STRING
         && (size_t)(tok->end - tok->start) == slen
         && memcmp(json + tok->start, s, slen) == 0);
}

/* Helper: extract a token's value as an arena-owned string */
static const char *json_tok_str(arena_t *arena, const char *json,
                                const jsmntok_t *tok) {
    size_t len = (size_t)(tok->end - tok->start);
    char *s = arena_alloc(arena, len + 1);
    if (!s) return NULL;
    memcpy(s, json + tok->start, len);
    s[len] = '\0';
    return s;
}

/* Parse tool_calls from an Anthropic Messages API response.
 *
 * Anthropic returns:
 *   { "content": [ {"type":"text","text":"..."},
 *                   {"type":"tool_use","id":"...","name":"...","input":{...}} ],
 *     "usage": {"input_tokens":N, "output_tokens":N} }
 *
 * OpenAI returns:
 *   { "choices": [{"message": {"content":"...",
 *       "tool_calls": [{"id":"...","function":{"name":"...","arguments":"..."}}]
 *     }}], "usage": {"prompt_tokens":N, "completion_tokens":N} }
 */
AdamLLMResponse adam_parse_response(arena_t *arena, const char *json,
                                    size_t json_len, AdamProviderType type) {
    AdamLLMResponse resp = {0};

    /* Allocate jsmn tokens from arena (estimate: 1 token per 8 chars) */
    size_t max_tokens = json_len / 4 + 64;
    jsmntok_t *tokens = arena_alloc(arena, max_tokens * sizeof(jsmntok_t));
    if (!tokens) { resp.error = ADAM_ERR_ALLOC; return resp; }

    jsmn_parser parser;
    jsmn_init(&parser);
    int ntok = jsmn_parse(&parser, json, json_len, tokens, (unsigned)max_tokens);
    if (ntok < 0) {
        resp.error = ADAM_ERR_PROVIDER;
        resp.error_msg = arena_strdup(arena, "JSON parse error");
        return resp;
    }

    /* Provider-specific extraction using token walk */
    if (type == ADAM_PROVIDER_ANTHROPIC) {
        adam_parse_anthropic(arena, json, tokens, ntok, &resp);
    } else {
        adam_parse_openai(arena, json, tokens, ntok, &resp);
    }
    return resp;
}

/* Build the request JSON body for an Anthropic Messages API call.
 * Uses arena for scratch space, returns an arena-owned JSON string. */
const char *adam_build_request_anthropic(
    arena_t *arena,
    const AdamMessage *msgs, size_t msg_count,
    const AdamToolDef *tools, size_t tool_count,
    const AdamProvider *provider
) {
    /* Estimate buffer size: system prompt + messages + tool schemas */
    size_t est = 4096;
    for (size_t i = 0; i < msg_count; i++)
        est += msgs[i].content_len + 128;
    for (size_t i = 0; i < tool_count; i++)
        est += strlen(tools[i].parameters_json) + 256;

    char *buf = arena_alloc(arena, est);
    if (!buf) return NULL;
    size_t pos = 0;

    /* Build JSON manually (no snprintf dependency for core parts) */
    pos += snprintf(buf + pos, est - pos,
        "{\"model\":\"%s\",\"max_tokens\":%d,\"temperature\":%.2f,",
        provider->model, provider->max_tokens, provider->temperature);

    /* System message (first message if role == system) */
    if (msg_count > 0 && msgs[0].role == ADAM_ROLE_SYSTEM) {
        pos += snprintf(buf + pos, est - pos,
            "\"system\":[{\"type\":\"text\",\"text\":");
        pos += adam_json_escape(buf + pos, est - pos,
                               msgs[0].content, msgs[0].content_len);
        pos += snprintf(buf + pos, est - pos,
            ",\"cache_control\":{\"type\":\"ephemeral\"}}],");
    }

    /* Messages array */
    pos += snprintf(buf + pos, est - pos, "\"messages\":[");
    int first = 1;
    for (size_t i = (msgs[0].role == ADAM_ROLE_SYSTEM ? 1 : 0);
         i < msg_count; i++) {
        if (!first) buf[pos++] = ',';
        first = 0;
        const char *role_str = (msgs[i].role == ADAM_ROLE_USER) ? "user"
                             : (msgs[i].role == ADAM_ROLE_ASSISTANT) ? "assistant"
                             : "user"; /* tool results wrapped as user */
        pos += snprintf(buf + pos, est - pos,
            "{\"role\":\"%s\",\"content\":", role_str);
        pos += adam_json_escape(buf + pos, est - pos,
                               msgs[i].content, msgs[i].content_len);
        buf[pos++] = '}';
    }
    pos += snprintf(buf + pos, est - pos, "]");

    /* Tools array */
    if (tool_count > 0) {
        pos += snprintf(buf + pos, est - pos, ",\"tools\":[");
        for (size_t i = 0; i < tool_count; i++) {
            if (i > 0) buf[pos++] = ',';
            pos += snprintf(buf + pos, est - pos,
                "{\"name\":\"%s\",\"description\":\"%s\","
                "\"input_schema\":%s}",
                tools[i].name, tools[i].description,
                tools[i].parameters_json);
        }
        buf[pos++] = ']';
    }

    buf[pos++] = '}';
    buf[pos] = '\0';
    return buf;
}
```

### 6.3 HTTP Layer with libcurl + mbedtls

libcurl handles all networking — LLM API calls, web_fetch tool, and future protocols (SMTP, SCP). mbedtls provides the TLS backend without requiring system OpenSSL.

```c
/* === adam_http.c === */

#include <curl/curl.h>
#include "adam.h"

/* Write callback: accumulates response body into arena */
typedef struct {
    arena_t *arena;
    char    *buf;
    size_t   len;
    size_t   cap;
} CurlWriteCtx;

static size_t curl_write_cb(char *data, size_t size, size_t nmemb,
                            void *userp) {
    CurlWriteCtx *ctx = (CurlWriteCtx *)userp;
    size_t bytes = size * nmemb;

    /* Grow buffer if needed (arena-allocated) */
    if (ctx->len + bytes >= ctx->cap) {
        size_t new_cap = (ctx->cap == 0) ? 8192 : ctx->cap * 2;
        while (new_cap < ctx->len + bytes + 1) new_cap *= 2;
        char *new_buf = arena_alloc(ctx->arena, new_cap);
        if (!new_buf) return 0;
        if (ctx->buf) memcpy(new_buf, ctx->buf, ctx->len);
        ctx->buf = new_buf;
        ctx->cap = new_cap;
        /* old buf stays in arena — freed on arena_reset */
    }

    memcpy(ctx->buf + ctx->len, data, bytes);
    ctx->len += bytes;
    ctx->buf[ctx->len] = '\0';
    return bytes;
}

/* Call an LLM provider API. All memory allocated from the per-turn arena. */
AdamLLMResponse adam_llm_call(
    arena_t *arena,
    const AdamProvider *provider,
    const AdamMessage *msgs, size_t msg_count,
    const AdamToolDef *tools, size_t tool_count
) {
    AdamLLMResponse resp = {0};
    CURL *curl = curl_easy_init();
    if (!curl) {
        resp.error = ADAM_ERR_CURL;
        resp.error_msg = arena_strdup(arena, "curl_easy_init failed");
        return resp;
    }

    /* Build request JSON (arena-allocated) */
    const char *body;
    if (provider->type == ADAM_PROVIDER_ANTHROPIC) {
        body = adam_build_request_anthropic(arena, msgs, msg_count,
                                           tools, tool_count, provider);
    } else {
        body = adam_build_request_openai(arena, msgs, msg_count,
                                        tools, tool_count, provider);
    }

    /* Set URL */
    const char *url;
    if (provider->base_url) {
        url = provider->base_url;
    } else if (provider->type == ADAM_PROVIDER_ANTHROPIC) {
        url = "https://api.anthropic.com/v1/messages";
    } else {
        url = "https://api.openai.com/v1/chat/completions";
    }
    curl_easy_setopt(curl, CURLOPT_URL, url);

    /* Headers (arena-allocated strings) */
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    if (provider->type == ADAM_PROVIDER_ANTHROPIC) {
        char *auth = arena_alloc(arena, strlen(provider->api_key) + 32);
        sprintf(auth, "x-api-key: %s", provider->api_key);
        headers = curl_slist_append(headers, auth);
        headers = curl_slist_append(headers,
                                    "anthropic-version: 2023-06-01");
    } else {
        char *auth = arena_alloc(arena, strlen(provider->api_key) + 32);
        sprintf(auth, "Authorization: Bearer %s", provider->api_key);
        headers = curl_slist_append(headers, auth);
    }
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    /* POST body */
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)strlen(body));

    /* Response accumulator */
    CurlWriteCtx write_ctx = { .arena = arena };
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &write_ctx);

    /* Timeouts */
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);

    /* Perform */
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        resp.error = ADAM_ERR_CURL;
        resp.error_msg = arena_strdup(arena, curl_easy_strerror(res));
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return resp;
    }

    /* Check HTTP status */
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    if (http_code == 429) {
        resp.error = 1; /* rate_limit */
        resp.error_msg = arena_strdup(arena, "Rate limited (429)");
    } else if (http_code == 401 || http_code == 403) {
        resp.error = 3; /* auth */
        resp.error_msg = arena_strdup(arena, "Auth error");
    } else if (http_code >= 400) {
        resp.error = ADAM_ERR_PROVIDER;
        resp.error_msg = arena_strdup(arena,
            write_ctx.buf ? write_ctx.buf : "HTTP error");
    } else {
        /* Parse response JSON */
        resp = adam_parse_response(arena, write_ctx.buf, write_ctx.len,
                                  provider->type);
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return resp;
}
```

### 6.4 The Core Agent Loop

The key insight: each tool-iteration uses a **per-turn arena** that is reset after the LLM response is processed. Strings that must survive (history messages) are copied to malloc'd storage. Everything else — JSON building, parsing, tool results — lives in the arena and vanishes on reset.

```c
/* === adam.c === */

#include "adam.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* --- History management (malloc-backed, survives across turns) --- */

static void history_grow(AdamHistory *h) {
    if (h->count >= h->capacity) {
        size_t new_cap = h->capacity ? h->capacity * 2 : 32;
        h->items = realloc(h->items, new_cap * sizeof(AdamMessage));
        h->capacity = new_cap;
    }
}

static void history_append(AdamHistory *h, AdamRole role,
                           const char *content, const char *tool_call_id) {
    history_grow(h);
    AdamMessage *m = &h->items[h->count++];
    memset(m, 0, sizeof(*m));
    m->role = role;
    m->content = strdup(content ? content : "");
    m->content_len = strlen(m->content);
    m->tool_call_id = tool_call_id ? strdup(tool_call_id) : NULL;
}

/* Append an assistant message that includes tool calls.
 * Copies tool call data from arena-owned to malloc'd storage. */
static void history_append_assistant_with_tools(
    AdamHistory *h, const char *content,
    const AdamToolCall *calls, size_t call_count
) {
    history_grow(h);
    AdamMessage *m = &h->items[h->count++];
    memset(m, 0, sizeof(*m));
    m->role = ADAM_ROLE_ASSISTANT;
    m->content = strdup(content ? content : "");
    m->content_len = strlen(m->content);
    if (call_count > 0 && calls) {
        m->tool_calls = malloc(call_count * sizeof(AdamToolCallEntry));
        m->tool_call_count = call_count;
        for (size_t i = 0; i < call_count; i++) {
            m->tool_calls[i].id = calls[i].id ? strdup(calls[i].id) : NULL;
            m->tool_calls[i].name = strdup(calls[i].name);
            m->tool_calls[i].arguments_json =
                calls[i].arguments_json ? strdup(calls[i].arguments_json) : NULL;
        }
    }
}

static void history_message_free(AdamMessage *m) {
    free(m->content);
    free(m->tool_call_id);
    for (size_t i = 0; i < m->tool_call_count; i++) {
        free(m->tool_calls[i].id);
        free(m->tool_calls[i].name);
        free(m->tool_calls[i].arguments_json);
    }
    free(m->tool_calls);
}

static void history_compress(AdamHistory *h) {
    /* Keep system message (index 0) + last 50% of remaining messages */
    if (h->count <= 4) return;
    size_t keep = h->count / 2;
    if (keep < 2) keep = 2;
    size_t drop = h->count - 1 - keep;
    for (size_t i = 1; i <= drop; i++)
        history_message_free(&h->items[i]);
    memmove(&h->items[1], &h->items[1 + drop], keep * sizeof(AdamMessage));
    h->count = 1 + keep;
}

/* --- Main Agent Loop --- */

AdamRunResult adam_run(AdamConfig *cfg, AdamHistory *history,
                       const char *user_message) {
    AdamRunResult result = {0};

    /* 1. Append user message to history */
    history_append(history, ADAM_ROLE_USER, user_message, NULL);

    /* 2. Ensure system prompt is at history[0] */
    if (history->count == 0 || history->items[0].role != ADAM_ROLE_SYSTEM) {
        /* Prepend: shift everything right by 1 */
        if (history->count >= history->capacity) {
            size_t new_cap = history->capacity ? history->capacity * 2 : 32;
            history->items = realloc(history->items,
                                     new_cap * sizeof(AdamMessage));
            history->capacity = new_cap;
        }
        memmove(&history->items[1], &history->items[0],
                history->count * sizeof(AdamMessage));
        history->items[0] = (AdamMessage){
            .role = ADAM_ROLE_SYSTEM,
            .content = strdup(cfg->system_prompt),
            .content_len = strlen(cfg->system_prompt),
        };
        history->count++;
    }

    /* 3. Create per-turn arena */
    size_t block_size = cfg->arena_block_size ? cfg->arena_block_size
                                              : 256 * 1024;
    arena_t *turn_arena = arena_create(block_size);
    if (!turn_arena) {
        result.status = ADAM_ERR_ALLOC;
        result.final_response = strdup("Arena allocation failed");
        return result;
    }

    /* 4. Tool iteration loop */
    int iteration = 0;
    int max_iter = cfg->max_iterations > 0 ? cfg->max_iterations : 25;

    while (iteration < max_iter) {
        iteration++;

        /* 4a. Reset arena for this iteration (all prior iteration
         *     scratch memory is freed in one shot) */
        arena_reset(turn_arena);

        /* 4b. Trim history if too long */
        int max_hist = cfg->max_history > 0 ? cfg->max_history : 50;
        if (history->count > (size_t)(max_hist + 1)) {
            history_compress(history);
        }

        /* 4c. Call LLM via libcurl (all response data is arena-owned) */
        AdamLLMResponse resp = adam_llm_call(
            turn_arena, &cfg->provider,
            history->items, history->count,
            cfg->tools.tools, cfg->tools.count
        );

        /* 4d. Handle errors */
        if (resp.error) {
            if (resp.error == 2 /* ctx_overflow */ && iteration <= 2) {
                history_compress(history);
                continue; /* retry with shorter history */
            }
            result.status = ADAM_ERR_PROVIDER;
            /* Copy error out of arena before it's destroyed */
            result.final_response = strdup(
                resp.error_msg ? resp.error_msg : "LLM error");
            break;
        }

        result.input_tokens_total += resp.input_tokens;
        result.output_tokens_total += resp.output_tokens;

        /* 4e. No tool calls → final text response */
        if (resp.tool_call_count == 0) {
            const char *text = resp.content ? resp.content : "";
            /* Copy out of arena into malloc'd result */
            result.final_response = strdup(text);
            /* Persist to history (malloc'd copy) */
            history_append(history, ADAM_ROLE_ASSISTANT, text, NULL);
            result.status = ADAM_OK;
            if (cfg->on_response)
                cfg->on_response(cfg->callback_ctx, text, strlen(text));
            break;
        }

        /* 4f. Has tool calls — persist assistant message with tool call IDs.
         *      This is required by the Anthropic API: assistant messages must
         *      contain the tool_use blocks, and tool results reference them. */
        history_append_assistant_with_tools(
            history, resp.content ? resp.content : "",
            resp.tool_calls, resp.tool_call_count);

        /* 4g. Execute each tool call */
        for (size_t i = 0; i < resp.tool_call_count; i++) {
            AdamToolCall *tc = &resp.tool_calls[i];

            if (cfg->on_tool_call)
                cfg->on_tool_call(cfg->callback_ctx, tc->name,
                                  tc->arguments_json);

            /* Find tool in registry */
            AdamToolResult tres = {0};
            int found = 0;
            for (size_t t = 0; t < cfg->tools.count; t++) {
                if (strcmp(cfg->tools.tools[t].name, tc->name) == 0) {
                    tres = cfg->tools.tools[t].execute(
                        turn_arena,
                        cfg->tools.tools[t].ctx,
                        tc->arguments_json,
                        tc->arguments_json ? strlen(tc->arguments_json) : 0
                    );
                    found = 1;
                    break;
                }
            }

            if (!found) {
                tres.for_llm = arena_strdup(turn_arena,
                                            "Error: tool not found");
                tres.success = 0;
            }

            /* Persist tool result to history (copies out of arena) */
            const char *result_text = tres.for_llm ? tres.for_llm
                                                   : "(no output)";
            history_append(history, ADAM_ROLE_TOOL, result_text, tc->id);

            /* Forward user-facing output */
            if (tres.for_user && cfg->on_response)
                cfg->on_response(cfg->callback_ctx, tres.for_user,
                                 strlen(tres.for_user));
        }

        /* Arena will be reset at top of next iteration.
         * All tool results, JSON buffers, curl responses — gone. */
    }

    if (iteration >= max_iter && result.status == 0
        && !result.final_response) {
        result.status = ADAM_ERR_MAX_ITERATIONS;
        result.final_response = strdup("Max tool iterations reached");
    }

    result.total_iterations = iteration;
    arena_destroy(turn_arena);
    return result;
}
```

### 6.5 Memory Model: Arena + Malloc + SQLite Layers

```
┌──────────────────────────────────────────────────────────────────────┐
│                       MEMORY ARCHITECTURE                            │
│                                                                      │
│  ┌──────────────────┐  ┌────────────────────┐  ┌──────────────────┐ │
│  │ MALLOC ZONE       │  │ PER-TURN ARENA     │  │ SQLITE (disk)    │ │
│  │ (survives turns)  │  │ (reset each iter)  │  │ (survives runs)  │ │
│  │                   │  │                    │  │                  │ │
│  │  AdamHistory      │  │  JSON request body │  │  Sessions        │ │
│  │    .items[]       │  │  curl response buf │  │    history rows   │ │
│  │    .content (dup) │  │  jsmn token array  │  │    summaries     │ │
│  │    .tool_calls[]  │  │  AdamLLMResponse   │  │                  │ │
│  │    .tool_call_id  │  │  AdamToolCall[]    │  │  Memory          │ │
│  │                   │  │  AdamToolResult    │  │    chunks + emb  │ │
│  │  AdamConfig       │  │  JSON escapes      │  │    FTS5 index    │ │
│  │  AdamToolRegistry │  │  memory results    │  │    vector index  │ │
│  │  final_response   │  │  stream chunks     │  │                  │ │
│  │  llama_context    │  │  scratch strings   │  │  Evolution       │ │
│  │                   │  │                    │  │    attempt logs   │ │
│  │  [long-lived]     │  │  [arena_reset]     │  │    strategies    │ │
│  └──────────────────┘  └────────────────────┘  └──────────────────┘ │
│                                                                      │
│  Rule: Arena for all per-iteration scratch. Malloc for in-memory     │
│  state that survives across iterations. SQLite for everything that   │
│  must survive across process restarts.                               │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.6 Library API Surface

```c
/* === adam.h — Public API === */

/* ---- Global lifecycle ---- */
void adam_global_init(void);       /* call once: curl_global_init + sqlite init */
void adam_global_cleanup(void);

/* ---- Agent lifecycle ---- */
AdamConfig  *adam_create(void);
void         adam_destroy(AdamConfig *cfg);

/* ---- Provider configuration ---- */
void adam_set_provider(AdamConfig *cfg, AdamProviderType type,
                       const char *api_key, const char *model);
void adam_set_base_url(AdamConfig *cfg, const char *url);
void adam_set_temperature(AdamConfig *cfg, float temperature);
void adam_set_max_tokens(AdamConfig *cfg, int max_tokens);
void adam_set_response_format(AdamConfig *cfg, const char *format);

#ifndef ADAM_NO_LOCAL
/* Local inference (llama.cpp) */
void adam_set_gguf(AdamConfig *cfg, const char *gguf_path,
                   int n_gpu_layers, int n_ctx);
#endif

/* ---- System prompt ---- */
void adam_set_identity(AdamConfig *cfg, const char *identity);
void adam_set_instructions(AdamConfig *cfg, const char *instructions);
void adam_add_bootstrap_file(AdamConfig *cfg, const char *path);
void adam_set_memory_enrichment(AdamConfig *cfg, int enabled);

/* ---- Tools ---- */
void adam_add_tool(AdamConfig *cfg, AdamToolDef tool);
void adam_remove_tool(AdamConfig *cfg, const char *name);

/* ---- Callbacks ---- */
void adam_set_callbacks(AdamConfig *cfg,
    void (*on_response)(void *, const char *, size_t),
    void (*on_tool_call)(void *, const char *, const char *),
    void (*on_error)(void *, int, const char *),
    void *ctx
);
void adam_set_stream_callback(AdamConfig *cfg, AdamStreamFn fn, void *ctx);
void adam_set_logger(AdamConfig *cfg, AdamLogFn fn, void *ctx,
                     AdamLogLevel min_level);

/* ---- Retry policy ---- */
void adam_set_retry_policy(AdamConfig *cfg, AdamRetryPolicy policy);

/* ---- Core execution ---- */
AdamRunResult adam_run(AdamConfig *cfg, AdamHistory *history, const char *msg);
void          adam_result_free(AdamRunResult *r);

/* ---- History management ---- */
AdamHistory  *adam_history_create(void);
void          adam_history_destroy(AdamHistory *h);
void          adam_history_clear(AdamHistory *h);
size_t        adam_history_estimate_tokens(const AdamHistory *h);

/* ---- Smart compression (uses the LLM to summarize old messages) ---- */
int adam_history_summarize(AdamConfig *cfg, AdamHistory *h,
                           size_t target_token_count);

/* ---- Memory system (SQLite + sqlite-memory + sqlite-vector) ---- */
adam_memory_t *adam_memory_open(const char *db_path);
void           adam_memory_close(adam_memory_t *mem);

/* Ingest knowledge */
int adam_memory_add(adam_memory_t *mem, const char *text,
                    const char *context, const char *source);
int adam_memory_add_file(adam_memory_t *mem, const char *path,
                         const char *context);
int adam_memory_add_directory(adam_memory_t *mem, const char *dir_path,
                              const char *context);

/* Hybrid search (BM25 + vector similarity via sqlite-memory) */
int adam_memory_search(adam_memory_t *mem, arena_t *arena,
                       const char *query, int limit,
                       AdamMemoryResult **results, size_t *count);

/* Lifecycle management */
int adam_memory_delete_context(adam_memory_t *mem, const char *context);
int adam_memory_clear(adam_memory_t *mem);

/* Sync memory between agents (shared SQLite database) */
int adam_memory_attach(adam_memory_t *mem, const char *remote_db_path,
                       const char *alias);
int adam_memory_sync(adam_memory_t *mem, const char *alias,
                     const char *context);

/* ---- Session persistence (SQLite-backed) ---- */
adam_session_t *adam_session_open(const char *db_path);
void            adam_session_close(adam_session_t *sess);

int  adam_session_save(adam_session_t *sess, const char *session_id,
                       const AdamHistory *h);
int  adam_session_load(adam_session_t *sess, const char *session_id,
                       AdamHistory *h);
int  adam_session_list(adam_session_t *sess, char ***ids, size_t *count);
int  adam_session_delete(adam_session_t *sess, const char *session_id);

/* ---- Token estimation ---- */
size_t adam_estimate_tokens(const char *text, size_t len);

/* ---- Cancellation ---- */
void adam_abort(AdamConfig *cfg);          /* set abort_flag = 1 */
void adam_reset_abort(AdamConfig *cfg);    /* set abort_flag = 0 */

/* ---- Rate limiting ---- */
void adam_set_rate_limit(AdamConfig *cfg, AdamRateLimit limit);

/* ---- Model registry (context window + cost lookup) ---- */
void adam_register_model(const AdamModelInfo *info);
int  adam_get_context_window(const char *model);   /* returns 0 if unknown */
float adam_estimate_cost(const char *model, int input_tokens, int output_tokens);

/* ---- Multimodal: attach media to a message ---- */
int adam_message_attach(AdamMessage *msg, AdamMediaType type,
                        const unsigned char *data, size_t len,
                        const char *filename);
int adam_message_attach_file(AdamMessage *msg, const char *file_path);

/* ---- Built-in tools (use libcurl internally) ---- */
AdamToolResult adam_tool_web_fetch(arena_t *arena, void *ctx,
                                   const char *args_json, size_t args_len);
AdamToolResult adam_tool_memory_search(arena_t *arena, void *ctx,
                                       const char *args_json, size_t args_len);
AdamToolResult adam_tool_memory_add(arena_t *arena, void *ctx,
                                    const char *args_json, size_t args_len);
```

### 6.7 What the Library Provides vs What the Embedder Provides

| **Adam library provides** | **Embedder provides** |
|---------------------------|-----------------------|
| Agent loop state machine | Custom tool implementations (shell, file, etc.) |
| LLM HTTP calls (Anthropic + OpenAI) via libcurl | API keys and model selection |
| Local inference via llama.cpp (GGUF models) | GGUF model files |
| JSON request building & response parsing via jsmn | Custom tool definitions |
| Arena-based per-turn memory management | Custom callbacks for UI/output |
| History management & smart summarization | Evaluation functions (for evolution) |
| Session persistence (SQLite-backed) | Additional protocol tools (SMTP, SCP) |
| Long-term memory (sqlite-memory + sqlite-vector) | |
| Memory sync between agents (SQLite attach) | |
| System prompt builder with memory enrichment | |
| Streaming responses (SSE + llama.cpp tokens) | |
| Token estimation & context budget management | |
| Retry with exponential backoff | |
| Structured logging | |
| Built-in tools: `web_fetch`, `memory_search`, `memory_add` | |
| Error classification (rate limit, auth, overflow) | |

### 6.8 Build

```makefile
# === Step 1: Build dependencies (static libraries) ===

# mbedtls
cd deps/mbedtls && cmake -DENABLE_TESTING=OFF -DENABLE_PROGRAMS=OFF \
    -DCMAKE_BUILD_TYPE=Release . && make -j$(nproc)

# libcurl with mbedtls backend
cd deps/curl && cmake -DCURL_USE_MBEDTLS=ON -DCURL_USE_OPENSSL=OFF \
    -DHTTP_ONLY=OFF -DBUILD_SHARED_LIBS=OFF \
    -DMBEDTLS_INCLUDE_DIRS=../mbedtls/include \
    -DMBEDTLS_LIBRARIES="../mbedtls/library/libmbedtls.a;../mbedtls/library/libmbedx509.a;../mbedtls/library/libmbedcrypto.a" \
    -DCMAKE_BUILD_TYPE=Release . && make -j$(nproc)

# llama.cpp (native only)
cd deps/llama.cpp && cmake -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release . && make -j$(nproc)

# SQLite amalgamation (compiled as part of Adam)
# sqlite-memory and sqlite-vector are compiled as loadable extensions or statically linked

# === Step 2: Build Adam ===

CFLAGS = -O2 -Wall -Wextra -std=c11 -Isrc \
         -Ideps/curl/include -Ideps/llama.cpp/include \
         -Ideps/sqlite -Ideps/sqlite-memory -Ideps/sqlite-vector

LDFLAGS = deps/curl/lib/libcurl.a \
          deps/mbedtls/library/libmbedtls.a \
          deps/mbedtls/library/libmbedx509.a \
          deps/mbedtls/library/libmbedcrypto.a \
          deps/llama.cpp/lib/libllama.a \
          deps/llama.cpp/lib/libggml.a \
          -lpthread -lz -lm -lstdc++

SRCS = src/arena.c src/adam.c src/adam_json.c src/adam_http.c \
       src/adam_local.c src/adam_memory.c src/adam_session.c \
       src/adam_stream.c src/adam_context.c src/adam_evolution.c \
       src/adam_tools.c src/adam_log.c \
       deps/sqlite/sqlite3.c

# Full build (all features)
libadam.a: $(SRCS:.c=.o)
	ar rcs $@ $^

adam: libadam.a examples/main.c
	$(CC) $(CFLAGS) examples/main.c -L. -ladam $(LDFLAGS) -o $@

# API-only build (no llama.cpp)
libadam-api.a:
	$(CC) $(CFLAGS) -DADAM_NO_LOCAL -c $(filter-out src/adam_local.c,$(SRCS))
	ar rcs $@ $(notdir $(SRCS:.c=.o))

# WASM build (no curl, no llama.cpp, no pthreads, SQLite in-memory only)
adam.wasm:
	emcc -O2 -s WASM=1 -DADAM_NO_CURL -DADAM_NO_LOCAL -DADAM_NO_PTHREADS \
	     -s EXPORTED_FUNCTIONS='["_adam_create","_adam_run","_adam_result_free", \
	        "_adam_memory_open","_adam_memory_search","_adam_memory_add"]' \
	     -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
	     $(filter-out src/adam_http.c src/adam_local.c,$(SRCS)) \
	     -o adam.wasm
```

---

## 7. Self-Evolving Agent Architecture

The goal: an agent that given a task runs indefinitely, evolving itself to become the most efficient and best solution. Evolution is tracked through markdown files, continuously updated, learning from past mistakes. Multiple agents can evolve in parallel.

### 7.1 Concept: Evolution Through Markdown

```
┌──────────────────────────────────────────────────────┐
│                   EVOLUTION LOOP                      │
│                                                       │
│  ┌─────────┐    ┌──────────┐    ┌──────────────────┐ │
│  │ TASK.md  │───▶│  AGENT   │───▶│ EVOLUTION.md     │ │
│  │ (goal)   │    │  (runs)  │    │ (learns)         │ │
│  └─────────┘    └────┬─────┘    └────────┬─────────┘ │
│                      │                    │           │
│                      ▼                    ▼           │
│               ┌────────────┐     ┌───────────────┐   │
│               │ ATTEMPT.md │     │ STRATEGY.md   │   │
│               │ (log #N)   │     │ (best known)  │   │
│               └────────────┘     └───────────────┘   │
│                      │                    │           │
│                      └────────┬───────────┘           │
│                               ▼                       │
│                     ┌──────────────────┐              │
│                     │   NEXT ATTEMPT   │              │
│                     │ (informed by all │              │
│                     │  prior history)  │              │
│                     └──────────────────┘              │
└──────────────────────────────────────────────────────┘
```

### 7.2 Evolution File Structure

```
evolution/
├── TASK.md                    # Immutable: the original goal
├── STRATEGY.md                # Mutable: current best approach
├── METRICS.md                 # Mutable: scoring criteria & results
├── attempts/
│   ├── 001_initial.md         # What was tried, what happened, score
│   ├── 002_refine_prompt.md
│   ├── 003_add_tool.md
│   └── ...
├── insights/
│   ├── wrong_assumptions.md   # Lessons learned (anti-patterns)
│   ├── effective_patterns.md  # What works (reuse these)
│   └── open_questions.md      # Unknowns to explore
└── artifacts/
    └── best_solution/         # Current best output
```

### 7.3 Evolution Loop in C

```c
/* === adam_evolution.c === */

#include "adam.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

typedef struct {
    char  *task_md;              /* original goal (immutable, malloc'd) */
    char  *strategy_md;          /* current best approach (malloc'd) */
    char  *metrics_md;           /* scoring criteria (malloc'd) */
    int    attempt_count;
    int    current_score;        /* best score so far (0-100) */
    char   base_path[256];       /* e.g. "evolution/" */
} AdamEvolution;

typedef struct {
    int    score;                /* 0-100 */
    char  *reasoning;            /* malloc'd: why this score */
    char  *next_suggestion;      /* malloc'd: what to try next */
} AdamEvalResult;

/* File I/O helpers (the only place we touch the filesystem) */
static char *read_file_malloc(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return strdup("");
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)len + 1);
    if (buf) { fread(buf, 1, (size_t)len, f); buf[len] = '\0'; }
    fclose(f);
    return buf ? buf : strdup("");
}

static void write_file(const char *path, const char *content) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    fputs(content, f);
    fclose(f);
}

/* Build a prompt that includes all evolution state.
 * Uses a scratch arena for string concatenation. */
static char *build_evolution_prompt(arena_t *arena, AdamEvolution *evo,
                                     int attempt_num) {
    /* Load all attempt files + insight files */
    char *wrong = read_file_malloc("evolution/insights/wrong_assumptions.md");
    char *effective = read_file_malloc("evolution/insights/effective_patterns.md");
    char *open_q = read_file_malloc("evolution/insights/open_questions.md");

    /* Load last 5 attempts (don't overwhelm context) */
    size_t est = strlen(evo->task_md) + strlen(evo->strategy_md)
               + strlen(evo->metrics_md) + strlen(wrong)
               + strlen(effective) + strlen(open_q) + 8192;
    char *prompt = arena_alloc(arena, est);
    if (!prompt) { free(wrong); free(effective); free(open_q); return NULL; }

    snprintf(prompt, est,
        "You are a self-improving agent on attempt #%d.\n\n"
        "## TASK (immutable)\n%s\n\n"
        "## CURRENT STRATEGY (score: %d/100)\n%s\n\n"
        "## SCORING CRITERIA\n%s\n\n"
        "## WRONG ASSUMPTIONS (avoid these)\n%s\n\n"
        "## EFFECTIVE PATTERNS (use these)\n%s\n\n"
        "## OPEN QUESTIONS\n%s\n\n"
        "## INSTRUCTIONS\n"
        "1. Produce a BETTER solution than all previous attempts.\n"
        "2. Explain what you changed and why.\n"
        "3. If you discover a wrong assumption, state it clearly.\n"
        "4. If you find an effective pattern, state it clearly.\n",
        attempt_num,
        evo->task_md, evo->current_score, evo->strategy_md,
        evo->metrics_md, wrong, effective, open_q);

    free(wrong); free(effective); free(open_q);
    return prompt;
}

/* Run the evolution loop until should_stop returns true */
void adam_evolution_run(
    AdamConfig *cfg,
    AdamEvolution *evo,
    AdamEvalResult (*evaluate)(const char *attempt_output),
    int (*should_stop)(int attempt_num, int best_score)
) {
    /* Scratch arena for prompt building (reset each attempt) */
    arena_t *scratch = arena_create(128 * 1024);
    if (!scratch) return;

    while (!should_stop(evo->attempt_count, evo->current_score)) {
        evo->attempt_count++;
        arena_reset(scratch);

        /* 1. Build evolution prompt */
        char *prompt = build_evolution_prompt(scratch, evo,
                                              evo->attempt_count);
        if (!prompt) break;

        /* 2. Run the inner agent loop (fresh history each attempt) */
        AdamHistory *history = adam_history_create();
        AdamRunResult result = adam_run(cfg, history, prompt);

        /* 3. Evaluate */
        AdamEvalResult eval = evaluate(
            result.final_response ? result.final_response : "");

        /* 4. Log the attempt */
        char path[256];
        snprintf(path, sizeof(path), "%sattempts/%03d.md",
                 evo->base_path, evo->attempt_count);
        arena_reset(scratch);
        char *log = arena_alloc(scratch, 4096 + strlen(result.final_response ? result.final_response : ""));
        if (log) {
            snprintf(log, 4096,
                "# Attempt %d — Score: %d/100\n\n"
                "## Output\n%s\n\n"
                "## Evaluation\n%s\n",
                evo->attempt_count, eval.score,
                result.final_response ? result.final_response : "(empty)",
                eval.reasoning ? eval.reasoning : "(none)");
            write_file(path, log);
        }

        /* 5. Update strategy if best score improved */
        if (eval.score > evo->current_score) {
            evo->current_score = eval.score;

            /* Ask agent to refine strategy */
            arena_reset(scratch);
            char *strat_prompt = arena_alloc(scratch, 4096);
            if (strat_prompt) {
                snprintf(strat_prompt, 4096,
                    "Attempt #%d scored %d (new best). "
                    "Update the strategy:\n\nPrevious:\n%s\n\n"
                    "What worked: %s\nSuggestion: %s\n\n"
                    "Write the updated strategy as markdown.",
                    evo->attempt_count, eval.score,
                    evo->strategy_md,
                    eval.reasoning ? eval.reasoning : "",
                    eval.next_suggestion ? eval.next_suggestion : "");

                AdamRunResult update = adam_run(cfg, history, strat_prompt);
                if (update.final_response) {
                    write_file("evolution/STRATEGY.md",
                               update.final_response);
                    free(evo->strategy_md);
                    evo->strategy_md = strdup(update.final_response);
                }
                adam_result_free(&update);
            }
        }

        /* 6. Update insights (always — failures teach too) */
        arena_reset(scratch);
        char *insight_prompt = arena_alloc(scratch, 4096);
        if (insight_prompt) {
            snprintf(insight_prompt, 4096,
                "Attempt #%d scored %d.\nReasoning: %s\n\n"
                "Based on this, update:\n"
                "1. WRONG ASSUMPTIONS (things we now know are false)\n"
                "2. EFFECTIVE PATTERNS (things that work)\n"
                "3. OPEN QUESTIONS (things still unknown)\n\n"
                "Format each as a markdown section.",
                evo->attempt_count, eval.score,
                eval.reasoning ? eval.reasoning : "");

            AdamRunResult insight = adam_run(cfg, history, insight_prompt);
            if (insight.final_response) {
                /* Parse sections and write to separate files */
                write_file("evolution/insights/wrong_assumptions.md",
                           insight.final_response);  /* TODO: extract section */
                write_file("evolution/insights/effective_patterns.md",
                           insight.final_response);  /* TODO: extract section */
            }
            adam_result_free(&insight);
        }

        /* 7. Cleanup */
        adam_result_free(&result);
        adam_history_destroy(history);
        if (eval.reasoning) free(eval.reasoning);
        if (eval.next_suggestion) free(eval.next_suggestion);
    }

    arena_destroy(scratch);
}
```

### 7.4 Parallel Evolution (Multiple Agents)

Multiple workers run attempts concurrently, each with its own model/provider. They share the evolution state through a mutex-protected structure and the filesystem.

```c
typedef struct {
    int              worker_id;
    AdamConfig      *cfg;           /* each worker can use a different model */
    AdamEvolution   *shared_evo;    /* protected by mutex */
    pthread_mutex_t *evo_lock;
    AdamEvalResult (*evaluate)(const char *output);
    int            (*should_stop)(int attempt, int best_score);
} AdamParallelWorker;

static void *evolution_worker_thread(void *arg) {
    AdamParallelWorker *w = (AdamParallelWorker *)arg;
    arena_t *scratch = arena_create(128 * 1024);
    if (!scratch) return NULL;

    while (1) {
        /* 1. Snapshot shared state under lock */
        pthread_mutex_lock(w->evo_lock);
        if (w->should_stop(w->shared_evo->attempt_count,
                           w->shared_evo->current_score)) {
            pthread_mutex_unlock(w->evo_lock);
            break;
        }
        int attempt_num = ++(w->shared_evo->attempt_count);
        /* Copy strategy & task (they may be updated by another worker) */
        char *task_copy = strdup(w->shared_evo->task_md);
        char *strategy_copy = strdup(w->shared_evo->strategy_md);
        int best_score = w->shared_evo->current_score;
        pthread_mutex_unlock(w->evo_lock);

        /* 2. Run attempt independently (no lock held) */
        arena_reset(scratch);
        AdamEvolution local_evo = {
            .task_md = task_copy,
            .strategy_md = strategy_copy,
            .current_score = best_score,
            .attempt_count = attempt_num,
        };
        snprintf(local_evo.base_path, sizeof(local_evo.base_path),
                 "evolution/");

        char *prompt = build_evolution_prompt(scratch, &local_evo,
                                              attempt_num);
        AdamHistory *history = adam_history_create();
        AdamRunResult result = adam_run(w->cfg, history, prompt);
        AdamEvalResult eval = w->evaluate(
            result.final_response ? result.final_response : "");

        /* 3. Log attempt (worker-specific filename) */
        char path[256];
        snprintf(path, sizeof(path), "evolution/attempts/%03d_w%d.md",
                 attempt_num, w->worker_id);
        char log_buf[4096];
        snprintf(log_buf, sizeof(log_buf),
            "# Attempt %d (worker %d) — Score: %d/100\n\n%s\n",
            attempt_num, w->worker_id, eval.score,
            result.final_response ? result.final_response : "");
        write_file(path, log_buf);

        /* 4. Merge results back under lock */
        pthread_mutex_lock(w->evo_lock);
        if (eval.score > w->shared_evo->current_score) {
            w->shared_evo->current_score = eval.score;
            free(w->shared_evo->strategy_md);
            w->shared_evo->strategy_md = strdup(
                result.final_response ? result.final_response : "");
            write_file("evolution/STRATEGY.md",
                       w->shared_evo->strategy_md);
        }
        pthread_mutex_unlock(w->evo_lock);

        /* 5. Cleanup */
        adam_result_free(&result);
        adam_history_destroy(history);
        free(task_copy);
        free(strategy_copy);
        if (eval.reasoning) free(eval.reasoning);
        if (eval.next_suggestion) free(eval.next_suggestion);
    }

    arena_destroy(scratch);
    return NULL;
}

/* Launch N parallel evolution workers */
void adam_evolution_parallel(
    AdamConfig *configs[],          /* different models per worker */
    int worker_count,
    AdamEvolution *evo,
    AdamEvalResult (*evaluate)(const char *output),
    int (*should_stop)(int attempt, int best_score)
) {
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_t *threads = malloc(sizeof(pthread_t) * (size_t)worker_count);
    AdamParallelWorker *workers = malloc(
        sizeof(AdamParallelWorker) * (size_t)worker_count);

    for (int i = 0; i < worker_count; i++) {
        workers[i] = (AdamParallelWorker){
            .worker_id = i,
            .cfg = configs[i],
            .shared_evo = evo,
            .evo_lock = &lock,
            .evaluate = evaluate,
            .should_stop = should_stop,
        };
        pthread_create(&threads[i], NULL, evolution_worker_thread,
                       &workers[i]);
    }

    for (int i = 0; i < worker_count; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(workers);
    pthread_mutex_destroy(&lock);
}
```

### 7.5 Key Design Principles

1. **Arena per turn** — Each tool iteration resets the arena. All JSON building, HTTP response buffers, jsmn token arrays, and tool result strings vanish in one `arena_reset()` call. Only history messages (which must survive) are `strdup`'d into malloc. This eliminates an entire class of memory leaks.

2. **jsmn for zero-copy parsing** — jsmn produces token offsets into the raw JSON string. No intermediate allocations. When we need to extract a string, we use `arena_strdup()` or `arena_alloc()` — it goes into the same arena as everything else and is freed automatically.

3. **libcurl + mbedtls for full protocol support** — The library natively handles HTTPS to LLM APIs. The same libcurl instance powers the `web_fetch` built-in tool. Future tools can use SMTP (email notifications), SCP/SFTP (file transfer), FTP, and any other protocol libcurl supports — with zero additional dependencies.

4. **llama.cpp for offline inference** — Run any GGUF model locally. No API keys, no network, full privacy. The same `AdamProvider` interface works for both remote and local models — the agent loop doesn't know the difference.

5. **SQLite for everything persistent** — Sessions, memory, evolution state, and agent configuration all live in SQLite databases. Single-file, zero-config, ACID-compliant. sqlite-memory provides intelligent chunking and hybrid search. sqlite-vector provides fast k-NN similarity search with quantization.

6. **Thread-safe by isolation** — Each `AdamConfig` + `AdamHistory` + `arena_t` triplet is independent. Parallel evolution workers each get their own arena and history. Only the shared `AdamEvolution` state requires a mutex, and the lock is held only for brief snapshots and merges.

7. **Evolution files are the API** — The markdown files ARE the program's state. Any external tool can read/write them. The agent reads them as input and writes them as output. This makes the system inspectable, debuggable, and resumable after crashes. If the process is killed mid-evolution, just restart — it reads the files and continues from where it left off.

8. **Scoring is pluggable** — The `evaluate()` callback can be another LLM call (ask a different model to judge), a test suite (compile & run), a human (prompt for score), or a metric function (BLEU, accuracy). The evolution loop doesn't care.

9. **Memory sync between agents** — Since sqlite-memory uses a standard SQLite database, agents can share knowledge by ATTACHing each other's databases. One agent ingests documentation, another agent searches it. Evolution workers share insights through the same mechanism.

10. **Embeddable** — Link `libadam.a` + static deps. Call `adam_global_init()`, configure, run. No runtime dependencies beyond libc and pthreads.

### 7.6 Example: Full-Featured Agent

```c
/* examples/main.c — Agent with memory, sessions, local + remote models */
#include "adam.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Stream callback: print tokens as they arrive */
static void on_stream(void *ctx, const char *chunk, size_t len, int is_done) {
    (void)ctx;
    fwrite(chunk, 1, len, stdout);
    if (is_done) printf("\n");
    fflush(stdout);
}

/* Logger */
static void on_log(void *ctx, AdamLogLevel level, const char *msg, size_t len) {
    (void)ctx;
    const char *labels[] = {"ERROR","WARN","INFO","DEBUG","TRACE"};
    fprintf(stderr, "[%s] %.*s\n", labels[level], (int)len, msg);
}

int main(int argc, char **argv) {
    adam_global_init();

    /* --- Open persistent stores --- */
    adam_memory_t  *memory  = adam_memory_open("adam_memory.db");
    adam_session_t *sessions = adam_session_open("adam_sessions.db");

    /* --- Ingest knowledge (first run only) --- */
    adam_memory_add_file(memory, "docs/architecture.md", "project");
    adam_memory_add_directory(memory, "src/", "codebase");

    /* --- Configure agent --- */
    AdamConfig *cfg = adam_create();

    /* Provider: use local GGUF model if available, else Anthropic API */
    if (argc > 1 && strstr(argv[1], ".gguf")) {
        adam_set_provider(cfg, ADAM_PROVIDER_LOCAL, NULL, argv[1]);
        adam_set_gguf(cfg, argv[1], /*n_gpu_layers=*/-1, /*n_ctx=*/8192);
    } else {
        adam_set_provider(cfg, ADAM_PROVIDER_ANTHROPIC,
                          getenv("ANTHROPIC_API_KEY"),
                          "claude-sonnet-4-20250514");
    }

    /* System prompt from bootstrap files */
    adam_set_identity(cfg, "You are Adam, a self-evolving AI agent.");
    adam_set_instructions(cfg, "Be concise. Use tools when needed.");
    adam_add_bootstrap_file(cfg, "SOUL.md");
    adam_set_memory_enrichment(cfg, 1); /* auto-inject relevant memories */

    /* Attach memory and sessions */
    cfg->memory = memory;
    cfg->sessions = sessions;
    cfg->memory_context = "default";

    /* Streaming + logging */
    adam_set_stream_callback(cfg, on_stream, NULL);
    adam_set_logger(cfg, on_log, NULL, ADAM_LOG_INFO);

    /* Retry policy */
    adam_set_retry_policy(cfg, (AdamRetryPolicy){
        .max_retries = 3,
        .backoff_ms = {1000, 2000, 5000, 10000},
        .rate_limit_cooldown_ms = 30000,
    });

    /* Built-in tools */
    adam_add_tool(cfg, (AdamToolDef){
        .name = "web_fetch",
        .description = "Fetch a URL and return its content",
        .parameters_json = "{\"type\":\"object\",\"properties\":"
            "{\"url\":{\"type\":\"string\"}},\"required\":[\"url\"]}",
        .execute = adam_tool_web_fetch,
    });
    adam_add_tool(cfg, (AdamToolDef){
        .name = "memory_search",
        .description = "Search long-term memory for relevant knowledge",
        .parameters_json = "{\"type\":\"object\",\"properties\":"
            "{\"query\":{\"type\":\"string\"},\"limit\":{\"type\":\"integer\"}}"
            ",\"required\":[\"query\"]}",
        .execute = adam_tool_memory_search,
        .ctx = memory,
    });

    /* --- Interactive loop --- */
    const char *session_id = "interactive-001";
    AdamHistory *history = adam_history_create();

    /* Resume previous session if it exists */
    adam_session_load(sessions, session_id, history);

    char input[4096];
    while (printf("\n> "), fgets(input, sizeof(input), stdin)) {
        /* Strip newline */
        size_t len = strlen(input);
        if (len > 0 && input[len-1] == '\n') input[len-1] = '\0';
        if (strlen(input) == 0) continue;
        if (strcmp(input, "/quit") == 0) break;

        /* Run agent */
        AdamRunResult result = adam_run(cfg, history, input);

        if (result.status != ADAM_OK) {
            fprintf(stderr, "Error: %s\n", result.final_response);
        }

        /* Auto-save session after each turn */
        adam_session_save(sessions, session_id, history);

        /* Summarize if history is getting long */
        if (adam_history_estimate_tokens(history) > 100000) {
            adam_history_summarize(cfg, history, 50000);
        }

        adam_result_free(&result);
    }

    /* --- Cleanup --- */
    adam_history_destroy(history);
    adam_memory_close(memory);
    adam_session_close(sessions);
    adam_destroy(cfg);
    adam_global_cleanup();
    return 0;
}
```

### 7.7 Example: Parallel Evolution with Shared Memory

```c
/* examples/evolve.c — Multiple agents evolving in parallel,
 * sharing knowledge through sqlite-memory */
#include "adam.h"
#include <stdio.h>
#include <stdlib.h>

static AdamEvalResult evaluate_with_llm(const char *output) {
    /* Use a separate LLM call to judge the output */
    AdamConfig *judge = adam_create();
    adam_set_provider(judge, ADAM_PROVIDER_ANTHROPIC,
                      getenv("ANTHROPIC_API_KEY"), "claude-sonnet-4-20250514");

    AdamHistory *h = adam_history_create();
    char prompt[8192];
    snprintf(prompt, sizeof(prompt),
        "Rate this solution 0-100. Respond as JSON: "
        "{\"score\":N,\"reasoning\":\"...\",\"suggestion\":\"...\"}\n\n%s",
        output);

    adam_set_response_format(judge, "json");
    AdamRunResult r = adam_run(judge, h, prompt);

    /* Parse JSON result (simplified) */
    AdamEvalResult eval = {0};
    if (r.final_response) {
        /* In practice: use jsmn to parse score, reasoning, suggestion */
        eval.score = 50; /* placeholder */
        eval.reasoning = strdup(r.final_response);
        eval.next_suggestion = strdup("Try a different approach");
    }

    adam_result_free(&r);
    adam_history_destroy(h);
    adam_destroy(judge);
    return eval;
}

int main(void) {
    adam_global_init();

    /* Shared memory: all workers read/write the same knowledge base */
    adam_memory_t *shared_mem = adam_memory_open("evolution_memory.db");

    /* Worker 0: Claude (remote) */
    AdamConfig *claude = adam_create();
    adam_set_provider(claude, ADAM_PROVIDER_ANTHROPIC,
                      getenv("ANTHROPIC_API_KEY"), "claude-sonnet-4-20250514");
    claude->memory = shared_mem;

    /* Worker 1: local GGUF model */
    AdamConfig *local = adam_create();
    adam_set_provider(local, ADAM_PROVIDER_LOCAL, NULL, "qwen2.5-coder-32b");
    adam_set_gguf(local, "models/qwen2.5-coder-32b-q4.gguf", -1, 16384);
    local->memory = shared_mem;

    /* Worker 2: GPT (remote) */
    AdamConfig *gpt = adam_create();
    adam_set_provider(gpt, ADAM_PROVIDER_OPENAI,
                      getenv("OPENAI_API_KEY"), "gpt-4o");
    gpt->memory = shared_mem;

    /* Evolution state */
    AdamEvolution evo = {
        .task_md = read_file_malloc("evolution/TASK.md"),
        .strategy_md = read_file_malloc("evolution/STRATEGY.md"),
        .metrics_md = read_file_malloc("evolution/METRICS.md"),
    };
    snprintf(evo.base_path, sizeof(evo.base_path), "evolution/");

    /* Launch 3 parallel workers: Claude + local GGUF + GPT */
    AdamConfig *configs[] = { claude, local, gpt };
    adam_evolution_parallel(configs, 3, &evo,
                           evaluate_with_llm,
                           /*should_stop=*/NULL /* run indefinitely */);

    adam_memory_close(shared_mem);
    adam_destroy(claude);
    adam_destroy(local);
    adam_destroy(gpt);
    adam_global_cleanup();
    return 0;
}
```

---

## 8. Component Deep-Dive: Local Inference (llama.cpp)

### 8.1 Integration Architecture

```c
/* === adam_local.c === */
/* Wraps llama.cpp to present the same AdamLLMResponse interface
 * as the HTTP providers. The agent loop doesn't know the difference. */

#ifndef ADAM_NO_LOCAL

#include "llama.h"
#include "adam.h"

typedef struct {
    struct llama_model   *model;
    struct llama_context *ctx;
    struct llama_sampler *sampler;
    int                   n_ctx;
} AdamLocalCtx;

/* Initialize a local model (called once, reused across turns) */
AdamLocalCtx *adam_local_init(const AdamProvider *provider) {
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = provider->n_gpu_layers;

    struct llama_model *model = llama_model_load_from_file(
        provider->gguf_path, mparams);
    if (!model) return NULL;

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = provider->n_ctx > 0 ? provider->n_ctx : 8192;
    cparams.n_batch = provider->n_batch > 0 ? provider->n_batch : 512;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { llama_model_free(model); return NULL; }

    AdamLocalCtx *lctx = calloc(1, sizeof(AdamLocalCtx));
    lctx->model = model;
    lctx->ctx = ctx;
    lctx->n_ctx = cparams.n_ctx;

    /* Set up sampler chain */
    lctx->sampler = llama_sampler_chain_init(
        (struct llama_sampler_chain_params){0});
    llama_sampler_chain_add(lctx->sampler,
        llama_sampler_init_temp(provider->temperature));
    llama_sampler_chain_add(lctx->sampler,
        llama_sampler_init_dist(0));

    return lctx;
}

/* Generate a response locally. All output strings are arena-owned. */
AdamLLMResponse adam_local_generate(
    AdamLocalCtx *lctx,
    arena_t *arena,
    const AdamMessage *msgs, size_t msg_count,
    const AdamToolDef *tools, size_t tool_count,
    AdamStreamFn on_stream, void *stream_ctx
) {
    AdamLLMResponse resp = {0};

    /* 1. Build prompt string from messages (Chatml or llama format) */
    const char *prompt = adam_build_chatml_prompt(arena, msgs, msg_count,
                                                  tools, tool_count);

    /* 2. Tokenize (arena-allocated token buffer) */
    int n_prompt_tokens = -llama_tokenize(
        llama_model_from_context(lctx->ctx), prompt, strlen(prompt),
        NULL, 0, true, true);
    llama_token *tokens = arena_alloc(arena,
        (size_t)n_prompt_tokens * sizeof(llama_token));
    llama_tokenize(llama_model_from_context(lctx->ctx),
        prompt, strlen(prompt),
        tokens, n_prompt_tokens, true, true);
    resp.input_tokens = n_prompt_tokens;

    /* 3. Decode prompt */
    llama_batch batch = llama_batch_get_one(tokens, n_prompt_tokens);
    if (llama_decode(lctx->ctx, batch) != 0) {
        resp.error = ADAM_ERR_LOCAL;
        resp.error_msg = arena_strdup(arena, "Failed to decode prompt");
        return resp;
    }

    /* 4. Generate tokens one by one */
    size_t out_cap = 4096;
    char *output = arena_alloc(arena, out_cap);
    size_t out_len = 0;
    int max_tokens = 4096;
    int n_generated = 0;

    while (n_generated < max_tokens) {
        llama_token new_token = llama_sampler_sample(
            lctx->sampler, lctx->ctx, -1);

        if (llama_token_is_eog(
                llama_model_from_context(lctx->ctx), new_token))
            break;

        /* Detokenize */
        char piece[256];
        int n = llama_token_to_piece(
            llama_model_from_context(lctx->ctx),
            new_token, piece, sizeof(piece), 0, true);
        if (n < 0) break;

        /* Grow output buffer if needed */
        if (out_len + (size_t)n >= out_cap) {
            out_cap *= 2;
            char *new_buf = arena_alloc(arena, out_cap);
            memcpy(new_buf, output, out_len);
            output = new_buf;
        }
        memcpy(output + out_len, piece, (size_t)n);
        out_len += (size_t)n;
        n_generated++;

        /* Stream callback */
        if (on_stream)
            on_stream(stream_ctx, piece, (size_t)n, 0);

        /* Prepare next batch */
        batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(lctx->ctx, batch) != 0) break;
    }

    output[out_len] = '\0';
    if (on_stream) on_stream(stream_ctx, "", 0, 1); /* signal done */

    resp.content = output;
    resp.output_tokens = n_generated;

    /* 5. Parse tool calls from output (if model used tool-call format) */
    adam_parse_tool_calls_from_text(arena, output, out_len,
                                    &resp.tool_calls, &resp.tool_call_count);

    /* 6. Clear KV cache for next turn */
    llama_kv_self_clear(lctx->ctx);

    return resp;
}

void adam_local_free(AdamLocalCtx *lctx) {
    if (!lctx) return;
    llama_sampler_free(lctx->sampler);
    llama_free(lctx->ctx);
    llama_model_free(lctx->model);
    free(lctx);
}

#endif /* ADAM_NO_LOCAL */
```

### 8.2 Unified Provider Dispatch

Inside `adam.c`, the provider dispatch is transparent:

```c
static AdamLLMResponse dispatch_llm_call(
    AdamConfig *cfg, arena_t *arena,
    const AdamMessage *msgs, size_t msg_count
) {
#ifndef ADAM_NO_LOCAL
    if (cfg->provider.type == ADAM_PROVIDER_LOCAL) {
        /* Lazy-init the llama.cpp context on first call */
        if (!cfg->_local_ctx)
            cfg->_local_ctx = adam_local_init(&cfg->provider);
        return adam_local_generate(
            cfg->_local_ctx, arena, msgs, msg_count,
            cfg->tools.tools, cfg->tools.count,
            cfg->on_stream, cfg->stream_ctx);
    }
#endif
    /* Remote HTTP provider (Anthropic, OpenAI, OpenAI-compat) */
    if (cfg->http_fn) {
        /* WASM path: embedder-provided HTTP */
        return adam_llm_call_via_callback(
            arena, cfg, msgs, msg_count,
            cfg->tools.tools, cfg->tools.count);
    }
#ifndef ADAM_NO_CURL
    return adam_llm_call(
        arena, &cfg->provider, msgs, msg_count,
        cfg->tools.tools, cfg->tools.count);
#else
    return (AdamLLMResponse){
        .error = ADAM_ERR_PROVIDER,
        .error_msg = arena_strdup(arena, "No HTTP provider available")
    };
#endif
}
```

---

## 9. Component Deep-Dive: Memory System

### 9.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ADAM MEMORY SYSTEM                           │
│                                                                  │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐ │
│  │ adam_memory.c │  │ sqlite-memory │  │ sqlite-vector        │ │
│  │ (Adam API)   │──│ (chunking +   │──│ (k-NN similarity     │ │
│  │              │  │  embedding +  │  │  search with          │ │
│  │ add/search/  │  │  hybrid srch) │  │  quantization)       │ │
│  │ sync/clear   │  │              │  │                      │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘ │
│         │                 │                      │              │
│         └─────────────────┼──────────────────────┘              │
│                           ▼                                     │
│                    ┌──────────────┐                              │
│                    │   SQLite DB  │  ← single file on disk      │
│                    │              │                              │
│                    │  dbmem_content (text + metadata)           │
│                    │  FTS5 index  (keyword search)              │
│                    │  vector index (similarity search)          │
│                    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Memory Enrichment Flow

Before each agent turn, Adam automatically enriches the system prompt with relevant memories:

```c
/* === adam_context.c === */

/* Build the system prompt with memory enrichment.
 * Called at the start of each agent turn. */
const char *adam_build_system_prompt(
    arena_t *arena,
    const AdamSystemPrompt *sp,
    adam_memory_t *memory,
    const char *user_message       /* used as search query */
) {
    size_t est = 16384;
    char *buf = arena_alloc(arena, est);
    size_t pos = 0;

    /* 1. Identity */
    if (sp->identity)
        pos += snprintf(buf + pos, est - pos, "%s\n\n", sp->identity);

    /* 2. Instructions */
    if (sp->instructions)
        pos += snprintf(buf + pos, est - pos, "%s\n\n", sp->instructions);

    /* 3. Bootstrap files (loaded from disk, cached) */
    for (size_t i = 0; i < sp->bootstrap_count; i++) {
        char *content = read_file_arena(arena, sp->bootstrap_files[i]);
        if (content)
            pos += snprintf(buf + pos, est - pos,
                "## %s\n%s\n\n", sp->bootstrap_files[i], content);
    }

    /* 4. Memory enrichment: search for relevant knowledge */
    if (sp->inject_memory && memory && user_message) {
        AdamMemoryResult *results = NULL;
        size_t count = 0;
        adam_memory_search(memory, arena, user_message, 5,
                          &results, &count);

        if (count > 0) {
            pos += snprintf(buf + pos, est - pos,
                "## Relevant Knowledge\n");
            for (size_t i = 0; i < count; i++) {
                pos += snprintf(buf + pos, est - pos,
                    "- [%s] %s\n", results[i].source, results[i].content);
            }
            pos += snprintf(buf + pos, est - pos, "\n");
        }
    }

    /* 5. Dynamic context */
    if (sp->inject_datetime) {
        /* Inject current date/time */
        time_t now = time(NULL);
        struct tm *tm = localtime(&now);
        pos += snprintf(buf + pos, est - pos,
            "Current date: %04d-%02d-%02d %02d:%02d\n\n",
            tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
            tm->tm_hour, tm->tm_min);
    }

    if (sp->session_info)
        pos += snprintf(buf + pos, est - pos, "%s\n", sp->session_info);

    buf[pos] = '\0';
    return buf;
}
```

### 9.3 Memory Sync Between Agents

Since sqlite-memory uses a standard SQLite database, agents can share knowledge using SQLite's `ATTACH DATABASE` mechanism:

```c
/* === adam_memory.c (sync section) === */

/* Attach another agent's memory database */
int adam_memory_attach(adam_memory_t *mem, const char *remote_db_path,
                       const char *alias) {
    char sql[512];
    snprintf(sql, sizeof(sql),
        "ATTACH DATABASE '%s' AS %s", remote_db_path, alias);
    return sqlite3_exec(mem->db, sql, NULL, NULL, NULL);
}

/* Sync: copy memories from a remote agent's database into ours.
 * Only copies memories in the specified context (namespace).
 * Deduplicates by content hash. */
int adam_memory_sync(adam_memory_t *mem, const char *alias,
                     const char *context) {
    char sql[1024];
    snprintf(sql, sizeof(sql),
        "INSERT OR IGNORE INTO dbmem_content "
        "(hash, content, context, source, created_at) "
        "SELECT hash, content, context, source, created_at "
        "FROM %s.dbmem_content WHERE context = ?", alias);

    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(mem->db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) return rc;
    sqlite3_bind_text(stmt, 1, context, -1, SQLITE_STATIC);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return (rc == SQLITE_DONE) ? SQLITE_OK : rc;
}
```

This enables several patterns:

| Pattern | How |
|---------|-----|
| **Shared knowledge base** | Multiple agents open the same `adam_memory.db` |
| **Knowledge federation** | Each agent has its own DB; attach + sync specific contexts |
| **Evolution knowledge** | Workers share insights through `adam_memory_sync()` |
| **Export/import** | Copy a `.db` file to another machine — it's just SQLite |

---

## 10. Component Deep-Dive: Session Persistence

```c
/* === adam_session.c === */

/* Sessions are stored in a SQLite database with two tables:
 *   sessions(id, created_at, updated_at)
 *   messages(session_id, idx, role, content, tool_call_id,
 *            tool_calls_json, created_at)
 *
 * The tool_calls_json column stores the serialized AdamToolCallEntry
 * array for assistant messages that include tool calls. */

adam_session_t *adam_session_open(const char *db_path) {
    adam_session_t *s = calloc(1, sizeof(adam_session_t));
    int rc = sqlite3_open(db_path, &s->db);
    if (rc != SQLITE_OK) { free(s); return NULL; }

    /* Create tables if they don't exist */
    sqlite3_exec(s->db,
        "CREATE TABLE IF NOT EXISTS sessions("
        "  id TEXT PRIMARY KEY,"
        "  created_at INTEGER DEFAULT (unixepoch()),"
        "  updated_at INTEGER DEFAULT (unixepoch()));"
        "CREATE TABLE IF NOT EXISTS messages("
        "  session_id TEXT NOT NULL,"
        "  idx INTEGER NOT NULL,"
        "  role INTEGER NOT NULL,"
        "  content TEXT,"
        "  tool_call_id TEXT,"
        "  tool_calls_json TEXT,"
        "  created_at INTEGER DEFAULT (unixepoch()),"
        "  PRIMARY KEY (session_id, idx),"
        "  FOREIGN KEY (session_id) REFERENCES sessions(id));",
        NULL, NULL, NULL);

    return s;
}

int adam_session_save(adam_session_t *sess, const char *session_id,
                      const AdamHistory *h) {
    /* Upsert session */
    sqlite3_exec(sess->db,
        "INSERT OR REPLACE INTO sessions(id, updated_at) "
        "VALUES(?, unixepoch())", NULL, NULL, NULL);

    /* Delete existing messages for this session, re-insert all */
    sqlite3_stmt *del;
    sqlite3_prepare_v2(sess->db,
        "DELETE FROM messages WHERE session_id = ?", -1, &del, NULL);
    sqlite3_bind_text(del, 1, session_id, -1, SQLITE_STATIC);
    sqlite3_step(del);
    sqlite3_finalize(del);

    /* Insert all current messages */
    sqlite3_stmt *ins;
    sqlite3_prepare_v2(sess->db,
        "INSERT INTO messages(session_id, idx, role, content, "
        "tool_call_id, tool_calls_json) VALUES(?,?,?,?,?,?)",
        -1, &ins, NULL);

    for (size_t i = 0; i < h->count; i++) {
        const AdamMessage *m = &h->items[i];
        sqlite3_bind_text(ins, 1, session_id, -1, SQLITE_STATIC);
        sqlite3_bind_int(ins, 2, (int)i);
        sqlite3_bind_int(ins, 3, m->role);
        sqlite3_bind_text(ins, 4, m->content, -1, SQLITE_STATIC);
        sqlite3_bind_text(ins, 5, m->tool_call_id, -1, SQLITE_STATIC);
        /* Serialize tool_calls to JSON if present */
        if (m->tool_call_count > 0) {
            char *tc_json = serialize_tool_calls(m->tool_calls,
                                                  m->tool_call_count);
            sqlite3_bind_text(ins, 6, tc_json, -1, SQLITE_TRANSIENT);
            free(tc_json);
        } else {
            sqlite3_bind_null(ins, 6);
        }
        sqlite3_step(ins);
        sqlite3_reset(ins);
    }
    sqlite3_finalize(ins);
    return SQLITE_OK;
}

int adam_session_load(adam_session_t *sess, const char *session_id,
                      AdamHistory *h) {
    adam_history_clear(h);

    sqlite3_stmt *sel;
    sqlite3_prepare_v2(sess->db,
        "SELECT role, content, tool_call_id, tool_calls_json "
        "FROM messages WHERE session_id = ? ORDER BY idx",
        -1, &sel, NULL);
    sqlite3_bind_text(sel, 1, session_id, -1, SQLITE_STATIC);

    while (sqlite3_step(sel) == SQLITE_ROW) {
        AdamRole role = (AdamRole)sqlite3_column_int(sel, 0);
        const char *content = (const char *)sqlite3_column_text(sel, 1);
        const char *tc_id = (const char *)sqlite3_column_text(sel, 2);
        const char *tc_json = (const char *)sqlite3_column_text(sel, 3);

        if (tc_json && role == ADAM_ROLE_ASSISTANT) {
            /* Deserialize tool calls and use history_append_assistant_with_tools */
            AdamToolCall *calls = NULL;
            size_t count = 0;
            deserialize_tool_calls(tc_json, &calls, &count);
            /* ... append with tool calls ... */
        } else {
            history_append(h, role, content, tc_id);
        }
    }
    sqlite3_finalize(sel);
    return SQLITE_OK;
}
```

---

## 11. Component Deep-Dive: Streaming

```c
/* === adam_stream.c === */

/* SSE (Server-Sent Events) parser for Anthropic/OpenAI streaming.
 * Used with CURLOPT_WRITEFUNCTION to process chunks incrementally. */

typedef struct {
    arena_t      *arena;
    AdamStreamFn  on_chunk;
    void         *stream_ctx;
    char         *line_buf;       /* partial line accumulator */
    size_t        line_len;
    size_t        line_cap;
    /* Accumulated full response (for final parsing) */
    char         *full_response;
    size_t        full_len;
    size_t        full_cap;
} SSEParser;

static size_t sse_write_cb(char *data, size_t size, size_t nmemb,
                            void *userp) {
    SSEParser *parser = (SSEParser *)userp;
    size_t bytes = size * nmemb;

    for (size_t i = 0; i < bytes; i++) {
        char c = data[i];
        if (c == '\n') {
            /* Process complete line */
            parser->line_buf[parser->line_len] = '\0';

            if (strncmp(parser->line_buf, "data: ", 6) == 0) {
                const char *payload = parser->line_buf + 6;
                if (strcmp(payload, "[DONE]") != 0) {
                    /* Parse JSON chunk, extract delta text */
                    const char *delta = sse_extract_delta(
                        parser->arena, payload, strlen(payload));
                    if (delta && parser->on_chunk) {
                        parser->on_chunk(parser->stream_ctx,
                            delta, strlen(delta), 0);
                    }
                    /* Accumulate for full response */
                    if (delta) {
                        size_t dlen = strlen(delta);
                        sse_append_full(parser, delta, dlen);
                    }
                }
            }
            parser->line_len = 0;
        } else {
            /* Accumulate line */
            if (parser->line_len >= parser->line_cap - 1) {
                parser->line_cap *= 2;
                parser->line_buf = realloc(parser->line_buf,
                                           parser->line_cap);
            }
            parser->line_buf[parser->line_len++] = c;
        }
    }
    return bytes;
}
```

---

## 12. Component Deep-Dive: Retry & Backoff

```c
/* === Inside adam.c — retry wrapper around dispatch_llm_call === */

static AdamLLMResponse dispatch_with_retry(
    AdamConfig *cfg, arena_t *arena,
    const AdamMessage *msgs, size_t msg_count
) {
    AdamRetryPolicy *policy = &cfg->retry;
    int max_retries = policy->max_retries > 0 ? policy->max_retries : 3;
    int attempt = 0;

    while (1) {
        AdamLLMResponse resp = dispatch_llm_call(cfg, arena, msgs, msg_count);

        /* Success or non-retriable error */
        if (resp.error == 0) return resp;
        if (resp.error == 3 /* auth */) return resp; /* don't retry auth */

        attempt++;
        if (attempt > max_retries) return resp;

        /* Log the retry */
        if (cfg->log_fn) {
            char msg[256];
            int n = snprintf(msg, sizeof(msg),
                "LLM call failed (error=%d: %s), retry %d/%d",
                resp.error, resp.error_msg ? resp.error_msg : "unknown",
                attempt, max_retries);
            cfg->log_fn(cfg->log_ctx, ADAM_LOG_WARN, msg, (size_t)n);
        }

        /* Context overflow: compress history and retry */
        if (resp.error == 2 /* ctx_overflow */) {
            /* Handled by caller (history_compress + arena_reset) */
            return resp;
        }

        /* Rate limit: longer cooldown */
        int delay_ms;
        if (resp.error == 1 /* rate_limit */) {
            delay_ms = policy->rate_limit_cooldown_ms > 0
                       ? policy->rate_limit_cooldown_ms : 30000;
        } else {
            delay_ms = (attempt <= 4)
                       ? policy->backoff_ms[attempt - 1] : 10000;
        }

        /* Sleep (milliseconds) */
        struct timespec ts = {
            .tv_sec = delay_ms / 1000,
            .tv_nsec = (delay_ms % 1000) * 1000000L
        };
        nanosleep(&ts, NULL);

        /* Reset arena for retry (discard previous response data) */
        arena_reset(arena);
    }
}
```

---

## 13. Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              ADAM AGENT                                 │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        EMBEDDER / APPLICATION                     │  │
│  │  CLI, Server, GUI, WASM app, Microcontroller, Another agent...   │  │
│  └────────────────────────────┬──────────────────────────────────────┘  │
│                               │ adam_run() / adam_evolution_parallel()  │
│  ┌────────────────────────────▼──────────────────────────────────────┐  │
│  │                          AGENT LOOP (adam.c)                       │  │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌─────────────────┐ │  │
│  │  │ Context  │  │ Tool      │  │ Retry &  │  │ History         │ │  │
│  │  │ Builder  │  │ Dispatch  │  │ Backoff  │  │ Compression &   │ │  │
│  │  │ (prompt  │  │ (execute  │  │ (policy) │  │ Summarization   │ │  │
│  │  │ +memory) │  │ +results) │  │          │  │                 │ │  │
│  │  └────┬─────┘  └─────┬────┘  └────┬─────┘  └────────┬────────┘ │  │
│  └───────┼──────────────┼────────────┼─────────────────┼──────────┘  │
│          │              │            │                  │              │
│  ┌───────▼──────────────▼────────────▼──────────────────▼──────────┐  │
│  │                      PROVIDER DISPATCH                          │  │
│  │                                                                  │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │  │
│  │  │ adam_http.c   │  │ adam_local.c  │  │ http_fn callback     │ │  │
│  │  │ libcurl +    │  │ llama.cpp    │  │ (WASM/custom)        │ │  │
│  │  │ mbedtls      │  │ GGUF models  │  │                      │ │  │
│  │  │              │  │              │  │                      │ │  │
│  │  │ Anthropic    │  │ Local infer  │  │ Embedder-provided    │ │  │
│  │  │ OpenAI       │  │ Streaming    │  │ HTTP function        │ │  │
│  │  │ OAI-compat   │  │ Token-by-tok │  │                      │ │  │
│  │  └──────────────┘  └──────────────┘  └────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      PERSISTENCE LAYER                           │   │
│  │                                                                   │   │
│  │  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────┐  │   │
│  │  │ Sessions     │  │ Memory            │  │ Evolution State   │  │   │
│  │  │ (SQLite)     │  │ (sqlite-memory    │  │ (markdown files   │  │   │
│  │  │              │  │  + sqlite-vector) │  │  + SQLite)        │  │   │
│  │  │ history rows │  │                   │  │                   │  │   │
│  │  │ tool calls   │  │ chunks + embed    │  │ TASK.md           │  │   │
│  │  │ summaries    │  │ FTS5 + vector idx │  │ STRATEGY.md       │  │   │
│  │  │              │  │ hybrid search     │  │ attempts/*.md     │  │   │
│  │  │              │  │ multi-agent sync  │  │ insights/*.md     │  │   │
│  │  └──────────────┘  └──────────────────┘  └───────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                        FOUNDATION                                │   │
│  │                                                                   │   │
│  │  ┌─────────┐  ┌────────┐  ┌──────────┐  ┌───────┐  ┌─────────┐│   │
│  │  │arena.h/c│  │ jsmn.h │  │ SQLite   │  │libcurl│  │llama.cpp││   │
│  │  │         │  │        │  │          │  │+mbed  │  │         ││   │
│  │  │ ~200 LOC│  │~450 LOC│  │amalgam.  │  │TLS    │  │GGUF     ││   │
│  │  └─────────┘  └────────┘  └──────────┘  └───────┘  └─────────┘│   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 14. Component Deep-Dive: Tool Call Format for Local Models

Local GGUF models (via llama.cpp) don't natively output OpenAI-style `tool_use` JSON. Adam needs a prompt template that teaches the model to emit tool calls in a parseable format, and a parser to extract them.

### 14.1 ChatML Tool Prompt Template

```c
/* === adam_local.c (prompt building) === */

/* Build a ChatML-style prompt that includes tool definitions
 * and teaches the model to emit tool calls in a parseable format. */
const char *adam_build_chatml_prompt(
    arena_t *arena,
    const AdamMessage *msgs, size_t msg_count,
    const AdamToolDef *tools, size_t tool_count
) {
    size_t est = 8192;
    for (size_t i = 0; i < msg_count; i++) est += msgs[i].content_len + 128;
    char *buf = arena_alloc(arena, est);
    size_t pos = 0;

    /* System message with tool definitions */
    if (msg_count > 0 && msgs[0].role == ADAM_ROLE_SYSTEM) {
        pos += snprintf(buf + pos, est - pos,
            "<|im_start|>system\n%s\n", msgs[0].content);

        /* Inject tool definitions */
        if (tool_count > 0) {
            pos += snprintf(buf + pos, est - pos,
                "\n## Available Tools\n"
                "When you need to use a tool, output EXACTLY this format:\n"
                "<tool_call>\n"
                "{\"name\": \"tool_name\", \"arguments\": {\"arg\": \"value\"}}\n"
                "</tool_call>\n\n"
                "You may call multiple tools. Wait for results before continuing.\n\n"
                "Tools:\n");
            for (size_t i = 0; i < tool_count; i++) {
                pos += snprintf(buf + pos, est - pos,
                    "- **%s**: %s\n  Parameters: %s\n",
                    tools[i].name, tools[i].description,
                    tools[i].parameters_json);
            }
        }
        pos += snprintf(buf + pos, est - pos, "<|im_end|>\n");
    }

    /* Conversation messages */
    for (size_t i = (msgs[0].role == ADAM_ROLE_SYSTEM ? 1 : 0);
         i < msg_count; i++) {
        const char *role = (msgs[i].role == ADAM_ROLE_USER) ? "user"
                         : (msgs[i].role == ADAM_ROLE_ASSISTANT) ? "assistant"
                         : "tool";
        pos += snprintf(buf + pos, est - pos,
            "<|im_start|>%s\n%s<|im_end|>\n", role, msgs[i].content);
    }

    /* Prompt for assistant response */
    pos += snprintf(buf + pos, est - pos, "<|im_start|>assistant\n");
    buf[pos] = '\0';
    return buf;
}
```

### 14.2 Tool Call Parser (XML tags from text)

```c
/* Parse <tool_call>...</tool_call> blocks from free-form text output.
 * Uses simple string scanning — no regex dependency. */
void adam_parse_tool_calls_from_text(
    arena_t *arena,
    const char *text, size_t text_len,
    AdamToolCall **out_calls, size_t *out_count
) {
    /* Count occurrences first */
    size_t count = 0;
    const char *p = text;
    while ((p = strstr(p, "<tool_call>")) != NULL) { count++; p++; }

    if (count == 0) { *out_calls = NULL; *out_count = 0; return; }

    AdamToolCall *calls = arena_alloc(arena, count * sizeof(AdamToolCall));
    size_t idx = 0;
    p = text;

    while ((p = strstr(p, "<tool_call>")) != NULL) {
        p += 11; /* skip "<tool_call>" */
        const char *end = strstr(p, "</tool_call>");
        if (!end) break;

        size_t json_len = (size_t)(end - p);
        /* Trim whitespace */
        while (json_len > 0 && (p[0] == '\n' || p[0] == ' ')) { p++; json_len--; }
        while (json_len > 0 && (p[json_len-1] == '\n' || p[json_len-1] == ' '))
            json_len--;

        /* Parse the JSON inside with jsmn */
        jsmntok_t tokens[64];
        jsmn_parser parser;
        jsmn_init(&parser);
        int ntok = jsmn_parse(&parser, p, json_len, tokens, 64);
        if (ntok < 1) { p = end + 12; continue; }

        AdamToolCall *tc = &calls[idx];
        tc->id = NULL; /* local models don't generate IDs — Adam assigns them */
        tc->name = NULL;
        tc->arguments_json = NULL;

        /* Walk tokens to find "name" and "arguments" */
        for (int t = 1; t < ntok; t++) {
            if (json_tok_eq(p, &tokens[t], "name") && t + 1 < ntok) {
                tc->name = json_tok_str(arena, p, &tokens[t + 1]);
                t++;
            } else if (json_tok_eq(p, &tokens[t], "arguments") && t + 1 < ntok) {
                /* Extract raw JSON substring for arguments */
                jsmntok_t *arg_tok = &tokens[t + 1];
                size_t alen = (size_t)(arg_tok->end - arg_tok->start);
                char *args = arena_alloc(arena, alen + 1);
                memcpy(args, p + arg_tok->start, alen);
                args[alen] = '\0';
                tc->arguments_json = args;
                t++;
            }
        }

        /* Generate a unique tool call ID */
        char id_buf[32];
        snprintf(id_buf, sizeof(id_buf), "call_%04zu", idx);
        tc->id = arena_strdup(arena, id_buf);

        if (tc->name) idx++;
        p = end + 12;
    }

    *out_calls = calls;
    *out_count = idx;
}
```

---

## 15. Component Deep-Dive: Cancellation

```c
/* === Inside adam.c — abort check in the main loop === */

/* The abort_flag is a volatile int in AdamConfig.
 * Set it to 1 from any thread (signal handler, UI thread, etc.)
 * to gracefully stop the current adam_run(). */

/* In the tool iteration loop, check before each LLM call: */
while (iteration < max_iter) {
    iteration++;

    /* Check for abort */
    if (cfg->abort_flag) {
        result.status = ADAM_OK;
        result.final_response = strdup("(aborted by user)");
        break;
    }

    arena_reset(turn_arena);
    /* ... rest of iteration ... */
}

/* Usage from another thread or signal handler: */
void adam_abort(AdamConfig *cfg) { cfg->abort_flag = 1; }
void adam_reset_abort(AdamConfig *cfg) { cfg->abort_flag = 0; }

/* Example: Ctrl+C handler */
static AdamConfig *g_cfg = NULL;
void sigint_handler(int sig) {
    (void)sig;
    if (g_cfg) adam_abort(g_cfg);
}
/* In main(): signal(SIGINT, sigint_handler); */
```

---

## 16. Component Deep-Dive: Cost Tracking & Model Registry

```c
/* === adam_models.c === */

/* Built-in model registry with context windows and pricing.
 * Users can extend with adam_register_model(). */

static AdamModelInfo g_models[] = {
    /* Anthropic */
    {"claude-opus-4",      200000,  15.0,  75.0},
    {"claude-sonnet-4",    200000,   3.0,  15.0},
    {"claude-haiku-4",     200000,   0.8,   4.0},
    /* OpenAI */
    {"gpt-4o",             128000,   2.5,  10.0},
    {"gpt-4o-mini",        128000,   0.15,  0.60},
    {"o3",                 200000,  10.0,  40.0},
    {"o4-mini",            200000,   1.1,   4.4},
    /* Local (zero cost) */
    {"local",                   0,   0.0,   0.0},
    {NULL, 0, 0, 0} /* sentinel */
};

/* Lookup by longest prefix match */
const AdamModelInfo *adam_lookup_model(const char *model) {
    const AdamModelInfo *best = NULL;
    size_t best_len = 0;
    for (int i = 0; g_models[i].model_prefix; i++) {
        size_t plen = strlen(g_models[i].model_prefix);
        if (strncmp(model, g_models[i].model_prefix, plen) == 0
            && plen > best_len) {
            best = &g_models[i];
            best_len = plen;
        }
    }
    return best;
}

int adam_get_context_window(const char *model) {
    const AdamModelInfo *info = adam_lookup_model(model);
    return info ? info->context_window : 0;
}

float adam_estimate_cost(const char *model, int input_tok, int output_tok) {
    const AdamModelInfo *info = adam_lookup_model(model);
    if (!info) return 0.0f;
    return (input_tok / 1e6f) * info->input_cost_per_mtok
         + (output_tok / 1e6f) * info->output_cost_per_mtok;
}

/* Proactive context budget: before calling the LLM, check if
 * history + system prompt + tools would exceed the model's window. */
static int should_summarize_proactively(AdamConfig *cfg, AdamHistory *h) {
    int ctx_window = adam_get_context_window(cfg->provider.model);
    if (ctx_window == 0) return 0; /* unknown model — skip */
    size_t est_tokens = adam_history_estimate_tokens(h);
    /* Trigger summarization at 75% of context window */
    return (est_tokens > (size_t)(ctx_window * 3 / 4));
}
```

---

## 17. Component Deep-Dive: Rate Limiter

```c
/* === Inside adam.c — proactive rate limiting === */

#include <time.h>

static void rate_limit_wait(AdamConfig *cfg) {
    if (cfg->rate_limit.requests_per_minute == 0
        && cfg->rate_limit.tokens_per_minute == 0)
        return; /* no limit configured */

    long now = (long)time(NULL);

    /* Reset window every 60 seconds */
    if (now - cfg->_rate_window_start >= 60) {
        cfg->_rate_window_start = now;
        cfg->_rate_request_count = 0;
        cfg->_rate_token_count = 0;
    }

    /* Check request limit */
    if (cfg->rate_limit.requests_per_minute > 0
        && cfg->_rate_request_count >= cfg->rate_limit.requests_per_minute) {
        long sleep_sec = 60 - (now - cfg->_rate_window_start);
        if (sleep_sec > 0) {
            if (cfg->log_fn) {
                char msg[128];
                int n = snprintf(msg, sizeof(msg),
                    "Rate limit: %d/%d RPM, sleeping %lds",
                    cfg->_rate_request_count,
                    cfg->rate_limit.requests_per_minute, sleep_sec);
                cfg->log_fn(cfg->log_ctx, ADAM_LOG_INFO, msg, (size_t)n);
            }
            struct timespec ts = { .tv_sec = sleep_sec };
            nanosleep(&ts, NULL);
        }
        cfg->_rate_window_start = (long)time(NULL);
        cfg->_rate_request_count = 0;
        cfg->_rate_token_count = 0;
    }

    /* Check token limit */
    if (cfg->rate_limit.tokens_per_minute > 0
        && cfg->_rate_token_count >= cfg->rate_limit.tokens_per_minute) {
        long sleep_sec = 60 - (now - cfg->_rate_window_start);
        if (sleep_sec > 0) {
            struct timespec ts = { .tv_sec = sleep_sec };
            nanosleep(&ts, NULL);
        }
        cfg->_rate_window_start = (long)time(NULL);
        cfg->_rate_request_count = 0;
        cfg->_rate_token_count = 0;
    }

    cfg->_rate_request_count++;
}

/* Called after each LLM response to update token counters */
static void rate_limit_record_tokens(AdamConfig *cfg, int tokens) {
    cfg->_rate_token_count += tokens;
}
```

---

## 18. Component Deep-Dive: Multimodal Messages

```c
/* === adam_multimodal.c === */

/* Attach binary media to a message.
 * The data is copied into malloc'd storage (survives across turns). */
int adam_message_attach(AdamMessage *msg, AdamMediaType type,
                        const unsigned char *data, size_t len,
                        const char *filename) {
    size_t new_count = msg->attachment_count + 1;
    msg->attachments = realloc(msg->attachments,
                               new_count * sizeof(AdamAttachment));
    AdamAttachment *a = &msg->attachments[msg->attachment_count];
    a->type = type;
    a->data = malloc(len);
    if (!a->data) return -1;
    memcpy(a->data, data, len);
    a->data_len = len;
    a->filename = filename ? strdup(filename) : NULL;
    msg->attachment_count = new_count;
    return 0;
}

/* Convenience: load from file, auto-detect type by extension */
int adam_message_attach_file(AdamMessage *msg, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *data = malloc((size_t)len);
    if (!data) { fclose(f); return -1; }
    fread(data, 1, (size_t)len, f);
    fclose(f);

    /* Detect type by extension */
    AdamMediaType type = ADAM_MEDIA_IMAGE_PNG; /* default */
    const char *ext = strrchr(path, '.');
    if (ext) {
        if (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0)
            type = ADAM_MEDIA_IMAGE_JPEG;
        else if (strcmp(ext, ".gif") == 0)  type = ADAM_MEDIA_IMAGE_GIF;
        else if (strcmp(ext, ".webp") == 0) type = ADAM_MEDIA_IMAGE_WEBP;
        else if (strcmp(ext, ".wav") == 0)  type = ADAM_MEDIA_AUDIO_WAV;
        else if (strcmp(ext, ".mp3") == 0)  type = ADAM_MEDIA_AUDIO_MP3;
        else if (strcmp(ext, ".pdf") == 0)  type = ADAM_MEDIA_PDF;
    }

    int rc = adam_message_attach(msg, type, data, (size_t)len,
                                 strrchr(path, '/') ? strrchr(path, '/') + 1
                                                    : path);
    free(data);
    return rc;
}

/* In adam_json.c — serialize attachments for Anthropic API:
 *
 * { "role": "user", "content": [
 *     { "type": "image", "source": {
 *         "type": "base64",
 *         "media_type": "image/png",
 *         "data": "<base64>"
 *     }},
 *     { "type": "text", "text": "What's in this image?" }
 * ]}
 *
 * For llama.cpp multimodal models (e.g. LLaVA), images are passed
 * through llama_image_embed_make_with_bytes(). */
```

---

## 19. What Makes Adam Unique — Pros & Cons

### 19.1 What Makes Adam Unique

**No other agent framework offers all of these simultaneously:**

| Unique Combination | Closest Alternative | Adam's Advantage |
|---|---|---|
| **C embeddable library** | All others are Go/Python/TS/Zig | Link into any C/C++/Rust/Swift project, or load as shared lib in any language via FFI |
| **Arena allocator for agent loop** | None use arenas | Zero memory leaks by construction; predictable allocation; cache-friendly |
| **Local + remote in same binary** | Most are API-only; llama.cpp is inference-only | Same agent loop, same tools, same memory — just swap the provider |
| **SQLite as the universal store** | Redis, Postgres, custom formats | Single file, zero config, ACID, works everywhere (including WASM via MEMFS) |
| **sqlite-memory + sqlite-vector** | Custom vector DBs, Qdrant, Pinecone | Hybrid BM25 + vector search in a single SQLite file. No external services |
| **Multi-agent memory sync via ATTACH** | Custom protocols, message passing | Zero-overhead: it's just `ATTACH DATABASE`. Agents share knowledge through SQL |
| **Self-evolving loop as first-class** | Bolted-on, prompt-chaining hacks | Evolution is a core architectural concept with dedicated state, scoring, insights |
| **WASM-portable** | None of the studied agents | Same C code compiles to native + WASM. Run agents in the browser |
| **Protocol-rich via libcurl** | HTTP-only or custom | HTTP, HTTPS, SMTP, SCP, SFTP, FTP, WebSocket — all through one dep |

**The one-sentence pitch**: Adam is the SQLite of agent frameworks — small, embeddable, zero-config, and it just works everywhere.

### 19.2 Pros

1. **Smallest footprint** — The core library (without deps) is ~3,000 lines of C. The full binary (with libcurl, mbedtls, SQLite, llama.cpp) is still smaller than a Node.js `node_modules` directory. Runs on devices where Python/Node/Go cannot.

2. **Zero memory leaks by design** — The arena allocator makes it structurally impossible to leak memory in the hot path. `arena_reset()` at iteration boundaries guarantees all scratch memory is reclaimed. This is not a convention — it's enforced by the architecture.

3. **Universal embedding** — C is the universal FFI. Adam can be called from Python (`ctypes`/`cffi`), Ruby (`ffi`), Swift (direct bridging), Rust (`extern "C"`), Java (JNI), C# (P/Invoke), Lua, and every other language. No other agent framework can claim this.

4. **Offline-capable** — With llama.cpp and SQLite, Adam can run entirely offline. No API keys, no network, no cloud dependency. The same code works in an airplane, a submarine, or an air-gapped server.

5. **Single-file persistence** — Everything is in SQLite databases (sessions, memory, vectors) or markdown files (evolution). Copy a `.db` file and you've migrated an agent's entire knowledge. No migrations, no schema tools, no ORM.

6. **WASM portability** — The same codebase compiles to WASM for browser embedding. SQLite runs in WASM (via MEMFS/OPFS). Only the HTTP layer needs a JavaScript bridge.

7. **Multi-agent by default** — Memory sync via `ATTACH DATABASE`, parallel evolution with pthreads, shared knowledge bases. Not an add-on — it's how the system was designed.

8. **Protocol diversity** — libcurl gives Adam HTTP, HTTPS, SMTP, SCP, SFTP, FTP, LDAP, and more. An agent can fetch web pages, send emails, transfer files, and query LDAP directories — all through the same dependency.

9. **Predictable performance** — No garbage collector, no JIT warmup, no event loop stalls. Memory usage is bounded by arena size. Startup is instant. Perfect for latency-sensitive or resource-constrained environments.

10. **Inspectable evolution** — Evolution state is markdown files on disk. You can `cat STRATEGY.md` to see what the agent learned. You can `git diff` to see how its strategy evolved. You can edit files manually to steer it. No opaque internal state.

### 19.3 Cons

1. **C is dangerous** — Buffer overflows, use-after-free (across arena boundaries), null pointer dereferences. The arena mitigates many issues, but C offers no safety guarantees. A bug in a tool implementation can crash the entire process. Mitigation: extensive fuzzing, ASan/UBSan in CI, strict coding standards.

2. **No garbage collector** — The malloc/arena split requires discipline. Strings that cross the arena boundary must be explicitly `strdup`'d. Forget once and you get a use-after-free. Mitigation: clear documentation, consistent patterns, a single boundary rule.

3. **Build complexity** — Five C/C++ dependencies (libcurl, mbedtls, llama.cpp, SQLite, sqlite-extensions) need to be compiled. Cross-compilation (ARM, RISC-V) requires careful toolchain setup. Mitigation: provide pre-built static libraries and a Docker-based build.

4. **Smaller ecosystem** — Python has LangChain, LlamaIndex, thousands of tools. TypeScript has Vercel AI SDK. Adam starts with a handful of built-in tools. Users must implement most tools themselves (or wrap existing C libraries). Mitigation: the tool interface is simple (`AdamToolFn`), and C can call into any library.

5. **No async I/O** — The agent loop is synchronous (blocks on LLM calls). For high-concurrency servers, you'd need one thread per agent. Go/Python/Node handle this more naturally with goroutines/asyncio/event-loop. Mitigation: pthreads + thread pool for parallel execution.

6. **Tool call format for local models is fragile** — Local GGUF models must be prompted to emit `<tool_call>` XML tags. Smaller models may not follow the format reliably, leading to parse failures. Mitigation: fallback parsing (try JSON first, then XML, then regex), and model-specific prompt templates.

7. **No plugin system** — Unlike OpenClaw's extension hooks, Adam has no dynamic plugin loading. All tools must be compiled in or provided as function pointers at startup. Mitigation: shared library loading (`dlopen`) could be added as an optional feature.

8. **Limited streaming for tools** — Tool execution is synchronous. Long-running tools (web scraping, file processing) block the iteration. No async tool pattern like PicoClaw's `AsyncExecutor`. Mitigation: tools can spawn threads internally and return a "task started" result.

9. **C++ dependency via llama.cpp** — llama.cpp is C++. This means the full build requires a C++ compiler and links `libstdc++`. Pure C purists may object. Mitigation: `ADAM_NO_LOCAL` build config excludes llama.cpp entirely.

10. **Manual JSON construction** — Building JSON request bodies with `snprintf` is error-prone (escaping, buffer sizing). jsmn only parses, it doesn't generate. Mitigation: a small `adam_json_escape()` helper and careful buffer estimation.

### 19.4 Comparison: Adam vs Existing Frameworks

| | **Adam (C)** | **NullClaw (Zig)** | **LangChain (Python)** | **Vercel AI SDK (TS)** |
|---|---|---|---|---|
| **Binary size** | ~5 MB | ~678 KB | N/A (runtime) | N/A (runtime) |
| **RAM (idle)** | ~2 MB | ~1 MB | ~50+ MB | ~80+ MB |
| **Startup** | <5 ms | <2 ms | ~2 s | ~1 s |
| **Embeddable** | Yes (any FFI) | No (Zig only) | No | No |
| **WASM** | Yes | No | No | Partial |
| **Offline** | Yes (llama.cpp) | No | No | No |
| **Memory system** | SQLite + vector | SQLite FTS5 + vector | Many (Pinecone, etc.) | External only |
| **Multi-agent sync** | ATTACH DATABASE | N/A | Custom | Custom |
| **Evolution loop** | Built-in | N/A | N/A | N/A |
| **Language FFI** | Universal | Zig/C | Python only | JS/TS only |
| **Safety** | Manual (arenas help) | Compile-time | GC managed | GC managed |
| **Ecosystem** | Small (growing) | Small | Massive | Large |

---

## 20. Cross-Device Sync & Mobile Application

### 20.1 Goal

Build an Adam mobile application (iOS and Android) that can continue a session started on desktop, and vice versa. The mobile UI is a chat-like app where Adam posts output and the user can input prompts in text or voice. Sessions can be created, continued, and evolved independently on any device.

### 20.2 Architecture

```
Desktop (macOS/Linux)         SQLite Cloud            Mobile (iOS/Android)
┌──────────────┐             ┌─────────────┐         ┌──────────────┐
│  libadam.a   │             │             │         │  libadam.a   │
│  (C library) │             │   SQLite    │         │  (C library) │
│              │◀─── sync ──▶│   Cloud    │◀── sync ─▶│              │
│  sessions DB │    (CRDT)   │             │  (CRDT)  │  sessions DB │
│  memory DB   │             │             │         │  memory DB   │
│  documents   │             └─────────────┘         │  documents   │
└──────────────┘                                     └──────────────┘
```

- Same `libadam.a` C library compiled for each platform (arm64-ios, arm64-android, x86_64-linux, arm64-macos)
- SQLite as the **single source of truth** — all state lives in the database
- Sync via [sqlite-sync](https://github.com/sqliteai/sqlite-sync) extension (CRDT-based, conflict-free, bidirectional)
- Offline-first: every device works independently, merges automatically on reconnect

### 20.3 Sync Layer — sqlite-sync

The [sqlite-sync](https://github.com/sqliteai/sqlite-sync) extension adds CRDT-based synchronization to SQLite. Changes are tracked per-column and merged deterministically without manual conflict resolution.

**Integration:**

```c
// At database open — enable sync on all Adam tables
cloudsync_init('sessions');
cloudsync_init('messages');
cloudsync_init('documents');

// Configure cloud endpoint
cloudsync_network_init("sqlitecloud://your-project.sqlite.cloud:8860/adam.db");
cloudsync_network_set_apikey("your-api-key");

// Sync — call periodically or on app foreground/background
cloudsync_network_sync();
```

**Public API addition:**

```c
// adam.h
adam_status_t adam_sync_init(adam_session_t *sess, const char *connection_string,
                             const char *api_key);
adam_status_t adam_sync(adam_session_t *sess);  // bidirectional sync
```

### 20.4 Schema Changes Required

**1. UUIDv7 primary keys** — replace sequential IDs with globally unique, time-ordered UUIDs so two devices creating sessions simultaneously won't collide:

```sql
-- Use cloudsync_uuid() for all new rows
INSERT INTO sessions(id, ...) VALUES(cloudsync_uuid(), ...);
```

**2. Documents table** — store markdown content (evolution files, bootstrap, memory notes) in SQLite instead of the filesystem:

```sql
CREATE TABLE documents (
    id TEXT PRIMARY KEY DEFAULT (cloudsync_uuid()),
    path TEXT UNIQUE NOT NULL,      -- "evolution/TASK.md", "SOUL.md", etc.
    content TEXT NOT NULL,
    device_id TEXT,                  -- which device last edited
    updated_at INTEGER DEFAULT (unixepoch())
);
```

Desktop reads/writes `.md` files and syncs them into this table. Mobile reads/writes only through SQLite. One source of truth.

**3. Device tracking** — identify which device made each change:

```sql
ALTER TABLE sessions ADD COLUMN device_id TEXT;
ALTER TABLE messages ADD COLUMN device_id TEXT;
```

### 20.5 Mobile Build

libadam.a compiles for mobile with the same source:

```makefile
# iOS (arm64)
xcrun -sdk iphoneos cc -arch arm64 -std=c11 -O2 \
    -DADAM_NO_LOCAL -DADAM_NO_CURL \
    -Isrc -Imodules/sqlite -Imodules/miniaudio \
    -c src/adam.c src/adam_json.c src/adam_http.c \
      src/adam_voice.c src/adam_audio.c src/adam_session.c \
      src/adam_net_apple.m modules/sqlite/sqlite3.c

# Android (arm64, via NDK)
$NDK/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android34-clang \
    -std=c11 -O2 -DADAM_NO_LOCAL \
    -Isrc -Imodules/sqlite -Imodules/miniaudio \
    -c src/adam.c src/adam_json.c src/adam_http.c \
      src/adam_voice.c src/adam_audio.c src/adam_session.c \
      src/adam_net_curl.c modules/sqlite/sqlite3.c
```

**Platform bindings:**
- **iOS**: Swift calls C via bridging header. NSURLSession for HTTP (already in adam_net_apple.m). AVAudioSession for mic permissions.
- **Android**: Kotlin calls C via JNI. OkHttp or the curl-based adam_net_curl.c for HTTP. MediaRecorder for mic.

### 20.6 Mobile Chat UI

Minimal chat interface wrapping the C API:

```
┌─────────────────────────────────────┐
│  Adam                          ≡    │
├─────────────────────────────────────┤
│                                     │
│  ┌─────────────────────────────┐    │
│  │ You: What's the weather?    │    │
│  └─────────────────────────────┘    │
│                                     │
│  ┌─────────────────────────────┐    │
│  │ 🔧 get_weather("Rome")     │    │
│  └─────────────────────────────┘    │
│                                     │
│  ┌─────────────────────────────┐    │
│  │ Adam: It's 22°C and sunny   │    │
│  │ in Rome.                    │    │
│  └─────────────────────────────┘    │
│                                     │
├─────────────────────────────────────┤
│  [🎤]  Type a message...    [Send]  │
└─────────────────────────────────────┘
```

- Messages rendered from `adam_history_t` (loaded from session DB)
- Text input → `adam_run(settings, history, text)`
- Mic button → `adam_voice_run(settings, history, audio, len, format)`
- Tool calls shown as collapsible cards
- Session picker in nav bar (from `adam_session_list()`)
- Sync indicator (last sync time, pending changes count)

### 20.7 Task Checklist

- [ ] Add sqlite-sync as submodule (`modules/sqlite-sync/`)
- [ ] Migrate primary keys to UUIDv7 (`cloudsync_uuid()`)
- [ ] Add `documents` table for markdown content storage
- [ ] Add `device_id` column to sessions and messages tables
- [ ] Add `adam_sync_init()` / `adam_sync()` to public API
- [ ] Add `adam_documents_read()` / `adam_documents_write()` API
- [ ] Modify `build_system_prompt()` to read bootstrap files from `documents` table
- [ ] Modify evolution loop to use `documents` table instead of filesystem
- [ ] Create iOS Xcode project with Swift bridging header
- [ ] Create Android project with JNI bindings
- [ ] Build libadam.a for arm64-ios and arm64-android
- [ ] Implement chat UI (SwiftUI for iOS, Jetpack Compose for Android)
- [ ] Add mic permission handling and audio session management
- [ ] Add sync trigger on app foreground / background transitions
- [ ] Add session picker and sync status indicator
- [ ] Test offline-first: create session on device A offline, sync to device B
- [ ] Test concurrent edits: both devices modify same session, verify CRDT merge
