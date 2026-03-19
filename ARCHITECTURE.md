# Architecture

Low-level internals of the Adam agent library. Covers data flow, storage, memory management, and the exact sequence of operations for every major subsystem.

## Table of Contents

- [Memory Model](#memory-model)
- [Agent Run Loop](#agent-run-loop)
- [System Prompt Construction](#system-prompt-construction)
- [LLM Dispatch Chain](#llm-dispatch-chain)
- [Tool Execution](#tool-execution)
- [Session Persistence](#session-persistence)
- [Long-Term Memory](#long-term-memory)
- [Response Cache](#response-cache)
- [Evolution Loop](#evolution-loop)
- [Research Mode](#research-mode)
- [Arena Allocator](#arena-allocator)
- [JSON Wire Formats](#json-wire-formats)
- [SQLite Schema](#sqlite-schema)

---

## Memory Model

Adam uses two allocation strategies:

**Arena allocator** (`arena_t`) — for per-iteration temporary data. Created at the start of `adam_run()`, reset after each tool iteration, destroyed on return. Used for LLM response parsing, system prompt building, tool result strings, and JSON construction. Zero-copy: no individual frees needed. Everything is released in bulk via `arena_reset()`.

**malloc/strdup** — for long-lived data that survives across iterations. Used for conversation history messages, settings strings, session data, evolution state, and the final response. The caller is responsible for freeing via the appropriate `_destroy()` or `_free()` function.

The rule: if a pointer comes from the arena, it's only valid until the next `arena_reset()` or `arena_destroy()`. If it needs to outlive the current iteration, it must be copied to malloc'd memory (typically via `strdup`).

---

## Agent Run Loop

`adam_run(settings, history, user_message)` is the core entry point. Here is the exact sequence:

### Phase 1: Setup

```
1. Validate inputs (settings, history non-NULL, provider configured)
2. Reset abort_flag to 0
3. Record start time (clock_gettime CLOCK_MONOTONIC)
4. Append user_message to history via adam_history_append_user()
   → strdup(content), store in history->items[count++]
5. Create per-turn arena: arena_create(arena_block_size)
   → allocates first 256KB block
6. Build system prompt (see System Prompt Construction below)
7. Insert or update system message at history->items[0]
   → if no system message exists: shift all messages right, insert at [0]
   → if exists: free old content, replace with new
```

### Phase 2: Iteration Loop

Runs up to `max_iterations` (default 25) times:

```
For each iteration:
  8.  Check abort_flag → if set, return ADAM_ERR_ABORTED
  9.  Check timeout_ms → if elapsed >= timeout_ms, return ADAM_ERR_TIMEOUT
  10. Reset arena (keeps first block, frees extras, zeroes used counters)

  11. Smart context window check:
      → estimate tokens in history (content_len/4 + tool_calls*50 per message)
      → if estimate > 75% of model context window:
         call adam_history_summarize() (LLM-based compression)
         fallback: history_compress() (drop oldest 50%)

  12. Count-based compression fallback:
      → if history->count > max_history: history_compress()

  13. Pre-send guardrail:
      → if on_before_send callback returns non-zero: return ADAM_ERR_GUARDRAIL

  14. Cache lookup:
      → compute FNV-1a hash of (model + all message roles + contents)
      → lookup in hash table → if hit: strdup cached response, break

  15. Rate limiter:
      → if requests_per_min exceeded: sleep until window resets
      → increment request counter

  16. LLM dispatch with retry (see LLM Dispatch Chain below):
      → up to retry_max attempts with exponential backoff
      → on rate limit: sleep retry_rate_limit_ms (default 30s)
      → on auth error or context overflow: no retry

  17. Handle errors:
      → context overflow on iteration <= 2: summarize and retry
      → other errors: set status, break

  18. Accumulate token counts (input_tokens, output_tokens)

  19. Post-receive guardrail:
      → if on_after_receive callback returns non-zero: return ADAM_ERR_GUARDRAIL

  20. If no tool calls → final response:
      → strdup response text
      → append assistant message to history
      → fire on_response callback
      → store in cache (if enabled)
      → break

  21. If tool calls → execute tools (see Tool Execution below):
      → append assistant message with tool call IDs to history
      → for each tool call:
         find tool in registry by name
         call tool->execute(arena, ctx, args_json, args_len)
         append tool result to history
         fire on_response callback for user-facing output
      → continue to next iteration
```

### Phase 3: Post-Run

```
  22. Compute cost: (input_tokens / 1M) * input_cost + (output_tokens / 1M) * output_cost
  23. Auto-save session:
      → if session_id && memory && auto_save:
         call adam_session_save(memory, session_id, history)
  24. Memory extraction:
      → if memory_extract && memory && status == ADAM_OK:
         build prompt from last 10 messages
         ask LLM: "Extract key facts, preferences, decisions..."
         if response != "NONE": call adam_memory_add(memory, response, context, "conversation")
  25. Compute elapsed_ms
  26. Destroy arena
  27. Return result
```

---

## System Prompt Construction

`build_system_prompt(arena, settings, user_message)` assembles the system prompt from multiple sources, all into a single arena-allocated buffer:

```
1. Identity (default: "You are a helpful assistant.")
2. Instructions (if set)
3. Bootstrap files:
   → for each file path: fopen, fread entire file, inject as "## filename\n{contents}\n"
   → buffer grows dynamically within the arena if needed
4. Memory enrichment (if inject_memory && memory):
   → call adam_memory_search(memory, arena, user_message, context, max_results)
   → SQL: SELECT path, snippet, context, ranking FROM memory_search WHERE query=?1 LIMIT ?3
   → inject as "## Relevant memories\n- {snippet}\n- {snippet}\n"
5. Datetime injection (if inject_datetime):
   → "Current date: 2026-03-18 14:30\n"
6. Session info (if set)
```

The buffer starts at 8KB and doubles when needed (arena-allocated, so no individual frees).

---

## LLM Dispatch Chain

`dispatch_llm(arena, settings, msgs, msg_count)` tries providers in priority order:

```
Priority 1: Custom LLM callback (llm_fn)
  → settings->llm_fn(llm_ctx, arena, msgs, msg_count, tools, tool_count)
  → Used for testing (mock LLM) and custom backends

Priority 2: Local inference (llama.cpp)
  → if gguf_path is set: adam_llm_call_local(arena, settings, msgs, ...)
  → Loads GGUF model, runs inference on GPU/CPU
  → Tool definitions injected into system prompt, tool calls parsed from output
  → If mmproj_path set + image attachments: vision path via mtmd
  → Log output suppressed unless local_verbose=1

Priority 3: Custom HTTP callback (http_fn)
  → For WASM or custom transports

Priority 4: HTTP API
  → if on_stream set: try adam_llm_call_http_stream() first
  → fallback: adam_llm_call_http()
  → Builds JSON request (Anthropic, OpenAI, or Gemini format)
  → POST to base_url with auth headers
  → Parse JSON response
```

The retry wrapper `dispatch_with_retry()` wraps the dispatch:
- Up to `retry_max` attempts (default 3)
- Non-retriable: `ADAM_ERR_AUTH`, `ADAM_ERR_CONTEXT_OVERFLOW`
- Rate limit: sleep `retry_rate_limit_ms` (default 30s)
- Other errors: exponential backoff (1s, 2s, 5s, 10s)
- Arena reset between retries

---

## Tool Execution

When the LLM returns tool calls, the agent loop executes them sequentially:

```
For each tool_call in response.tool_calls:
  1. Fire on_tool_call callback (notification)
  2. Search tools[] array for matching name (linear scan, strcmp)
  3. If found:
     → call tool->execute(arena, tool->ctx, args_json, strlen(args_json))
     → tool allocates for_llm and for_user strings from the arena
  4. If not found:
     → set for_llm = "Error: tool not found"
  5. Append tool result to history:
     → adam_history_append_tool(history, for_llm, tool_call_id)
     → strdup'd into malloc'd memory (survives arena reset)
  6. If for_user: fire on_response callback
```

Tool results are arena-allocated (valid only for current iteration) but immediately copied into the history via strdup. The arena is reset at the top of the next iteration.

---

## Session Persistence

Sessions are stored in the same SQLite database as long-term memory. Two tables manage sessions.

### Creating a Session

`adam_session_create(memory, out_id, out_id_size)`:

```
1. Generate UUIDv7 via dbmem_uuid_v7()
   → 48-bit millisecond timestamp + random bytes
   → format: "018e5a3c-7b2a-7d00-8000-1a2b3c4d5e6f" (36 chars + null)
   → naturally time-ordered (sortable)
2. INSERT INTO sessions(id) VALUES(?1)
3. Copy UUID to out_id
```

### Saving a Session

`adam_session_save(memory, session_id, history)`:

```
BEGIN IMMEDIATE;

INSERT INTO sessions(id, updated_at) VALUES(?1, unixepoch())
  ON CONFLICT(id) DO UPDATE SET updated_at=unixepoch();

DELETE FROM messages WHERE session_id=?1;

For each message in history (index 0..count-1):
  INSERT INTO messages(session_id, idx, role, content, content_len,
                        tool_call_id, tool_calls_json)
  VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7);

  → role: integer (0=system, 1=user, 2=assistant, 3=tool)
  → tool_calls_json: serialized as JSON array:
    [{"id":"call_001","name":"search","arguments":"{\"q\":\"test\"}"}]
  → JSON escaping: " → \", \ → \\, \n → \n

COMMIT;
```

### Loading a Session

`adam_session_load(memory, session_id, history)`:

```
1. Verify session exists:
   SELECT 1 FROM sessions WHERE id=?1

2. Clear existing history: adam_history_clear(h)

3. Load messages in order:
   SELECT role, content, tool_call_id, tool_calls_json
   FROM messages WHERE session_id=?1 ORDER BY idx

4. For each row:
   → role=1 (user): adam_history_append_user(h, content)
   → role=2 (assistant):
     if tool_calls_json: deserialize array → adam_tool_call_t[]
     adam_history_append_assistant(h, content, tool_calls, count)
   → role=3 (tool): adam_history_append_tool(h, content, tool_call_id)
   → role=0 (system): adam_history_append_user then overwrite role
```

Tool call deserialization parses the JSON array, extracting `id`, `name`, and `arguments` fields with unescaping of backslash sequences.

### Auto-Save

When `session_id` and `auto_save` are set, `adam_session_save()` is called automatically at the end of every `adam_run()`, after cost tracking but before elapsed time calculation. This means every conversation turn is persisted.

---

## Long-Term Memory

Memory is powered by two SQLite extensions loaded into the same database: **sqlite-memory** (chunking, embedding, BM25 full-text search) and **sqlite-vector** (vector similarity search).

### Database Initialization

`adam_memory_open(db_path)`:

```
1. sqlite3_open(db_path)
2. PRAGMA journal_mode=WAL
3. PRAGMA synchronous=NORMAL
4. sqlite3_vector_init() — load vector extension
5. sqlite3_memory_init() — load memory extension
6. Create session tables (sessions, messages) — see above
7. PRAGMA foreign_keys=ON
```

### Internal Tables (created by sqlite-memory)

```sql
-- Chunk content and metadata
dbmem_content (
    hash INTEGER PRIMARY KEY,   -- content hash
    path TEXT NOT NULL UNIQUE,   -- file path or UUIDv7
    value TEXT,                  -- original text (if save_content enabled)
    length INTEGER NOT NULL,    -- byte length
    context TEXT,               -- grouping label (e.g., "project", "preferences")
    created_at INTEGER,
    last_accessed INTEGER
)

-- Vector embeddings for each chunk
dbmem_vault (
    hash INTEGER NOT NULL,      -- references dbmem_content.hash
    seq INTEGER NOT NULL,       -- chunk sequence number
    embedding BLOB NOT NULL,    -- float vector
    offset INTEGER NOT NULL,    -- byte offset in source
    length INTEGER NOT NULL,    -- chunk byte length
    PRIMARY KEY (hash, seq)
)

-- Full-text search index
dbmem_vault_fts USING fts5 (
    content,                    -- chunk text
    hash UNINDEXED,
    seq UNINDEXED,
    context UNINDEXED           -- for context-filtered FTS
)

-- Embedding cache (avoid re-computing)
dbmem_cache (
    text_hash INTEGER,
    provider TEXT,
    model TEXT,
    embedding BLOB,
    dimension INTEGER,
    PRIMARY KEY (text_hash, provider, model)
)

-- Extension configuration
dbmem_settings (key TEXT PRIMARY KEY, value TEXT)
```

### Adding Knowledge

`adam_memory_add(mem, text, context, source)`:

```
1. SELECT memory_add_text(?1, ?2)
   → sqlite-memory extension processes the text:
     a. Chunk text into segments (markdown-aware splitting)
     b. For each chunk:
        - Compute embedding via configured provider (local or API)
        - Store in dbmem_vault (hash, seq, embedding, offset, length)
        - Index in dbmem_vault_fts (content, hash, seq, context)
     c. Store metadata in dbmem_content (hash, path, value, context, ...)

2. If markdown_output enabled:
   → Write {markdown_dir}/{sanitized_source}.md
   → YAML frontmatter: context, source, date
   → Body: raw text
```

### Searching

`adam_memory_search(mem, arena, query, context, limit)`:

```sql
SELECT path, snippet, context, ranking
FROM memory_search
WHERE query = ?1 [AND context = ?2]
LIMIT ?3
```

The `memory_search` virtual table (defined by sqlite-memory) performs:
1. **BM25 full-text search** via `dbmem_vault_fts`
2. **Vector similarity search** via `dbmem_vault` + `sqlite-vector`
3. **Hybrid ranking** combining both scores
4. Returns snippet (text around match), path (source file), context, and ranking score

Results are arena-allocated. The caller does not need to free them individually.

### Memory Extraction

After a successful `adam_run()`, if `memory_extract` is enabled:

```
1. Collect last 10 messages from history
2. Format as "Role: content\n" for each
3. Build extraction prompt:
   System: "Extract key facts, preferences, and decisions..."
   User: formatted messages
4. Call dispatch_llm() (separate LLM call, not cached)
5. If response is not "NONE":
   adam_memory_add(memory, response, memory_context, "conversation")
```

This creates a feedback loop: conversations generate knowledge that enriches future conversations.

---

## Response Cache

The cache is an LRU hash table that maps message history to LLM responses.

### Data Structure

```
adam_cache_t:
  buckets[bucket_count]     — hash table (array of collision chains)
  lru_head → ... → lru_tail — doubly-linked list, most to least recent
  count, max_entries         — current size and capacity
  hits, misses               — statistics

adam_cache_entry_t:
  hash (uint64_t)           — FNV-1a of model + messages
  response (char *)         — malloc'd response text
  input_tokens, output_tokens
  lru_prev, lru_next        — LRU list pointers
  bucket_next               — collision chain pointer
```

### Hash Function (FNV-1a)

```
h = 0xcbf29ce484222325 (offset basis)
for each byte in model name:
    h ^= byte; h *= 0x100000001b3
for each message in history:
    h ^= role (as 4 bytes); h *= 0x100000001b3
    for each byte in content:
        h ^= byte; h *= 0x100000001b3
```

Tool definitions are NOT included in the key. Same messages with different tools produce the same hash. This is intentional — tool availability doesn't change the response for non-tool queries, and for tool queries the response includes tool calls which are not cached (only final text responses are cached).

### Lookup

```
1. idx = hash % bucket_count
2. Walk buckets[idx] collision chain
3. If match: move to LRU front, return response, increment hits
4. No match: increment misses, return NULL
```

### Store

```
1. Check if hash already exists → if yes, update in place, move to LRU front
2. While count >= max_entries: evict LRU tail
   → remove from LRU list
   → remove from bucket chain
   → free entry
3. Create new entry: calloc, strdup response
4. Insert into bucket chain (prepend)
5. Push to LRU front
```

### Integration with adam_run

The cache key is computed inside the iteration loop (recomputed each iteration since history changes after tool calls). Only final text responses (no tool calls) are cached. Cache hits skip the LLM call entirely — the cached response is strdup'd and returned immediately.

---

## Evolution Loop

`adam_evolve(settings, config)` runs a self-improving loop where the agent iterates on a task, scores each attempt, and refines its strategy.

### Per-Iteration Flow

```
For each iteration (0 to max_iterations-1):

  1. Build attempt prompt:
     "You are a self-improving agent on iteration #N.
      ## TASK (immutable)
      {task}
      ## CURRENT STRATEGY
      {strategy or "No strategy yet."}
      ## SCORING CRITERIA
      {metrics}
      ## INSIGHTS (accumulated lessons)
      {insights or "No insights yet."}
      ## PREVIOUS ATTEMPTS
      - Iteration 1 (score 45): output preview...
      - Iteration 2 (score 62): output preview...
      Produce your best attempt."

  2. Clear history, call adam_run(settings, history, prompt)
     → fresh conversation each attempt (no carry-over)

  3. Call eval_fn(ctx, output, iteration) → score 0-100
     → -1 signals error, stops loop

  4. Store in ring buffer (FIFO, max 10 entries)
     → if full: shift left, overwrite oldest

  5. If score > best_score:
     a. Update best (output, score, iteration)
     b. Reset plateau counter
     c. Refine strategy via LLM:
        "The score improved from X to Y.
         ## OLD STRATEGY
         {old}
         ## ATTEMPT THAT SCORED Y
         {output}
         Refine the strategy."
     d. Replace strategy with LLM response

  6. Extract insights via LLM (always, even on failure):
     "Iteration N scored X/100.
      ## ATTEMPT
      {output}
      ## CURRENT INSIGHTS
      {insights}
      Update the insights with lessons learned."
     → Replace insights with LLM response

  7. Check stop conditions:
     → score >= target_score: STOP_SCORE
     → iterations_without_improvement >= plateau_iters: STOP_PLATEAU
     → last iteration: STOP_MAX_ITERS
```

### LLM Calls Per Iteration

- **Attempt**: 1 call (always)
- **Strategy refine**: 1 call (only on improvement)
- **Insight extraction**: 1 call (always)
- Total: 2-3 LLM calls per iteration

### Ring Buffer

The `attempts[]` array holds the last 10 entries. When full, the oldest entry is evicted by shifting all entries left and inserting at the end. This gives the LLM context about recent attempts without unbounded growth.

---

## Research Mode

`adam_research(settings, config)` runs an autonomous information-gathering loop where the agent uses tools across multiple iterations to build a research report.

### Key Difference from Evolution

- **Evolution**: fresh history each iteration, LLM produces an "attempt"
- **Research**: persistent history across iterations, LLM uses tools to gather information

### Protocol

The agent follows a structured protocol injected via system prompt:

```
1. Use tools to search for information
2. Report findings as:
   FINDING: <fact>
   SOURCE: <where found>
3. Identify gaps and search further
4. When complete: RESEARCH_COMPLETE followed by report
```

### Per-Iteration Flow

```
For each iteration (0 to max_iterations):

  1. Build continuation prompt:
     → First iteration: "Begin your research. Use tools..."
     → Subsequent: "Research iteration N. Findings so far (M):
       - finding 1 [source 1]
       - finding 2
       Identify gaps and search for more."

  2. Call adam_run(settings, history, prompt)
     → History persists! Agent sees previous tool calls and results
     → Tools (web_fetch, memory_search, etc.) are available

  3. Parse findings from response:
     → Scan for "FINDING:" lines (max 100 per response)
     → For each: extract content, check next line for "SOURCE:"
     → Append to findings[] array (realloc as needed)

  4. Check completeness:
     → If response contains "RESEARCH_COMPLETE":
        extract text after marker as report
     → Else if is_complete callback returns 1: stop
     → Else if last iteration: stop

  5. If stopped without report and findings exist:
     → Synthesis call: "Synthesize all findings into a report..."
     → One additional LLM call with all findings listed
     → Store response as report

  6. Fallback: if still no report, use last assistant message
```

### Finding Storage

```c
adam_research_finding_t {
    char *content;      // malloc'd: "The speed of light is 299,792,458 m/s"
    char *source;       // malloc'd: "Wikipedia" (or NULL)
    int iteration;      // which iteration found this (0-indexed)
};
```

Findings grow via `realloc`. Each finding's content and source are `strdup`'d from the parsed response. The entire array is freed by `adam_research_result_free()`.

---

## Arena Allocator

The arena provides fast, bulk-freed allocation for per-iteration temporary data.

### Block Chain

```
arena_t
  ├─ head → block_0 [256KB, used=4200] → block_1 [256KB, used=800] → NULL
  ├─ current → block_1
  └─ default_cap = 262144 (256KB)

arena_block_t:
  next: pointer to next block
  used: bytes consumed
  capacity: size of data[]
  data[]: flexible array member
```

### Allocation

```
arena_alloc(arena, size):
  1. Align offset to 8 bytes: offset = (current->used + 7) & ~7
  2. If offset + size <= current->capacity:
     → current->used = offset + size
     → return current->data + offset
  3. Else: allocate new block
     → cap = max(size, default_cap)
     → malloc(sizeof(block) + cap)
     → link: current->next = new_block
     → current = new_block
     → new_block->used = size
     → return new_block->data
```

### Reset

```
arena_reset(arena):
  1. Keep head block
  2. Walk head->next chain, free all subsequent blocks
  3. head->next = NULL
  4. head->used = 0
  5. current = head
```

This makes the arena reusable without reallocating the first block. Ideal for the iteration loop where the same arena is reset 25+ times.

### Destroy

```
arena_destroy(arena):
  1. Walk block chain from head, free each block
  2. Free arena struct
```

---

## JSON Wire Formats

Adam supports three JSON formats for LLM APIs.

### Anthropic Messages API

```json
{
  "model": "claude-sonnet-4-20250514",
  "max_tokens": 4096,
  "temperature": 0.7,
  "top_p": 0.9,
  "system": [{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}],
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
      {"type": "text", "text": "Describe this image"}
    ]},
    {"role": "assistant", "content": [
      {"type": "text", "text": "Let me search."},
      {"type": "tool_use", "id": "call_001", "name": "search", "input": {"q": "test"}}
    ]},
    {"role": "user", "content": [
      {"type": "tool_result", "tool_use_id": "call_001", "content": "Found 42 results."}
    ]}
  ],
  "tools": [
    {"name": "search", "description": "...", "input_schema": {"type": "object", ...}}
  ]
}
```

Key differences: system message outside messages array, tool results as `"user"` role with `type: "tool_result"`, tool calls as `type: "tool_use"` with `input` (raw JSON object), cache control hint on system message, images as `type: "image"` with base64 source.

### OpenAI Chat Completions

```json
{
  "model": "gpt-4o",
  "max_completion_tokens": 4096,
  "temperature": 0.7,
  "top_p": 0.9,
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      {"type": "text", "text": "Describe this image"}
    ]},
    {"role": "assistant", "content": "Let me search.", "tool_calls": [
      {"id": "call_001", "type": "function", "function": {"name": "search", "arguments": "{\"q\":\"test\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_001", "content": "Found 42 results."}
  ],
  "tools": [
    {"type": "function", "function": {"name": "search", "description": "...", "parameters": {"type": "object", ...}}}
  ]
}
```

Key differences: system message inline, tool results as `"tool"` role with `tool_call_id`, tool calls as `function.arguments` (stringified JSON), images as `type: "image_url"` with data URI.

### Google Gemini

```json
{
  "systemInstruction": {"role": "user", "parts": [{"text": "..."}]},
  "contents": [
    {"role": "user", "parts": [
      {"inlineData": {"mimeType": "image/jpeg", "data": "..."}},
      {"text": "Describe this image"}
    ]},
    {"role": "model", "parts": [{"functionCall": {"name": "search", "args": {"q": "test"}}}]},
    {"role": "user", "parts": [{"functionResponse": {"name": "search", "response": {"content": "Found 42 results."}}}]}
  ],
  "tools": [{"functionDeclarations": [{"name": "search", "description": "...", "parameters": {...}}]}],
  "generationConfig": {"temperature": 0.7, "maxOutputTokens": 4096, "topP": 0.9}
}
```

Key differences: system as `systemInstruction`, roles are `"user"`/`"model"`, tool calls as `functionCall` with `args` (raw JSON), tool results as `functionResponse`, images as `inlineData` with mimeType, generation params in separate `generationConfig` object. Streaming uses `alt=sse` URL parameter (no `stream:true` in body). Gemini has no tool call IDs — the library generates synthetic IDs (`gemini_0`, `gemini_1`).

### Local Model (llama.cpp)

Local models use the model's native chat template (via `llama_model_chat_template()`). Tool definitions are injected into the system prompt as plain text. The model outputs tool calls as `<tool_call>{"name":"...","arguments":{...}}</tool_call>` tags which are parsed from the text output. Synthetic IDs are generated (`local_0`, `local_1`). When tools are registered, streaming is buffered until tool call parsing is complete to prevent raw tag text from leaking to the user.

### JSON Building

All JSON is built using an arena-backed string buffer (`abuf_t`). Image attachments are base64-encoded inline via `abuf_base64()`. The `media_type_string()` helper maps `adam_media_type_t` enums to MIME type strings.

### SSE Streaming

The SSE parser (`adam_stream.c`) handles all three cloud providers:
- Text and tool call arguments are unescaped from JSON string encoding (`\n` → newline, `\uXXXX` → UTF-8, surrogate pairs for emoji)
- Anthropic: `content_block_delta` with `text_delta` or `partial_json`
- OpenAI: `choices[].delta.content` or `tool_calls[].function.arguments`
- Gemini: `candidates[].content.parts[].text` or `functionCall`

### Response Parsing

Uses jsmn (a minimal JSON tokenizer, ~400 lines, header-only). All parsing is zero-allocation — tokens reference positions in the original JSON string. Extracted values are copied to the arena via `arena_strdup`.

---

## SQLite Schema

All persistent data lives in a single SQLite database file. The schema has two layers:

### Adam Layer (created in `adam_memory_open`)

```sql
-- Conversation sessions (UUIDv7 primary keys)
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,                    -- UUIDv7: "018e5a3c-7b2a-..."
    sync INTEGER DEFAULT 0,                -- 1 = include in multi-agent sync
    created_at INTEGER DEFAULT (unixepoch()),
    updated_at INTEGER DEFAULT (unixepoch())
);

-- Messages within sessions
CREATE TABLE messages (
    session_id TEXT NOT NULL,               -- FK → sessions.id
    idx INTEGER NOT NULL,                   -- message index (0-based)
    role INTEGER NOT NULL,                  -- 0=system, 1=user, 2=assistant, 3=tool
    content TEXT,                           -- message text
    content_len INTEGER DEFAULT 0,
    tool_call_id TEXT,                      -- for role=3 (tool results)
    tool_calls_json TEXT,                   -- for role=2 (assistant tool calls)
    created_at INTEGER DEFAULT (unixepoch()),
    PRIMARY KEY (session_id, idx),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
```

### sqlite-memory Layer (created by extension)

```sql
-- Content metadata
CREATE TABLE dbmem_content (
    hash INTEGER PRIMARY KEY,               -- content hash
    path TEXT NOT NULL UNIQUE,              -- file path or generated UUIDv7
    value TEXT DEFAULT NULL,                -- original text (optional)
    length INTEGER NOT NULL,                -- byte length
    context TEXT DEFAULT NULL,              -- grouping label
    created_at INTEGER DEFAULT 0,
    last_accessed INTEGER DEFAULT 0
);

-- Vector embeddings per chunk
CREATE TABLE dbmem_vault (
    hash INTEGER NOT NULL,                  -- FK → dbmem_content.hash
    seq INTEGER NOT NULL,                   -- chunk sequence number
    embedding BLOB NOT NULL,                -- float32 vector
    offset INTEGER NOT NULL,                -- byte offset in source
    length INTEGER NOT NULL,                -- chunk byte length
    PRIMARY KEY (hash, seq)
);

-- Full-text search index (FTS5)
CREATE VIRTUAL TABLE dbmem_vault_fts USING fts5 (
    content,                                -- chunk text
    hash UNINDEXED,
    seq UNINDEXED,
    context UNINDEXED                       -- for context-filtered search
);

-- Embedding computation cache
CREATE TABLE dbmem_cache (
    text_hash INTEGER NOT NULL,
    provider TEXT NOT NULL,                 -- "local", "openai", "voyage"
    model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    dimension INTEGER NOT NULL,
    PRIMARY KEY (text_hash, provider, model)
);

-- Extension settings
CREATE TABLE dbmem_settings (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Hybrid search virtual table (read-only, query interface)
CREATE VIRTUAL TABLE memory_search (...);
-- Columns: query (hidden), max_entries (hidden), context (hidden),
--           hash, seq, ranking, path, snippet
```

### Pragmas

```sql
PRAGMA journal_mode=WAL;        -- concurrent readers, single writer
PRAGMA synchronous=NORMAL;      -- balance durability and speed
PRAGMA foreign_keys=ON;         -- cascade deletes (sessions → messages)
```

WAL mode allows `adam_run()` to read memory while another thread writes, without blocking. This is important for the thread pool where multiple agents may share a memory database.
