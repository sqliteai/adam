//
//  adam.h
//  Adam — Embeddable AI Agent Library
//
//  Created by Marco Bambini on 14/03/26.
//

#pragma once

#include "arena.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// MARK: - Version
// ============================================================================

#define ADAM_VERSION_MAJOR       0
#define ADAM_VERSION_MINOR       1
#define ADAM_VERSION_PATCH       0
#define ADAM_VERSION_STRING      "0.1.0"

// ============================================================================
// MARK: - Build Configuration
// ============================================================================

// Define these before including adam.h to disable features:
//   ADAM_NO_CURL       — no libcurl (no HTTP, must provide http_fn callback)
//   ADAM_NO_LOCAL      — no llama.cpp (no local GGUF inference)
//   ADAM_NO_PTHREADS   — no pthreads (no thread pool, single-threaded only)
//   ADAM_NO_SQLITE     — no SQLite (no memory, no sessions)

// ============================================================================
// MARK: - Forward Declarations
// ============================================================================

typedef struct adam_settings_t   adam_settings_t;
typedef struct adam_agent_t      adam_agent_t;
typedef struct adam_pool_t       adam_pool_t;
typedef struct adam_memory_t     adam_memory_t;
typedef struct adam_session_t    adam_session_t;

// ============================================================================
// MARK: - Status Codes
// ============================================================================

typedef enum {
    ADAM_OK                      = 0,
    ADAM_ERR_ALLOC               = 1,   // memory allocation failed
    ADAM_ERR_PROVIDER             = 2,   // LLM provider returned an error
    ADAM_ERR_RATE_LIMIT          = 3,   // rate limited (HTTP 429)
    ADAM_ERR_AUTH                = 4,   // authentication failed (HTTP 401/403)
    ADAM_ERR_CONTEXT_OVERFLOW    = 5,   // context window exceeded
    ADAM_ERR_MAX_ITERATIONS      = 6,   // tool loop hit max_iterations
    ADAM_ERR_ABORTED             = 7,   // cancelled via abort flag
    ADAM_ERR_CURL                = 8,   // libcurl error
    ADAM_ERR_LOCAL               = 9,   // llama.cpp error
    ADAM_ERR_SQLITE              = 10,  // SQLite / memory / session error
    ADAM_ERR_JSON                = 11,  // JSON parse or build error
    ADAM_ERR_TOOL_NOT_FOUND      = 12,  // tool name not in registry
    ADAM_ERR_INVALID_PARAM       = 13,  // invalid parameter passed to API
    ADAM_ERR_NO_PROVIDER         = 14,  // no provider configured
} adam_status_t;

const char *adam_status_string(adam_status_t status);

// ============================================================================
// MARK: - Logging
// ============================================================================

typedef enum {
    ADAM_LOG_ERROR               = 0,
    ADAM_LOG_WARN                = 1,
    ADAM_LOG_INFO                = 2,
    ADAM_LOG_DEBUG               = 3,
    ADAM_LOG_TRACE               = 4,
} adam_log_level_t;

// Callback: the library never writes to stderr/stdout on its own.
// msg is NOT null-terminated — use len. The pointer is only valid
// for the duration of the callback.
typedef void (*adam_log_fn)(void *ctx, adam_log_level_t level,
                            const char *msg, size_t len);

// ============================================================================
// MARK: - API Format
// ============================================================================

// There are only two JSON wire formats for LLM APIs.
// Every provider uses one of these. The user sets api_format + base_url.
typedef enum {
    ADAM_API_NONE                = 0,   // not configured
    ADAM_API_ANTHROPIC           = 1,   // Anthropic Messages API format
    ADAM_API_OPENAI              = 2,   // OpenAI Chat Completions format
                                       //   (also: Groq, Together, Ollama,
                                       //    vLLM, LM Studio, any compatible)
} adam_api_format_t;

// ============================================================================
// MARK: - Message Types
// ============================================================================

typedef enum {
    ADAM_ROLE_SYSTEM             = 0,
    ADAM_ROLE_USER               = 1,
    ADAM_ROLE_ASSISTANT          = 2,
    ADAM_ROLE_TOOL               = 3,
} adam_role_t;

// --- Multimodal attachments ---

typedef enum {
    ADAM_MEDIA_IMAGE_PNG         = 0,
    ADAM_MEDIA_IMAGE_JPEG        = 1,
    ADAM_MEDIA_IMAGE_GIF         = 2,
    ADAM_MEDIA_IMAGE_WEBP        = 3,
    ADAM_MEDIA_AUDIO_WAV         = 4,
    ADAM_MEDIA_AUDIO_MP3         = 5,
    ADAM_MEDIA_PDF               = 6,
} adam_media_type_t;

typedef struct {
    adam_media_type_t   type;
    uint8_t            *data;           // malloc'd: raw bytes
    size_t              data_len;
    char               *filename;       // malloc'd: optional display name (or NULL)
} adam_attachment_t;

// --- Tool call (on assistant messages) ---

typedef struct {
    char               *id;             // malloc'd: unique call ID
    char               *name;           // malloc'd: tool name
    char               *arguments_json; // malloc'd: raw JSON string
} adam_tool_call_entry_t;

// --- Message ---

typedef struct {
    adam_role_t          role;
    char                *content;           // malloc'd, null-terminated
    size_t               content_len;

    // For ROLE_ASSISTANT: outbound tool calls
    adam_tool_call_entry_t *tool_calls;     // malloc'd array (or NULL)
    size_t               tool_call_count;

    // For ROLE_TOOL: which tool call this result belongs to
    char                *tool_call_id;      // malloc'd (or NULL)

    // Multimodal attachments (primarily for ROLE_USER)
    adam_attachment_t   *attachments;        // malloc'd array (or NULL)
    size_t               attachment_count;
} adam_message_t;

// --- Conversation history ---

typedef struct {
    adam_message_t      *items;             // malloc'd growable array
    size_t               count;
    size_t               capacity;
} adam_history_t;

// ============================================================================
// MARK: - Tool System
// ============================================================================

// Result returned by a tool (arena-owned strings — valid until arena_reset).
typedef struct {
    const char          *for_llm;           // arena-owned: text sent back to LLM
    const char          *for_user;          // arena-owned: text shown to user (optional)
    int                  success;           // 1 = success, 0 = error
} adam_tool_result_t;

// Parsed tool call from LLM response (arena-owned strings).
typedef struct {
    const char          *id;                // arena-owned
    const char          *name;              // arena-owned
    const char          *arguments_json;    // arena-owned
} adam_tool_call_t;

// Tool execution callback.
// - arena: allocate for_llm/for_user from this arena
// - ctx: user-provided context (from adam_tool_def_t.ctx)
// - args_json/args_len: raw JSON arguments string
typedef adam_tool_result_t (*adam_tool_fn)(
    arena_t             *arena,
    void                *ctx,
    const char          *args_json,
    size_t               args_len
);

typedef struct {
    const char          *name;              // tool name (must be unique)
    const char          *description;       // human-readable description
    const char          *parameters_json;   // JSON Schema for arguments
    adam_tool_fn         execute;            // execution callback
    void                *ctx;               // user context passed to execute
} adam_tool_def_t;

// ============================================================================
// MARK: - Streaming
// ============================================================================

// Called for each chunk of a streaming response.
// - chunk/len: partial text (may be empty on final call)
// - is_done: 1 on the final chunk, 0 otherwise
typedef void (*adam_stream_fn)(void *ctx, const char *chunk, size_t len,
                               int is_done);

// ============================================================================
// MARK: - LLM Response (internal, arena-owned)
// ============================================================================

typedef struct {
    const char          *content;           // assistant text (NULL if tool_calls only)
    adam_tool_call_t    *tool_calls;        // arena-allocated array
    size_t               tool_call_count;
    int                  input_tokens;
    int                  output_tokens;
    adam_status_t        error;             // ADAM_OK if no error
    const char          *error_msg;         // arena-owned (NULL if no error)
} adam_llm_response_t;

// ============================================================================
// MARK: - LLM Callback (for testing / custom backends)
// ============================================================================

// When non-NULL, this callback replaces the built-in LLM dispatch entirely.
// The embedder receives the full message history and tool definitions,
// and returns a response. All strings in the response must be allocated
// from the provided arena.
//
// This is the primary injection point for testing: provide a mock LLM
// that returns canned responses or simulates tool-call sequences.
typedef adam_llm_response_t (*adam_llm_fn)(
    void                *ctx,
    arena_t             *arena,
    const adam_message_t *msgs,
    size_t               msg_count,
    const adam_tool_def_t *tools,
    size_t               tool_count
);

// ============================================================================
// MARK: - HTTP Callback (for WASM / custom transports)
// ============================================================================

// When non-NULL and llm_fn is NULL, this callback replaces libcurl
// for HTTP requests. The embedder must perform the HTTP call and
// return the response. All strings must be allocated from arena.
typedef adam_llm_response_t (*adam_http_fn)(
    void                *ctx,
    const char          *url,
    const char          *method,            // "POST"
    const char          *headers_json,      // [["key","val"],...]
    const char          *body,
    size_t               body_len,
    arena_t             *arena
);

// ============================================================================
// MARK: - Run Result
// ============================================================================

typedef struct {
    adam_status_t        status;
    char                *final_response;    // malloc'd — caller must free (or NULL)
    int                  total_iterations;  // tool loop iterations executed
    int                  input_tokens;      // total input tokens consumed
    int                  output_tokens;     // total output tokens generated
    float                cost_usd;          // estimated cost ($0.0 for local models)
    double               elapsed_ms;        // wall-clock time for adam_run()
} adam_run_result_t;

// ============================================================================
// MARK: - Settings (all fields have defaults)
// ============================================================================

// Created by adam_create_settings(). All fields are initialized to sensible
// defaults. Modify only what you need, then pass to adam_run() or adam_agent_create().
//
// Defaults are documented inline. Fields marked [required] must be set
// before calling adam_run().

struct adam_settings_t {

    // --- Remote LLM API ---
    // Set api_format + api_key + model to use a remote provider.
    // base_url is optional: defaults to the official endpoint for the format.
    //   Anthropic: https://api.anthropic.com/v1/messages
    //   OpenAI:    https://api.openai.com/v1/chat/completions
    // Override base_url to point to any compatible endpoint
    // (Groq, Together, Ollama, vLLM, LM Studio, etc.)

    adam_api_format_t    api_format;         // default: ADAM_API_NONE
    const char          *api_key;           // default: NULL (required for remote)
    const char          *base_url;          // default: NULL (uses format default)
    const char          *model;             // default: "claude-sonnet-4-20250514"

#ifndef ADAM_NO_LOCAL
    // --- Local inference (llama.cpp) ---
    // Set gguf_path to enable local inference. When set, the remote API
    // is not used. api_format/api_key/base_url are ignored.

    const char          *gguf_path;         // default: NULL (set to enable local)
    int                  local_gpu_layers;   // default: -1 (all layers on GPU)
    int                  local_ctx_size;     // default: 0 (use model default)
    int                  local_batch_size;   // default: 512
#endif

    // --- Generation parameters ---

    float                temperature;        // default: 0.7
    int                  max_tokens;         // default: 4096
    const char          *response_format;    // default: NULL (text). "json" for JSON mode
    float                top_p;              // default: 1.0
    float                frequency_penalty;  // default: 0.0
    float                presence_penalty;   // default: 0.0

    // --- Agent loop settings ---

    int                  max_iterations;     // default: 25 (max tool-call iterations)
    int                  max_history;        // default: 100 (max messages before compression)
    int                  summarize_threshold;// default: 0 (0 = auto: 75% of context window)

    // --- Arena ---

    size_t               arena_block_size;   // default: 256 * 1024 (256 KB)

    // --- System prompt ---

    const char          *identity;           // default: NULL ("You are a helpful assistant.")
    const char          *instructions;       // default: NULL
    const char         **bootstrap_files;    // default: NULL (array of file paths)
    size_t               bootstrap_count;    // default: 0
    int                  inject_memory;      // default: 0 (1 = auto-enrich from memory)
    int                  inject_datetime;    // default: 1 (1 = inject current date/time)
    const char          *session_info;       // default: NULL

    // --- Tools ---

    adam_tool_def_t     *tools;              // default: NULL
    size_t               tool_count;         // default: 0
    size_t               tool_capacity;      // default: 0 (internal)

    // --- Retry policy ---

    int                  retry_max;          // default: 3
    int                  retry_backoff_ms[4];// default: {1000, 2000, 5000, 10000}
    int                  retry_rate_limit_ms;// default: 30000 (30s cooldown on 429)

    // --- Rate limiting (proactive) ---

    int                  rate_requests_per_min; // default: 0 (unlimited)
    int                  rate_tokens_per_min;   // default: 0 (unlimited)

    // --- Streaming ---

    adam_stream_fn       on_stream;          // default: NULL (blocking mode)
    void                *stream_ctx;         // default: NULL

    // --- Callbacks ---

    void (*on_response)(void *ctx, const char *text, size_t len);
    void (*on_tool_call)(void *ctx, const char *name, const char *args);
    void (*on_error)(void *ctx, adam_status_t code, const char *msg);
    void                *callback_ctx;       // default: NULL

    // --- Logging ---

    adam_log_fn          log_fn;             // default: NULL (no logging)
    void                *log_ctx;            // default: NULL
    adam_log_level_t     log_level;          // default: ADAM_LOG_WARN

    // --- Memory & sessions ---

    adam_memory_t       *memory;             // default: NULL (disabled)
    adam_session_t      *sessions;           // default: NULL (disabled)
    const char          *memory_context;     // default: "default"

    // --- Cancellation ---

    volatile int         abort_flag;         // set to 1 from any thread to stop

    // --- Custom LLM backend (highest priority — overrides everything) ---

    adam_llm_fn          llm_fn;             // default: NULL
    void                *llm_ctx;            // default: NULL

    // --- WASM / custom HTTP ---

    adam_http_fn         http_fn;            // default: NULL (use libcurl)
    void                *http_ctx;           // default: NULL

    // --- Internal state (managed by the library — do not touch) ---

    void                *_local_ctx;         // llama.cpp context
    int                  _rate_req_count;    // requests in current window
    int                  _rate_tok_count;    // tokens in current window
    int64_t              _rate_window_start; // timestamp of window start (ms)
};

// ============================================================================
// MARK: - Global Lifecycle
// ============================================================================

// Call once at program start. Initializes libcurl, SQLite, llama.cpp backend.
// Returns ADAM_OK on success.
adam_status_t   adam_init(void);

// Call once at program exit. Cleans up all global state.
void            adam_cleanup(void);

// ============================================================================
// MARK: - Settings
// ============================================================================

// Create a settings struct with all defaults filled in.
// The returned pointer must be freed with adam_settings_destroy().
adam_settings_t *adam_create_settings(void);

// Destroy a settings struct and all owned resources.
void            adam_settings_destroy(adam_settings_t *settings);

// --- Convenience setters (validate and copy strings) ---

// Remote API provider.
adam_status_t   adam_settings_set_provider(adam_settings_t *s,
                    adam_api_format_t format, const char *api_key,
                    const char *model);

// Override the default API endpoint URL.
adam_status_t   adam_settings_set_base_url(adam_settings_t *s, const char *url);

#ifndef ADAM_NO_LOCAL
// Local GGUF model via llama.cpp. Overrides remote API when set.
adam_status_t   adam_settings_set_local(adam_settings_t *s,
                    const char *gguf_path, int gpu_layers, int ctx_size);
#endif

adam_status_t   adam_settings_set_identity(adam_settings_t *s, const char *text);
adam_status_t   adam_settings_set_instructions(adam_settings_t *s, const char *text);
adam_status_t   adam_settings_add_bootstrap_file(adam_settings_t *s, const char *path);

adam_status_t   adam_settings_add_tool(adam_settings_t *s, adam_tool_def_t tool);
adam_status_t   adam_settings_remove_tool(adam_settings_t *s, const char *name);

adam_status_t   adam_settings_set_stream(adam_settings_t *s,
                    adam_stream_fn fn, void *ctx);

adam_status_t   adam_settings_set_logger(adam_settings_t *s,
                    adam_log_fn fn, void *ctx, adam_log_level_t level);

adam_status_t   adam_settings_set_callbacks(adam_settings_t *s,
                    void (*on_response)(void *, const char *, size_t),
                    void (*on_tool_call)(void *, const char *, const char *),
                    void (*on_error)(void *, adam_status_t, const char *),
                    void *ctx);

adam_status_t   adam_settings_set_llm_callback(adam_settings_t *s,
                    adam_llm_fn fn, void *ctx);

adam_status_t   adam_settings_set_http_callback(adam_settings_t *s,
                    adam_http_fn fn, void *ctx);

// ============================================================================
// MARK: - History
// ============================================================================

adam_history_t *adam_history_create(void);
void            adam_history_destroy(adam_history_t *h);
void            adam_history_clear(adam_history_t *h);
size_t          adam_history_count(const adam_history_t *h);

// Append a message. Strings are copied (strdup'd).
adam_status_t   adam_history_append_user(adam_history_t *h,
                    const char *content);
adam_status_t   adam_history_append_assistant(adam_history_t *h,
                    const char *content,
                    const adam_tool_call_t *tool_calls, size_t tool_call_count);
adam_status_t   adam_history_append_tool(adam_history_t *h,
                    const char *content, const char *tool_call_id);

// Multimodal: attach media to the last message.
adam_status_t   adam_history_attach(adam_history_t *h,
                    adam_media_type_t type,
                    const uint8_t *data, size_t len,
                    const char *filename);
adam_status_t   adam_history_attach_file(adam_history_t *h,
                    const char *file_path);

// Token estimation.
size_t          adam_history_estimate_tokens(const adam_history_t *h);

// Smart compression: use the LLM to summarize old messages.
adam_status_t   adam_history_summarize(adam_settings_t *s, adam_history_t *h,
                    size_t target_token_count);

// ============================================================================
// MARK: - Agent Run (single-shot, blocking)
// ============================================================================

// Run the agent loop: send user_message, iterate tool calls until the LLM
// produces a final text response or max_iterations is reached.
//
// history is updated in-place with all messages exchanged.
// Returns a result struct — caller must call adam_run_result_free().
adam_run_result_t adam_run(adam_settings_t *settings,
                           adam_history_t *history,
                           const char *user_message);

void            adam_run_result_free(adam_run_result_t *result);

// ============================================================================
// MARK: - Cancellation
// ============================================================================

// Thread-safe: can be called from any thread or signal handler.
void            adam_abort(adam_settings_t *s);
void            adam_abort_reset(adam_settings_t *s);

// ============================================================================
// MARK: - Thread Pool
// ============================================================================

#ifndef ADAM_NO_PTHREADS

// A thread pool for running multiple agents concurrently.
// Each worker thread has its own arena. Jobs are queued and
// dispatched to the first available worker.

// Job completion callback — called from the worker thread.
typedef void (*adam_job_done_fn)(void *ctx, adam_run_result_t result);

typedef struct {
    adam_settings_t     *settings;       // agent settings (not shared — one per job)
    adam_history_t      *history;        // conversation history (not shared)
    const char          *user_message;   // input message
    adam_job_done_fn     on_done;        // completion callback
    void                *on_done_ctx;    // callback context
} adam_job_t;

// Create a thread pool with `num_workers` threads.
// If num_workers == 0, defaults to the number of CPU cores.
adam_pool_t    *adam_pool_create(int num_workers);

// Submit a job to the pool. The job struct is copied internally.
// Returns ADAM_OK if the job was queued successfully.
adam_status_t   adam_pool_submit(adam_pool_t *pool, adam_job_t job);

// Wait for all queued jobs to complete, then destroy the pool.
// Blocks until all workers have finished.
void            adam_pool_destroy(adam_pool_t *pool);

// Query pool status.
int             adam_pool_pending(const adam_pool_t *pool);
int             adam_pool_active(const adam_pool_t *pool);

#endif // ADAM_NO_PTHREADS

// ============================================================================
// MARK: - Memory System (SQLite + sqlite-memory + sqlite-vector)
// ============================================================================

#ifndef ADAM_NO_SQLITE

typedef struct {
    const char          *content;        // arena-owned
    const char          *source;         // arena-owned
    const char          *context;        // arena-owned
    float                score;          // relevance score (0.0 - 1.0)
} adam_memory_result_t;

adam_memory_t  *adam_memory_open(const char *db_path);
void            adam_memory_close(adam_memory_t *mem);

// Ingest knowledge.
adam_status_t   adam_memory_add(adam_memory_t *mem, const char *text,
                    const char *context, const char *source);
adam_status_t   adam_memory_add_file(adam_memory_t *mem, const char *path,
                    const char *context);
adam_status_t   adam_memory_add_directory(adam_memory_t *mem,
                    const char *dir_path, const char *context);

// Hybrid search (BM25 + vector similarity).
// Results are arena-allocated — valid until arena_reset/destroy.
adam_status_t   adam_memory_search(adam_memory_t *mem, arena_t *arena,
                    const char *query, int limit,
                    adam_memory_result_t **results, size_t *count);

// Lifecycle.
adam_status_t   adam_memory_delete_context(adam_memory_t *mem, const char *ctx);
adam_status_t   adam_memory_clear(adam_memory_t *mem);

// Multi-agent sync via ATTACH DATABASE.
adam_status_t   adam_memory_attach(adam_memory_t *mem,
                    const char *remote_db_path, const char *alias);
adam_status_t   adam_memory_sync(adam_memory_t *mem,
                    const char *alias, const char *context);

// ============================================================================
// MARK: - Session Persistence (SQLite-backed)
// ============================================================================

adam_session_t *adam_session_open(const char *db_path);
void            adam_session_close(adam_session_t *sess);

adam_status_t   adam_session_save(adam_session_t *sess, const char *session_id,
                    const adam_history_t *h);
adam_status_t   adam_session_load(adam_session_t *sess, const char *session_id,
                    adam_history_t *h);
adam_status_t   adam_session_list(adam_session_t *sess,
                    char ***ids, size_t *count);
adam_status_t   adam_session_delete(adam_session_t *sess, const char *session_id);

#endif // ADAM_NO_SQLITE

// ============================================================================
// MARK: - Model Registry
// ============================================================================

typedef struct {
    const char          *model_prefix;      // e.g. "claude-sonnet-4"
    int                  context_window;    // e.g. 200000
    float                input_cost_mtok;   // $ per million input tokens
    float                output_cost_mtok;  // $ per million output tokens
} adam_model_info_t;

// Register a custom model (extends the built-in table).
adam_status_t   adam_model_register(const adam_model_info_t *info);

// Lookup by longest prefix match. Returns NULL if not found.
const adam_model_info_t *adam_model_lookup(const char *model);

// Convenience.
int             adam_model_context_window(const char *model);
float           adam_model_estimate_cost(const char *model,
                    int input_tokens, int output_tokens);

// ============================================================================
// MARK: - Token Estimation
// ============================================================================

// Rough heuristic: ~4 characters per token (good enough for budgeting).
size_t          adam_estimate_tokens(const char *text, size_t len);

// ============================================================================
// MARK: - Built-in Tools
// ============================================================================

// These match the adam_tool_fn signature and can be passed directly
// to adam_settings_add_tool().

#ifndef ADAM_NO_CURL
// Fetch a URL. args: {"url":"...","method":"GET"} (method optional).
adam_tool_result_t adam_tool_web_fetch(arena_t *arena, void *ctx,
                                       const char *args_json, size_t args_len);
#endif

#ifndef ADAM_NO_SQLITE
// Search long-term memory. args: {"query":"...","limit":5}.
adam_tool_result_t adam_tool_memory_search(arena_t *arena, void *ctx,
                                           const char *args_json, size_t args_len);

// Add to long-term memory. args: {"text":"...","context":"...","source":"..."}.
adam_tool_result_t adam_tool_memory_add(arena_t *arena, void *ctx,
                                        const char *args_json, size_t args_len);
#endif

#ifdef __cplusplus
}
#endif
