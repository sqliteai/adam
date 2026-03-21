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

#define UNUSED_PARAM(p) ((void)(p))

#define ADAM_VERSION_MAJOR       0
#define ADAM_VERSION_MINOR       7
#define ADAM_VERSION_PATCH       0
#define ADAM_VERSION_STRING      "0.7.0"

// ============================================================================
// MARK: - Build Configuration
// ============================================================================

// Define these before including adam.h to disable features:
//   ADAM_NO_CURL       — no libcurl (no HTTP, must provide http_fn callback)
//   ADAM_NO_LOCAL      — no llama.cpp (no local GGUF inference)
//   ADAM_NO_PTHREADS   — no pthreads (no thread pool, no voice thread)
//   ADAM_NO_SQLITE     — no SQLite (no memory, no sessions)
//   ADAM_NO_VOICE      — no voice subsystem
//   ADAM_NO_FILESYSTEM — no file_read/file_write/list_directory tools
//   ADAM_NO_SHELL      — no shell_exec tool

// ============================================================================
// MARK: - Forward Declarations
// ============================================================================

typedef struct adam_settings_t   adam_settings_t;
typedef struct adam_agent_t      adam_agent_t;
typedef struct adam_pool_t       adam_pool_t;
typedef struct adam_memory_t     adam_memory_t;
typedef struct adam_voice_t      adam_voice_t;
typedef struct adam_cache_t      adam_cache_t;

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
    ADAM_ERR_VOICE               = 15,  // voice subsystem error (STT/TTS)
    ADAM_ERR_NOT_IMPLEMENTED     = 16,  // feature stub (not yet implemented)
    ADAM_ERR_TIMEOUT             = 17,  // wall-clock timeout exceeded
    ADAM_ERR_GUARDRAIL           = 18,  // blocked by guardrail callback
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
    ADAM_API_GEMINI              = 3,   // Google Gemini API format
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
// MARK: - Voice (STT / TTS)
// ============================================================================

#if !defined(ADAM_NO_VOICE) && !defined(ADAM_NO_PTHREADS)

// --- STT/TTS backend selection ---

typedef enum {
    ADAM_STT_NONE               = 0,   // disabled
    ADAM_STT_CLOUD              = 1,   // cloud API via libcurl (OpenAI Whisper, etc.)
    ADAM_STT_LOCAL              = 2,   // local engine (not yet implemented)
} adam_stt_backend_t;

typedef enum {
    ADAM_TTS_NONE               = 0,   // disabled
    ADAM_TTS_CLOUD              = 1,   // cloud API (OpenAI TTS, ElevenLabs, etc.)
    ADAM_TTS_LOCAL              = 2,   // local engine (not yet implemented)
    ADAM_TTS_SYSTEM             = 3,   // OS built-in TTS (always available, no deps):
                                       //   macOS/iOS: AVSpeechSynthesizer
                                       //   Android:   android.speech.tts.TextToSpeech
                                       //   Linux:     espeak-ng / spd-say
                                       //   Windows:   SAPI (System.Speech)
                                       //   WASM:      window.speechSynthesis (via tts_fn)
} adam_tts_backend_t;

// --- Audio format ---

typedef enum {
    ADAM_AUDIO_PCM16            = 0,   // raw PCM 16-bit signed, little-endian
    ADAM_AUDIO_WAV              = 1,   // WAV container
    ADAM_AUDIO_MP3              = 2,   // MP3
    ADAM_AUDIO_OGG_OPUS         = 3,   // OGG/Opus
    ADAM_AUDIO_FLAC             = 4,   // FLAC
} adam_audio_format_t;

// --- STT callback (custom backend) ---
// Transcribe audio to text. All output strings must be arena-allocated.
typedef adam_status_t (*adam_stt_fn)(
    void                *ctx,
    arena_t             *arena,
    const uint8_t       *audio_data,
    size_t               audio_len,
    adam_audio_format_t  format,
    int                  sample_rate,       // e.g. 16000
    const char          *language,          // e.g. "en" or NULL for auto
    const char         **out_text           // arena-owned result
);

// --- TTS callback (custom backend) ---
// Synthesize text to audio. Output buffer must be arena-allocated.
typedef adam_status_t (*adam_tts_fn)(
    void                *ctx,
    arena_t             *arena,
    const char          *text,
    const char          *voice,             // voice ID or name
    adam_audio_format_t  out_format,         // desired output format
    uint8_t            **out_audio,          // arena-owned result
    size_t              *out_len
);

// --- Voice event callback ---
// Called from the voice thread when a voice command is transcribed.
// The transcribed text is then fed into the main agent loop.
// Return 0 to accept the command, non-zero to discard it.
typedef int (*adam_voice_event_fn)(
    void                *ctx,
    const char          *transcribed_text,   // what the user said
    float                confidence          // 0.0 - 1.0
);

// --- Audio playback callback ---
// Called to play synthesized audio. The embedder handles platform-specific
// playback (CoreAudio, ALSA, etc.). If NULL, adam uses a default player
// (afplay on macOS, aplay on Linux).
typedef adam_status_t (*adam_audio_play_fn)(
    void                *ctx,
    const uint8_t       *audio_data,
    size_t               audio_len,
    adam_audio_format_t  format
);

#endif // !ADAM_NO_VOICE && !ADAM_NO_PTHREADS

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
    int                  http_status;       // HTTP status code (0 for non-HTTP errors)
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
// MARK: - Guardrails
// ============================================================================

// Called before sending messages to the LLM.
// Return 0 to allow, non-zero to deny (adam_run returns ADAM_ERR_GUARDRAIL).
typedef int (*adam_guardrail_fn)(void *ctx, const adam_message_t *msgs,
                                  size_t msg_count);

// Called after receiving an LLM response, before processing.
// Return 0 to allow, non-zero to deny.
typedef int (*adam_guardrail_response_fn)(void *ctx, const char *content,
                                           const adam_tool_call_t *tool_calls,
                                           size_t tool_call_count);

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

    // Structured error context (populated on failure):
    int                  http_status;       // HTTP status code (0 if not an HTTP error)
} adam_run_result_t;

// ============================================================================
// MARK: - Settings (all fields have defaults)
// ============================================================================

// Created by adam_create_settings(). All fields are initialized to sensible
// defaults. Modify only what you need, then pass to adam_run() or adam_agent_create().
//
// Defaults are documented inline. Fields marked [required] must be set
// before calling adam_run().
//
// THREAD SAFETY: A settings struct is NOT thread-safe. It contains mutable
// internal state (_local_ctx, _curl_*, _rate_*) that is modified during
// adam_run(). Each thread or concurrent agent MUST use its own settings
// instance. The thread pool (adam_pool_t) enforces this by requiring a
// separate settings per job. Sharing a settings struct across threads
// will cause data races and undefined behavior.

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
    const char          *mmproj_path;       // default: NULL (multimodal projector GGUF for vision)
    int                  local_gpu_layers;   // default: -1 (all layers on GPU)
    int                  local_ctx_size;     // default: 0 (use model default)
    int                  local_batch_size;   // default: 512
    int                  local_verbose;      // default: 0 (1 = show llama.cpp/ggml log output)
#endif

    // --- Generation parameters ---

    float                temperature;        // default: 0.7
    int                  max_tokens;         // default: 4096
    const char          *response_format;    // default: NULL (text). "json" for JSON mode
    float                top_p;              // default: 1.0

    // --- Agent loop settings ---

    int                  max_iterations;     // default: 25 (max tool-call iterations)
    int                  max_history;        // default: 100 (max messages before compression)
    int                  summarize_threshold;// default: 0 (0 = auto: 75% of context window)
    int                  timeout_ms;        // default: 0 (0 = no timeout, ms wall-clock)

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

    // --- Filesystem sandbox ---
    // Tools that access the filesystem (file_read, file_write, list_directory,
    // shell_exec) are restricted to these directories. Paths are resolved
    // with realpath() to prevent symlink/.. escapes.

    const char         **allowed_dirs;       // default: NULL (no filesystem access)
    size_t               allowed_dir_count;  // default: 0
    size_t               _allowed_dir_cap;   // internal

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
    const char          *memory_context;     // default: NULL (search all contexts)

    // --- Memory configuration (embedding & search) ---

    const char          *memory_embedding_provider; // default: "local"
    const char          *memory_embedding_model;    // default: NULL (must set to enable)
    const char          *memory_api_key;            // default: NULL (falls back to api_key)
    const char          *memory_dir;                // default: NULL (markdown output dir)
    int                  memory_markdown_output;    // default: 0 (1 = write .md files)
    int                  memory_max_results;        // default: 5 (enrichment results)
    float                memory_min_score;          // default: 0.0 (0 = sqlite-memory default)

    // --- Session auto-save ---

    const char          *session_id;            // default: NULL (no auto-save)
    int                  auto_save;             // default: 1 (save after each adam_run when session_id set)

    // --- Memory extraction ---

    int                  memory_extract;        // default: 0 (1 = extract facts after conversation)

    // --- Voice (STT / TTS) ---

#if !defined(ADAM_NO_VOICE) && !defined(ADAM_NO_PTHREADS)
    int                  voice_enabled;      // default: 0 (1 = start voice thread)

    // STT settings
    adam_stt_backend_t   stt_backend;        // default: ADAM_STT_NONE
    const char          *stt_api_url;        // default: NULL (uses provider default)
    const char          *stt_api_key;        // default: NULL (falls back to api_key)
    const char          *stt_model;          // default: "gpt-4o-mini-transcribe"
    const char          *stt_language;       // default: NULL (auto-detect)
    int                  stt_sample_rate;    // default: 16000
    adam_stt_fn          stt_fn;             // default: NULL (use built-in cloud/local)
    void                *stt_ctx;            // default: NULL

    // TTS settings
    adam_tts_backend_t   tts_backend;        // default: ADAM_TTS_NONE
    const char          *tts_api_url;        // default: NULL (uses provider default)
    const char          *tts_api_key;        // default: NULL (falls back to api_key)
    const char          *tts_model;          // default: "gpt-4o-mini-tts"
    const char          *tts_voice;          // default: "coral"
    adam_audio_format_t  tts_format;         // default: ADAM_AUDIO_MP3
    adam_tts_fn          tts_fn;             // default: NULL (use built-in cloud/local)
    void                *tts_ctx;            // default: NULL

    // Voice event callback (called when speech is transcribed)
    adam_voice_event_fn  on_voice;           // default: NULL (accept all)
    void                *voice_ctx;          // default: NULL

    // Audio playback
    adam_audio_play_fn   audio_play_fn;      // default: NULL (use system player)
    void                *audio_play_ctx;     // default: NULL

    // Silence detection
    float                voice_silence_sec;  // default: 1.0 (seconds of silence to trigger STT)
    float                voice_energy_threshold; // default: 0.02 (energy level to detect speech)
#endif

    // --- Guardrails ---

    adam_guardrail_fn             on_before_send;     // default: NULL
    void                         *before_send_ctx;    // default: NULL
    adam_guardrail_response_fn    on_after_receive;    // default: NULL
    void                         *after_receive_ctx;   // default: NULL

    // --- Response cache ---

    adam_cache_t        *cache;              // default: NULL (no caching)

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
    void                *_mtmd_ctx;          // multimodal context (mtmd)
    int                  _rate_req_count;    // requests in current window
    int                  _rate_tok_count;    // tokens in current window
    int64_t              _rate_window_start; // timestamp of window start (ms)
    void                *_curl_llm;          // persistent CURL* for LLM calls
    void                *_curl_stt;          // persistent CURL* for STT calls
    void                *_curl_tts;          // persistent CURL* for TTS calls
#if !defined(ADAM_NO_VOICE) && !defined(ADAM_NO_PTHREADS)
    adam_voice_t        *_voice;             // voice thread state
#endif
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

// Set multimodal projector for local vision models (e.g. Gemma-3, LLaVA).
// The mmproj_path points to a GGUF file containing the vision encoder.
adam_status_t   adam_settings_set_mmproj(adam_settings_t *s,
                    const char *mmproj_path);
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

// Filesystem sandbox: grant access to a directory for file/shell tools.
// The path is resolved with realpath(). Can be called multiple times.
adam_status_t   adam_settings_allow_dir(adam_settings_t *s, const char *dir_path);

// Memory embedding configuration.
adam_status_t   adam_settings_set_memory(adam_settings_t *s,
                    const char *embedding_provider,
                    const char *embedding_model,
                    const char *api_key);

#if !defined(ADAM_NO_VOICE) && !defined(ADAM_NO_PTHREADS)
// Enable voice and configure STT backend.
adam_status_t   adam_settings_set_stt(adam_settings_t *s,
                    adam_stt_backend_t backend,
                    const char *api_url,       // NULL = provider default
                    const char *api_key,       // NULL = falls back to s->api_key
                    const char *model);        // NULL = "whisper-1"

// Enable voice and configure TTS backend.
adam_status_t   adam_settings_set_tts(adam_settings_t *s,
                    adam_tts_backend_t backend,
                    const char *api_url,       // NULL = provider default
                    const char *api_key,       // NULL = falls back to s->api_key
                    const char *model,         // NULL = "tts-1"
                    const char *voice);        // NULL = "alloy"

// Custom STT/TTS backends (override built-in cloud/local).
adam_status_t   adam_settings_set_stt_callback(adam_settings_t *s,
                    adam_stt_fn fn, void *ctx);
adam_status_t   adam_settings_set_tts_callback(adam_settings_t *s,
                    adam_tts_fn fn, void *ctx);

// Voice event filter (called when speech is transcribed).
adam_status_t   adam_settings_set_voice_callback(adam_settings_t *s,
                    adam_voice_event_fn fn, void *ctx);
#endif

// ============================================================================
// MARK: - History
// ============================================================================

adam_history_t *adam_history_create(void);
void            adam_history_destroy(adam_history_t *h);
void            adam_history_clear(adam_history_t *h);
size_t          adam_history_count(const adam_history_t *h);
adam_history_t *adam_history_clone(const adam_history_t *h);

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

// --- Structured JSON output ---

typedef struct {
    adam_run_result_t   base;           // embedded base result
    int                 json_valid;     // 1 if response was valid JSON
    int                 retries_used;   // retries needed for valid JSON
} adam_json_result_t;

// Run the agent and validate that the response is valid JSON.
// Retries up to max_retries times if the response isn't valid JSON.
// json_hint is appended to the prompt to guide JSON structure (can be NULL).
adam_json_result_t adam_run_json(adam_settings_t *s, adam_history_t *h,
                                 const char *user_message,
                                 const char *json_hint,
                                 int max_retries);

void            adam_json_result_free(adam_json_result_t *r);

// Extract a string value for a top-level key from a JSON string.
// Returns a malloc'd string or NULL if key not found. Caller must free().
char           *adam_json_extract(const char *json, const char *key);

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
// MARK: - Voice Runtime
// ============================================================================

#if !defined(ADAM_NO_VOICE) && !defined(ADAM_NO_PTHREADS)

// Start the voice thread. Listens for audio input, transcribes via STT,
// and feeds commands into the agent loop. The agent's on_response text
// is automatically sent to TTS if tts_backend is configured.
//
// Requires voice_enabled=1 and at least stt_backend set in settings.
// The voice thread runs until adam_voice_stop() is called.
adam_status_t   adam_voice_start(adam_settings_t *s, adam_history_t *h);

// Stop the voice thread and wait for it to finish.
void            adam_voice_stop(adam_settings_t *s);

// Check if the voice thread is currently running.
int             adam_voice_is_running(const adam_settings_t *s);

// One-shot STT: transcribe audio data to text.
// Useful for processing pre-recorded audio without the voice thread.
// Result is arena-allocated.
adam_status_t   adam_stt_transcribe(adam_settings_t *s, arena_t *arena,
                    const uint8_t *audio, size_t audio_len,
                    adam_audio_format_t format,
                    const char **out_text);

// One-shot TTS: synthesize text to audio.
// Result is arena-allocated.
adam_status_t   adam_tts_synthesize(adam_settings_t *s, arena_t *arena,
                    const char *text,
                    uint8_t **out_audio, size_t *out_len);

// Speak text aloud: synthesize + play via audio_play_fn or system player.
adam_status_t   adam_tts_speak(adam_settings_t *s, const char *text);

// Play raw audio data via audio_play_fn or system player.
adam_status_t   adam_audio_play(adam_settings_t *s,
                    const uint8_t *audio, size_t len,
                    adam_audio_format_t format);

// Full voice turn: transcribe audio → run agent → speak response.
// Convenience function that chains STT → adam_run → TTS → play.
adam_run_result_t adam_voice_run(adam_settings_t *s, adam_history_t *h,
                    const uint8_t *audio, size_t audio_len,
                    adam_audio_format_t format);

#endif // !ADAM_NO_VOICE && !ADAM_NO_PTHREADS

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

// Apply embedding settings from adam_settings_t. Call after open, before add/search.
adam_status_t   adam_memory_configure(adam_memory_t *mem, const adam_settings_t *s);

// Ingest knowledge.
adam_status_t   adam_memory_add(adam_memory_t *mem, const char *text,
                    const char *context, const char *source);
adam_status_t   adam_memory_add_file(adam_memory_t *mem, const char *path,
                    const char *context);
adam_status_t   adam_memory_add_directory(adam_memory_t *mem,
                    const char *dir_path, const char *context);

// Hybrid search (BM25 + vector similarity).
// context: NULL = search all, non-NULL = filter by context label.
// Results are arena-allocated — valid until arena_reset/destroy.
adam_status_t   adam_memory_search(adam_memory_t *mem, arena_t *arena,
                    const char *query, const char *context, int limit,
                    adam_memory_result_t **results, size_t *count);

// Lifecycle.
adam_status_t   adam_memory_delete_context(adam_memory_t *mem, const char *ctx);
adam_status_t   adam_memory_clear(adam_memory_t *mem);

// Multi-agent sync.
adam_status_t   adam_memory_sync(adam_memory_t *mem,
                    const char *alias, const char *context);

// ============================================================================
// MARK: - Session Persistence (stored in the memory database)
// ============================================================================

// Create a new session with a UUIDv7 primary key.
// Returns the session ID in out_id (must be at least 37 bytes).
adam_status_t   adam_session_create(adam_memory_t *mem, char *out_id, size_t out_id_size);

adam_status_t   adam_session_save(adam_memory_t *mem, const char *session_id,
                    const adam_history_t *h);
adam_status_t   adam_session_load(adam_memory_t *mem, const char *session_id,
                    adam_history_t *h);
adam_status_t   adam_session_list(adam_memory_t *mem,
                    char ***ids, size_t *count);
adam_status_t   adam_session_delete(adam_memory_t *mem, const char *session_id);

// Mark/unmark a session for sync (sync=1 means include in sync).
adam_status_t   adam_session_set_sync(adam_memory_t *mem, const char *session_id,
                    int sync);

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
// MARK: - Response Cache
// ============================================================================

// LRU cache for LLM responses. Keyed on model + message history.
adam_cache_t   *adam_cache_create(size_t max_entries);
void            adam_cache_destroy(adam_cache_t *c);
void            adam_cache_clear(adam_cache_t *c);
size_t          adam_cache_count(const adam_cache_t *c);
size_t          adam_cache_hits(const adam_cache_t *c);
size_t          adam_cache_misses(const adam_cache_t *c);

// ============================================================================
// MARK: - Token Estimation
// ============================================================================

// Estimate token count. Uses the local model's tokenizer if loaded,
// otherwise falls back to ~4 characters per token heuristic.
size_t          adam_estimate_tokens(const char *text, size_t len);

// Estimate tokens using the local model's actual tokenizer.
// Returns 0 if no local model is loaded. Available when !ADAM_NO_LOCAL.
size_t          adam_estimate_tokens_local(const adam_settings_t *s,
                    const char *text, size_t len);

// ============================================================================
// MARK: - Evolution Loop (self-improving agent)
// ============================================================================

// Stop reason: why the evolution loop terminated.
typedef enum {
    ADAM_EVOLVE_STOP_MAX_ITERS   = 0,   // hit max_iterations
    ADAM_EVOLVE_STOP_SCORE       = 1,   // reached target_score
    ADAM_EVOLVE_STOP_PLATEAU     = 2,   // no improvement for plateau_iters
    ADAM_EVOLVE_STOP_ABORTED     = 3,   // abort_flag was set
    ADAM_EVOLVE_STOP_ERROR       = 4,   // LLM or eval error
} adam_evolve_stop_t;

#define ADAM_EVOLVE_MAX_ATTEMPTS 10

// A single attempt in the evolution history.
typedef struct {
    char               *output;         // malloc'd: the attempt's output
    int                 score;          // 0-100
    int                 iteration;      // which iteration produced this
} adam_evolve_attempt_t;

// Evaluation callback: scores an attempt.
// Return 0-100 for a valid score, or -1 to signal error (stops the loop).
typedef int (*adam_evolve_eval_fn)(void *ctx, const char *output, int iteration);

// Progress callback: called after each iteration (optional).
typedef void (*adam_evolve_progress_fn)(void *ctx, int iteration, int score,
                                        int best_score, const char *output);

// Configuration for the evolution loop.
typedef struct {
    const char                *task;            // [required] the immutable goal
    const char                *initial_strategy;// starting approach (or NULL)
    const char                *metrics;         // scoring criteria description (or NULL)

    int                        max_iterations;  // default: 10
    int                        target_score;    // default: 95 (stop if score >= this)
    int                        plateau_iters;   // default: 3 (stop after N iters w/o improvement)

    adam_evolve_eval_fn        eval_fn;         // [required]
    void                      *eval_ctx;
    adam_evolve_progress_fn    progress_fn;     // optional
    void                      *progress_ctx;
} adam_evolve_config_t;

// Result of the evolution loop.
typedef struct {
    adam_status_t              status;
    adam_evolve_stop_t         stop_reason;

    // Best attempt
    char                      *best_output;     // malloc'd (or NULL)
    int                        best_score;
    int                        best_iteration;

    // Final evolved state
    char                      *strategy;        // malloc'd: final refined strategy
    char                      *insights;        // malloc'd: accumulated insights

    // Ring buffer of recent attempts
    adam_evolve_attempt_t      attempts[ADAM_EVOLVE_MAX_ATTEMPTS];
    int                        attempt_count;   // stored entries (0..10)
    int                        attempt_total;   // total iterations completed

    // Stats
    int                        total_input_tokens;
    int                        total_output_tokens;
    float                      total_cost_usd;
    double                     elapsed_ms;
} adam_evolve_result_t;

// Create a config with sensible defaults. Caller must set task + eval_fn.
adam_evolve_config_t adam_evolve_config_defaults(void);

// Run the evolution loop. Blocks until a stop condition is reached.
adam_evolve_result_t adam_evolve(adam_settings_t *settings,
                                 const adam_evolve_config_t *config);

// Free all malloc'd strings in the result.
void adam_evolve_result_free(adam_evolve_result_t *result);

// ============================================================================
// MARK: - Research Mode (autonomous information gathering)
// ============================================================================

// Stop reason for the research loop.
typedef enum {
    ADAM_RESEARCH_STOP_COMPLETE  = 0,   // agent declared research complete
    ADAM_RESEARCH_STOP_MAX_ITERS = 1,   // hit max_iterations
    ADAM_RESEARCH_STOP_ABORTED   = 2,   // abort_flag was set
    ADAM_RESEARCH_STOP_ERROR     = 3,   // LLM or tool error
    ADAM_RESEARCH_STOP_CALLBACK  = 4,   // is_complete callback returned 1
} adam_research_stop_t;

// A single finding from the research.
typedef struct {
    char               *content;        // malloc'd: the finding text
    char               *source;         // malloc'd: source/citation (or NULL)
    int                 iteration;      // which iteration found this
} adam_research_finding_t;

// Completeness callback: return 1 if research is done, 0 to continue.
// Called after each iteration with the current report draft.
typedef int (*adam_research_complete_fn)(void *ctx, const char *report,
                                         int iteration);

// Progress callback: called after each iteration.
typedef void (*adam_research_progress_fn)(void *ctx, int iteration,
                                          const char *status);

// Configuration for the research loop.
typedef struct {
    const char                *question;        // [required] the research question
    const char                *instructions;    // additional research instructions (or NULL)
    int                        max_iterations;  // default: 5
    adam_research_complete_fn   is_complete;     // optional (NULL = agent self-assesses)
    void                      *complete_ctx;
    adam_research_progress_fn  on_progress;     // optional
    void                      *progress_ctx;
} adam_research_config_t;

// Result of the research loop.
typedef struct {
    adam_status_t              status;
    adam_research_stop_t       stop_reason;

    // Final report (synthesized by LLM from all findings).
    char                      *report;          // malloc'd (or NULL)

    // Individual findings accumulated across iterations.
    adam_research_finding_t   *findings;         // malloc'd array (or NULL)
    size_t                     finding_count;

    // Stats
    int                        total_iterations;
    int                        total_input_tokens;
    int                        total_output_tokens;
    float                      total_cost_usd;
    double                     elapsed_ms;
} adam_research_result_t;

// Create a config with sensible defaults. Caller must set question.
adam_research_config_t adam_research_config_defaults(void);

// Run the research loop. The agent uses tools from settings to gather
// information, accumulates findings, and produces a synthesized report.
// The conversation history persists across iterations.
adam_research_result_t adam_research(adam_settings_t *settings,
                                     const adam_research_config_t *config);

// Free all malloc'd strings in the result.
void adam_research_result_free(adam_research_result_t *result);

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

// Deep research on a topic. ctx must be adam_settings_t*.
// args: {"question":"...","instructions":"...","max_iterations":5}
// (instructions, max_iterations optional).
adam_tool_result_t adam_tool_research(arena_t *arena, void *ctx,
                                      const char *args_json, size_t args_len);

// --- Multi-agent: invoke a sub-agent as a tool ---

// Context for the sub-agent tool. Embedder allocates and configures this.
typedef struct {
    adam_settings_t    *settings;    // sub-agent's settings (required)
    adam_history_t     *history;     // NULL = fresh each call, non-NULL = persistent
} adam_agent_tool_ctx_t;

// Sub-agent tool. ctx must be adam_agent_tool_ctx_t*.
// args: {"message":"..."}.
adam_tool_result_t adam_tool_agent(arena_t *arena, void *ctx,
                                   const char *args_json, size_t args_len);

// --- Filesystem tools (ctx must be adam_settings_t* for sandbox checks) ---

// Read a file. args: {"path":"...","max_bytes":65536} (max_bytes optional, default 64KB).
adam_tool_result_t adam_tool_file_read(arena_t *arena, void *ctx,
                                       const char *args_json, size_t args_len);

// Write a file. args: {"path":"...","content":"...","append":false} (append optional).
adam_tool_result_t adam_tool_file_write(arena_t *arena, void *ctx,
                                        const char *args_json, size_t args_len);

// List directory contents. args: {"path":"..."}.
adam_tool_result_t adam_tool_list_directory(arena_t *arena, void *ctx,
                                            const char *args_json, size_t args_len);

// --- Shell execution (ctx must be adam_settings_t* for sandbox checks) ---

// Execute a shell command. args: {"command":"...","timeout":30} (timeout in seconds, default 30).
// Working directory is restricted to allowed_dirs. stdout+stderr returned.
adam_tool_result_t adam_tool_shell_exec(arena_t *arena, void *ctx,
                                        const char *args_json, size_t args_len);

// --- Computation ---

// Evaluate a math expression. args: {"expression":"..."}.
// Supports: +, -, *, /, %, ^, (), unary minus. No ctx needed.
adam_tool_result_t adam_tool_calculator(arena_t *arena, void *ctx,
                                        const char *args_json, size_t args_len);

// --- Web search (ctx must be adam_settings_t* for API key access) ---

// Search the web via Brave Search API. args: {"query":"...","count":5}.
// Requires brave_api_key set in settings or BRAVE_API_KEY env var.
adam_tool_result_t adam_tool_web_search(arena_t *arena, void *ctx,
                                        const char *args_json, size_t args_len);

// --- HTTP ---

// POST JSON to an endpoint. args: {"url":"...","body":"...","headers":{"key":"val"}}.
adam_tool_result_t adam_tool_http_post(arena_t *arena, void *ctx,
                                       const char *args_json, size_t args_len);

#ifndef ADAM_NO_SQLITE
// Execute SQL against a database. ctx must be adam_memory_t*.
// args: {"sql":"...","params":["..."]} (params optional).
adam_tool_result_t adam_tool_sql_query(arena_t *arena, void *ctx,
                                       const char *args_json, size_t args_len);
#endif

#ifdef __cplusplus
}
#endif
