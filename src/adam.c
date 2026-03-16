//
//  adam.c
//  Adam — Core agent loop, settings, history, thread pool
//
//  Created by Marco Bambini on 14/03/26.
//

#include "adam.h"
#include "adam_net.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#ifndef ADAM_NO_PTHREADS
#include <pthread.h>
#endif

#if !defined(ADAM_NO_CURL) && !defined(__APPLE__)
#include <curl/curl.h>
#endif

#ifndef ADAM_NO_SQLITE
#include "sqlite3.h"
#endif

// ============================================================================
// MARK: - Internal: Logging Helper
// ============================================================================


#define ADAM_LOG(s, level, ...)                                              \
    do {                                                                    \
        if ((s)->log_fn && (level) <= (s)->log_level) {                     \
            char _buf[512];                                                 \
            int _n = snprintf(_buf, sizeof(_buf), __VA_ARGS__);             \
            if (_n > 0) (s)->log_fn((s)->log_ctx, (level), _buf, (size_t)_n); \
        }                                                                   \
    } while (0)

// ============================================================================
// MARK: - Status Strings
// ============================================================================

static const char *g_status_strings[] = {
    [ADAM_OK]                   = "ok",
    [ADAM_ERR_ALLOC]            = "allocation failed",
    [ADAM_ERR_PROVIDER]         = "provider error",
    [ADAM_ERR_RATE_LIMIT]       = "rate limited",
    [ADAM_ERR_AUTH]              = "authentication error",
    [ADAM_ERR_CONTEXT_OVERFLOW]  = "context overflow",
    [ADAM_ERR_MAX_ITERATIONS]   = "max iterations reached",
    [ADAM_ERR_ABORTED]          = "aborted",
    [ADAM_ERR_CURL]             = "curl error",
    [ADAM_ERR_LOCAL]            = "local inference error",
    [ADAM_ERR_SQLITE]           = "sqlite error",
    [ADAM_ERR_JSON]             = "json error",
    [ADAM_ERR_TOOL_NOT_FOUND]   = "tool not found",
    [ADAM_ERR_INVALID_PARAM]    = "invalid parameter",
    [ADAM_ERR_NO_PROVIDER]      = "no provider configured",
    [ADAM_ERR_VOICE]            = "voice error",
    [ADAM_ERR_NOT_IMPLEMENTED]  = "not yet implemented",
};

const char *adam_status_string(adam_status_t status) {
    if (status < 0 || status > ADAM_ERR_NOT_IMPLEMENTED) return "unknown error";
    return g_status_strings[status];
}

// ============================================================================
// MARK: - Global Lifecycle
// ============================================================================

adam_status_t adam_init(void) {
#if !defined(ADAM_NO_CURL) && !defined(__APPLE__)
    curl_global_init(CURL_GLOBAL_DEFAULT);
#endif
#ifndef ADAM_NO_SQLITE
    sqlite3_initialize();
#endif
    return ADAM_OK;
}

void adam_cleanup(void) {
#if !defined(ADAM_NO_CURL) && !defined(__APPLE__)
    curl_global_cleanup();
#endif
#ifndef ADAM_NO_SQLITE
    sqlite3_shutdown();
#endif
}

// ============================================================================
// MARK: - Settings
// ============================================================================

adam_settings_t *adam_create_settings(void) {
    adam_settings_t *s = calloc(1, sizeof(adam_settings_t));
    if (!s) return NULL;

    // Provider defaults
    s->api_format        = ADAM_API_NONE;
    s->model                = "claude-sonnet-4-20250514";

#ifndef ADAM_NO_LOCAL
    s->local_gpu_layers     = -1;
    s->local_ctx_size       = 0;
    s->local_batch_size     = 512;
#endif

    // Generation defaults
    s->temperature          = 0.7f;
    s->max_tokens           = 4096;
    s->top_p                = 1.0f;
    s->frequency_penalty    = 0.0f;
    s->presence_penalty     = 0.0f;

    // Agent loop defaults
    s->max_iterations       = 25;
    s->max_history          = 100;
    s->summarize_threshold  = 0;    // auto

    // Arena default
    s->arena_block_size     = 256 * 1024;

    // System prompt defaults
    s->inject_datetime      = 1;

    // Retry defaults
    s->retry_max            = 3;
    s->retry_backoff_ms[0]  = 1000;
    s->retry_backoff_ms[1]  = 2000;
    s->retry_backoff_ms[2]  = 5000;
    s->retry_backoff_ms[3]  = 10000;
    s->retry_rate_limit_ms  = 30000;

    // Logging default
    s->log_level            = ADAM_LOG_WARN;

    // Memory default context
    s->memory_context       = "default";

    // Voice defaults
#if !defined(ADAM_NO_VOICE) && !defined(ADAM_NO_PTHREADS)
    s->voice_enabled        = 0;
    s->stt_backend          = ADAM_STT_NONE;
    s->stt_model            = "gpt-4o-mini-transcribe";
    s->stt_sample_rate      = 16000;
    s->tts_backend          = ADAM_TTS_NONE;
    s->tts_model            = "gpt-4o-mini-tts";
    s->tts_voice            = "coral";
    s->tts_format           = ADAM_AUDIO_MP3;
    s->voice_silence_sec    = 1.0f;
    s->voice_energy_threshold = 0.02f;
#endif

    return s;
}

void adam_settings_destroy(adam_settings_t *s) {
    if (!s) return;
    adam_net_cleanup(s);
#ifndef ADAM_NO_LOCAL
    extern void adam_local_cleanup(adam_settings_t *);
    adam_local_cleanup(s);
#endif
    free(s->tools);
    free(s->bootstrap_files);
    free(s);
}

// --- Convenience setters ---

adam_status_t adam_settings_set_provider(adam_settings_t *s,
                                         adam_api_format_t format,
                                         const char *api_key,
                                         const char *model) {
    if (!s) return ADAM_ERR_INVALID_PARAM;
    s->api_format = format;
    s->api_key = api_key;
    if (model) s->model = model;
    return ADAM_OK;
}

adam_status_t adam_settings_set_base_url(adam_settings_t *s, const char *url) {
    if (!s) return ADAM_ERR_INVALID_PARAM;
    s->base_url = url;
    return ADAM_OK;
}

#ifndef ADAM_NO_LOCAL
adam_status_t adam_settings_set_local(adam_settings_t *s,
                                      const char *gguf_path,
                                      int gpu_layers, int ctx_size) {
    if (!s || !gguf_path) return ADAM_ERR_INVALID_PARAM;
    s->gguf_path = gguf_path;
    s->local_gpu_layers = gpu_layers;
    s->local_ctx_size = ctx_size;
    return ADAM_OK;
}
#endif

adam_status_t adam_settings_set_identity(adam_settings_t *s, const char *text) {
    if (!s) return ADAM_ERR_INVALID_PARAM;
    s->identity = text;
    return ADAM_OK;
}

adam_status_t adam_settings_set_instructions(adam_settings_t *s, const char *text) {
    if (!s) return ADAM_ERR_INVALID_PARAM;
    s->instructions = text;
    return ADAM_OK;
}

adam_status_t adam_settings_add_bootstrap_file(adam_settings_t *s, const char *path) {
    if (!s || !path) return ADAM_ERR_INVALID_PARAM;
    size_t new_count = s->bootstrap_count + 1;
    const char **new_arr = realloc(s->bootstrap_files,
                                    new_count * sizeof(const char *));
    if (!new_arr) return ADAM_ERR_ALLOC;
    new_arr[s->bootstrap_count] = path;
    s->bootstrap_files = new_arr;
    s->bootstrap_count = new_count;
    return ADAM_OK;
}

adam_status_t adam_settings_add_tool(adam_settings_t *s, adam_tool_def_t tool) {
    if (!s || !tool.name || !tool.execute) return ADAM_ERR_INVALID_PARAM;
    if (s->tool_count >= s->tool_capacity) {
        size_t new_cap = s->tool_capacity ? s->tool_capacity * 2 : 8;
        adam_tool_def_t *new_arr = realloc(s->tools,
                                            new_cap * sizeof(adam_tool_def_t));
        if (!new_arr) return ADAM_ERR_ALLOC;
        s->tools = new_arr;
        s->tool_capacity = new_cap;
    }
    s->tools[s->tool_count++] = tool;
    return ADAM_OK;
}

adam_status_t adam_settings_remove_tool(adam_settings_t *s, const char *name) {
    if (!s || !name) return ADAM_ERR_INVALID_PARAM;
    for (size_t i = 0; i < s->tool_count; i++) {
        if (strcmp(s->tools[i].name, name) == 0) {
            memmove(&s->tools[i], &s->tools[i + 1],
                    (s->tool_count - i - 1) * sizeof(adam_tool_def_t));
            s->tool_count--;
            return ADAM_OK;
        }
    }
    return ADAM_ERR_TOOL_NOT_FOUND;
}

adam_status_t adam_settings_set_stream(adam_settings_t *s,
                                       adam_stream_fn fn, void *ctx) {
    if (!s) return ADAM_ERR_INVALID_PARAM;
    s->on_stream = fn;
    s->stream_ctx = ctx;
    return ADAM_OK;
}

adam_status_t adam_settings_set_logger(adam_settings_t *s,
                                       adam_log_fn fn, void *ctx,
                                       adam_log_level_t level) {
    if (!s) return ADAM_ERR_INVALID_PARAM;
    s->log_fn = fn;
    s->log_ctx = ctx;
    s->log_level = level;
    return ADAM_OK;
}

adam_status_t adam_settings_set_callbacks(adam_settings_t *s,
                    void (*on_response)(void *, const char *, size_t),
                    void (*on_tool_call)(void *, const char *, const char *),
                    void (*on_error)(void *, adam_status_t, const char *),
                    void *ctx) {
    if (!s) return ADAM_ERR_INVALID_PARAM;
    s->on_response = on_response;
    s->on_tool_call = on_tool_call;
    s->on_error = on_error;
    s->callback_ctx = ctx;
    return ADAM_OK;
}

adam_status_t adam_settings_set_llm_callback(adam_settings_t *s,
                                              adam_llm_fn fn, void *ctx) {
    if (!s) return ADAM_ERR_INVALID_PARAM;
    s->llm_fn = fn;
    s->llm_ctx = ctx;
    return ADAM_OK;
}

adam_status_t adam_settings_set_http_callback(adam_settings_t *s,
                                              adam_http_fn fn, void *ctx) {
    if (!s) return ADAM_ERR_INVALID_PARAM;
    s->http_fn = fn;
    s->http_ctx = ctx;
    return ADAM_OK;
}

// ============================================================================
// MARK: - Voice Settings (setters only — runtime is in adam_voice.c)
// ============================================================================

#if !defined(ADAM_NO_VOICE) && !defined(ADAM_NO_PTHREADS)

adam_status_t adam_settings_set_stt(adam_settings_t *s,
                                    adam_stt_backend_t backend,
                                    const char *api_url,
                                    const char *api_key,
                                    const char *model) {
    if (!s) return ADAM_ERR_INVALID_PARAM;
    s->voice_enabled = 1;
    s->stt_backend = backend;
    if (api_url) s->stt_api_url = api_url;
    if (api_key) s->stt_api_key = api_key;
    if (model)   s->stt_model = model;
    return ADAM_OK;
}

adam_status_t adam_settings_set_tts(adam_settings_t *s,
                                    adam_tts_backend_t backend,
                                    const char *api_url,
                                    const char *api_key,
                                    const char *model,
                                    const char *voice) {
    if (!s) return ADAM_ERR_INVALID_PARAM;
    s->tts_backend = backend;
    if (api_url) s->tts_api_url = api_url;
    if (api_key) s->tts_api_key = api_key;
    if (model)   s->tts_model = model;
    if (voice)   s->tts_voice = voice;
    return ADAM_OK;
}

adam_status_t adam_settings_set_stt_callback(adam_settings_t *s,
                                             adam_stt_fn fn, void *ctx) {
    if (!s) return ADAM_ERR_INVALID_PARAM;
    s->stt_fn = fn;
    s->stt_ctx = ctx;
    return ADAM_OK;
}

adam_status_t adam_settings_set_tts_callback(adam_settings_t *s,
                                             adam_tts_fn fn, void *ctx) {
    if (!s) return ADAM_ERR_INVALID_PARAM;
    s->tts_fn = fn;
    s->tts_ctx = ctx;
    return ADAM_OK;
}

adam_status_t adam_settings_set_voice_callback(adam_settings_t *s,
                                               adam_voice_event_fn fn,
                                               void *ctx) {
    if (!s) return ADAM_ERR_INVALID_PARAM;
    s->on_voice = fn;
    s->voice_ctx = ctx;
    return ADAM_OK;
}

#endif // !ADAM_NO_VOICE && !ADAM_NO_PTHREADS

// ============================================================================
// MARK: - Cancellation
// ============================================================================

void adam_abort(adam_settings_t *s) {
    if (s) s->abort_flag = 1;
}

void adam_abort_reset(adam_settings_t *s) {
    if (s) s->abort_flag = 0;
}

// ============================================================================
// MARK: - Token Estimation
// ============================================================================

size_t adam_estimate_tokens(const char *text, size_t len) {
    if (!text) return 0;
    if (len == 0) len = strlen(text);
    return (len + 3) / 4; // ~4 chars per token heuristic
}

// ============================================================================
// MARK: - History
// ============================================================================

adam_history_t *adam_history_create(void) {
    adam_history_t *h = calloc(1, sizeof(adam_history_t));
    return h;
}

static void adam_message_free(adam_message_t *m) {
    if (!m) return;
    free(m->content);
    free(m->tool_call_id);
    for (size_t i = 0; i < m->tool_call_count; i++) {
        free(m->tool_calls[i].id);
        free(m->tool_calls[i].name);
        free(m->tool_calls[i].arguments_json);
    }
    free(m->tool_calls);
    for (size_t i = 0; i < m->attachment_count; i++) {
        free(m->attachments[i].data);
        free(m->attachments[i].filename);
    }
    free(m->attachments);
}

void adam_history_destroy(adam_history_t *h) {
    if (!h) return;
    for (size_t i = 0; i < h->count; i++)
        adam_message_free(&h->items[i]);
    free(h->items);
    free(h);
}

void adam_history_clear(adam_history_t *h) {
    if (!h) return;
    for (size_t i = 0; i < h->count; i++)
        adam_message_free(&h->items[i]);
    h->count = 0;
}

size_t adam_history_count(const adam_history_t *h) {
    return h ? h->count : 0;
}

static adam_status_t history_grow(adam_history_t *h) {
    if (h->count >= h->capacity) {
        size_t new_cap = h->capacity ? h->capacity * 2 : 32;
        adam_message_t *new_items = realloc(h->items,
                                             new_cap * sizeof(adam_message_t));
        if (!new_items) return ADAM_ERR_ALLOC;
        h->items = new_items;
        h->capacity = new_cap;
    }
    return ADAM_OK;
}

adam_status_t adam_history_append_user(adam_history_t *h, const char *content) {
    if (!h || !content) return ADAM_ERR_INVALID_PARAM;
    adam_status_t rc = history_grow(h);
    if (rc != ADAM_OK) return rc;

    adam_message_t *m = &h->items[h->count++];
    memset(m, 0, sizeof(*m));
    m->role = ADAM_ROLE_USER;
    m->content = strdup(content);
    m->content_len = strlen(content);
    return ADAM_OK;
}

adam_status_t adam_history_append_assistant(adam_history_t *h,
                                            const char *content,
                                            const adam_tool_call_t *tool_calls,
                                            size_t tool_call_count) {
    if (!h) return ADAM_ERR_INVALID_PARAM;
    adam_status_t rc = history_grow(h);
    if (rc != ADAM_OK) return rc;

    adam_message_t *m = &h->items[h->count++];
    memset(m, 0, sizeof(*m));
    m->role = ADAM_ROLE_ASSISTANT;
    m->content = strdup(content ? content : "");
    m->content_len = strlen(m->content);

    if (tool_call_count > 0 && tool_calls) {
        m->tool_calls = malloc(tool_call_count * sizeof(adam_tool_call_entry_t));
        if (!m->tool_calls) return ADAM_ERR_ALLOC;
        m->tool_call_count = tool_call_count;
        for (size_t i = 0; i < tool_call_count; i++) {
            m->tool_calls[i].id =
                tool_calls[i].id ? strdup(tool_calls[i].id) : NULL;
            m->tool_calls[i].name =
                tool_calls[i].name ? strdup(tool_calls[i].name) : NULL;
            m->tool_calls[i].arguments_json =
                tool_calls[i].arguments_json
                    ? strdup(tool_calls[i].arguments_json) : NULL;
        }
    }
    return ADAM_OK;
}

adam_status_t adam_history_append_tool(adam_history_t *h,
                                       const char *content,
                                       const char *tool_call_id) {
    if (!h || !content) return ADAM_ERR_INVALID_PARAM;
    adam_status_t rc = history_grow(h);
    if (rc != ADAM_OK) return rc;

    adam_message_t *m = &h->items[h->count++];
    memset(m, 0, sizeof(*m));
    m->role = ADAM_ROLE_TOOL;
    m->content = strdup(content);
    m->content_len = strlen(content);
    m->tool_call_id = tool_call_id ? strdup(tool_call_id) : NULL;
    return ADAM_OK;
}

adam_status_t adam_history_attach(adam_history_t *h,
                                  adam_media_type_t type,
                                  const uint8_t *data, size_t len,
                                  const char *filename) {
    if (!h || h->count == 0 || !data || len == 0) return ADAM_ERR_INVALID_PARAM;
    adam_message_t *m = &h->items[h->count - 1];

    size_t new_count = m->attachment_count + 1;
    adam_attachment_t *new_arr = realloc(m->attachments,
                                         new_count * sizeof(adam_attachment_t));
    if (!new_arr) return ADAM_ERR_ALLOC;
    m->attachments = new_arr;

    adam_attachment_t *a = &m->attachments[m->attachment_count];
    a->type = type;
    a->data = malloc(len);
    if (!a->data) return ADAM_ERR_ALLOC;
    memcpy(a->data, data, len);
    a->data_len = len;
    a->filename = filename ? strdup(filename) : NULL;
    m->attachment_count = new_count;
    return ADAM_OK;
}

size_t adam_history_estimate_tokens(const adam_history_t *h) {
    if (!h) return 0;
    size_t total = 0;
    for (size_t i = 0; i < h->count; i++) {
        total += adam_estimate_tokens(h->items[i].content,
                                      h->items[i].content_len);
        // Tool calls add some overhead
        total += h->items[i].tool_call_count * 50;
    }
    return total;
}

// ============================================================================
// MARK: - Internal: History Compression
// ============================================================================

static void history_compress(adam_history_t *h) {
    // Keep system message (index 0) + last 50% of remaining messages.
    // Ensures we don't drop tool results without their tool calls.
    if (h->count <= 4) return;
    size_t keep = h->count / 2;
    if (keep < 2) keep = 2;
    size_t drop = h->count - 1 - keep;
    for (size_t i = 1; i <= drop; i++)
        adam_message_free(&h->items[i]);
    memmove(&h->items[1], &h->items[1 + drop],
            keep * sizeof(adam_message_t));
    h->count = 1 + keep;
}

// ============================================================================
// MARK: - Internal: System Prompt
// ============================================================================

static const char *build_system_prompt(arena_t *arena, adam_settings_t *s,
                                        const char *user_message) {
    size_t est = 8192;
    char *buf = arena_alloc(arena, est);
    if (!buf) return NULL;
    size_t pos = 0;

    // Identity
    const char *identity = s->identity ? s->identity
                                       : "You are a helpful assistant.";
    pos += snprintf(buf + pos, est - pos, "%s\n\n", identity);

    // Instructions
    if (s->instructions) {
        pos += snprintf(buf + pos, est - pos, "%s\n\n", s->instructions);
    }

    // Bootstrap files
    for (size_t i = 0; i < s->bootstrap_count; i++) {
        FILE *f = fopen(s->bootstrap_files[i], "r");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        long flen = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (flen > 0) {
            // Grow buffer if needed
            size_t need = (size_t)flen + 256;
            if (pos + need >= est) {
                size_t new_est = est;
                while (new_est < pos + need + 1) new_est *= 2;
                char *new_buf = arena_alloc(arena, new_est);
                if (new_buf) {
                    memcpy(new_buf, buf, pos);
                    buf = new_buf;
                    est = new_est;
                }
            }
            if (pos + need < est) {
                pos += (size_t)snprintf(buf + pos, est - pos, "## %s\n",
                                        s->bootstrap_files[i]);
                size_t read = fread(buf + pos, 1, (size_t)flen, f);
                pos += read;
                buf[pos++] = '\n';
                buf[pos++] = '\n';
            }
        }
        fclose(f);
    }

    // Memory enrichment
    // TODO: implement when adam_memory.c is ready
    // if (s->inject_memory && s->memory && user_message) { ... }
    UNUSED_PARAM(user_message);

    // Date/time injection
    if (s->inject_datetime) {
        time_t now = time(NULL);
        struct tm *tm = localtime(&now);
        if (tm) {
            pos += snprintf(buf + pos, est - pos,
                "Current date: %04d-%02d-%02d %02d:%02d\n\n",
                tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
                tm->tm_hour, tm->tm_min);
        }
    }

    // Session info
    if (s->session_info) {
        pos += snprintf(buf + pos, est - pos, "%s\n", s->session_info);
    }

    buf[pos] = '\0';
    return buf;
}

// ============================================================================
// MARK: - Internal: Rate Limiter
// ============================================================================

static void rate_limit_wait(adam_settings_t *s) {
    if (s->rate_requests_per_min == 0 && s->rate_tokens_per_min == 0)
        return;

    int64_t now = (int64_t)time(NULL);

    // Reset window every 60 seconds
    if (now - s->_rate_window_start >= 60) {
        s->_rate_window_start = now;
        s->_rate_req_count = 0;
        s->_rate_tok_count = 0;
    }

    // Wait if request limit exceeded
    if (s->rate_requests_per_min > 0
        && s->_rate_req_count >= s->rate_requests_per_min) {
        int64_t sleep_sec = 60 - (now - s->_rate_window_start);
        if (sleep_sec > 0) {
            ADAM_LOG(s, ADAM_LOG_INFO,
                     "Rate limit: %d/%d RPM, waiting %llds",
                     s->_rate_req_count, s->rate_requests_per_min,
                     (long long)sleep_sec);
            struct timespec ts = { .tv_sec = (time_t)sleep_sec };
            nanosleep(&ts, NULL);
        }
        s->_rate_window_start = (int64_t)time(NULL);
        s->_rate_req_count = 0;
        s->_rate_tok_count = 0;
    }

    s->_rate_req_count++;
}

static void rate_limit_record(adam_settings_t *s, int tokens) {
    s->_rate_tok_count += tokens;
}

// ============================================================================
// MARK: - Internal: LLM Dispatch (stub — implemented in adam_http.c / adam_local.c)
// ============================================================================

// Forward declarations for provider-specific implementations.
// These will be implemented in adam_http.c and adam_local.c.
// For now, we declare them as weak symbols or stubs.

extern adam_llm_response_t adam_llm_call_http(
    arena_t *arena, adam_settings_t *s,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
);

#ifndef ADAM_NO_LOCAL
extern adam_llm_response_t adam_llm_call_local(
    arena_t *arena, adam_settings_t *s,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
);
#endif

static adam_llm_response_t dispatch_llm(
    arena_t *arena, adam_settings_t *s,
    const adam_message_t *msgs, size_t msg_count
) {
    // Priority 1: direct LLM callback (testing, custom backends)
    if (s->llm_fn) {
        return s->llm_fn(s->llm_ctx, arena, msgs, msg_count,
                          s->tools, s->tool_count);
    }

#ifndef ADAM_NO_LOCAL
    // Priority 2: local inference when gguf_path is set
    if (s->gguf_path) {
        return adam_llm_call_local(arena, s, msgs, msg_count,
                                   s->tools, s->tool_count);
    }
#endif

    UNUSED_PARAM(msgs);
    UNUSED_PARAM(msg_count);

    // WASM / custom HTTP path
    if (s->http_fn) {
        // TODO: build request JSON, call http_fn, parse response
        return (adam_llm_response_t){
            .error = ADAM_ERR_PROVIDER,
            .error_msg = arena_strdup(arena, "http_fn not yet implemented"),
        };
    }

#if !defined(ADAM_NO_CURL) || defined(__APPLE__)
    // adam_llm_call_http uses adam_net which has platform-specific backends:
    // Apple → NSURLSession, Linux/Windows → libcurl
    return adam_llm_call_http(arena, s, msgs, msg_count,
                              s->tools, s->tool_count);
#else
    return (adam_llm_response_t){
        .error = ADAM_ERR_NO_PROVIDER,
        .error_msg = arena_strdup(arena, "no HTTP provider available"),
    };
#endif
}

// ============================================================================
// MARK: - Internal: Retry Wrapper
// ============================================================================

static adam_llm_response_t dispatch_with_retry(
    arena_t *arena, adam_settings_t *s,
    const adam_message_t *msgs, size_t msg_count
) {
    int max_retries = s->retry_max > 0 ? s->retry_max : 3;

    for (int attempt = 0; attempt <= max_retries; attempt++) {
        adam_llm_response_t resp = dispatch_llm(arena, s, msgs, msg_count);

        // Success or non-retriable error
        if (resp.error == ADAM_OK) return resp;
        if (resp.error == ADAM_ERR_AUTH) return resp;
        if (resp.error == ADAM_ERR_CONTEXT_OVERFLOW) return resp;

        // Last attempt — return whatever we got
        if (attempt == max_retries) return resp;

        // Compute delay
        int delay_ms;
        if (resp.error == ADAM_ERR_RATE_LIMIT) {
            delay_ms = s->retry_rate_limit_ms > 0
                       ? s->retry_rate_limit_ms : 30000;
        } else {
            delay_ms = (attempt < 4) ? s->retry_backoff_ms[attempt] : 10000;
        }

        ADAM_LOG(s, ADAM_LOG_WARN,
                 "LLM call failed (%s), retry %d/%d in %dms",
                 resp.error_msg ? resp.error_msg : adam_status_string(resp.error),
                 attempt + 1, max_retries, delay_ms);

        struct timespec ts = {
            .tv_sec = delay_ms / 1000,
            .tv_nsec = (long)(delay_ms % 1000) * 1000000L,
        };
        nanosleep(&ts, NULL);

        // Reset arena before retry (discard previous response)
        arena_reset(arena);
    }

    // Unreachable, but just in case
    return (adam_llm_response_t){
        .error = ADAM_ERR_PROVIDER,
        .error_msg = arena_strdup(arena, "max retries exhausted"),
    };
}

// ============================================================================
// MARK: - Internal: Cost Tracking
// ============================================================================

static float compute_cost(const char *model, int input_tok, int output_tok) {
    const adam_model_info_t *info = adam_model_lookup(model);
    if (!info) return 0.0f;
    return (input_tok / 1e6f) * info->input_cost_mtok
         + (output_tok / 1e6f) * info->output_cost_mtok;
}

// ============================================================================
// MARK: - Agent Run
// ============================================================================

adam_run_result_t adam_run(adam_settings_t *s, adam_history_t *history,
                           const char *user_message) {
    adam_run_result_t result = {0};

    // Validate inputs
    if (!s) {
        result.status = ADAM_ERR_INVALID_PARAM;
        result.final_response = strdup("settings is NULL");
        return result;
    }
    if (!history) {
        result.status = ADAM_ERR_INVALID_PARAM;
        result.final_response = strdup("history is NULL");
        return result;
    }
    int has_remote = (s->api_format != ADAM_API_NONE);
    int has_local = 0;
#ifndef ADAM_NO_LOCAL
    has_local = (s->gguf_path != NULL);
#endif
    if (!has_remote && !has_local && !s->http_fn && !s->llm_fn) {
        result.status = ADAM_ERR_NO_PROVIDER;
        result.final_response = strdup("no provider configured");
        return result;
    }

    // Reset abort flag
    s->abort_flag = 0;

    // Record start time
    struct timespec ts_start;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    ADAM_LOG(s, ADAM_LOG_INFO, "adam_run: starting (model=%s, tools=%zu)",
             s->model, s->tool_count);

    // 1. Append user message to history
    if (user_message && strlen(user_message) > 0) {
        adam_status_t rc = adam_history_append_user(history, user_message);
        if (rc != ADAM_OK) {
            result.status = rc;
            result.final_response = strdup("failed to append user message");
            return result;
        }
    }

    // 2. Create per-turn arena
    arena_t *turn = arena_create(s->arena_block_size);
    if (!turn) {
        result.status = ADAM_ERR_ALLOC;
        result.final_response = strdup("arena creation failed");
        return result;
    }

    // 3. Build system prompt and ensure it's at history[0]
    const char *sys_prompt = build_system_prompt(turn, s, user_message);
    if (history->count == 0 || history->items[0].role != ADAM_ROLE_SYSTEM) {
        // Prepend system message
        adam_status_t rc = history_grow(history);
        if (rc != ADAM_OK) {
            arena_destroy(turn);
            result.status = rc;
            result.final_response = strdup("failed to prepend system message");
            return result;
        }
        memmove(&history->items[1], &history->items[0],
                history->count * sizeof(adam_message_t));
        memset(&history->items[0], 0, sizeof(adam_message_t));
        history->items[0].role = ADAM_ROLE_SYSTEM;
        history->items[0].content = strdup(sys_prompt ? sys_prompt : "");
        history->items[0].content_len = strlen(history->items[0].content);
        history->count++;
    } else {
        // Update existing system message
        free(history->items[0].content);
        history->items[0].content = strdup(sys_prompt ? sys_prompt : "");
        history->items[0].content_len = strlen(history->items[0].content);
    }

    // 4. Tool iteration loop
    int max_iter = s->max_iterations > 0 ? s->max_iterations : 25;

    for (int iteration = 0; iteration < max_iter; iteration++) {
        // Check abort
        if (s->abort_flag) {
            ADAM_LOG(s, ADAM_LOG_INFO, "adam_run: aborted at iteration %d",
                     iteration);
            result.status = ADAM_ERR_ABORTED;
            result.final_response = strdup("(aborted)");
            break;
        }

        // Reset arena for this iteration
        arena_reset(turn);

        // Trim history if too long
        if ((int)history->count > s->max_history + 1) {
            ADAM_LOG(s, ADAM_LOG_DEBUG, "compressing history (%zu messages)",
                     history->count);
            history_compress(history);
        }

        // Rate limiter
        rate_limit_wait(s);

        // Call LLM (with retry)
        adam_llm_response_t resp = dispatch_with_retry(
            turn, s, history->items, history->count);

        // Handle errors
        if (resp.error != ADAM_OK) {
            if (resp.error == ADAM_ERR_CONTEXT_OVERFLOW && iteration <= 2) {
                ADAM_LOG(s, ADAM_LOG_WARN,
                         "context overflow, compressing and retrying");
                history_compress(history);
                continue;
            }
            result.status = resp.error;
            result.final_response = resp.error_msg
                ? strdup(resp.error_msg) : strdup(adam_status_string(resp.error));
            if (s->on_error) {
                s->on_error(s->callback_ctx, resp.error, result.final_response);
            }
            break;
        }

        result.input_tokens += resp.input_tokens;
        result.output_tokens += resp.output_tokens;
        rate_limit_record(s, resp.input_tokens + resp.output_tokens);
        result.total_iterations = iteration + 1;

        // No tool calls → final response
        if (resp.tool_call_count == 0) {
            const char *text = resp.content ? resp.content : "";
            result.status = ADAM_OK;
            result.final_response = strdup(text);
            adam_history_append_assistant(history, text, NULL, 0);

            if (s->on_response) {
                s->on_response(s->callback_ctx, text, strlen(text));
            }

            ADAM_LOG(s, ADAM_LOG_INFO,
                     "adam_run: done (iterations=%d, in=%d, out=%d)",
                     result.total_iterations, result.input_tokens,
                     result.output_tokens);
            break;
        }

        // Has tool calls — persist assistant message with tool call IDs
        adam_history_append_assistant(history, resp.content ? resp.content : "",
                                      resp.tool_calls, resp.tool_call_count);

        ADAM_LOG(s, ADAM_LOG_DEBUG, "iteration %d: %zu tool call(s)",
                 iteration, resp.tool_call_count);

        // Execute each tool call
        for (size_t i = 0; i < resp.tool_call_count; i++) {
            adam_tool_call_t *tc = &resp.tool_calls[i];

            if (s->on_tool_call) {
                s->on_tool_call(s->callback_ctx, tc->name,
                                tc->arguments_json);
            }

            ADAM_LOG(s, ADAM_LOG_DEBUG, "  tool: %s(%s)",
                     tc->name, tc->arguments_json ? tc->arguments_json : "");

            // Find tool in registry
            adam_tool_result_t tres = {
                .for_llm = NULL, .for_user = NULL, .success = 0
            };
            int found = 0;

            for (size_t t = 0; t < s->tool_count; t++) {
                if (strcmp(s->tools[t].name, tc->name) == 0) {
                    tres = s->tools[t].execute(
                        turn, s->tools[t].ctx,
                        tc->arguments_json,
                        tc->arguments_json ? strlen(tc->arguments_json) : 0
                    );
                    found = 1;
                    break;
                }
            }

            if (!found) {
                ADAM_LOG(s, ADAM_LOG_WARN, "tool not found: %s", tc->name);
                tres.for_llm = arena_strdup(turn, "Error: tool not found");
                tres.success = 0;
            }

            // Persist tool result (copies from arena to malloc)
            const char *result_text = tres.for_llm ? tres.for_llm
                                                   : "(no output)";
            adam_history_append_tool(history, result_text, tc->id);

            // Forward user-facing output
            if (tres.for_user && s->on_response) {
                s->on_response(s->callback_ctx, tres.for_user,
                               strlen(tres.for_user));
            }
        }

        // Check if this was the last iteration
        if (iteration == max_iter - 1) {
            result.status = ADAM_ERR_MAX_ITERATIONS;
            result.final_response = strdup("max tool iterations reached");
            ADAM_LOG(s, ADAM_LOG_WARN, "adam_run: max iterations reached (%d)",
                     max_iter);
        }

        // Arena will be reset at top of next iteration
    }

    // Cost tracking
    result.cost_usd = compute_cost(s->model, result.input_tokens,
                                    result.output_tokens);

    // Elapsed time
    struct timespec ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    result.elapsed_ms = (ts_end.tv_sec - ts_start.tv_sec) * 1000.0
                      + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e6;

    arena_destroy(turn);
    return result;
}

void adam_run_result_free(adam_run_result_t *r) {
    if (!r) return;
    free(r->final_response);
    r->final_response = NULL;
}

// ============================================================================
// MARK: - Model Registry
// ============================================================================

static adam_model_info_t g_builtin_models[] = {
    // Anthropic
    { "claude-opus-4",      200000, 15.0f, 75.0f },
    { "claude-sonnet-4",    200000,  3.0f, 15.0f },
    { "claude-haiku-4",     200000,  0.8f,  4.0f },
    // OpenAI
    { "gpt-4o",             128000,  2.5f, 10.0f },
    { "gpt-4o-mini",        128000,  0.15f, 0.60f },
    { "o3",                 200000, 10.0f, 40.0f },
    { "o4-mini",            200000,  1.1f,  4.4f },
    // Sentinel
    { NULL, 0, 0, 0 },
};

// User-registered models (simple fixed-size table for now)
#define ADAM_MAX_CUSTOM_MODELS 32
static adam_model_info_t g_custom_models[ADAM_MAX_CUSTOM_MODELS];
static size_t g_custom_model_count = 0;

adam_status_t adam_model_register(const adam_model_info_t *info) {
    if (!info || !info->model_prefix) return ADAM_ERR_INVALID_PARAM;
    if (g_custom_model_count >= ADAM_MAX_CUSTOM_MODELS) return ADAM_ERR_ALLOC;
    g_custom_models[g_custom_model_count++] = *info;
    return ADAM_OK;
}

const adam_model_info_t *adam_model_lookup(const char *model) {
    if (!model) return NULL;

    const adam_model_info_t *best = NULL;
    size_t best_len = 0;

    // Search custom models first (allow overrides)
    for (size_t i = 0; i < g_custom_model_count; i++) {
        size_t plen = strlen(g_custom_models[i].model_prefix);
        if (strncmp(model, g_custom_models[i].model_prefix, plen) == 0
            && plen > best_len) {
            best = &g_custom_models[i];
            best_len = plen;
        }
    }

    // Search built-in models
    for (int i = 0; g_builtin_models[i].model_prefix; i++) {
        size_t plen = strlen(g_builtin_models[i].model_prefix);
        if (strncmp(model, g_builtin_models[i].model_prefix, plen) == 0
            && plen > best_len) {
            best = &g_builtin_models[i];
            best_len = plen;
        }
    }

    return best;
}

int adam_model_context_window(const char *model) {
    const adam_model_info_t *info = adam_model_lookup(model);
    return info ? info->context_window : 0;
}

float adam_model_estimate_cost(const char *model,
                                int input_tokens, int output_tokens) {
    return compute_cost(model, input_tokens, output_tokens);
}

// ============================================================================
// MARK: - Thread Pool
// ============================================================================

#ifndef ADAM_NO_PTHREADS

typedef struct adam_job_node_t {
    adam_job_t               job;
    struct adam_job_node_t  *next;
} adam_job_node_t;

struct adam_pool_t {
    pthread_t              *threads;
    int                     num_workers;
    adam_job_node_t        *queue_head;
    adam_job_node_t        *queue_tail;
    int                     pending;     // jobs in queue
    int                     active;      // jobs being processed
    int                     shutdown;
    pthread_mutex_t         mutex;
    pthread_cond_t          cond;
};

static void *pool_worker(void *arg) {
    adam_pool_t *pool = (adam_pool_t *)arg;

    while (1) {
        pthread_mutex_lock(&pool->mutex);

        // Wait for a job or shutdown
        while (!pool->queue_head && !pool->shutdown) {
            pthread_cond_wait(&pool->cond, &pool->mutex);
        }

        if (pool->shutdown && !pool->queue_head) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }

        // Dequeue job
        adam_job_node_t *node = pool->queue_head;
        pool->queue_head = node->next;
        if (!pool->queue_head) pool->queue_tail = NULL;
        pool->pending--;
        pool->active++;
        pthread_mutex_unlock(&pool->mutex);

        // Execute the job
        adam_job_t job = node->job;
        free(node);

        adam_run_result_t result = adam_run(job.settings, job.history,
                                            job.user_message);

        // Call completion callback
        if (job.on_done) {
            job.on_done(job.on_done_ctx, result);
        } else {
            adam_run_result_free(&result);
        }

        pthread_mutex_lock(&pool->mutex);
        pool->active--;
        pthread_mutex_unlock(&pool->mutex);
    }

    return NULL;
}

adam_pool_t *adam_pool_create(int num_workers) {
    if (num_workers <= 0) {
        // Default: 4 workers (a reasonable fallback)
        num_workers = 4;
    }

    adam_pool_t *pool = calloc(1, sizeof(adam_pool_t));
    if (!pool) return NULL;

    pool->num_workers = num_workers;
    pool->threads = calloc((size_t)num_workers, sizeof(pthread_t));
    if (!pool->threads) { free(pool); return NULL; }

    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->cond, NULL);

    for (int i = 0; i < num_workers; i++) {
        pthread_create(&pool->threads[i], NULL, pool_worker, pool);
    }

    return pool;
}

adam_status_t adam_pool_submit(adam_pool_t *pool, adam_job_t job) {
    if (!pool || !job.settings || !job.history)
        return ADAM_ERR_INVALID_PARAM;

    adam_job_node_t *node = calloc(1, sizeof(adam_job_node_t));
    if (!node) return ADAM_ERR_ALLOC;
    node->job = job;

    pthread_mutex_lock(&pool->mutex);
    if (pool->queue_tail) {
        pool->queue_tail->next = node;
    } else {
        pool->queue_head = node;
    }
    pool->queue_tail = node;
    pool->pending++;
    pthread_cond_signal(&pool->cond);
    pthread_mutex_unlock(&pool->mutex);

    return ADAM_OK;
}

void adam_pool_destroy(adam_pool_t *pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->cond);
    pthread_mutex_unlock(&pool->mutex);

    for (int i = 0; i < pool->num_workers; i++) {
        pthread_join(pool->threads[i], NULL);
    }

    // Free any remaining queued jobs
    adam_job_node_t *node = pool->queue_head;
    while (node) {
        adam_job_node_t *next = node->next;
        free(node);
        node = next;
    }

    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->cond);
    free(pool->threads);
    free(pool);
}

int adam_pool_pending(const adam_pool_t *pool) {
    return pool ? pool->pending : 0;
}

int adam_pool_active(const adam_pool_t *pool) {
    return pool ? pool->active : 0;
}

#endif // ADAM_NO_PTHREADS
