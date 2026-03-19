//
//  adam_http.c
//  Adam — LLM HTTP provider + remote embedding provider (platform-independent)
//
//  Uses adam_net.h abstraction for actual HTTP calls.
//  Works on Apple (NSURLSession), Linux/Windows (libcurl), WASM (callback).
//
//  Created by Marco Bambini on 15/03/26.
//

#include "adam.h"
#include "adam_json.h"
#include "adam_net.h"

#ifndef ADAM_NO_SQLITE
#define SQLITE_CORE
#include "sqlite3.h"
#include "sqlite-memory.h"
#define JSMN_STATIC
#include "jsmn.h"
#endif

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Auth header overhead
#define AUTH_HDR_EXTRA 32

adam_llm_response_t adam_llm_call_http(
    arena_t *arena, adam_settings_t *s,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
) {
    adam_llm_response_t resp = {0};

    // Build request JSON
    const char *body = adam_json_build_request(
        arena, s->api_format, s->model,
        msgs, msg_count, tools, tool_count,
        s->temperature, s->max_tokens, s->top_p,
        s->response_format
    );
    if (!body) {
        resp.error = ADAM_ERR_JSON;
        resp.error_msg = arena_strdup(arena, "failed to build request JSON");
        return resp;
    }



    // Determine URL
    const char *url = s->base_url;
    char *gemini_url = NULL;
    if (!url) {
        if (s->api_format == ADAM_API_GEMINI) {
            // Gemini: model in URL path, API key in query string
            const char *model = s->model ? s->model : "gemini-2.0-flash";
            size_t gurl_len = strlen(model) + (s->api_key ? strlen(s->api_key) : 0) + 128;
            gemini_url = arena_alloc(arena, gurl_len);
            if (gemini_url) {
                snprintf(gemini_url, gurl_len,
                    "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s",
                    model, s->api_key ? s->api_key : "");
                url = gemini_url;
            }
        } else if (s->api_format == ADAM_API_ANTHROPIC) {
            url = "https://api.anthropic.com/v1/messages";
        } else {
            url = "https://api.openai.com/v1/chat/completions";
        }
    }

    // Build auth header
    if (!s->api_key) {
        resp.error = ADAM_ERR_AUTH;
        resp.error_msg = arena_strdup(arena, "api_key is NULL");
        return resp;
    }
    size_t auth_len = strlen(s->api_key) + AUTH_HDR_EXTRA;
    char *auth = arena_alloc(arena, auth_len);
    if (!auth) {
        resp.error = ADAM_ERR_ALLOC;
        return resp;
    }

    const char **extra = NULL;
    const char *anthropic_extra[] = { "anthropic-version: 2023-06-01", NULL };

    if (s->api_format == ADAM_API_ANTHROPIC) {
        snprintf(auth, auth_len, "x-api-key: %s", s->api_key);
        extra = anthropic_extra;
    } else if (s->api_format == ADAM_API_GEMINI) {
        // Gemini: no auth header (key is in URL query string)
        auth[0] = '\0';
    } else {
        snprintf(auth, auth_len, "Authorization: Bearer %s", s->api_key);
    }

    // HTTP POST
    adam_net_response_t net = adam_net_post_json(
        s, arena, url, auth, body, extra, /*handle_id=*/0
    );

    if (net.error != ADAM_OK) {
        resp.error = net.error;
        resp.error_msg = arena_strdup(arena, "HTTP request failed");
        return resp;
    }

    // Classify HTTP status and parse response
    resp.http_status = (int)net.http_code;

    if (net.http_code == 429) {
        resp = adam_json_parse_response(arena, s->api_format,
                                        (const char *)net.data, net.data_len);
        resp.http_status = (int)net.http_code;
        if (resp.error == ADAM_OK) resp.error = ADAM_ERR_RATE_LIMIT;
        if (!resp.error_msg) resp.error_msg = arena_strdup(arena, "rate limited (429)");
    } else if (net.http_code == 401 || net.http_code == 403) {
        resp = adam_json_parse_response(arena, s->api_format,
                                        (const char *)net.data, net.data_len);
        resp.http_status = (int)net.http_code;
        if (resp.error == ADAM_OK) resp.error = ADAM_ERR_AUTH;
        if (!resp.error_msg) resp.error_msg = arena_strdup(arena, "authentication error");
    } else if (net.http_code >= 400) {
        resp = adam_json_parse_response(arena, s->api_format,
                                        (const char *)net.data, net.data_len);
        resp.http_status = (int)net.http_code;
        if (resp.error == ADAM_OK) resp.error = ADAM_ERR_PROVIDER;
        if (!resp.error_msg) {
            resp.error_msg = (net.data && net.data_len > 0)
                ? arena_strdup(arena, (const char *)net.data)
                : arena_strdup(arena, "HTTP error");
        }
    } else {
        resp = adam_json_parse_response(arena, s->api_format,
                                        (const char *)net.data, net.data_len);
        resp.http_status = (int)net.http_code;
    }

    return resp;
}

// ============================================================================
// MARK: - Remote Embedding Provider (registered into sqlite-memory)
// ============================================================================
//
// This custom provider uses adam_net_post_json() so it works on all platforms
// (NSURLSession on macOS, libcurl on Linux) without sqlite-memory needing its
// own HTTP code. It supports Voyage (for Anthropic users) and OpenAI embeddings
// using the same OpenAI-compatible request/response format.
//
// Voyage API:  https://api.voyageai.com/v1/embeddings
// OpenAI API:  https://api.openai.com/v1/embeddings
//
// Request:  {"model":"...","input":"..."}
// Response: {"data":[{"embedding":[...]}],"usage":{"prompt_tokens":...},"model":"..."}

#ifndef ADAM_NO_SQLITE

#define EMBED_DEFAULT_BUF_SIZE  (128 * 1024)

// Passed via dbmem_provider_t.xdata to all callbacks.
typedef struct {
    adam_settings_t *settings;      // borrowed — must outlive the provider
    char            *url;           // embedding API endpoint
} adam_embed_xdata_t;

typedef struct {
    adam_embed_xdata_t *xd;         // borrowed — outlives the engine
    char            *api_key;
    char            *model;
    float           *embedding;
    size_t           embedding_cap;
    char            *request_buf;
    size_t           request_cap;
} adam_embed_engine_t;

// Determine the embedding API URL from provider name.
static const char *embed_url_for_provider(const char *provider) {
    if (!provider) return "https://api.voyageai.com/v1/embeddings";
    if (strcmp(provider, "voyage") == 0)
        return "https://api.voyageai.com/v1/embeddings";
    if (strcmp(provider, "openai") == 0)
        return "https://api.openai.com/v1/embeddings";
    if (strcmp(provider, "grok") == 0 || strcmp(provider, "xai") == 0)
        return "https://api.x.ai/v1/embeddings";
    // Default: treat as OpenAI-compatible
    return "https://api.openai.com/v1/embeddings";
}

static void *adam_embed_init(const char *model, const char *api_key,
                              void *xdata, char err_msg[1024]) {
    adam_embed_xdata_t *xd = (adam_embed_xdata_t *)xdata;
    if (!api_key || !api_key[0]) {
        snprintf(err_msg, 1024, "API key required for remote embeddings");
        return NULL;
    }
    if (!model || !model[0]) {
        snprintf(err_msg, 1024, "Model name required for remote embeddings");
        return NULL;
    }
    if (!xd || !xd->settings) {
        snprintf(err_msg, 1024, "adam_settings not configured for embedding provider");
        return NULL;
    }

    adam_embed_engine_t *eng = calloc(1, sizeof(adam_embed_engine_t));
    if (!eng) {
        snprintf(err_msg, 1024, "Failed to allocate embedding engine");
        return NULL;
    }

    eng->xd = xd;
    eng->api_key = strdup(api_key);
    eng->model = strdup(model);
    if (!eng->api_key || !eng->model) {
        free(eng->api_key);
        free(eng->model);
        free(eng);
        snprintf(err_msg, 1024, "Failed to allocate engine strings");
        return NULL;
    }

    eng->request_buf = malloc(EMBED_DEFAULT_BUF_SIZE);
    eng->request_cap = eng->request_buf ? EMBED_DEFAULT_BUF_SIZE : 0;

    return eng;
}

// JSON-escape text into dst. Returns bytes written (excluding NUL).
static size_t json_escape_text(char *dst, size_t dst_cap,
                                const char *src, int src_len) {
    size_t w = 0;
    for (int i = 0; i < src_len && w + 6 < dst_cap; i++) {
        unsigned char c = (unsigned char)src[i];
        switch (c) {
            case '"':  dst[w++] = '\\'; dst[w++] = '"'; break;
            case '\\': dst[w++] = '\\'; dst[w++] = '\\'; break;
            case '\n': dst[w++] = '\\'; dst[w++] = 'n'; break;
            case '\r': dst[w++] = '\\'; dst[w++] = 'r'; break;
            case '\t': dst[w++] = '\\'; dst[w++] = 't'; break;
            default:
                if (c < 0x20) {
                    static const char hex[] = "0123456789abcdef";
                    dst[w++] = '\\'; dst[w++] = 'u';
                    dst[w++] = '0'; dst[w++] = '0';
                    dst[w++] = hex[(c >> 4) & 0xF];
                    dst[w++] = hex[c & 0xF];
                } else {
                    dst[w++] = (char)c;
                }
        }
    }
    dst[w] = '\0';
    return w;
}

static int adam_embed_compute(void *engine, const char *text, int text_len,
                               void *xdata, dbmem_embedding_result_t *result) {
    (void)xdata;
    adam_embed_engine_t *eng = (adam_embed_engine_t *)engine;
    if (!eng || !eng->xd || !eng->xd->settings || !text || text_len <= 0)
        return -1;

    // Ensure request buffer is large enough: text * 2 (escaping) + overhead
    size_t need = (size_t)text_len * 2 + strlen(eng->model) + 256;
    if (need > eng->request_cap) {
        free(eng->request_buf);
        eng->request_buf = malloc(need);
        eng->request_cap = eng->request_buf ? need : 0;
        if (!eng->request_buf) return -1;
    }

    // Build JSON: {"model":"...","input":"..."}
    size_t pos = 0;
    pos += (size_t)snprintf(eng->request_buf + pos, eng->request_cap - pos,
        "{\"model\":\"%s\",\"input\":\"", eng->model);
    pos += json_escape_text(eng->request_buf + pos, eng->request_cap - pos,
        text, text_len);
    pos += (size_t)snprintf(eng->request_buf + pos, eng->request_cap - pos,
        "\"}");

    // Build auth header
    char auth[512];
    snprintf(auth, sizeof(auth), "Authorization: Bearer %s", eng->api_key);

    // POST via adam_net
    arena_t *arena = arena_create(EMBED_DEFAULT_BUF_SIZE);
    if (!arena) return -1;

    adam_net_response_t net = adam_net_post_json(
        eng->xd->settings, arena, eng->xd->url, auth, eng->request_buf,
        NULL, /*handle_id=*/3  // dedicated handle for embeddings
    );

    if (net.error != ADAM_OK || net.http_code != 200) {
        arena_destroy(arena);
        return -1;
    }

    // Parse response: {"data":[{"embedding":[...]}],"usage":{"prompt_tokens":N}}
    // Two-pass jsmn: count tokens, then parse.
    const char *json = (const char *)net.data;
    size_t json_len = net.data_len;

    jsmn_parser parser;
    jsmn_init(&parser);
    int ntok = jsmn_parse(&parser, json, json_len, NULL, 0);
    if (ntok < 1) { arena_destroy(arena); return -1; }

    jsmntok_t *tokens = arena_alloc(arena, (size_t)ntok * sizeof(jsmntok_t));
    if (!tokens) { arena_destroy(arena); return -1; }

    jsmn_init(&parser);
    jsmn_parse(&parser, json, json_len, tokens, (unsigned)ntok);

    // Extract embedding array and prompt_tokens
    int emb_start = -1, emb_count = 0, prompt_tokens = 0;

    for (int i = 0; i < ntok - 1; i++) {
        if (tokens[i].type != JSMN_STRING) continue;
        int klen = tokens[i].end - tokens[i].start;
        const char *key = json + tokens[i].start;

        if (klen == 9 && memcmp(key, "embedding", 9) == 0
            && tokens[i + 1].type == JSMN_ARRAY) {
            emb_count = tokens[i + 1].size;
            emb_start = i + 2;
        } else if (klen == 13 && memcmp(key, "prompt_tokens", 13) == 0
                   && tokens[i + 1].type == JSMN_PRIMITIVE) {
            prompt_tokens = atoi(json + tokens[i + 1].start);
        }
    }

    if (emb_start < 0 || emb_count == 0) {
        arena_destroy(arena);
        return -1;
    }

    // Grow embedding buffer if needed
    if (eng->embedding_cap < (size_t)emb_count) {
        free(eng->embedding);
        eng->embedding = malloc(sizeof(float) * (size_t)emb_count);
        eng->embedding_cap = eng->embedding ? (size_t)emb_count : 0;
        if (!eng->embedding) { arena_destroy(arena); return -1; }
    }

    // Parse floats
    for (int i = 0; i < emb_count; i++) {
        eng->embedding[i] = strtof(json + tokens[emb_start + i].start, NULL);
    }

    arena_destroy(arena);

    result->n_embd = emb_count;
    result->n_tokens = prompt_tokens;
    result->n_tokens_truncated = 0;
    result->embedding = eng->embedding;
    return 0;
}

static void adam_embed_free(void *engine, void *xdata) {
    (void)xdata;
    if (!engine) return;
    adam_embed_engine_t *eng = (adam_embed_engine_t *)engine;
    free(eng->api_key);
    free(eng->model);
    free(eng->embedding);
    free(eng->request_buf);
    free(eng);
}

// Public: register the adam remote embedding provider on a sqlite-memory db.
// provider_name: "voyage", "openai", etc.
// Must be called before memory_set_model() triggers the init callback.
adam_status_t adam_embed_provider_register(sqlite3 *db,
                                           adam_settings_t *s,
                                           const char *provider_name) {
    if (!db || !s || !provider_name) return ADAM_ERR_INVALID_PARAM;

    // Allocate xdata (freed when the memory db is closed — sqlite-memory
    // doesn't free it, so it leaks on close. Acceptable: one per db lifetime.)
    adam_embed_xdata_t *xd = calloc(1, sizeof(adam_embed_xdata_t));
    if (!xd) return ADAM_ERR_ALLOC;
    xd->settings = s;
    xd->url = strdup(embed_url_for_provider(provider_name));
    if (!xd->url) { free(xd); return ADAM_ERR_ALLOC; }

    dbmem_provider_t provider = {
        .init    = adam_embed_init,
        .compute = adam_embed_compute,
        .free    = adam_embed_free,
        .xdata   = xd,
    };

    int rc = sqlite3_memory_register_provider(db, provider_name, &provider);
    if (rc != SQLITE_OK) { free(xd->url); free(xd); return ADAM_ERR_SQLITE; }

    return ADAM_OK;
}

#endif // ADAM_NO_SQLITE
