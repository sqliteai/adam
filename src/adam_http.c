//
//  adam_http.c
//  Adam — HTTP LLM provider via libcurl + mbedtls
//
//  Created by Marco Bambini on 15/03/26.
//

#ifndef ADAM_NO_CURL

#include "adam.h"
#include "adam_json.h"
#include <curl/curl.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// MARK: - curl write callback
// ============================================================================

typedef struct {
    arena_t *arena;
    char    *buf;
    size_t   len;
    size_t   cap;
} curl_buf_t;

static size_t write_cb(char *data, size_t size, size_t nmemb, void *userp) {
    curl_buf_t *ctx = (curl_buf_t *)userp;
    size_t bytes = size * nmemb;

    if (ctx->len + bytes >= ctx->cap) {
        size_t new_cap = ctx->cap == 0 ? 8192 : ctx->cap * 2;
        while (new_cap < ctx->len + bytes + 1) new_cap *= 2;
        char *new_buf = arena_alloc(ctx->arena, new_cap);
        if (!new_buf) return 0;
        if (ctx->buf) memcpy(new_buf, ctx->buf, ctx->len);
        ctx->buf = new_buf;
        ctx->cap = new_cap;
    }

    memcpy(ctx->buf + ctx->len, data, bytes);
    ctx->len += bytes;
    ctx->buf[ctx->len] = '\0';
    return bytes;
}

// ============================================================================
// MARK: - LLM HTTP call
// ============================================================================

adam_llm_response_t adam_llm_call_http(
    arena_t *arena, adam_settings_t *s,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
) {
    adam_llm_response_t resp = {0};

    CURL *curl = curl_easy_init();
    if (!curl) {
        resp.error = ADAM_ERR_CURL;
        resp.error_msg = arena_strdup(arena, "curl_easy_init failed");
        return resp;
    }

    // Determine URL
    const char *url = s->base_url;
    if (!url) {
        if (s->api_format == ADAM_API_ANTHROPIC) {
            url = "https://api.anthropic.com/v1/messages";
        } else {
            url = "https://api.openai.com/v1/chat/completions";
        }
    }

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);

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
        curl_easy_cleanup(curl);
        return resp;
    }
    // Use COPYPOSTFIELDS so curl owns a copy of the body.
    // (POSTFIELDS stores only a pointer, which can be invalidated if
    // the write callback allocates from the same arena.)
    curl_easy_setopt(curl, CURLOPT_COPYPOSTFIELDS, body);


    // Headers
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    if (s->api_key) {
        char *auth = arena_alloc(arena, strlen(s->api_key) + 64);
        if (auth) {
            if (s->api_format == ADAM_API_ANTHROPIC) {
                snprintf(auth, strlen(s->api_key) + 64, "x-api-key: %s", s->api_key);
                headers = curl_slist_append(headers, auth);
                headers = curl_slist_append(headers,
                    "anthropic-version: 2023-06-01");
            } else {
                snprintf(auth, strlen(s->api_key) + 64, "Authorization: Bearer %s", s->api_key);
                headers = curl_slist_append(headers, auth);
            }
        }
    }
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    // Response buffer
    curl_buf_t write_ctx = { .arena = arena };
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &write_ctx);

    // Perform request
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        resp.error = ADAM_ERR_CURL;
        resp.error_msg = arena_strdup(arena, curl_easy_strerror(res));
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return resp;
    }

    // Check HTTP status
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);


    if (http_code == 429) {
        // Try to parse error message from body, fall back to generic
        resp = adam_json_parse_response(arena, s->api_format,
                                        write_ctx.buf, write_ctx.len);
        if (resp.error == ADAM_OK) resp.error = ADAM_ERR_RATE_LIMIT;
        if (!resp.error_msg) resp.error_msg = arena_strdup(arena, "rate limited (429)");
    } else if (http_code == 401 || http_code == 403) {
        resp = adam_json_parse_response(arena, s->api_format,
                                        write_ctx.buf, write_ctx.len);
        if (resp.error == ADAM_OK) resp.error = ADAM_ERR_AUTH;
        if (!resp.error_msg) resp.error_msg = arena_strdup(arena, "authentication error");
    } else if (http_code >= 400) {
        // Parse error body to extract structured error message
        resp = adam_json_parse_response(arena, s->api_format,
                                        write_ctx.buf, write_ctx.len);
        if (resp.error == ADAM_OK) resp.error = ADAM_ERR_PROVIDER;
        if (!resp.error_msg) {
            resp.error_msg = write_ctx.buf
                ? arena_strdup(arena, write_ctx.buf)
                : arena_strdup(arena, "HTTP error");
        }
    } else {
        // Parse successful response JSON
        resp = adam_json_parse_response(
            arena, s->api_format,
            write_ctx.buf, write_ctx.len
        );
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return resp;
}

#endif // ADAM_NO_CURL
