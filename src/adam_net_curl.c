//
//  adam_net_curl.c
//  Adam — HTTP layer via libcurl (Linux, Windows)
//
//  Created by Marco Bambini on 16/03/26.
//

#if !defined(ADAM_NO_CURL) && !defined(__APPLE__)

#include "adam_net.h"
#include <curl/curl.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// MARK: - Write callback (accumulates into arena buffer)
// ============================================================================

typedef struct {
    arena_t *arena;
    uint8_t *buf;
    size_t   len;
    size_t   cap;
} net_buf_t;

static size_t write_cb(char *data, size_t size, size_t nmemb, void *userp) {
    net_buf_t *ctx = (net_buf_t *)userp;
    size_t bytes = size * nmemb;
    if (ctx->len + bytes >= ctx->cap) {
        size_t new_cap = ctx->cap == 0 ? 8192 : ctx->cap * 2;
        while (new_cap < ctx->len + bytes + 1) new_cap *= 2;
        uint8_t *new_buf = arena_alloc(ctx->arena, new_cap);
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
// MARK: - Stream write callback
// ============================================================================

typedef struct {
    adam_net_stream_fn fn;
    void             *ctx;
    int               aborted;
} stream_cb_ctx_t;

static size_t stream_write_cb(char *data, size_t size, size_t nmemb, void *userp) {
    stream_cb_ctx_t *sc = (stream_cb_ctx_t *)userp;
    size_t bytes = size * nmemb;
    if (sc->aborted) return 0;
    if (sc->fn(sc->ctx, (const uint8_t *)data, bytes) != 0) {
        sc->aborted = 1;
        return 0;
    }
    return bytes;
}

// ============================================================================
// MARK: - Persistent handle management
// ============================================================================

static CURL *get_handle(adam_settings_t *s, int handle_id) {
    void **slot;
    switch (handle_id) {
    case 0: slot = &s->_curl_llm; break;
    case 1: slot = &s->_curl_stt; break;
    case 2: slot = &s->_curl_tts; break;
    default: return curl_easy_init();
    }
    if (!*slot) *slot = curl_easy_init();
    CURL *h = (CURL *)*slot;
    if (h) curl_easy_reset(h);
    return h;
}

// ============================================================================
// MARK: - Cleanup
// ============================================================================

void adam_net_cleanup(adam_settings_t *s) {
    if (s->_curl_llm) { curl_easy_cleanup(s->_curl_llm); s->_curl_llm = NULL; }
    if (s->_curl_stt) { curl_easy_cleanup(s->_curl_stt); s->_curl_stt = NULL; }
    if (s->_curl_tts) { curl_easy_cleanup(s->_curl_tts); s->_curl_tts = NULL; }
}

// ============================================================================
// MARK: - POST JSON
// ============================================================================

adam_net_response_t adam_net_post_json(
    adam_settings_t *s, arena_t *arena,
    const char *url, const char *auth_header,
    const char *body, const char **extra_headers, int handle_id
) {
    adam_net_response_t resp = {0};
    CURL *curl = get_handle(s, handle_id);
    if (!curl) { resp.error = ADAM_ERR_CURL; return resp; }

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    if (auth_header) headers = curl_slist_append(headers, auth_header);
    if (extra_headers) {
        for (int i = 0; extra_headers[i]; i++)
            headers = curl_slist_append(headers, extra_headers[i]);
    }

    net_buf_t write_ctx = { .arena = arena };

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_COPYPOSTFIELDS, body);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &write_ctx);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        resp.error = ADAM_ERR_CURL;
    } else {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp.http_code);
    }

    resp.data = write_ctx.buf;
    resp.data_len = write_ctx.len;
    curl_slist_free_all(headers);
    return resp;
}

// ============================================================================
// MARK: - POST Multipart
// ============================================================================

adam_net_response_t adam_net_post_multipart(
    adam_settings_t *s, arena_t *arena,
    const char *url, const char *auth_header,
    const adam_net_field_t *fields, size_t field_count, int handle_id
) {
    adam_net_response_t resp = {0};
    CURL *curl = get_handle(s, handle_id);
    if (!curl) { resp.error = ADAM_ERR_CURL; return resp; }

    curl_mime *mime = curl_mime_init(curl);
    for (size_t i = 0; i < field_count; i++) {
        curl_mimepart *part = curl_mime_addpart(mime);
        curl_mime_name(part, fields[i].name);
        if (fields[i].value) {
            curl_mime_data(part, fields[i].value, CURL_ZERO_TERMINATED);
        } else {
            curl_mime_data(part, (const char *)fields[i].data, fields[i].data_len);
            if (fields[i].filename) curl_mime_filename(part, fields[i].filename);
            if (fields[i].content_type) curl_mime_type(part, fields[i].content_type);
        }
    }

    struct curl_slist *headers = NULL;
    if (auth_header) headers = curl_slist_append(headers, auth_header);

    net_buf_t write_ctx = { .arena = arena };

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &write_ctx);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        resp.error = ADAM_ERR_CURL;
    } else {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp.http_code);
    }

    resp.data = write_ctx.buf;
    resp.data_len = write_ctx.len;
    curl_mime_free(mime);
    curl_slist_free_all(headers);
    return resp;
}

// ============================================================================
// MARK: - POST Streaming
// ============================================================================

adam_net_response_t adam_net_post_streaming(
    adam_settings_t *s, const char *url, const char *auth_header,
    const char *body, adam_net_stream_fn on_chunk, void *stream_ctx,
    int handle_id
) {
    adam_net_response_t resp = {0};
    CURL *curl = get_handle(s, handle_id);
    if (!curl) { resp.error = ADAM_ERR_CURL; return resp; }

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    if (auth_header) headers = curl_slist_append(headers, auth_header);

    stream_cb_ctx_t sc = { .fn = on_chunk, .ctx = stream_ctx };

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_COPYPOSTFIELDS, body);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, stream_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &sc);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        resp.error = ADAM_ERR_CURL;
    } else {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp.http_code);
    }

    curl_slist_free_all(headers);
    return resp;
}

#endif // !ADAM_NO_CURL && !__APPLE__
