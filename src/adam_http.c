//
//  adam_http.c
//  Adam — LLM HTTP provider (platform-independent)
//
//  Uses adam_net.h abstraction for actual HTTP calls.
//  Works on Apple (NSURLSession), Linux/Windows (libcurl), WASM (callback).
//
//  Created by Marco Bambini on 15/03/26.
//

#include "adam.h"
#include "adam_json.h"
#include "adam_net.h"
#include <string.h>
#include <stdio.h>

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
    if (!url) {
        url = (s->api_format == ADAM_API_ANTHROPIC)
            ? "https://api.anthropic.com/v1/messages"
            : "https://api.openai.com/v1/chat/completions";
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
    if (net.http_code == 429) {
        resp = adam_json_parse_response(arena, s->api_format,
                                        (const char *)net.data, net.data_len);
        if (resp.error == ADAM_OK) resp.error = ADAM_ERR_RATE_LIMIT;
        if (!resp.error_msg) resp.error_msg = arena_strdup(arena, "rate limited (429)");
    } else if (net.http_code == 401 || net.http_code == 403) {
        resp = adam_json_parse_response(arena, s->api_format,
                                        (const char *)net.data, net.data_len);
        if (resp.error == ADAM_OK) resp.error = ADAM_ERR_AUTH;
        if (!resp.error_msg) resp.error_msg = arena_strdup(arena, "authentication error");
    } else if (net.http_code >= 400) {
        resp = adam_json_parse_response(arena, s->api_format,
                                        (const char *)net.data, net.data_len);
        if (resp.error == ADAM_OK) resp.error = ADAM_ERR_PROVIDER;
        if (!resp.error_msg) {
            resp.error_msg = (net.data && net.data_len > 0)
                ? arena_strdup(arena, (const char *)net.data)
                : arena_strdup(arena, "HTTP error");
        }
    } else {
        resp = adam_json_parse_response(arena, s->api_format,
                                        (const char *)net.data, net.data_len);
    }

    return resp;
}
