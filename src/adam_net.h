//
//  adam_net.h
//  Adam — Internal HTTP abstraction layer
//
//  Provides platform-independent HTTP operations used by adam_http.c
//  and adam_voice.c. Implementations:
//    - adam_net_curl.c   (Linux, Windows — libcurl + mbedtls)
//    - adam_net_apple.m  (macOS, iOS — NSURLSession)
//    - WASM: embedder provides via adam_settings_t.http_fn callback
//
//  Created by Marco Bambini on 16/03/26.
//

#pragma once

#include "adam.h"
#include "arena.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// MARK: - HTTP Response
// ============================================================================

typedef struct {
    uint8_t    *data;           // arena-allocated response body
    size_t      data_len;
    long        http_code;      // HTTP status code (200, 429, etc.)
    adam_status_t error;        // ADAM_OK if HTTP call succeeded
} adam_net_response_t;

// ============================================================================
// MARK: - Streaming callback
// ============================================================================

// Called for each chunk of data as it arrives from the server.
// Return 0 to continue, non-zero to abort the transfer.
typedef int (*adam_net_stream_fn)(
    void        *ctx,
    const uint8_t *chunk,
    size_t       chunk_len
);

// ============================================================================
// MARK: - Multipart form field
// ============================================================================

typedef struct {
    const char  *name;          // field name
    const char  *value;         // field value (text) or NULL for binary
    const uint8_t *data;        // binary data (if value is NULL)
    size_t       data_len;      // binary data length
    const char  *filename;      // filename for binary fields (e.g. "audio.wav")
    const char  *content_type;  // MIME type for binary fields (e.g. "audio/wav")
} adam_net_field_t;

// ============================================================================
// MARK: - HTTP Operations
// ============================================================================

// Initialize/cleanup the HTTP subsystem (per settings instance).
// Called from adam_settings_destroy.
void adam_net_cleanup(adam_settings_t *s);

// POST JSON, receive response body.
// auth_header: "Authorization: Bearer xxx" or "x-api-key: xxx"
adam_net_response_t adam_net_post_json(
    adam_settings_t *s,
    arena_t         *arena,
    const char      *url,
    const char      *auth_header,
    const char      *body,
    const char     **extra_headers,  // NULL-terminated array, or NULL
    int              handle_id       // 0=LLM, 1=STT, 2=TTS (for connection reuse)
);

// POST multipart form, receive response body.
adam_net_response_t adam_net_post_multipart(
    adam_settings_t *s,
    arena_t         *arena,
    const char      *url,
    const char      *auth_header,
    const adam_net_field_t *fields,
    size_t           field_count,
    int              handle_id
);

// POST JSON, stream response body via callback.
// Used for streaming TTS (PCM chunks piped to audio player).
adam_net_response_t adam_net_post_streaming(
    adam_settings_t *s,
    const char      *url,
    const char      *auth_header,
    const char      *body,
    adam_net_stream_fn on_chunk,
    void            *stream_ctx,
    int              handle_id
);

#ifdef __cplusplus
}
#endif
