//
//  adam_voice.c
//  Adam — Voice subsystem: cloud STT/TTS via libcurl, audio playback
//
//  Created by Marco Bambini on 15/03/26.
//

#if !defined(ADAM_NO_VOICE) && !defined(ADAM_NO_PTHREADS)

#include "adam.h"
#include "adam_audio.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define UNUSED_PARAM(p) ((void)(p))

// ============================================================================
// MARK: - Internal: curl helpers (shared with adam_http.c)
// ============================================================================

#ifndef ADAM_NO_CURL
#include <curl/curl.h>

typedef struct {
    arena_t *arena;
    char    *buf;
    size_t   len;
    size_t   cap;
} voice_curl_buf_t;

static size_t voice_write_cb(char *data, size_t size, size_t nmemb, void *userp) {
    voice_curl_buf_t *ctx = (voice_curl_buf_t *)userp;
    size_t bytes = size * nmemb;

    if (ctx->len + bytes >= ctx->cap) {
        size_t new_cap = ctx->cap == 0 ? 16384 : ctx->cap * 2;
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
// MARK: - Cloud STT (OpenAI Whisper API)
// ============================================================================

// POST multipart/form-data to:
//   https://api.openai.com/v1/audio/transcriptions
//
// Fields:
//   file:     audio data (binary)
//   model:    "whisper-1"
//   language: optional ("en", "it", etc.)
//
// Response (JSON):
//   {"text": "transcribed text here"}

static adam_status_t cloud_stt(
    adam_settings_t *s, arena_t *arena,
    const uint8_t *audio, size_t audio_len,
    adam_audio_format_t format,
    const char **out_text
) {
    // Reuse persistent CURL handle
    if (!s->_curl_stt) s->_curl_stt = curl_easy_init();
    CURL *curl = (CURL *)s->_curl_stt;
    if (!curl) return ADAM_ERR_CURL;
    curl_easy_reset(curl);

    const char *url = s->stt_api_url
        ? s->stt_api_url
        : "https://api.openai.com/v1/audio/transcriptions";

    const char *key = s->stt_api_key ? s->stt_api_key : s->api_key;
    if (!key) return ADAM_ERR_AUTH;

    // Determine file extension for the mime type
    const char *filename;
    switch (format) {
    case ADAM_AUDIO_WAV:      filename = "audio.wav";  break;
    case ADAM_AUDIO_MP3:      filename = "audio.mp3";  break;
    case ADAM_AUDIO_OGG_OPUS: filename = "audio.ogg";  break;
    case ADAM_AUDIO_FLAC:     filename = "audio.flac"; break;
    default:                  filename = "audio.wav";  break;
    }

    // Build multipart form
    curl_mime *mime = curl_mime_init(curl);

    // File field
    curl_mimepart *part = curl_mime_addpart(mime);
    curl_mime_name(part, "file");
    curl_mime_data(part, (const char *)audio, audio_len);
    curl_mime_filename(part, filename);
    curl_mime_type(part, format == ADAM_AUDIO_MP3 ? "audio/mpeg" : "audio/wav");

    // Model field
    part = curl_mime_addpart(mime);
    curl_mime_name(part, "model");
    curl_mime_data(part, s->stt_model ? s->stt_model : "whisper-1",
                   CURL_ZERO_TERMINATED);

    // Language field (optional)
    if (s->stt_language) {
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "language");
        curl_mime_data(part, s->stt_language, CURL_ZERO_TERMINATED);
    }

    // Response format
    part = curl_mime_addpart(mime);
    curl_mime_name(part, "response_format");
    curl_mime_data(part, "json", CURL_ZERO_TERMINATED);

    // Auth header
    char *auth_hdr = arena_alloc(arena, strlen(key) + 32);
    snprintf(auth_hdr, strlen(key) + 32, "Authorization: Bearer %s", key);

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, auth_hdr);

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

    voice_curl_buf_t write_ctx = { .arena = arena };
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, voice_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &write_ctx);

    CURLcode res = curl_easy_perform(curl);
    adam_status_t status = ADAM_OK;
    long http_code = 0;

    if (res != CURLE_OK) {
        status = ADAM_ERR_CURL;
    } else {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code >= 400) {
            // Extract error message from JSON response for better diagnostics
            if (write_ctx.buf) {
                const char *msg = strstr(write_ctx.buf, "\"message\"");
                if (msg) {
                    msg = strchr(msg + 9, '"');
                    if (msg) {
                        msg++;
                        const char *end = strchr(msg, '"');
                        if (end) {
                            size_t mlen = (size_t)(end - msg);
                            char *err = arena_alloc(arena, mlen + 32);
                            snprintf(err, mlen + 32, "STT error (HTTP %ld): %.*s",
                                     http_code, (int)mlen, msg);
                            // Store as out_text so caller can see the error
                            fprintf(stderr, "  [STT] %s\n", err);
                        }
                    }
                }
            }
            if (http_code == 429)
                status = ADAM_ERR_RATE_LIMIT;
            else if (http_code == 401 || http_code == 403)
                status = ADAM_ERR_AUTH;
            else
                status = ADAM_ERR_VOICE;
        }
    }

    if (status == ADAM_OK && write_ctx.buf) {
        // Parse {"text": "..."} — find the "text" key specifically
        // (not "type" or "text_tokens" etc.)
        const char *p = write_ctx.buf;
        while ((p = strstr(p, "\"text\"")) != NULL) {
            // Verify this is a top-level key by checking what's after the value
            const char *colon = p + 6;
            while (*colon == ' ' || *colon == '\t') colon++;
            if (*colon == ':') {
                colon++;
                while (*colon == ' ' || *colon == '\t' || *colon == '\n') colon++;
                if (*colon == '"') {
                    colon++;
                    const char *end = colon;
                    while (*end && !(*end == '"' && *(end-1) != '\\')) end++;
                    size_t tlen = (size_t)(end - colon);
                    char *text = arena_alloc(arena, tlen + 1);
                    memcpy(text, colon, tlen);
                    text[tlen] = '\0';
                    *out_text = text;
                    break;
                }
            }
            p++;
        }
        if (!*out_text) status = ADAM_ERR_JSON;
    }

    curl_mime_free(mime);
    curl_slist_free_all(headers);
    return status;
}

// ============================================================================
// MARK: - Cloud TTS (OpenAI TTS API)
// ============================================================================

// POST JSON to:
//   https://api.openai.com/v1/audio/speech
//
// Body:
//   {"model":"tts-1","input":"text","voice":"alloy"}
//
// Response: raw audio bytes (mp3 by default)

static adam_status_t cloud_tts(
    adam_settings_t *s, arena_t *arena,
    const char *text,
    uint8_t **out_audio, size_t *out_len
) {
    // Reuse persistent CURL handle
    if (!s->_curl_tts) s->_curl_tts = curl_easy_init();
    CURL *curl = (CURL *)s->_curl_tts;
    if (!curl) return ADAM_ERR_CURL;
    curl_easy_reset(curl);

    const char *url = s->tts_api_url
        ? s->tts_api_url
        : "https://api.openai.com/v1/audio/speech";

    const char *key = s->tts_api_key ? s->tts_api_key : s->api_key;
    if (!key) return ADAM_ERR_AUTH;

    // Determine response_format string for the API
    const char *fmt_str;
    switch (s->tts_format) {
    case ADAM_AUDIO_MP3:      fmt_str = "mp3";  break;
    case ADAM_AUDIO_OGG_OPUS: fmt_str = "opus"; break;
    case ADAM_AUDIO_FLAC:     fmt_str = "flac"; break;
    case ADAM_AUDIO_WAV:      fmt_str = "wav";  break;
    case ADAM_AUDIO_PCM16:    fmt_str = "pcm";  break;
    default:                  fmt_str = "mp3";  break;
    }

    // Build JSON body — escape text properly
    size_t text_len = strlen(text);
    size_t body_cap = text_len * 2 + 256;
    char *body = arena_alloc(arena, body_cap);
    if (!body) return ADAM_ERR_ALLOC;

    // Simple JSON escape for the text
    size_t pos = 0;
    pos += (size_t)snprintf(body + pos, body_cap - pos,
        "{\"model\":\"%s\",\"voice\":\"%s\",\"response_format\":\"%s\",\"input\":\"",
        s->tts_model ? s->tts_model : "tts-1",
        s->tts_voice ? s->tts_voice : "alloy",
        fmt_str);

    for (size_t i = 0; i < text_len && pos < body_cap - 10; i++) {
        char c = text[i];
        switch (c) {
        case '"':  body[pos++] = '\\'; body[pos++] = '"';  break;
        case '\\': body[pos++] = '\\'; body[pos++] = '\\'; break;
        case '\n': body[pos++] = '\\'; body[pos++] = 'n';  break;
        case '\r': body[pos++] = '\\'; body[pos++] = 'r';  break;
        case '\t': body[pos++] = '\\'; body[pos++] = 't';  break;
        default:   body[pos++] = c;
        }
    }
    pos += (size_t)snprintf(body + pos, body_cap - pos, "\"}");

    // Headers
    char *auth_hdr = arena_alloc(arena, strlen(key) + 32);
    snprintf(auth_hdr, strlen(key) + 32, "Authorization: Bearer %s", key);

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, auth_hdr);

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

    // Response is raw audio bytes
    voice_curl_buf_t write_ctx = { .arena = arena };
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, voice_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &write_ctx);

    CURLcode res = curl_easy_perform(curl);
    adam_status_t status = ADAM_OK;

    if (res != CURLE_OK) {
        status = ADAM_ERR_CURL;
    } else {
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code >= 400) {
            if (http_code == 429)
                status = ADAM_ERR_RATE_LIMIT;
            else if (http_code == 401 || http_code == 403)
                status = ADAM_ERR_AUTH;
            else
                status = ADAM_ERR_VOICE;
        }
    }

    if (status == ADAM_OK && write_ctx.buf && write_ctx.len > 0) {
        *out_audio = (uint8_t *)write_ctx.buf;
        *out_len = write_ctx.len;
    } else if (status == ADAM_OK) {
        status = ADAM_ERR_VOICE;
    }

    curl_slist_free_all(headers);
    return status;
}

#endif // ADAM_NO_CURL

// ============================================================================
// MARK: - Default audio playback (via miniaudio)
// ============================================================================

static adam_status_t default_audio_play(
    const uint8_t *audio, size_t len, adam_audio_format_t format
) {
    return adam_audio_play_miniaudio(audio, len, format);
}

// ============================================================================
// MARK: - Retry helper
// ============================================================================

#include <time.h>

static void voice_sleep_ms(int ms) {
    struct timespec ts = { .tv_sec = ms / 1000, .tv_nsec = (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

// ============================================================================
// MARK: - Public API implementations
// ============================================================================

adam_status_t adam_stt_transcribe(adam_settings_t *s, arena_t *arena,
                                  const uint8_t *audio, size_t audio_len,
                                  adam_audio_format_t format,
                                  const char **out_text) {
    if (!s || !arena || !audio || !out_text) return ADAM_ERR_INVALID_PARAM;
    *out_text = NULL;

    // Custom callback takes priority
    if (s->stt_fn) {
        return s->stt_fn(s->stt_ctx, arena, audio, audio_len,
                          format, s->stt_sample_rate, s->stt_language,
                          out_text);
    }

    switch (s->stt_backend) {
    case ADAM_STT_CLOUD: {
#ifndef ADAM_NO_CURL
        // Retry with backoff on rate limit (429)
        int backoff[] = {2000, 5000, 10000, 20000};
        int max_retries = 4;
        adam_status_t rc = ADAM_OK;
        for (int attempt = 0; attempt <= max_retries; attempt++) {
            rc = cloud_stt(s, arena, audio, audio_len, format, out_text);
            if (rc != ADAM_ERR_AUTH && rc != ADAM_ERR_RATE_LIMIT) return rc;
            if (rc == ADAM_OK) return rc;
            if (attempt == max_retries) return rc;
            int delay = backoff[attempt < 4 ? attempt : 3];
            fprintf(stderr, "  [STT] rate limited, retrying in %dms (%d/%d)\n",
                    delay, attempt + 1, max_retries);
            voice_sleep_ms(delay);
            *out_text = NULL;  // reset for retry
        }
        return rc;
#else
        return ADAM_ERR_NOT_IMPLEMENTED;
#endif
    }

    case ADAM_STT_LOCAL:
        return ADAM_ERR_NOT_IMPLEMENTED;

    case ADAM_STT_NONE:
    default:
        return ADAM_ERR_INVALID_PARAM;
    }
}

adam_status_t adam_tts_synthesize(adam_settings_t *s, arena_t *arena,
                                  const char *text,
                                  uint8_t **out_audio, size_t *out_len) {
    if (!s || !arena || !text || !out_audio || !out_len)
        return ADAM_ERR_INVALID_PARAM;
    *out_audio = NULL;
    *out_len = 0;

    // Custom callback takes priority
    if (s->tts_fn) {
        return s->tts_fn(s->tts_ctx, arena, text, s->tts_voice,
                          s->tts_format, out_audio, out_len);
    }

    switch (s->tts_backend) {
    case ADAM_TTS_CLOUD: {
#ifndef ADAM_NO_CURL
        int backoff[] = {2000, 5000, 10000, 20000};
        int max_retries = 4;
        adam_status_t rc = ADAM_OK;
        for (int attempt = 0; attempt <= max_retries; attempt++) {
            rc = cloud_tts(s, arena, text, out_audio, out_len);
            if (rc != ADAM_ERR_RATE_LIMIT) return rc;
            if (attempt == max_retries) return rc;
            int delay = backoff[attempt < 4 ? attempt : 3];
            fprintf(stderr, "  [TTS] rate limited, retrying in %dms (%d/%d)\n",
                    delay, attempt + 1, max_retries);
            voice_sleep_ms(delay);
            *out_audio = NULL;
            *out_len = 0;
        }
        return rc;
#else
        return ADAM_ERR_NOT_IMPLEMENTED;
#endif
    }

    case ADAM_TTS_LOCAL:
        return ADAM_ERR_NOT_IMPLEMENTED;

    case ADAM_TTS_NONE:
    default:
        return ADAM_ERR_INVALID_PARAM;
    }
}

adam_status_t adam_audio_play(adam_settings_t *s,
                              const uint8_t *audio, size_t len,
                              adam_audio_format_t format) {
    if (!s || !audio || len == 0) return ADAM_ERR_INVALID_PARAM;

    if (s->audio_play_fn) {
        return s->audio_play_fn(s->audio_play_ctx, audio, len, format);
    }

    return default_audio_play(audio, len, format);
}

adam_status_t adam_tts_speak(adam_settings_t *s, const char *text) {
    if (!s || !text) return ADAM_ERR_INVALID_PARAM;

    arena_t *arena = arena_create(256 * 1024);
    if (!arena) return ADAM_ERR_ALLOC;

    uint8_t *audio = NULL;
    size_t audio_len = 0;

    adam_status_t rc = adam_tts_synthesize(s, arena, text, &audio, &audio_len);
    if (rc != ADAM_OK) {
        arena_destroy(arena);
        return rc;
    }

    rc = adam_audio_play(s, audio, audio_len, s->tts_format);
    arena_destroy(arena);
    return rc;
}

adam_run_result_t adam_voice_run(adam_settings_t *s, adam_history_t *h,
                                 const uint8_t *audio, size_t audio_len,
                                 adam_audio_format_t format) {
    adam_run_result_t result = {0};
    if (!s || !h || !audio) {
        result.status = ADAM_ERR_INVALID_PARAM;
        result.final_response = strdup("invalid parameters");
        return result;
    }

    // Step 1: Transcribe audio to text
    arena_t *stt_arena = arena_create(64 * 1024);
    if (!stt_arena) {
        result.status = ADAM_ERR_ALLOC;
        result.final_response = strdup("arena alloc failed");
        return result;
    }

    const char *transcribed = NULL;
    adam_status_t stt_rc = adam_stt_transcribe(s, stt_arena, audio, audio_len,
                                               format, &transcribed);
    if (stt_rc != ADAM_OK || !transcribed) {
        result.status = stt_rc;
        result.final_response = strdup("STT transcription failed");
        arena_destroy(stt_arena);
        return result;
    }

    // Copy transcribed text before destroying arena
    char *user_text = strdup(transcribed);
    arena_destroy(stt_arena);

    // Step 2: Run agent with transcribed text
    result = adam_run(s, h, user_text);
    free(user_text);

    // Step 3: Speak the response (if TTS is configured and we got a response)
    if (result.status == ADAM_OK && result.final_response
        && s->tts_backend != ADAM_TTS_NONE) {
        adam_tts_speak(s, result.final_response);
    }

    return result;
}

// ============================================================================
// MARK: - Voice thread stubs (unchanged)
// ============================================================================

adam_status_t adam_voice_start(adam_settings_t *s, adam_history_t *h) {
    UNUSED_PARAM(h);
    if (!s) return ADAM_ERR_INVALID_PARAM;
    if (!s->voice_enabled) return ADAM_ERR_INVALID_PARAM;
    if (s->stt_backend == ADAM_STT_NONE) return ADAM_ERR_INVALID_PARAM;
    return ADAM_ERR_NOT_IMPLEMENTED;
}

void adam_voice_stop(adam_settings_t *s) {
    if (!s) return;
}

int adam_voice_is_running(const adam_settings_t *s) {
    if (!s) return 0;
    return (s->_voice != NULL);
}

#endif // !ADAM_NO_VOICE && !ADAM_NO_PTHREADS
