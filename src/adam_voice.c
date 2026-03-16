//
//  adam_voice.c
//  Adam — Voice subsystem: cloud STT/TTS + audio playback
//
//  Uses adam_net.h for HTTP (platform-independent).
//
//  Created by Marco Bambini on 15/03/26.
//

#if !defined(ADAM_NO_VOICE) && !defined(ADAM_NO_PTHREADS)

#include "adam.h"
#include "adam_net.h"
#include "adam_audio.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#define AUTH_HDR_EXTRA 32

// System TTS (implemented in adam_tts_system.m / adam_tts_system.c)
extern adam_status_t adam_tts_system_speak(const char *text, const char *language);

// ============================================================================
// MARK: - Helpers
// ============================================================================

static void voice_sleep_ms(int ms) {
    struct timespec ts = { .tv_sec = ms / 1000, .tv_nsec = (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

static char *make_auth(arena_t *arena, const char *key) {
    size_t len = strlen(key) + AUTH_HDR_EXTRA;
    char *h = arena_alloc(arena, len);
    if (h) snprintf(h, len, "Authorization: Bearer %s", key);
    return h;
}

static char *make_auth_malloc(const char *key) {
    size_t len = strlen(key) + AUTH_HDR_EXTRA;
    char *h = malloc(len);
    if (h) snprintf(h, len, "Authorization: Bearer %s", key);
    return h;
}

// ============================================================================
// MARK: - Cloud STT
// ============================================================================

static adam_status_t cloud_stt(
    adam_settings_t *s, arena_t *arena,
    const uint8_t *audio, size_t audio_len,
    adam_audio_format_t format,
    const char **out_text
) {
    const char *url = s->stt_api_url
        ? s->stt_api_url
        : "https://api.openai.com/v1/audio/transcriptions";
    const char *key = s->stt_api_key ? s->stt_api_key : s->api_key;
    if (!key) return ADAM_ERR_AUTH;

    const char *filename;
    const char *mimetype;
    switch (format) {
    case ADAM_AUDIO_WAV:      filename = "audio.wav";  mimetype = "audio/wav";  break;
    case ADAM_AUDIO_MP3:      filename = "audio.mp3";  mimetype = "audio/mpeg"; break;
    case ADAM_AUDIO_OGG_OPUS: filename = "audio.ogg";  mimetype = "audio/ogg";  break;
    case ADAM_AUDIO_FLAC:     filename = "audio.flac"; mimetype = "audio/flac"; break;
    default:                  filename = "audio.wav";  mimetype = "audio/wav";  break;
    }

    adam_net_field_t fields[4];
    size_t nfields = 0;

    fields[nfields++] = (adam_net_field_t){
        .name = "file", .data = audio, .data_len = audio_len,
        .filename = filename, .content_type = mimetype
    };
    fields[nfields++] = (adam_net_field_t){
        .name = "model", .value = s->stt_model ? s->stt_model : "gpt-4o-mini-transcribe"
    };
    if (s->stt_language) {
        fields[nfields++] = (adam_net_field_t){
            .name = "language", .value = s->stt_language
        };
    }
    fields[nfields++] = (adam_net_field_t){
        .name = "response_format", .value = "json"
    };

    char *auth = make_auth(arena, key);
    if (!auth) return ADAM_ERR_ALLOC;

    adam_net_response_t net = adam_net_post_multipart(
        s, arena, url, auth, fields, nfields, /*handle_id=*/1
    );

    if (net.error != ADAM_OK) return ADAM_ERR_CURL;

    if (net.http_code >= 400) {
        if (net.data && net.data_len > 0)
            fprintf(stderr, "  [STT] HTTP %ld: %.200s\n",
                    net.http_code, (const char *)net.data);
        if (net.http_code == 429) return ADAM_ERR_RATE_LIMIT;
        if (net.http_code == 401 || net.http_code == 403) return ADAM_ERR_AUTH;
        return ADAM_ERR_VOICE;
    }

    // Parse {"text": "..."} from response
    if (net.data && net.data_len > 0) {
        const char *p = (const char *)net.data;
        while ((p = strstr(p, "\"text\"")) != NULL) {
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
                    if (text) { memcpy(text, colon, tlen); text[tlen] = '\0'; }
                    *out_text = text;
                    return ADAM_OK;
                }
            }
            p++;
        }
    }

    return ADAM_ERR_JSON;
}

// ============================================================================
// MARK: - Cloud TTS (non-streaming)
// ============================================================================

static const char *tts_build_body(arena_t *arena, adam_settings_t *s,
                                   const char *text, const char *fmt_str) {
    size_t text_len = strlen(text);
    size_t cap = text_len * 2 + 256;
    char *body = arena_alloc(arena, cap);
    if (!body) return NULL;

    size_t pos = 0;
    pos += (size_t)snprintf(body + pos, cap - pos,
        "{\"model\":\"%s\",\"voice\":\"%s\",\"response_format\":\"%s\",\"input\":\"",
        s->tts_model ? s->tts_model : "gpt-4o-mini-tts",
        s->tts_voice ? s->tts_voice : "coral", fmt_str);

    for (size_t i = 0; i < text_len && pos < cap - 10; i++) {
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
    pos += (size_t)snprintf(body + pos, cap - pos, "\"}");
    return body;
}

static adam_status_t cloud_tts(
    adam_settings_t *s, arena_t *arena,
    const char *text, uint8_t **out_audio, size_t *out_len
) {
    const char *url = s->tts_api_url
        ? s->tts_api_url
        : "https://api.openai.com/v1/audio/speech";
    const char *key = s->tts_api_key ? s->tts_api_key : s->api_key;
    if (!key) return ADAM_ERR_AUTH;

    const char *fmt_str;
    switch (s->tts_format) {
    case ADAM_AUDIO_MP3:      fmt_str = "mp3";  break;
    case ADAM_AUDIO_OGG_OPUS: fmt_str = "opus"; break;
    case ADAM_AUDIO_FLAC:     fmt_str = "flac"; break;
    case ADAM_AUDIO_WAV:      fmt_str = "wav";  break;
    case ADAM_AUDIO_PCM16:    fmt_str = "pcm";  break;
    default:                  fmt_str = "mp3";  break;
    }

    const char *body = tts_build_body(arena, s, text, fmt_str);
    if (!body) return ADAM_ERR_ALLOC;

    char *auth = make_auth(arena, key);
    if (!auth) return ADAM_ERR_ALLOC;

    adam_net_response_t net = adam_net_post_json(
        s, arena, url, auth, body, NULL, /*handle_id=*/2
    );

    if (net.error != ADAM_OK) return ADAM_ERR_CURL;

    if (net.http_code >= 400) {
        if (net.http_code == 429) return ADAM_ERR_RATE_LIMIT;
        if (net.http_code == 401 || net.http_code == 403) return ADAM_ERR_AUTH;
        return ADAM_ERR_VOICE;
    }

    if (net.data && net.data_len > 0) {
        *out_audio = net.data;
        *out_len = net.data_len;
        return ADAM_OK;
    }
    return ADAM_ERR_VOICE;
}

// ============================================================================
// MARK: - Streaming TTS (PCM: HTTP → ring buffer → miniaudio)
// ============================================================================

typedef struct {
    adam_pcm_player_t *player;
    uint8_t            prebuf[12288];
    size_t             prebuf_len;
    int                is_error;
} stream_tts_ctx_t;

static int stream_tts_chunk(void *ctx, const uint8_t *chunk, size_t len) {
    stream_tts_ctx_t *sc = (stream_tts_ctx_t *)ctx;
    if (sc->is_error) return 0; // swallow

    // Buffer initial data to detect JSON errors vs PCM
    if (!sc->player) {
        size_t space = sizeof(sc->prebuf) - sc->prebuf_len;
        size_t copy = len < space ? len : space;
        memcpy(sc->prebuf + sc->prebuf_len, chunk, copy);
        sc->prebuf_len += copy;

        // JSON errors start with '{'
        const uint8_t *p = sc->prebuf;
        size_t plen = sc->prebuf_len;
        while (plen > 0 && (*p == ' ' || *p == '\n' || *p == '\r')) { p++; plen--; }
        if (plen > 0 && *p == '{') { sc->is_error = 1; return 0; }

        // Wait for ~200ms of audio (9600 bytes at 24kHz s16 mono)
        if (sc->prebuf_len < 9600 && len == copy
            && sc->prebuf_len < sizeof(sc->prebuf))
            return 0; // keep buffering

        // Fade-in (~2ms = 48 samples)
        size_t fade = 48;
        size_t prebuf_samp = sc->prebuf_len / 2;
        if (prebuf_samp > fade) {
            int16_t *samples = (int16_t *)sc->prebuf;
            for (size_t i = 0; i < fade; i++)
                samples[i] = (int16_t)((int32_t)samples[i] * (int32_t)i / (int32_t)fade);
        }

        sc->player = adam_pcm_player_start(24000, 1);
        if (!sc->player) { sc->is_error = 1; return 0; }

        adam_pcm_player_feed(sc->player, sc->prebuf, sc->prebuf_len);
        if (copy < len)
            adam_pcm_player_feed(sc->player, chunk + copy, len - copy);
        return 0;
    }

    adam_pcm_player_feed(sc->player, chunk, len);
    return 0;
}

static adam_status_t cloud_tts_streaming(adam_settings_t *s, const char *text) {
    const char *url = s->tts_api_url
        ? s->tts_api_url
        : "https://api.openai.com/v1/audio/speech";
    const char *key = s->tts_api_key ? s->tts_api_key : s->api_key;
    if (!key) return ADAM_ERR_AUTH;

    // Build body with PCM format (malloc'd — arena not available here)
    size_t text_len = strlen(text);
    size_t cap = text_len * 2 + 256;
    char *body = malloc(cap);
    if (!body) return ADAM_ERR_ALLOC;

    size_t pos = 0;
    pos += (size_t)snprintf(body + pos, cap - pos,
        "{\"model\":\"%s\",\"voice\":\"%s\",\"response_format\":\"pcm\",\"input\":\"",
        s->tts_model ? s->tts_model : "gpt-4o-mini-tts",
        s->tts_voice ? s->tts_voice : "coral");
    for (size_t i = 0; i < text_len && pos < cap - 10; i++) {
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
    pos += (size_t)snprintf(body + pos, cap - pos, "\"}");

    char *auth = make_auth_malloc(key);
    if (!auth) { free(body); return ADAM_ERR_ALLOC; }

    stream_tts_ctx_t sc = {0};

    adam_net_response_t net = adam_net_post_streaming(
        s, url, auth, body, NULL, stream_tts_chunk, &sc, /*handle_id=*/2
    );

    free(auth);
    free(body);

    adam_status_t status = ADAM_OK;
    if (net.error != ADAM_OK) {
        status = ADAM_ERR_CURL;
    } else if (net.http_code == 429) {
        status = ADAM_ERR_RATE_LIMIT;
    } else if (net.http_code == 401 || net.http_code == 403) {
        status = ADAM_ERR_AUTH;
    } else if (net.http_code >= 400 || sc.is_error) {
        status = ADAM_ERR_VOICE;
    }

    if (sc.player) adam_pcm_player_finish(sc.player);
    return status;
}

// ============================================================================
// MARK: - Default audio playback
// ============================================================================

static adam_status_t default_audio_play(
    const uint8_t *audio, size_t len, adam_audio_format_t format
) {
    return adam_audio_play_miniaudio(audio, len, format);
}

// ============================================================================
// MARK: - Public API
// ============================================================================

adam_status_t adam_stt_transcribe(adam_settings_t *s, arena_t *arena,
                                  const uint8_t *audio, size_t audio_len,
                                  adam_audio_format_t format,
                                  const char **out_text) {
    if (!s || !arena || !audio || !out_text) return ADAM_ERR_INVALID_PARAM;
    *out_text = NULL;

    if (s->stt_fn)
        return s->stt_fn(s->stt_ctx, arena, audio, audio_len,
                          format, s->stt_sample_rate, s->stt_language, out_text);

    switch (s->stt_backend) {
    case ADAM_STT_CLOUD: {
        int backoff[] = {2000, 5000, 10000, 20000};
        adam_status_t rc = ADAM_OK;
        for (int attempt = 0; attempt <= 4; attempt++) {
            rc = cloud_stt(s, arena, audio, audio_len, format, out_text);
            if (rc != ADAM_ERR_RATE_LIMIT) return rc;
            if (attempt == 4) return rc;
            fprintf(stderr, "  [STT] rate limited, retrying in %dms (%d/4)\n",
                    backoff[attempt], attempt + 1);
            voice_sleep_ms(backoff[attempt]);
            *out_text = NULL;
        }
        return rc;
    }
    case ADAM_STT_LOCAL:  return ADAM_ERR_NOT_IMPLEMENTED;
    default:             return ADAM_ERR_INVALID_PARAM;
    }
}

adam_status_t adam_tts_synthesize(adam_settings_t *s, arena_t *arena,
                                  const char *text,
                                  uint8_t **out_audio, size_t *out_len) {
    if (!s || !arena || !text || !out_audio || !out_len)
        return ADAM_ERR_INVALID_PARAM;
    *out_audio = NULL; *out_len = 0;

    if (s->tts_fn)
        return s->tts_fn(s->tts_ctx, arena, text, s->tts_voice,
                          s->tts_format, out_audio, out_len);

    switch (s->tts_backend) {
    case ADAM_TTS_CLOUD: {
        int backoff[] = {2000, 5000, 10000, 20000};
        adam_status_t rc = ADAM_OK;
        for (int attempt = 0; attempt <= 4; attempt++) {
            rc = cloud_tts(s, arena, text, out_audio, out_len);
            if (rc != ADAM_ERR_RATE_LIMIT) return rc;
            if (attempt == 4) return rc;
            fprintf(stderr, "  [TTS] rate limited, retrying in %dms (%d/4)\n",
                    backoff[attempt], attempt + 1);
            voice_sleep_ms(backoff[attempt]);
            *out_audio = NULL; *out_len = 0;
        }
        return rc;
    }
    case ADAM_TTS_SYSTEM:
        // System TTS plays directly — can't return audio data.
        // Use adam_tts_speak() instead for ADAM_TTS_SYSTEM.
        return ADAM_ERR_NOT_IMPLEMENTED;
    case ADAM_TTS_LOCAL:  return ADAM_ERR_NOT_IMPLEMENTED;
    default:             return ADAM_ERR_INVALID_PARAM;
    }
}

adam_status_t adam_audio_play(adam_settings_t *s,
                              const uint8_t *audio, size_t len,
                              adam_audio_format_t format) {
    if (!s || !audio || len == 0) return ADAM_ERR_INVALID_PARAM;
    if (s->audio_play_fn)
        return s->audio_play_fn(s->audio_play_ctx, audio, len, format);
    return default_audio_play(audio, len, format);
}

adam_status_t adam_tts_speak(adam_settings_t *s, const char *text) {
    if (!s || !text) return ADAM_ERR_INVALID_PARAM;

    // System TTS: direct playback via OS engine (no audio data round-trip)
    if (s->tts_backend == ADAM_TTS_SYSTEM && !s->tts_fn) {
        return adam_tts_system_speak(text, s->stt_language);
    }

    // Streaming path for cloud TTS (overlaps download + playback)
    if (s->tts_backend == ADAM_TTS_CLOUD && !s->tts_fn) {
        int backoff[] = {2000, 5000, 10000, 20000};
        for (int attempt = 0; attempt <= 4; attempt++) {
            adam_status_t rc = cloud_tts_streaming(s, text);
            if (rc != ADAM_ERR_RATE_LIMIT) return rc;
            if (attempt == 4) return rc;
            fprintf(stderr, "  [TTS] rate limited, retrying in %dms (%d/4)\n",
                    backoff[attempt], attempt + 1);
            voice_sleep_ms(backoff[attempt]);
        }
    }

    // Fallback: non-streaming (synthesize fully, then play)
    arena_t *arena = arena_create(256 * 1024);
    if (!arena) return ADAM_ERR_ALLOC;
    uint8_t *audio = NULL; size_t audio_len = 0;
    adam_status_t rc = adam_tts_synthesize(s, arena, text, &audio, &audio_len);
    if (rc == ADAM_OK)
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

    char *user_text = strdup(transcribed);
    arena_destroy(stt_arena);

    result = adam_run(s, h, user_text);
    free(user_text);

    if (result.status == ADAM_OK && result.final_response
        && s->tts_backend != ADAM_TTS_NONE) {
        adam_tts_speak(s, result.final_response);
    }

    return result;
}

// ============================================================================
// MARK: - Voice thread stubs
// ============================================================================

adam_status_t adam_voice_start(adam_settings_t *s, adam_history_t *h) {
    UNUSED_PARAM(h);
    if (!s || !s->voice_enabled || s->stt_backend == ADAM_STT_NONE)
        return ADAM_ERR_INVALID_PARAM;
    return ADAM_ERR_NOT_IMPLEMENTED;
}

void adam_voice_stop(adam_settings_t *s) { UNUSED_PARAM(s); }

int adam_voice_is_running(const adam_settings_t *s) {
    return (s && s->_voice != NULL);
}

#endif // !ADAM_NO_VOICE && !ADAM_NO_PTHREADS
