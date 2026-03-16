//
//  adam_stt_local.c
//  Adam — Local STT via whisper.cpp
//
//  Created by Marco Bambini on 16/03/26.
//

#ifndef ADAM_NO_LOCAL

#include "adam.h"
#include "whisper.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// ============================================================================
// MARK: - Persistent whisper context
// ============================================================================

typedef struct {
    struct whisper_context *ctx;
} adam_whisper_ctx_t;

// ============================================================================
// MARK: - WAV PCM16 → float32 conversion
// ============================================================================

// Whisper expects 32-bit float samples at 16kHz mono.
// Our recorder produces 16-bit signed PCM at 16kHz mono in a WAV container.
// Skip the 44-byte WAV header and convert int16 → float.
static float *wav_to_float(arena_t *arena, const uint8_t *wav_data,
                            size_t wav_len, size_t *out_samples) {
    *out_samples = 0;
    if (!wav_data || wav_len < 44) return NULL;

    // Verify WAV header
    if (memcmp(wav_data, "RIFF", 4) != 0 || memcmp(wav_data + 8, "WAVE", 4) != 0)
        return NULL;

    // Find "data" chunk
    const uint8_t *p = wav_data + 12;
    const uint8_t *end = wav_data + wav_len;
    while (p + 8 <= end) {
        uint32_t chunk_size;
        memcpy(&chunk_size, p + 4, 4);
        if (memcmp(p, "data", 4) == 0) {
            p += 8; // skip "data" + size
            size_t pcm_bytes = (size_t)chunk_size;
            if (p + pcm_bytes > end) pcm_bytes = (size_t)(end - p);
            size_t n_samples = pcm_bytes / 2;
            float *out = arena_alloc(arena, n_samples * sizeof(float));
            if (!out) return NULL;
            const int16_t *src = (const int16_t *)p;
            for (size_t i = 0; i < n_samples; i++) {
                int16_t sample;
                memcpy(&sample, src + i, sizeof(int16_t)); // safe unaligned
                out[i] = (float)sample / 32768.0f;
            }
            *out_samples = n_samples;
            return out;
        }
        p += 8 + chunk_size;
        if (chunk_size % 2) p++; // WAV chunks are 2-byte aligned
    }
    return NULL;
}

// Also handle raw PCM16 (no WAV header)
static float *pcm16_to_float(arena_t *arena, const uint8_t *data,
                              size_t len, size_t *out_samples) {
    size_t n = len / 2;
    float *out = arena_alloc(arena, n * sizeof(float));
    if (!out) { *out_samples = 0; return NULL; }
    for (size_t i = 0; i < n; i++) {
        int16_t sample;
        memcpy(&sample, data + i * 2, 2);
        out[i] = (float)sample / 32768.0f;
    }
    *out_samples = n;
    return out;
}

// ============================================================================
// MARK: - Public: local STT transcription
// ============================================================================

adam_status_t adam_stt_local_transcribe(
    adam_settings_t *s, arena_t *arena,
    const uint8_t *audio, size_t audio_len,
    adam_audio_format_t format,
    const char **out_text
) {
    if (!s || !arena || !audio || audio_len == 0 || !out_text)
        return ADAM_ERR_INVALID_PARAM;
    *out_text = NULL;

    // We need a whisper model path — reuse stt_model as the GGUF path
    // for local STT (e.g. "models/ggml-base.en.bin")
    const char *model_path = s->stt_model;
    if (!model_path || model_path[0] == '\0')
        return ADAM_ERR_INVALID_PARAM;

    // Lazy-init whisper context
    // Store in a separate slot to avoid conflict with llama.cpp's _local_ctx.
    // We use the _voice pointer (adam_voice_t*) which is otherwise unused.
    adam_whisper_ctx_t *wctx = (adam_whisper_ctx_t *)s->_voice;
    if (!wctx) {
        struct whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = true;

        struct whisper_context *ctx = whisper_init_from_file_with_params(
            model_path, cparams);
        if (!ctx) {
            fprintf(stderr, "[adam_stt_local] failed to load whisper model: %s\n",
                    model_path);
            return ADAM_ERR_LOCAL;
        }

        wctx = calloc(1, sizeof(adam_whisper_ctx_t));
        if (!wctx) { whisper_free(ctx); return ADAM_ERR_ALLOC; }
        wctx->ctx = ctx;
        s->_voice = (adam_voice_t *)wctx;
    }

    // Convert audio to float32 samples
    float *samples = NULL;
    size_t n_samples = 0;

    if (format == ADAM_AUDIO_WAV) {
        samples = wav_to_float(arena, audio, audio_len, &n_samples);
    } else if (format == ADAM_AUDIO_PCM16) {
        samples = pcm16_to_float(arena, audio, audio_len, &n_samples);
    } else {
        // Unsupported format for local STT (MP3/OGG/FLAC would need a decoder)
        return ADAM_ERR_INVALID_PARAM;
    }

    if (!samples || n_samples == 0)
        return ADAM_ERR_VOICE;

    // Run whisper
    struct whisper_full_params wparams = whisper_full_default_params(
        WHISPER_SAMPLING_GREEDY);
    wparams.print_realtime = false;
    wparams.print_progress = false;
    wparams.print_timestamps = false;
    wparams.print_special = false;
    wparams.single_segment = false;
    wparams.no_timestamps = true;

    // Language hint (if set)
    if (s->stt_language)
        wparams.language = s->stt_language;
    else
        wparams.language = "auto";

    int rc = whisper_full(wctx->ctx, wparams, samples, (int)n_samples);
    if (rc != 0) {
        return ADAM_ERR_VOICE;
    }

    // Collect segments into a single string
    int n_seg = whisper_full_n_segments(wctx->ctx);
    if (n_seg <= 0) {
        *out_text = arena_strdup(arena, "");
        return ADAM_OK;
    }

    // Calculate total length
    size_t total_len = 0;
    for (int i = 0; i < n_seg; i++) {
        const char *seg = whisper_full_get_segment_text(wctx->ctx, i);
        if (seg) total_len += strlen(seg);
    }

    char *result = arena_alloc(arena, total_len + 1);
    if (!result) return ADAM_ERR_ALLOC;

    size_t pos = 0;
    for (int i = 0; i < n_seg; i++) {
        const char *seg = whisper_full_get_segment_text(wctx->ctx, i);
        if (seg) {
            size_t slen = strlen(seg);
            memcpy(result + pos, seg, slen);
            pos += slen;
        }
    }
    result[pos] = '\0';

    // Trim leading whitespace (whisper often prepends a space)
    const char *trimmed = result;
    while (*trimmed == ' ') trimmed++;

    *out_text = (trimmed != result) ? arena_strdup(arena, trimmed) : result;
    return ADAM_OK;
}

// ============================================================================
// MARK: - Cleanup
// ============================================================================

void adam_stt_local_cleanup(adam_settings_t *s) {
    if (!s || !s->_voice) return;
    adam_whisper_ctx_t *wctx = (adam_whisper_ctx_t *)s->_voice;
    if (wctx->ctx) whisper_free(wctx->ctx);
    free(wctx);
    s->_voice = NULL;
}

#endif // ADAM_NO_LOCAL
