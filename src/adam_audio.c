//
//  adam_audio.c
//  Adam — Cross-platform audio capture and playback via miniaudio
//
//  Replaces platform-specific code (adam_mic_macos.m, afplay, aplay)
//  with a single portable C implementation.
//
//  Created by Marco Bambini on 15/03/26.
//

#if !defined(ADAM_NO_VOICE) && !defined(ADAM_NO_PTHREADS)

// miniaudio: single-header library — define implementation once
#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING      // we don't need encoding (WAV writing done manually)
#define MA_NO_GENERATION    // we don't need waveform generation
#include "miniaudio.h"

#include "adam.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// ============================================================================
// MARK: - Audio Playback
// ============================================================================

// Playback state shared between play function and callback
typedef struct {
    ma_decoder  *decoder;
    volatile int done;
} playback_ctx_t;

// Miniaudio playback data callback — feeds decoded audio to device
static void playback_data_callback(ma_device *device, void *output,
                                    const void *input, ma_uint32 frame_count) {
    (void)input;
    playback_ctx_t *pc = (playback_ctx_t *)device->pUserData;
    ma_uint64 frames_read;
    ma_decoder_read_pcm_frames(pc->decoder, output, frame_count, &frames_read);
    if (frames_read < frame_count) {
        ma_uint32 bytes_per_frame = ma_get_bytes_per_frame(
            device->playback.format, device->playback.channels);
        memset((uint8_t *)output + frames_read * bytes_per_frame,
               0, (frame_count - (ma_uint32)frames_read) * bytes_per_frame);
        pc->done = 1;
    }
}

// Play audio data synchronously. Blocks until playback completes.
adam_status_t adam_audio_play_miniaudio(
    const uint8_t *audio_data, size_t audio_len, adam_audio_format_t format
) {
    if (!audio_data || audio_len == 0) return ADAM_ERR_INVALID_PARAM;

    // Write to temp file (miniaudio's decoder works best with files)
    const char *ext;
    switch (format) {
    case ADAM_AUDIO_WAV:      ext = "wav";  break;
    case ADAM_AUDIO_MP3:      ext = "mp3";  break;
    case ADAM_AUDIO_FLAC:     ext = "flac"; break;
    default:                  ext = "mp3";  break;
    }

    char tmppath[256];
    snprintf(tmppath, sizeof(tmppath), "/tmp/adam_play_%d.%s", (int)getpid(), ext);

    FILE *f = fopen(tmppath, "wb");
    if (!f) return ADAM_ERR_VOICE;
    fwrite(audio_data, 1, audio_len, f);
    fclose(f);

    // Decode and play
    ma_result result;
    ma_decoder decoder;
    result = ma_decoder_init_file(tmppath, NULL, &decoder);
    if (result != MA_SUCCESS) {
        remove(tmppath);
        return ADAM_ERR_VOICE;
    }

    ma_device_config config = ma_device_config_init(ma_device_type_playback);
    config.playback.format   = decoder.outputFormat;
    config.playback.channels = decoder.outputChannels;
    config.sampleRate        = decoder.outputSampleRate;

    // Playback state
    typedef struct {
        ma_decoder *decoder;
        volatile int done;
    } playback_ctx_t;

    playback_ctx_t pctx = { .decoder = &decoder, .done = 0 };

    config.dataCallback = playback_data_callback;
    config.pUserData    = &pctx;

    ma_device device;
    result = ma_device_init(NULL, &config, &device);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        remove(tmppath);
        return ADAM_ERR_VOICE;
    }

    result = ma_device_start(&device);
    if (result != MA_SUCCESS) {
        ma_device_uninit(&device);
        ma_decoder_uninit(&decoder);
        remove(tmppath);
        return ADAM_ERR_VOICE;
    }

    // Wait for playback to finish
    while (!pctx.done) {
        ma_sleep(50);
    }
    // Small tail delay to let the last buffer drain
    ma_sleep(100);

    ma_device_uninit(&device);
    ma_decoder_uninit(&decoder);
    remove(tmppath);
    return ADAM_OK;
}

// ============================================================================
// MARK: - Streaming PCM Playback
// ============================================================================

// Lock-free-ish ring buffer for producer (curl) → consumer (miniaudio).
// Single producer, single consumer. Sizes are power-of-two for fast modulo.

#define PCM_RING_SIZE (256 * 1024)  // 256 KB ≈ ~5s of 24kHz mono s16

#include <pthread.h>

typedef struct adam_pcm_player_t adam_pcm_player_t;

struct adam_pcm_player_t {
    ma_device    device;
    int16_t      ring[PCM_RING_SIZE / sizeof(int16_t)];
    volatile size_t write_pos;   // producer (curl thread) writes here (in samples)
    volatile size_t read_pos;    // consumer (audio thread) reads here (in samples)
    volatile int    finished;    // producer sets when no more data
    volatile int    started;     // set once device is running
    int             sample_rate;
    int             channels;
};

#define RING_SAMPLES (PCM_RING_SIZE / sizeof(int16_t))

static size_t ring_available(adam_pcm_player_t *p) {
    size_t w = p->write_pos;
    size_t r = p->read_pos;
    return (w >= r) ? (w - r) : (RING_SAMPLES - r + w);
}

static void pcm_playback_callback(ma_device *device, void *output,
                                    const void *input, ma_uint32 frame_count) {
    (void)input;
    adam_pcm_player_t *p = (adam_pcm_player_t *)device->pUserData;
    int16_t *out = (int16_t *)output;
    size_t frames_needed = frame_count * (size_t)p->channels;
    size_t avail = ring_available(p);

    size_t to_read = (avail < frames_needed) ? avail : frames_needed;
    for (size_t i = 0; i < to_read; i++) {
        out[i] = p->ring[p->read_pos % RING_SAMPLES];
        p->read_pos = (p->read_pos + 1) % RING_SAMPLES;
    }
    // Fill remainder with silence
    for (size_t i = to_read; i < frames_needed; i++) {
        out[i] = 0;
    }
}

adam_pcm_player_t *adam_pcm_player_start(int sample_rate, int channels) {
    adam_pcm_player_t *p = calloc(1, sizeof(adam_pcm_player_t));
    if (!p) return NULL;

    p->sample_rate = sample_rate;
    p->channels = channels;

    ma_device_config config = ma_device_config_init(ma_device_type_playback);
    config.playback.format   = ma_format_s16;
    config.playback.channels = (ma_uint32)channels;
    config.sampleRate        = (ma_uint32)sample_rate;
    config.dataCallback      = pcm_playback_callback;
    config.pUserData         = p;
    config.periodSizeInMilliseconds = 50;  // low latency

    if (ma_device_init(NULL, &config, &p->device) != MA_SUCCESS) {
        free(p);
        return NULL;
    }

    if (ma_device_start(&p->device) != MA_SUCCESS) {
        ma_device_uninit(&p->device);
        free(p);
        return NULL;
    }

    p->started = 1;
    return p;
}

void adam_pcm_player_feed(adam_pcm_player_t *p, const uint8_t *pcm_data, size_t len) {
    if (!p || !pcm_data || len == 0) return;

    // pcm_data is raw int16_t samples (little-endian)
    size_t samples = len / sizeof(int16_t);
    const int16_t *src = (const int16_t *)pcm_data;

    for (size_t i = 0; i < samples; i++) {
        // Spin-wait if ring is full (very rare — playback is real-time)
        size_t next = (p->write_pos + 1) % RING_SAMPLES;
        while (next == p->read_pos) {
            ma_sleep(1);
        }
        p->ring[p->write_pos] = src[i];
        p->write_pos = next;
    }
}

void adam_pcm_player_finish(adam_pcm_player_t *p) {
    if (!p) return;

    p->finished = 1;

    // Wait for ring buffer to drain
    while (ring_available(p) > 0) {
        ma_sleep(20);
    }
    // Extra delay to let the last audio buffer play out
    ma_sleep(150);

    ma_device_uninit(&p->device);
    free(p);
}

// ============================================================================
// MARK: - Audio Capture (Microphone Recording)
// ============================================================================

typedef struct {
    int16_t    *buffer;        // PCM16 sample buffer
    size_t      capacity;      // max samples
    size_t      count;         // samples written
    volatile int stop;         // set to 1 to stop
} capture_ctx_t;

static void capture_callback(ma_device *device, void *output,
                              const void *input, ma_uint32 frame_count) {
    (void)output;
    capture_ctx_t *ctx = (capture_ctx_t *)device->pUserData;
    if (ctx->stop) return;

    const int16_t *samples = (const int16_t *)input;
    size_t to_copy = frame_count;
    if (ctx->count + to_copy > ctx->capacity)
        to_copy = ctx->capacity - ctx->count;

    if (to_copy > 0) {
        memcpy(ctx->buffer + ctx->count, samples, to_copy * sizeof(int16_t));
        ctx->count += to_copy;
    }

    // Auto-stop if buffer is full
    if (ctx->count >= ctx->capacity)
        ctx->stop = 1;
}

// Record from the default microphone.
// Returns a malloc'd WAV buffer. Caller must free it.
// Blocks until max_seconds elapse or *stop_flag is set.
uint8_t *adam_audio_record(int sample_rate, int max_seconds,
                            volatile int *stop_flag, size_t *out_len) {
    *out_len = 0;

    size_t max_samples = (size_t)(sample_rate * max_seconds);
    capture_ctx_t ctx = {
        .buffer = malloc(max_samples * sizeof(int16_t)),
        .capacity = max_samples,
        .count = 0,
        .stop = 0,
    };
    if (!ctx.buffer) return NULL;

    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.format   = ma_format_s16;
    config.capture.channels = 1;
    config.sampleRate       = (ma_uint32)sample_rate;
    config.dataCallback     = capture_callback;
    config.pUserData        = &ctx;

    ma_device device;
    if (ma_device_init(NULL, &config, &device) != MA_SUCCESS) {
        free(ctx.buffer);
        return NULL;
    }

    if (ma_device_start(&device) != MA_SUCCESS) {
        ma_device_uninit(&device);
        free(ctx.buffer);
        return NULL;
    }

    // Wait for stop signal, timeout, or buffer full
    while (!ctx.stop) {
        if (stop_flag && *stop_flag) {
            ctx.stop = 1;
            break;
        }
        ma_sleep(50);
    }

    ma_device_uninit(&device);

    if (ctx.count == 0) {
        free(ctx.buffer);
        return NULL;
    }

    // Build WAV in memory
    uint32_t data_size = (uint32_t)(ctx.count * sizeof(int16_t));
    size_t wav_size = 44 + data_size; // 44 = WAV header
    uint8_t *wav = malloc(wav_size);
    if (!wav) { free(ctx.buffer); return NULL; }

    // Write header to a temp FILE* via fmemopen, or just do it manually
    // WAV header (44 bytes):
    uint32_t chunk_size = 36 + data_size;
    uint32_t sr = (uint32_t)sample_rate;
    uint32_t byte_rate = sr * 1 * 16 / 8;
    uint16_t block_align = 1 * 16 / 8;

    size_t p = 0;
    memcpy(wav + p, "RIFF", 4); p += 4;
    memcpy(wav + p, &chunk_size, 4); p += 4;
    memcpy(wav + p, "WAVE", 4); p += 4;
    memcpy(wav + p, "fmt ", 4); p += 4;
    uint32_t fmt_size = 16;
    memcpy(wav + p, &fmt_size, 4); p += 4;
    uint16_t audio_format = 1;
    memcpy(wav + p, &audio_format, 2); p += 2;
    uint16_t num_channels = 1;
    memcpy(wav + p, &num_channels, 2); p += 2;
    memcpy(wav + p, &sr, 4); p += 4;
    memcpy(wav + p, &byte_rate, 4); p += 4;
    memcpy(wav + p, &block_align, 2); p += 2;
    uint16_t bits = 16;
    memcpy(wav + p, &bits, 2); p += 2;
    memcpy(wav + p, "data", 4); p += 4;
    memcpy(wav + p, &data_size, 4); p += 4;
    // PCM data
    memcpy(wav + p, ctx.buffer, data_size);

    free(ctx.buffer);
    *out_len = wav_size;
    return wav;
}

#endif // !ADAM_NO_VOICE && !ADAM_NO_PTHREADS
