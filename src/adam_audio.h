//
//  adam_audio.h
//  Adam — Cross-platform audio via miniaudio (internal header)
//
//  Created by Marco Bambini on 15/03/26.
//

#pragma once

#include "adam.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(ADAM_NO_VOICE) && !defined(ADAM_NO_PTHREADS)

// Play audio data synchronously. Blocks until playback completes.
// Supports WAV, MP3, FLAC via miniaudio's built-in decoders.
adam_status_t adam_audio_play_miniaudio(
    const uint8_t *audio_data, size_t audio_len, adam_audio_format_t format);

// Streaming PCM playback: start device, then feed chunks, then finish.
// The device plays audio in real-time as chunks arrive.
// PCM format: signed 16-bit LE, 24000 Hz, mono (OpenAI TTS PCM output).
typedef struct adam_pcm_player_t adam_pcm_player_t;

adam_pcm_player_t *adam_pcm_player_start(int sample_rate, int channels);
void               adam_pcm_player_feed(adam_pcm_player_t *p,
                       const uint8_t *pcm_data, size_t len);
void               adam_pcm_player_finish(adam_pcm_player_t *p); // waits for drain, then stops

// Record from the default microphone.
// Returns a malloc'd WAV buffer (16kHz, mono, PCM16). Caller must free.
// Blocks until max_seconds elapse or *stop_flag is set to non-zero.
uint8_t *adam_audio_record(int sample_rate, int max_seconds,
                            volatile int *stop_flag, size_t *out_len);

#endif // !ADAM_NO_VOICE && !ADAM_NO_PTHREADS

#ifdef __cplusplus
}
#endif
