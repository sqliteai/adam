//
//  test_voice_interactive.c
//  Adam — Interactive voice conversation
//
//  Speak into your microphone, the agent listens, thinks, and speaks back.
//  Press Enter to stop recording. Type "quit" to exit.
//
//  LLM:  Grok (xAI) via OpenAI-compatible API
//  STT:  OpenAI Whisper API (if OPENAI_API_KEY set) or keyboard fallback
//  TTS:  macOS `say` command (built-in, no API needed)
//
//  Build & run:
//    make voice
//

#include "adam.h"
#include "adam_audio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>

#ifndef ADAM_NO_PTHREADS
#include <pthread.h>
#endif


// ============================================================================
// MARK: - .env loader
// ============================================================================

static char g_env_buf[4096];

static const char *env_get(const char *key) {
    FILE *f = fopen(".env", "r");
    if (!f) return NULL;
    size_t klen = strlen(key);
    while (fgets(g_env_buf, sizeof(g_env_buf), f)) {
        if (strncmp(g_env_buf, key, klen) == 0 && g_env_buf[klen] == '=') {
            fclose(f);
            char *val = g_env_buf + klen + 1;
            size_t len = strlen(val);
            while (len > 0 && (val[len-1] == '\n' || val[len-1] == '\r'))
                val[--len] = '\0';
            return val;
        }
    }
    fclose(f);
    return NULL;
}

// ============================================================================
// MARK: - macOS TTS via `say` command
// ============================================================================

static adam_status_t macos_tts_play(void *ctx, arena_t *arena,
                                     const char *text, const char *voice,
                                     adam_audio_format_t out_format,
                                     uint8_t **out_audio, size_t *out_len) {
    UNUSED_PARAM(ctx); UNUSED_PARAM(arena); UNUSED_PARAM(voice);
    UNUSED_PARAM(out_format); UNUSED_PARAM(out_audio); UNUSED_PARAM(out_len);

    // Escape single quotes in text for shell safety
    size_t tlen = strlen(text);
    char *escaped = malloc(tlen * 4 + 64);
    if (!escaped) return ADAM_ERR_ALLOC;

    size_t j = 0;
    for (size_t i = 0; i < tlen; i++) {
        if (text[i] == '\'') {
            escaped[j++] = '\'';
            escaped[j++] = '\\';
            escaped[j++] = '\'';
            escaped[j++] = '\'';
        } else if (text[i] == '"') {
            escaped[j++] = '\\';
            escaped[j++] = '"';
        } else {
            escaped[j++] = text[i];
        }
    }
    escaped[j] = '\0';

    char *cmd = malloc(j + 64);
    snprintf(cmd, j + 64, "say '%s'", escaped);
    int ret = system(cmd);
    free(escaped);
    free(cmd);

    // Return empty audio (the `say` command plays directly)
    *out_audio = arena_alloc(arena, 1);
    *out_len = 0;
    return (ret == 0) ? ADAM_OK : ADAM_ERR_VOICE;
}

// ============================================================================
// MARK: - Microphone recording (via miniaudio)
// ============================================================================

static volatile int g_stop_recording = 0;

#ifndef ADAM_NO_PTHREADS
static void *wait_for_enter(void *arg) {
    UNUSED_PARAM(arg);
    getchar();
    g_stop_recording = 1;
    return NULL;
}
#endif

static uint8_t *record_from_mic(size_t *out_len, int max_seconds) {
    g_stop_recording = 0;

#ifndef ADAM_NO_PTHREADS
    pthread_t enter_thread;
    pthread_create(&enter_thread, NULL, wait_for_enter, NULL);
#endif

    printf("  [Recording... press Enter to stop]\n");
    uint8_t *wav = adam_audio_record(16000, max_seconds,
                                     &g_stop_recording, out_len);

#ifndef ADAM_NO_PTHREADS
    g_stop_recording = 1; // signal thread to exit if still waiting
    pthread_cancel(enter_thread);
    pthread_join(enter_thread, NULL);
#endif

    if (!wav || *out_len == 0) {
        printf("  [Recording failed]\n");
        return NULL;
    }
    return wav;
}

// ============================================================================
// MARK: - Logger
// ============================================================================

static void on_log(void *ctx, adam_log_level_t level,
                    const char *msg, size_t len) {
    UNUSED_PARAM(ctx);
    if (level <= ADAM_LOG_INFO) {
        const char *labels[] = {"ERROR","WARN ","INFO ","DEBUG","TRACE"};
        fprintf(stderr, "  [%s] %.*s\n", labels[level], (int)len, msg);
    }
}

// ============================================================================
// MARK: - Main interactive loop
// ============================================================================

int main(void) {
    // Load keys (strdup because env_get uses a shared buffer)
    const char *_gk = env_get("GROK_API_KEY");
    char *grok_key = _gk ? strdup(_gk) : NULL;
    const char *_ak = env_get("ANTHROPIC_API_KEY");
    char *anthropic_key = _ak ? strdup(_ak) : NULL;
    const char *_ok = env_get("OPENAI_API_KEY");
    char *openai_key = _ok ? strdup(_ok) : NULL;

    const char *llm_key = NULL;
    const char *llm_model = NULL;
    const char *llm_url = NULL;
    adam_api_format_t llm_format = ADAM_API_NONE;

    // Pick best available LLM
    if (grok_key) {
        llm_key = grok_key;
        llm_model = "grok-3-mini-fast";
        llm_url = "https://api.x.ai/v1/chat/completions";
        llm_format = ADAM_API_OPENAI;
    } else if (anthropic_key) {
        llm_key = anthropic_key;
        llm_model = "claude-sonnet-4-20250514";
        llm_format = ADAM_API_ANTHROPIC;
    } else {
        fprintf(stderr, "Error: no API key found in .env\n");
        return 1;
    }

    int has_cloud_stt = (openai_key != NULL);

    adam_init();

    // Configure agent
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_provider(s, llm_format, llm_key, llm_model);
    if (llm_url) adam_settings_set_base_url(s, llm_url);

    adam_settings_set_identity(s,
        "You are a friendly voice assistant. Keep your responses short and "
        "conversational — 1-2 sentences max. You are speaking out loud, so "
        "avoid markdown, code blocks, or bullet points.");
    adam_settings_set_logger(s, on_log, NULL, ADAM_LOG_INFO);

    // TTS: macOS `say` (always available, no API needed)
    adam_settings_set_tts(s, ADAM_TTS_CLOUD, NULL, NULL, NULL, NULL);
    adam_settings_set_tts_callback(s, macos_tts_play, NULL);

    // STT: cloud Whisper if we have a key
    if (has_cloud_stt) {
        adam_settings_set_stt(s, ADAM_STT_CLOUD, NULL, openai_key, "whisper-1");
    }

    printf("\n");
    printf("  ╔══════════════════════════════════════════╗\n");
    printf("  ║     Adam Interactive Voice Assistant      ║\n");
    printf("  ╠══════════════════════════════════════════╣\n");
    printf("  ║  LLM: %-35s║\n", llm_model);
    printf("  ║  STT: %-35s║\n", has_cloud_stt ? "OpenAI Whisper (cloud)" : "keyboard (type input)");
    printf("  ║  TTS: %-35s║\n", "macOS say (built-in)");
    printf("  ╠══════════════════════════════════════════╣\n");
    if (has_cloud_stt) {
        printf("  ║  Press Enter to start recording.         ║\n");
        printf("  ║  Press Enter again to stop and send.     ║\n");
    } else {
        printf("  ║  Type your message and press Enter.      ║\n");
    }
    printf("  ║  Type 'quit' to exit.                    ║\n");
    printf("  ╚══════════════════════════════════════════╝\n\n");

    adam_history_t *h = adam_history_create();
    int turn = 0;

    while (1) {
        turn++;
        char input_text[4096] = {0};

        if (has_cloud_stt) {
            // Voice input mode
            printf("[Turn %d] Press Enter to start recording...", turn);
            fflush(stdout);

            // Wait for Enter to begin recording
            int c;
            while ((c = getchar()) != '\n' && c != EOF) {
                // Check for "quit" typed
                if (c == 'q') {
                    char rest[16];
                    if (fgets(rest, sizeof(rest), stdin) && strncmp(rest, "uit", 3) == 0)
                        goto done;
                }
            }
            if (c == EOF) break;

            // Record from microphone
            size_t audio_len = 0;
            uint8_t *audio = record_from_mic(&audio_len, 15);

            if (!audio || audio_len == 0) {
                printf("  [No audio captured, try again]\n\n");
                continue;
            }

            printf("  [Recorded %zu bytes, transcribing...]\n", audio_len);

            // Transcribe
            arena_t *stt_arena = arena_create(64 * 1024);
            const char *transcribed = NULL;
            adam_status_t stt_rc = adam_stt_transcribe(s, stt_arena,
                audio, audio_len, ADAM_AUDIO_WAV, &transcribed);
            free(audio);

            if (stt_rc != ADAM_OK || !transcribed || strlen(transcribed) == 0) {
                printf("  [Transcription failed: %s]\n\n",
                       adam_status_string(stt_rc));
                arena_destroy(stt_arena);
                continue;
            }

            printf("  You said: \"%s\"\n", transcribed);
            strncpy(input_text, transcribed, sizeof(input_text) - 1);
            arena_destroy(stt_arena);
        } else {
            // Keyboard input mode
            printf("[Turn %d] You: ", turn);
            fflush(stdout);
            if (!fgets(input_text, sizeof(input_text), stdin)) break;

            // Strip newline
            size_t len = strlen(input_text);
            while (len > 0 && (input_text[len-1] == '\n' || input_text[len-1] == '\r'))
                input_text[--len] = '\0';
        }

        if (strlen(input_text) == 0) continue;
        if (strcmp(input_text, "quit") == 0 || strcmp(input_text, "exit") == 0)
            break;

        // Run agent
        adam_run_result_t r = adam_run(s, h, input_text);

        if (r.status != ADAM_OK) {
            printf("  [Error: %s]\n\n",
                   r.final_response ? r.final_response
                                    : adam_status_string(r.status));
            adam_run_result_free(&r);
            continue;
        }

        printf("  Adam: %s\n", r.final_response ? r.final_response : "...");

        // Speak the response via TTS
        if (r.final_response) {
            adam_tts_speak(s, r.final_response);
        }

        printf("  [tokens: %d in/%d out | $%.6f | %.0fms]\n\n",
               r.input_tokens, r.output_tokens, r.cost_usd, r.elapsed_ms);

        adam_run_result_free(&r);
    }

done:
    printf("\nGoodbye!\n");

    adam_history_destroy(h);
    adam_settings_destroy(s);
    adam_cleanup();
    free(grok_key);
    free(anthropic_key);
    free(openai_key);
    return 0;
}
