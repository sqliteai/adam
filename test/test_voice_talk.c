//
//  test_voice_talk.c
//  Adam — Fully voice-interactive conversation
//
//  Speak into your microphone. The agent listens, thinks, and speaks back.
//  The conversation continues until you say "quit" or "exit".
//  Speaks in whatever language you use.
//
//  LLM:  Grok (xAI) via OpenAI-compatible API
//  STT:  OpenAI Whisper API
//  TTS:  OpenAI TTS API (voice: "nova") + miniaudio playback
//
//  Build & run:
//    make talk
//

#include "adam.h"
#include "adam_audio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <pthread.h>

#define UNUSED_PARAM(p) ((void)(p))

// ============================================================================
// MARK: - .env loader
// ============================================================================

static char *env_load(const char *key) {
    FILE *f = fopen(".env", "r");
    if (!f) return NULL;
    size_t klen = strlen(key);
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, key, klen) == 0 && line[klen] == '=') {
            fclose(f);
            char *val = line + klen + 1;
            size_t len = strlen(val);
            while (len > 0 && (val[len-1] == '\n' || val[len-1] == '\r'))
                val[--len] = '\0';
            return strdup(val);
        }
    }
    fclose(f);
    return NULL;
}

// ============================================================================
// MARK: - Logger
// ============================================================================

static void on_log(void *ctx, adam_log_level_t level,
                    const char *msg, size_t len) {
    UNUSED_PARAM(ctx);
    if (level <= ADAM_LOG_WARN) {
        const char *labels[] = {"ERROR","WARN ","INFO ","DEBUG","TRACE"};
        fprintf(stderr, "  [%s] %.*s\n", labels[level], (int)len, msg);
    }
}

// ============================================================================
// MARK: - Recording with Enter-to-stop
// ============================================================================

static volatile int g_stop_recording = 0;

static void *enter_thread_fn(void *arg) {
    UNUSED_PARAM(arg);
    getchar();
    g_stop_recording = 1;
    return NULL;
}

static uint8_t *record_speech(size_t *out_len) {
    g_stop_recording = 0;

    pthread_t tid;
    pthread_create(&tid, NULL, enter_thread_fn, NULL);

    printf("    Listening... (press Enter when done speaking)\n");
    uint8_t *wav = adam_audio_record(16000, 30, &g_stop_recording, out_len);

    g_stop_recording = 1;
    pthread_cancel(tid);
    pthread_join(tid, NULL);

    return wav;
}

// ============================================================================
// MARK: - Case-insensitive word check
// ============================================================================

static int contains_quit(const char *text) {
    if (!text) return 0;
    // Normalize to lowercase for matching
    const char *quit_words[] = {
        "quit", "exit", "stop", "bye", "goodbye",
        "esci", "basta", "fine", "arrivederci",   // Italian
        "salir", "adiós", "adios",                 // Spanish
        "quitter", "au revoir",                    // French
        "aufhören", "tschüss",                     // German
        NULL
    };
    char lower[512];
    size_t len = strlen(text);
    if (len >= sizeof(lower)) len = sizeof(lower) - 1;
    for (size_t i = 0; i < len; i++)
        lower[i] = (char)tolower((unsigned char)text[i]);
    lower[len] = '\0';

    for (int i = 0; quit_words[i]; i++) {
        if (strstr(lower, quit_words[i]))
            return 1;
    }
    return 0;
}

// ============================================================================
// MARK: - Main
// ============================================================================

int main(void) {
    char *grok_key = env_load("GROK_API_KEY");
    char *anthropic_key = env_load("ANTHROPIC_API_KEY");
    char *openai_key = env_load("OPENAI_API_KEY");

    if (!openai_key) {
        fprintf(stderr, "Error: OPENAI_API_KEY required in .env for STT/TTS\n");
        free(grok_key); free(anthropic_key);
        return 1;
    }

    // Pick LLM provider — prefer fastest for voice conversation
    const char *llm_key, *llm_model, *llm_url = NULL;
    adam_api_format_t llm_fmt;

    if (openai_key) {
        // GPT-4o-mini is the fastest for short conversational replies
        llm_key = openai_key;
        llm_model = "gpt-4o-mini";
        llm_fmt = ADAM_API_OPENAI;
    } else if (grok_key) {
        llm_key = grok_key;
        llm_model = "grok-3-mini-fast";
        llm_url = "https://api.x.ai/v1/chat/completions";
        llm_fmt = ADAM_API_OPENAI;
    } else if (anthropic_key) {
        llm_key = anthropic_key;
        llm_model = "claude-sonnet-4-20250514";
        llm_fmt = ADAM_API_ANTHROPIC;
    } else {
        llm_key = openai_key;
        llm_model = "gpt-4o-mini";
        llm_fmt = ADAM_API_OPENAI;
    }

    adam_init();

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_provider(s, llm_fmt, llm_key, llm_model);
    if (llm_url) adam_settings_set_base_url(s, llm_url);

    adam_settings_set_identity(s,
        "You are a friendly voice assistant having a natural conversation. "
        "Keep responses short and conversational — 1-2 sentences. "
        "Always reply in the same language the user speaks. "
        "If the user speaks Italian, reply in Italian. "
        "If the user speaks English, reply in English. "
        "Do not use markdown, bullet points, or code. "
        "Speak naturally as if in a real conversation.");

    s->max_tokens = 150;  // voice replies should be short
    adam_settings_set_logger(s, on_log, NULL, ADAM_LOG_WARN);

    // STT: OpenAI Whisper
    adam_settings_set_stt(s, ADAM_STT_CLOUD, NULL, openai_key, "whisper-1");

    // TTS: OpenAI TTS with neural voice
    adam_settings_set_tts(s, ADAM_TTS_CLOUD, NULL, openai_key, "tts-1", "nova");
    s->tts_format = ADAM_AUDIO_MP3;

    printf("\n");
    printf("  ╔══════════════════════════════════════════════╗\n");
    printf("  ║       Adam Voice Conversation                ║\n");
    printf("  ╠══════════════════════════════════════════════╣\n");
    printf("  ║  LLM: %-39s║\n", llm_model);
    printf("  ║  STT: OpenAI Whisper (cloud)                 ║\n");
    printf("  ║  TTS: OpenAI TTS nova (cloud) + miniaudio    ║\n");
    printf("  ╠══════════════════════════════════════════════╣\n");
    printf("  ║  Speak naturally. Press Enter when done.     ║\n");
    printf("  ║  Say \"quit\" or \"exit\" to end.               ║\n");
    printf("  ║  Speak any language — replies match yours.   ║\n");
    printf("  ╚══════════════════════════════════════════════╝\n\n");

    adam_history_t *h = adam_history_create();
    int turn = 0;
    float total_cost = 0.0f;

    while (1) {
        turn++;
        printf("  [Turn %d] ", turn);

        // 1. Record from microphone
        size_t audio_len = 0;
        uint8_t *audio = record_speech(&audio_len);

        if (!audio || audio_len < 1000) {
            printf("    (no speech detected, try again)\n\n");
            free(audio);
            continue;
        }
        printf("    Recorded %zu bytes, transcribing...\n", audio_len);

        // 2. Transcribe via Whisper
        arena_t *stt_arena = arena_create(64 * 1024);
        const char *transcript = NULL;
        adam_status_t stt_rc = adam_stt_transcribe(s, stt_arena,
            audio, audio_len, ADAM_AUDIO_WAV, &transcript);
        free(audio);

        if (stt_rc != ADAM_OK || !transcript || strlen(transcript) == 0) {
            printf("    (transcription failed: %s)\n\n",
                   adam_status_string(stt_rc));
            arena_destroy(stt_arena);
            continue;
        }

        printf("    You: \"%s\"\n", transcript);

        // 3. Check for quit command
        if (contains_quit(transcript)) {
            printf("    (quit command detected)\n");
            // Say goodbye
            adam_run_result_t bye = adam_run(s, h,
                "The user said goodbye. Say a brief, warm goodbye in their language.");
            if (bye.status == ADAM_OK && bye.final_response) {
                printf("    Adam: %s\n", bye.final_response);
                adam_tts_speak(s, bye.final_response);
            }
            adam_run_result_free(&bye);
            arena_destroy(stt_arena);
            break;
        }

        // Copy transcript before destroying arena
        char *user_text = strdup(transcript);
        arena_destroy(stt_arena);

        // 4. Run agent
        adam_run_result_t r = adam_run(s, h, user_text);
        free(user_text);

        if (r.status != ADAM_OK) {
            printf("    (agent error: %s)\n\n",
                   r.final_response ? r.final_response
                                    : adam_status_string(r.status));
            adam_run_result_free(&r);
            continue;
        }

        printf("    Adam: %s\n", r.final_response ? r.final_response : "...");
        total_cost += r.cost_usd;

        // 5. Speak the response
        if (r.final_response) {
            adam_tts_speak(s, r.final_response);
        }

        printf("    [%d in/%d out | $%.4f | %.0fms]\n\n",
               r.input_tokens, r.output_tokens, r.cost_usd, r.elapsed_ms);

        adam_run_result_free(&r);
    }

    printf("\n  Session: %d turns, $%.4f total cost\n", turn, total_cost);
    printf("  Goodbye!\n\n");

    adam_history_destroy(h);
    adam_settings_destroy(s);
    adam_cleanup();
    free(grok_key);
    free(anthropic_key);
    free(openai_key);
    return 0;
}
