//
//  test_voice_talk.c
//  Adam — Fully voice-interactive conversation
//
//  Speak into your microphone. The agent listens, thinks, and speaks back.
//  Supports both cloud and fully local (offline) modes.
//
//  Cloud mode (default):
//    make talk
//    STT: OpenAI gpt-4o-mini-transcribe
//    LLM: GPT-4o-mini / Anthropic / Grok
//    TTS: OpenAI gpt-4o-mini-tts + miniaudio
//
//  Local mode (fully offline):
//    make talk LOCAL=1
//    STT: whisper.cpp (models/ggml-base.en.bin)
//    LLM: llama.cpp  (models/*.gguf)
//    TTS: OS built-in (AVSpeechSynthesizer / espeak-ng)
//
//  Custom GGUF:
//    make talk LOCAL=1 GGUF=models/my-model.gguf
//

#include "adam.h"
#include "adam_audio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <pthread.h>

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
// MARK: - Cloud voice selection
// ============================================================================

static const char *g_voices[] = {
    "alloy", "ash", "ballad", "coral", "echo",
    "fable", "nova", "sage", "shimmer", "verse",
};
static const char *g_voice_desc[] = {
    "Neutral, balanced",
    "Warm, conversational",
    "Soft, gentle",
    "Clear, friendly",
    "Smooth, authoritative",
    "Expressive, storytelling",
    "Bright, energetic",
    "Calm, measured",
    "Light, upbeat",
    "Versatile, dynamic",
};
#define NUM_VOICES 10

static const char *pick_voice(void) {
    printf("  Available voices:\n");
    for (int i = 0; i < NUM_VOICES; i++) {
        printf("    %d. %-10s — %s%s\n", i, g_voices[i], g_voice_desc[i],
               i == 3 ? " (default)" : "");
    }
    printf("\n  Select voice [0-9, name, or Enter for coral]: ");
    fflush(stdout);

    char buf[64];
    if (!fgets(buf, sizeof(buf), stdin) || buf[0] == '\n')
        return "coral";

    size_t len = strlen(buf);
    while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r'))
        buf[--len] = '\0';

    if (len == 1 && buf[0] >= '0' && buf[0] <= '9')
        return g_voices[buf[0] - '0'];

    for (int i = 0; i < NUM_VOICES; i++)
        if (strcmp(buf, g_voices[i]) == 0) return g_voices[i];

    printf("  (unknown voice \"%s\", using coral)\n", buf);
    return "coral";
}

// ============================================================================
// MARK: - Signal handling
// ============================================================================

static volatile int g_quit = 0;

static void sigint_handler(int sig) {
    (void)sig;
    g_quit = 1;
    g_stop_recording = 1;
}

// ============================================================================
// MARK: - Detect local mode
// ============================================================================

static int is_local_mode(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--local") == 0 || strcmp(argv[i], "-l") == 0)
            return 1;
    }
    return 0;
}

static const char *find_gguf(int argc, char **argv, const char *fallback) {
    for (int i = 1; i < argc; i++) {
        if (strstr(argv[i], ".gguf")) return argv[i];
    }
    return fallback;
}

static const char *find_whisper(int argc, char **argv, const char *fallback) {
    for (int i = 1; i < argc; i++) {
        if (strstr(argv[i], ".bin") && strstr(argv[i], "ggml")) return argv[i];
    }
    return fallback;
}

// ============================================================================
// MARK: - Main
// ============================================================================

int main(int argc, char **argv) {
    int local = is_local_mode(argc, argv);

    char *grok_key = env_load("GROK_API_KEY");
    char *anthropic_key = env_load("ANTHROPIC_API_KEY");
    char *openai_key = env_load("OPENAI_API_KEY");

    // In cloud mode, we need an OpenAI key for STT/TTS
    if (!local && !openai_key) {
        fprintf(stderr, "Error: OPENAI_API_KEY required for cloud mode.\n");
        fprintf(stderr, "Use --local for fully offline mode.\n");
        free(grok_key); free(anthropic_key);
        return 1;
    }

    adam_init();

    adam_settings_t *s = adam_create_settings();

    const char *mode_str;
    const char *stt_str;
    const char *tts_str;
    const char *llm_str;

    if (local) {
        // ── LOCAL MODE: fully offline ──

        // LLM
        const char *gguf = find_gguf(argc, argv, "models/Qwen3.5-0.8B-Q8_0.gguf");
        s->gguf_path = gguf;
        s->local_gpu_layers = -1;
        s->local_ctx_size = 4096;
        llm_str = gguf;

        // STT: whisper.cpp
        const char *whisper = find_whisper(argc, argv, "models/ggml-base.en.bin");
        adam_settings_set_stt(s, ADAM_STT_LOCAL, NULL, NULL, whisper);
        stt_str = whisper;

        // TTS: OS built-in
        adam_settings_set_tts(s, ADAM_TTS_SYSTEM, NULL, NULL, NULL, NULL);
        tts_str = "System (AVSpeechSynthesizer)";

        mode_str = "LOCAL (offline)";

    } else {
        // ── CLOUD MODE ──

        // LLM: pick fastest available
        if (openai_key) {
            adam_settings_set_provider(s, ADAM_API_OPENAI, openai_key, "gpt-4o-mini");
            llm_str = "gpt-4o-mini (OpenAI)";
        } else if (grok_key) {
            adam_settings_set_provider(s, ADAM_API_OPENAI, grok_key, "grok-3-mini-fast");
            adam_settings_set_base_url(s, "https://api.x.ai/v1/chat/completions");
            llm_str = "grok-3-mini-fast (xAI)";
        } else if (anthropic_key) {
            adam_settings_set_provider(s, ADAM_API_ANTHROPIC, anthropic_key,
                                       "claude-sonnet-4-20250514");
            llm_str = "claude-sonnet-4 (Anthropic)";
        } else {
            llm_str = "(none)";
        }

        // STT: OpenAI
        adam_settings_set_stt(s, ADAM_STT_CLOUD, NULL, openai_key,
                              "gpt-4o-mini-transcribe");
        stt_str = "gpt-4o-mini-transcribe (cloud)";

        // TTS: voice selection
        printf("\n");
        const char *voice = pick_voice();
        adam_settings_set_tts(s, ADAM_TTS_CLOUD, NULL, openai_key,
                              "gpt-4o-mini-tts", voice);
        s->tts_format = ADAM_AUDIO_MP3;
        tts_str = voice;

        mode_str = "CLOUD";
    }

    adam_settings_set_identity(s,
        "You are a friendly voice assistant having a natural conversation. "
        "Keep responses short and conversational — 1-2 sentences. "
        "Always detect and reply in the same language the user speaks. "
        "Do not use markdown, bullet points, or code. "
        "Speak naturally as if in a real conversation.");

    s->max_tokens = 150;
    adam_settings_set_logger(s, on_log, NULL, ADAM_LOG_WARN);

    printf("\n");
    printf("  ╔══════════════════════════════════════════════╗\n");
    printf("  ║       Adam Voice Conversation                ║\n");
    printf("  ╠══════════════════════════════════════════════╣\n");
    printf("  ║  Mode: %-39s║\n", mode_str);
    printf("  ║  LLM:  %-39s║\n", llm_str);
    printf("  ║  STT:  %-39s║\n", stt_str);
    printf("  ║  TTS:  %-39s║\n", tts_str);
    printf("  ╠══════════════════════════════════════════════╣\n");
    printf("  ║  Speak naturally. Press Enter when done.     ║\n");
    printf("  ║  Press Ctrl+C to exit.                       ║\n");
    printf("  ║  Speak any language — replies match yours.   ║\n");
    printf("  ╚══════════════════════════════════════════════╝\n");

    // Warm up TTS (cloud only — system TTS is instant)
    if (!local) {
        printf("\n  Warming up TTS...");
        fflush(stdout);
        adam_tts_speak(s, ".");
        printf(" ready!\n");
    }
    printf("\n");

    signal(SIGINT, sigint_handler);

    adam_history_t *h = adam_history_create();
    int turn = 0;
    float total_cost = 0.0f;

    while (!g_quit) {
        turn++;
        printf("  [Turn %d] ", turn);

        // 1. Record
        size_t audio_len = 0;
        uint8_t *audio = record_speech(&audio_len);

        if (g_quit) { free(audio); break; }

        if (!audio || audio_len < 1000) {
            printf("    (no speech detected, try again)\n\n");
            free(audio);
            continue;
        }
        printf("    Recorded %zu bytes, transcribing...\n", audio_len);

        // 2. Transcribe
        arena_t *stt_arena = arena_create(64 * 1024);
        const char *transcript = NULL;
        adam_status_t stt_rc = adam_stt_transcribe(s, stt_arena,
            audio, audio_len, ADAM_AUDIO_WAV, &transcript);
        free(audio);

        if (g_quit) { arena_destroy(stt_arena); break; }

        if (stt_rc != ADAM_OK || !transcript || strlen(transcript) == 0) {
            printf("    (transcription failed: %s)\n\n",
                   adam_status_string(stt_rc));
            arena_destroy(stt_arena);
            continue;
        }

        printf("    You: \"%s\"\n", transcript);

        char *user_text = strdup(transcript);
        arena_destroy(stt_arena);

        // 3. Run agent
        adam_run_result_t r = adam_run(s, h, user_text);
        free(user_text);

        if (g_quit) { adam_run_result_free(&r); break; }

        if (r.status != ADAM_OK) {
            printf("    (agent error: %s)\n\n",
                   r.final_response ? r.final_response
                                    : adam_status_string(r.status));
            adam_run_result_free(&r);
            continue;
        }

        printf("    Adam: %s\n", r.final_response ? r.final_response : "...");
        total_cost += r.cost_usd;

        // 4. Speak
        if (r.final_response && !g_quit) {
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
