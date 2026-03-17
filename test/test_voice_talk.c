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
#ifndef ADAM_NO_LOCAL
#include "llama.h"
#include "whisper.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sys/select.h>
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

// Wait for Enter by polling stdin — no pthread_cancel needed.
// The thread checks g_stop_recording periodically and exits cleanly.
static void *enter_thread_fn(void *arg) {
    UNUSED_PARAM(arg);
    while (!g_stop_recording) {
        fd_set fds;
        struct timeval tv = {0, 100000}; // 100ms poll interval
        FD_ZERO(&fds);
        FD_SET(STDIN_FILENO, &fds);
        if (select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0) {
            int c = getchar();
            if (c == '\n' || c == EOF) {
                g_stop_recording = 1;
                return NULL;
            }
        }
    }
    return NULL;
}

static uint8_t *record_speech(size_t *out_len) {
    g_stop_recording = 0;

    int is_tty = isatty(STDIN_FILENO);
    pthread_t tid;

    if (is_tty) {
        // Interactive: Enter key stops recording
        pthread_create(&tid, NULL, enter_thread_fn, NULL);
        printf("    Listening... (press Enter when done speaking)\n");
    } else {
        // Non-interactive (piped): use 5s timeout only
        printf("    Listening... (5s timeout)\n");
    }

    int max_seconds = is_tty ? 30 : 5;
    uint8_t *wav = adam_audio_record(16000, max_seconds, &g_stop_recording, out_len);

    g_stop_recording = 1;
    if (is_tty) pthread_join(tid, NULL);

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
// MARK: - Strip thinking tags from local model output
// ============================================================================

// Qwen3.5 and other reasoning models wrap their thinking in
// <think>...</think> tags. Strip these for voice output.
static const char *strip_think_tags(const char *text) {
    if (!text) return text;
    // Find </think> — everything after it is the actual response
    const char *end_tag = strstr(text, "</think>");
    if (end_tag) {
        const char *after = end_tag + 8; // strlen("</think>") = 8
        while (*after == '\n' || *after == '\r' || *after == ' ') after++;
        return after;
    }
    // No think tags — check if it starts with <think> (incomplete)
    if (strncmp(text, "<think>", 7) == 0) {
        // Thinking block never closed — return empty
        return "";
    }
    return text;
}

// ============================================================================
// MARK: - Log suppression for local models
// ============================================================================

#ifndef ADAM_NO_LOCAL
static void silence_ggml_log(enum ggml_log_level level, const char *text, void *ctx) {
    UNUSED_PARAM(level); UNUSED_PARAM(text); UNUSED_PARAM(ctx);
}
#endif

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
        // Auto-detect whisper model: prefer multilingual, fall back to English-only
        const char *whisper = find_whisper(argc, argv, NULL);
        if (!whisper) {
            const char *candidates[] = {
                "models/ggml-base.bin",       // multilingual (preferred)
                "models/ggml-base.en.bin",    // English-only
                "models/ggml-small.bin",
                "models/ggml-tiny.bin",
                "models/ggml-tiny.en.bin",
                NULL
            };
            for (int i = 0; candidates[i]; i++) {
                FILE *f = fopen(candidates[i], "r");
                if (f) { fclose(f); whisper = candidates[i]; break; }
            }
            if (!whisper) {
                fprintf(stderr, "Error: no whisper model found in models/\n");
                fprintf(stderr, "Download from: https://huggingface.co/ggerganov/whisper.cpp\n");
                adam_settings_destroy(s);
                adam_cleanup();
                return 1;
            }
        }
        adam_settings_set_stt(s, ADAM_STT_LOCAL, NULL, NULL, whisper);
        // Don't force a language — let whisper auto-detect.
        // But if the user wants to force Italian: s->stt_language = "it";
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

    if (local) {
        // Local models (especially small ones like 0.8B) need explicit
        // instructions to not use thinking/reasoning and to be brief.
        // Also need more tokens since reasoning models use tokens for <think>.
        adam_settings_set_identity(s,
            "You are a voice assistant. "
            "Reply directly in 1 sentence. Do not think or reason. "
            "Do not use <think> tags. Do not use markdown. "
            "Reply in the same language the user speaks. "
            "Be very brief.");
        s->max_tokens = 256;
    } else {
        adam_settings_set_identity(s,
            "You are a friendly voice assistant having a natural conversation. "
            "Keep responses short and conversational — 1-2 sentences. "
            "Always detect and reply in the same language the user speaks. "
            "Do not use markdown, bullet points, or code. "
            "Speak naturally as if in a real conversation.");
        s->max_tokens = 150;
    }
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

    if (local) {
#ifndef ADAM_NO_LOCAL
        // Suppress all whisper.cpp / llama.cpp / ggml verbose logs
        whisper_log_set(silence_ggml_log, NULL);
        llama_log_set(silence_ggml_log, NULL);
#endif

        // Warm up: load whisper + llama models now
        printf("\n  Loading models...");
        fflush(stdout);

        // Trigger whisper model load with a tiny silent WAV
        {
            arena_t *a = arena_create(64 * 1024);
            int16_t silence[1600]; // 100ms at 16kHz
            memset(silence, 0, sizeof(silence));

            // Build a minimal WAV in-place
            size_t pcm_bytes = sizeof(silence);
            size_t wav_size = 44 + pcm_bytes;
            uint8_t *wav = arena_alloc(a, wav_size);
            if (wav) {
                memset(wav, 0, wav_size);
                uint32_t sr = 16000, br = 32000, ds = (uint32_t)pcm_bytes;
                uint32_t cs = 36 + ds;
                uint16_t ba = 2, bits = 16, af = 1, ch = 1;
                uint32_t fs = 16;
                size_t p = 0;
                memcpy(wav+p,"RIFF",4);p+=4; memcpy(wav+p,&cs,4);p+=4;
                memcpy(wav+p,"WAVE",4);p+=4; memcpy(wav+p,"fmt ",4);p+=4;
                memcpy(wav+p,&fs,4);p+=4; memcpy(wav+p,&af,2);p+=2;
                memcpy(wav+p,&ch,2);p+=2; memcpy(wav+p,&sr,4);p+=4;
                memcpy(wav+p,&br,4);p+=4; memcpy(wav+p,&ba,2);p+=2;
                memcpy(wav+p,&bits,2);p+=2; memcpy(wav+p,"data",4);p+=4;
                memcpy(wav+p,&ds,4);p+=4;
                memcpy(wav+p, silence, pcm_bytes);

                const char *text = NULL;
                adam_stt_transcribe(s, a, wav, wav_size, ADAM_AUDIO_WAV, &text);
            }
            arena_destroy(a);
        }

        // Trigger llama model load with a dummy run
        {
            adam_history_t *dummy = adam_history_create();
            adam_run_result_t r = adam_run(s, dummy, "hi");
            adam_run_result_free(&r);
            adam_history_destroy(dummy);
        }

        printf(" ready!\n");
    } else {
        // Cloud: warm up TTS
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

        if (stt_rc != ADAM_OK) {
            printf("    (transcription error: %s)\n\n",
                   adam_status_string(stt_rc));
            arena_destroy(stt_arena);
            continue;
        }
        if (!transcript || strlen(transcript) == 0) {
            printf("    (no speech detected, try again)\n\n");
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

        // Strip <think>...</think> tags from reasoning models
        const char *response = strip_think_tags(r.final_response);
        if (!response || !response[0]) response = "(no response)";

        printf("    Adam: %s\n", response);
        total_cost += r.cost_usd;

        // 4. Speak
        if (response[0] != '(' && !g_quit) {
            adam_tts_speak(s, response);
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
