//
//  voice/main.c
//  Voice agent — STT + TTS pipeline demonstration
//

#include "adam.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main(void) {
    adam_status_t status = adam_init();
    if (status != ADAM_OK) {
        fprintf(stderr, "adam_init failed: %s\n", adam_status_string(status));
        return 1;
    }

    char *api_key = env_load("ANTHROPIC_API_KEY");
    if (!api_key) {
        fprintf(stderr, "Missing ANTHROPIC_API_KEY in .env file\n");
        adam_cleanup();
        return 1;
    }

    char *openai_key = env_load("OPENAI_API_KEY");

    adam_settings_t *settings = adam_create_settings();
    adam_settings_set_provider(settings, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");
    adam_settings_set_identity(settings, "You are a friendly voice assistant. Keep responses brief and conversational.");

    // Configure STT: cloud-based Whisper via OpenAI
    adam_settings_set_stt(settings,
        ADAM_STT_CLOUD,
        NULL,                   // default API URL
        openai_key,             // OpenAI key for Whisper
        "whisper-1"             // model
    );

    // Configure TTS: use the system's built-in speech synthesizer
    // (macOS: AVSpeechSynthesizer, Linux: espeak-ng)
    adam_settings_set_tts(settings,
        ADAM_TTS_SYSTEM,
        NULL,                   // no API URL needed for system TTS
        NULL,                   // no API key needed
        NULL,                   // default model
        NULL                    // default voice
    );

    // Demonstrate TTS by speaking a greeting
    printf("Speaking a greeting using system TTS...\n");
    status = adam_tts_speak(settings, "Hello! I am Adam, your voice assistant. How can I help you today?");
    if (status != ADAM_OK) {
        fprintf(stderr, "TTS speak failed: %s\n", adam_status_string(status));
    } else {
        printf("Greeting spoken successfully.\n");
    }

    // In a real application, you would start the voice thread to
    // continuously listen, transcribe, run the agent, and speak responses:
    //
    //   adam_history_t *history = adam_history_create();
    //   settings->voice_enabled = 1;
    //   adam_voice_start(settings, history);
    //
    //   // ... voice loop runs in background ...
    //   // Press Ctrl+C or set abort_flag to stop
    //
    //   adam_voice_stop(settings);
    //   adam_history_destroy(history);

    printf("\nVoice pipeline configured. In a full application, call\n");
    printf("adam_voice_start() to begin the listen-transcribe-respond loop.\n");

    adam_settings_destroy(settings);
    free(api_key);
    free(openai_key);
    adam_cleanup();
    return 0;
}
