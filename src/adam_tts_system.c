//
//  adam_tts_system.c
//  Adam — System TTS fallback for non-Apple platforms
//
//  Linux:   espeak-ng or spd-say (speech-dispatcher)
//  Windows: PowerShell → System.Speech.Synthesis (SAPI)
//  WASM:    uses tts_fn callback (maps to window.speechSynthesis)
//
//  Created by Marco Bambini on 16/03/26.
//

#ifndef __APPLE__

#include "adam.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Escape single quotes for shell safety
static char *shell_escape(const char *text) {
    size_t len = strlen(text);
    // Worst case: every char is a single quote → 4x expansion
    char *escaped = malloc(len * 4 + 1);
    if (!escaped) return NULL;
    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        if (text[i] == '\'') {
            escaped[j++] = '\'';
            escaped[j++] = '\\';
            escaped[j++] = '\'';
            escaped[j++] = '\'';
        } else {
            escaped[j++] = text[i];
        }
    }
    escaped[j] = '\0';
    return escaped;
}

adam_status_t adam_tts_system_speak(const char *text, const char *language) {
    if (!text || !text[0]) return ADAM_OK;
    UNUSED_PARAM(language);

    char *escaped = shell_escape(text);
    if (!escaped) return ADAM_ERR_ALLOC;

    size_t cmd_len = strlen(escaped) + 256;
    char *cmd = malloc(cmd_len);
    if (!cmd) { free(escaped); return ADAM_ERR_ALLOC; }

#if defined(_WIN32) || defined(_WIN64)
    // Windows: use PowerShell with System.Speech.Synthesis (SAPI)
    snprintf(cmd, cmd_len,
        "powershell -Command \"Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Speak('%s')\"", escaped);
#elif defined(__linux__)
    // Linux: try espeak-ng first, fall back to spd-say
    snprintf(cmd, cmd_len,
        "espeak-ng '%s' 2>/dev/null || spd-say '%s' 2>/dev/null",
        escaped, escaped);
#else
    // Unknown platform — no-op
    free(cmd);
    free(escaped);
    return ADAM_ERR_NOT_IMPLEMENTED;
#endif

    int ret = system(cmd);
    free(cmd);
    free(escaped);
    return (ret == 0) ? ADAM_OK : ADAM_ERR_VOICE;
}

#endif // !__APPLE__
