//
//  google-gemini/main.c
//  Using Google Gemini models via the Gemini API
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

    char *api_key = env_load("GEMINI_API_KEY");
    if (!api_key) {
        fprintf(stderr, "Missing GEMINI_API_KEY in .env file\n");
        adam_cleanup();
        return 1;
    }

    adam_settings_t *settings = adam_create_settings();

    // Configure the Gemini provider
    adam_settings_set_provider(settings, ADAM_API_GEMINI, api_key, "gemini-2.5-flash");
    adam_settings_set_identity(settings, "You are a concise and helpful assistant.");

    adam_history_t *history = adam_history_create();

    printf("User: Explain the difference between TCP and UDP in three sentences.\n\n");

    adam_run_result_t result = adam_run(settings, history,
        "Explain the difference between TCP and UDP in three sentences.");

    if (result.status != ADAM_OK) {
        fprintf(stderr, "adam_run failed: %s\n", adam_status_string(result.status));
    } else {
        printf("Assistant: %s\n\n", result.final_response);
        printf("  [tokens: %d in / %d out, %.1f ms]\n",
               result.input_tokens, result.output_tokens, result.elapsed_ms);
    }

    adam_run_result_free(&result);
    adam_history_destroy(history);
    adam_settings_destroy(settings);
    free(api_key);
    adam_cleanup();
    return 0;
}
