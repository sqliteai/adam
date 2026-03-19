//
//  image-generation/main.c
//  Generating images with Gemini's image generation model
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

    // Use the Gemini image generation model
    adam_settings_set_provider(settings, ADAM_API_GEMINI, api_key,
                               "gemini-2.0-flash-preview-image-generation");

    adam_history_t *history = adam_history_create();

    const char *prompt = "Draw a friendly robot sitting at a desk writing code on a laptop. "
                         "Use a warm color palette with soft lighting.";

    printf("User: %s\n\n", prompt);

    adam_run_result_t result = adam_run(settings, history, prompt);

    if (result.status != ADAM_OK) {
        fprintf(stderr, "adam_run failed: %s\n", adam_status_string(result.status));
    } else {
        printf("Response received (%zu characters)\n\n", strlen(result.final_response));

        // The response may contain a data URI (data:image/png;base64,...) for the image,
        // along with any text the model provides.
        if (strstr(result.final_response, "data:image")) {
            printf("Image data URI found in response.\n");
            printf("First 100 chars: %.100s...\n\n", result.final_response);
        } else {
            printf("Assistant: %s\n\n", result.final_response);
        }

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
