//
//  streaming/main.c
//  Real-time token streaming from the LLM
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

// Stream callback: prints each token as it arrives
static void on_stream(void *ctx, const char *chunk, size_t len, int is_done) {
    (void)ctx;
    if (len > 0) {
        fwrite(chunk, 1, len, stdout);
        fflush(stdout);
    }
    if (is_done) {
        printf("\n");
    }
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

    adam_settings_t *settings = adam_create_settings();
    adam_settings_set_provider(settings, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");

    // Enable streaming with our callback
    adam_settings_set_stream(settings, on_stream, NULL);

    // Cap output length for this demo
    settings->max_tokens = 512;

    adam_history_t *history = adam_history_create();

    printf("User: Tell me a very short story about a robot learning to paint.\n\n");
    printf("Assistant: ");

    // adam_run still returns the full response, but on_stream fires for each token
    adam_run_result_t result = adam_run(settings, history,
        "Tell me a very short story about a robot learning to paint. Keep it under 200 words.");

    if (result.status != ADAM_OK) {
        fprintf(stderr, "\nRun failed: %s\n", adam_status_string(result.status));
    } else {
        printf("\n  [tokens: %d in / %d out, %.1f ms]\n",
               result.input_tokens, result.output_tokens, result.elapsed_ms);
    }

    adam_run_result_free(&result);
    adam_history_destroy(history);
    adam_settings_destroy(settings);
    free(api_key);
    adam_cleanup();
    return 0;
}
