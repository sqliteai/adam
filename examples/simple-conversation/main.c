//
//  simple-conversation/main.c
//  Multi-turn conversation with Anthropic Claude
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
    // Initialize the library
    adam_status_t status = adam_init();
    if (status != ADAM_OK) {
        fprintf(stderr, "adam_init failed: %s\n", adam_status_string(status));
        return 1;
    }

    // Load API key from .env file
    char *api_key = env_load("ANTHROPIC_API_KEY");
    if (!api_key) {
        fprintf(stderr, "Missing ANTHROPIC_API_KEY in .env file\n");
        adam_cleanup();
        return 1;
    }

    // Create settings and configure the Anthropic provider
    adam_settings_t *settings = adam_create_settings();
    adam_settings_set_provider(settings, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");
    adam_settings_set_identity(settings, "You are a friendly and knowledgeable assistant.");

    // Create conversation history (persists across turns)
    adam_history_t *history = adam_history_create();

    // --- Turn 1: Ask a question ---
    printf("User: What is the capital of France and what is it known for?\n\n");

    adam_run_result_t r1 = adam_run(settings, history, "What is the capital of France and what is it known for?");
    if (r1.status != ADAM_OK) {
        fprintf(stderr, "Turn 1 failed: %s\n", adam_status_string(r1.status));
        adam_run_result_free(&r1);
        goto cleanup;
    }

    printf("Assistant: %s\n\n", r1.final_response);
    printf("  [tokens: %d in / %d out, %.1f ms]\n\n", r1.input_tokens, r1.output_tokens, r1.elapsed_ms);
    adam_run_result_free(&r1);

    // --- Turn 2: Follow-up question (context is preserved in history) ---
    printf("User: How does its population compare to London?\n\n");

    adam_run_result_t r2 = adam_run(settings, history, "How does its population compare to London?");
    if (r2.status != ADAM_OK) {
        fprintf(stderr, "Turn 2 failed: %s\n", adam_status_string(r2.status));
        adam_run_result_free(&r2);
        goto cleanup;
    }

    printf("Assistant: %s\n\n", r2.final_response);
    printf("  [tokens: %d in / %d out, %.1f ms]\n\n", r2.input_tokens, r2.output_tokens, r2.elapsed_ms);
    adam_run_result_free(&r2);

    // --- Turn 3: Another follow-up ---
    printf("User: Which one would you recommend visiting and why?\n\n");

    adam_run_result_t r3 = adam_run(settings, history, "Which one would you recommend visiting and why?");
    if (r3.status != ADAM_OK) {
        fprintf(stderr, "Turn 3 failed: %s\n", adam_status_string(r3.status));
        adam_run_result_free(&r3);
        goto cleanup;
    }

    printf("Assistant: %s\n\n", r3.final_response);
    printf("  [tokens: %d in / %d out, %.1f ms]\n\n", r3.input_tokens, r3.output_tokens, r3.elapsed_ms);
    adam_run_result_free(&r3);

    printf("Conversation complete. Total messages in history: %zu\n", adam_history_count(history));

cleanup:
    adam_history_destroy(history);
    adam_settings_destroy(settings);
    free(api_key);
    adam_cleanup();
    return 0;
}
