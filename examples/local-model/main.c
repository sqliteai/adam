//
//  local-model/main.c
//  Offline inference with a local GGUF model via llama.cpp
//

#include "adam.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path-to-model.gguf>\n", argv[0]);
        return 1;
    }

    const char *gguf_path = argv[1];

    adam_status_t status = adam_init();
    if (status != ADAM_OK) {
        fprintf(stderr, "adam_init failed: %s\n", adam_status_string(status));
        return 1;
    }

    adam_settings_t *settings = adam_create_settings();

    // Configure local inference: GGUF path, all GPU layers (-1), default context size (0)
    status = adam_settings_set_local(settings, gguf_path, -1, 0);
    if (status != ADAM_OK) {
        fprintf(stderr, "Failed to set local model: %s\n", adam_status_string(status));
        adam_settings_destroy(settings);
        adam_cleanup();
        return 1;
    }

    settings->temperature = 0.8f;
    settings->max_tokens = 256;

    adam_history_t *history = adam_history_create();

    printf("Prompt: Write a haiku about programming.\n\n");

    adam_run_result_t result = adam_run(settings, history,
        "Write a haiku about programming.");

    if (result.status != ADAM_OK) {
        fprintf(stderr, "adam_run failed: %s\n", adam_status_string(result.status));
    } else {
        printf("Response:\n%s\n\n", result.final_response);
        printf("  [tokens: %d in / %d out, %.1f ms]\n",
               result.input_tokens, result.output_tokens, result.elapsed_ms);
    }

    adam_run_result_free(&result);
    adam_history_destroy(history);
    adam_settings_destroy(settings);
    adam_cleanup();
    return 0;
}
