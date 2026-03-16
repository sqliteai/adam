//
//  test_chat.c
//  Adam — Interactive text chat (keyboard in, text out)
//
//  Supports both remote APIs and local GGUF models.
//
//  Build & run:
//    make chat                          (remote: Anthropic/OpenAI/Grok)
//    make chat GGUF=models/model.gguf   (local: llama.cpp)
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

static void on_stream(void *ctx, const char *chunk, size_t len, int is_done) {
    UNUSED_PARAM(ctx);
    if (len > 0) fwrite(chunk, 1, len, stdout);
    if (is_done) printf("\n");
    fflush(stdout);
}

int main(int argc, char **argv) {
    adam_init();

    adam_settings_t *s = adam_create_settings();

    // Check for local GGUF model (passed as argument or GGUF env)
    const char *gguf = NULL;
    for (int i = 1; i < argc; i++) {
        if (strstr(argv[i], ".gguf")) { gguf = argv[i]; break; }
    }

#ifndef ADAM_NO_LOCAL
    if (gguf) {
        s->gguf_path = gguf;
        s->local_gpu_layers = -1;
        s->local_ctx_size = 4096;
        printf("  Model: %s (local)\n", gguf);
    } else
#endif
    {
        // Try remote providers
        char *anthropic = env_load("ANTHROPIC_API_KEY");
        char *openai = env_load("OPENAI_API_KEY");
        char *grok = env_load("GROK_API_KEY");

        if (anthropic) {
            adam_settings_set_provider(s, ADAM_API_ANTHROPIC, anthropic,
                                       "claude-sonnet-4-20250514");
            printf("  Model: claude-sonnet-4 (Anthropic)\n");
        } else if (grok) {
            adam_settings_set_provider(s, ADAM_API_OPENAI, grok,
                                       "grok-3-mini-fast");
            adam_settings_set_base_url(s,
                "https://api.x.ai/v1/chat/completions");
            printf("  Model: grok-3-mini-fast (xAI)\n");
        } else if (openai) {
            adam_settings_set_provider(s, ADAM_API_OPENAI, openai,
                                       "gpt-4o-mini");
            printf("  Model: gpt-4o-mini (OpenAI)\n");
        } else {
            fprintf(stderr, "No API key in .env and no GGUF model.\n");
            fprintf(stderr, "Usage: %s [model.gguf]\n", argv[0]);
            adam_settings_destroy(s);
            adam_cleanup();
            return 1;
        }
    }

    adam_settings_set_identity(s,
        "You are a helpful assistant. Be concise but thorough.");
    s->on_stream = on_stream;

    printf("  Type your message. Press Enter to send. Ctrl+C to quit.\n\n");

    adam_history_t *h = adam_history_create();
    char input[4096];
    int turn = 0;

    while (1) {
        turn++;
        printf("[%d] You: ", turn);
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;

        size_t len = strlen(input);
        while (len > 0 && (input[len-1] == '\n' || input[len-1] == '\r'))
            input[--len] = '\0';
        if (len == 0) continue;

        printf("[%d] Adam: ", turn);
        fflush(stdout);

        adam_run_result_t r = adam_run(s, h, input);

        if (r.status != ADAM_OK) {
            printf("(error: %s)\n",
                   r.final_response ? r.final_response
                                    : adam_status_string(r.status));
        }
        // If not streaming, print the response
        if (!s->on_stream && r.final_response) {
            printf("%s\n", r.final_response);
        }

        printf("  [%d in/%d out | $%.4f | %.0fms]\n\n",
               r.input_tokens, r.output_tokens, r.cost_usd, r.elapsed_ms);

        adam_run_result_free(&r);
    }

    printf("\nGoodbye!\n");
    adam_history_destroy(h);
    adam_settings_destroy(s);
    adam_cleanup();
    return 0;
}
