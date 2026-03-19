//
//  memory/main.c
//  Long-term memory with hybrid BM25 + vector search
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

    // Open an in-memory database for this demo
    adam_memory_t *mem = adam_memory_open(":memory:");
    if (!mem) {
        fprintf(stderr, "Failed to open memory database\n");
        free(api_key);
        adam_cleanup();
        return 1;
    }

    // Create settings and configure provider
    adam_settings_t *settings = adam_create_settings();
    adam_settings_set_provider(settings, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");

    // Configure memory embedding (using the same API key for embeddings)
    adam_settings_set_memory(settings, "openai", "text-embedding-3-small",
                            env_load("OPENAI_API_KEY"));

    // Attach memory to settings and apply configuration
    settings->memory = mem;
    adam_memory_configure(mem, settings);

    // Add facts that the LLM would not know on its own
    printf("Adding facts to memory...\n");
    adam_memory_add(mem, "Project Aurora's launch date is March 15, 2027.",
                    "projects", "internal-docs");
    adam_memory_add(mem, "The Aurora team lead is Dr. Sarah Chen from the Boston office.",
                    "projects", "internal-docs");
    adam_memory_add(mem, "Aurora's budget is $4.2 million, approved by the board in January.",
                    "projects", "internal-docs");
    printf("  Added 3 facts about Project Aurora.\n\n");

    // Enable automatic memory injection into the system prompt
    settings->inject_memory = 1;

    // Ask a question the agent can only answer from memory
    adam_history_t *history = adam_history_create();

    printf("User: Who is leading Project Aurora and when does it launch?\n\n");

    adam_run_result_t result = adam_run(settings, history,
        "Who is leading Project Aurora and when does it launch?");

    if (result.status == ADAM_OK) {
        printf("Assistant: %s\n\n", result.final_response);
        printf("  [tokens: %d in / %d out, %.1f ms]\n", result.input_tokens,
               result.output_tokens, result.elapsed_ms);
    } else {
        fprintf(stderr, "Run failed: %s\n", adam_status_string(result.status));
    }

    adam_run_result_free(&result);
    adam_history_destroy(history);
    adam_memory_close(mem);
    adam_settings_destroy(settings);
    free(api_key);
    adam_cleanup();
    return 0;
}
