//
//  response-cache/main.c
//  LRU response caching for repeated queries
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

    adam_settings_t *settings = adam_create_settings();
    adam_settings_set_provider(settings, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");

    // Create a response cache with up to 64 entries
    adam_cache_t *cache = adam_cache_create(64);
    settings->cache = cache;

    const char *query = "In one sentence, what is the speed of light in a vacuum?";

    // --- First run: cache miss, calls the LLM ---
    printf("--- Run 1 (expect cache miss) ---\n");
    adam_history_t *h1 = adam_history_create();
    adam_run_result_t r1 = adam_run(settings, h1, query);
    if (r1.status != ADAM_OK) {
        fprintf(stderr, "Run 1 failed: %s\n", adam_status_string(r1.status));
        adam_run_result_free(&r1);
        adam_history_destroy(h1);
        goto cleanup;
    }
    printf("Response: %s\n", r1.final_response);
    printf("  [%.1f ms]\n", r1.elapsed_ms);
    printf("  Cache: %zu hits, %zu misses, %zu entries\n\n",
           adam_cache_hits(cache), adam_cache_misses(cache), adam_cache_count(cache));
    adam_run_result_free(&r1);
    adam_history_destroy(h1);

    // --- Second run: same query, cache hit ---
    printf("--- Run 2 (expect cache hit) ---\n");
    adam_history_t *h2 = adam_history_create();
    adam_run_result_t r2 = adam_run(settings, h2, query);
    if (r2.status != ADAM_OK) {
        fprintf(stderr, "Run 2 failed: %s\n", adam_status_string(r2.status));
        adam_run_result_free(&r2);
        adam_history_destroy(h2);
        goto cleanup;
    }
    printf("Response: %s\n", r2.final_response);
    printf("  [%.1f ms]\n", r2.elapsed_ms);
    printf("  Cache: %zu hits, %zu misses, %zu entries\n\n",
           adam_cache_hits(cache), adam_cache_misses(cache), adam_cache_count(cache));
    adam_run_result_free(&r2);
    adam_history_destroy(h2);

cleanup:
    adam_cache_destroy(cache);
    adam_settings_destroy(settings);
    free(api_key);
    adam_cleanup();
    return 0;
}
