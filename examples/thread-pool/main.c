//
//  thread-pool/main.c
//  Concurrent agent execution with a thread pool
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

#define NUM_JOBS 10

// Context passed to the on_done callback
typedef struct {
    int job_id;
} job_ctx_t;

// Called from a worker thread when a job completes
static void on_job_done(void *ctx, adam_run_result_t result) {
    job_ctx_t *jc = (job_ctx_t *)ctx;
    if (result.status == ADAM_OK) {
        printf("[Job %2d] %s\n", jc->job_id, result.final_response);
        printf("         [%d in / %d out tokens, %.1f ms]\n",
               result.input_tokens, result.output_tokens, result.elapsed_ms);
    } else {
        printf("[Job %2d] ERROR: %s\n", jc->job_id, adam_status_string(result.status));
    }
    adam_run_result_free(&result);
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

    // Each job needs its own settings and history (not shared across threads)
    adam_settings_t *settings[NUM_JOBS];
    adam_history_t  *histories[NUM_JOBS];
    job_ctx_t        contexts[NUM_JOBS];
    char             prompts[NUM_JOBS][128];

    for (int i = 0; i < NUM_JOBS; i++) {
        settings[i] = adam_create_settings();
        adam_settings_set_provider(settings[i], ADAM_API_ANTHROPIC, api_key,
                                  "claude-sonnet-4-20250514");
        settings[i]->max_tokens = 100;
        histories[i] = adam_history_create();
        contexts[i].job_id = i + 1;
        snprintf(prompts[i], sizeof(prompts[i]),
                 "Tell me one random fun fact about the number %d. Keep it to one sentence.", i + 1);
    }

    // Create a pool with 4 worker threads
    adam_pool_t *pool = adam_pool_create(4);
    if (!pool) {
        fprintf(stderr, "Failed to create thread pool\n");
        goto cleanup;
    }

    printf("Submitting %d jobs to a 4-worker thread pool...\n\n", NUM_JOBS);

    // Submit all jobs
    for (int i = 0; i < NUM_JOBS; i++) {
        adam_job_t job = {
            .settings     = settings[i],
            .history      = histories[i],
            .user_message = prompts[i],
            .on_done      = on_job_done,
            .on_done_ctx  = &contexts[i]
        };
        adam_status_t s = adam_pool_submit(pool, job);
        if (s != ADAM_OK) {
            fprintf(stderr, "Failed to submit job %d: %s\n", i + 1, adam_status_string(s));
        }
    }

    // Wait for all jobs to finish and destroy the pool
    adam_pool_destroy(pool);
    printf("\nAll jobs completed.\n");

cleanup:
    for (int i = 0; i < NUM_JOBS; i++) {
        adam_history_destroy(histories[i]);
        adam_settings_destroy(settings[i]);
    }
    free(api_key);
    adam_cleanup();
    return 0;
}
