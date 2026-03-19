//
//  evolution/main.c
//  Self-improving agent via the evolution loop
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

// Evaluation function: scores output based on length and keyword presence
static int eval_fn(void *ctx, const char *output, int iteration) {
    (void)ctx;
    (void)iteration;

    if (!output || strlen(output) == 0) return 0;

    int score = 0;
    size_t len = strlen(output);

    // Reward sufficient length (up to 40 points)
    if (len >= 200) score += 40;
    else if (len >= 100) score += 25;
    else if (len >= 50) score += 10;

    // Reward presence of key terms (up to 40 points)
    const char *keywords[] = {"benefit", "example", "conclusion", "important", "because"};
    int keyword_count = sizeof(keywords) / sizeof(keywords[0]);
    for (int i = 0; i < keyword_count; i++) {
        if (strstr(output, keywords[i]) != NULL) {
            score += 8;
        }
    }

    // Reward structure: presence of numbered points or bullet markers (up to 20 points)
    if (strstr(output, "1.") != NULL && strstr(output, "2.") != NULL) {
        score += 20;
    } else if (strstr(output, "- ") != NULL) {
        score += 10;
    }

    return score > 100 ? 100 : score;
}

// Progress callback: print each iteration's score
static void on_progress(void *ctx, int iteration, int score, int best_score,
                         const char *output) {
    (void)ctx;
    (void)output;
    printf("  Iteration %d: score=%d (best so far: %d)\n", iteration, score, best_score);
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

    // Configure the evolution loop
    adam_evolve_config_t config = adam_evolve_config_defaults();
    config.task = "Write a persuasive paragraph explaining why learning to code "
                  "is important. Include concrete examples, use structured points, "
                  "and end with a strong conclusion.";
    config.initial_strategy = "Write a clear, structured paragraph with numbered points.";
    config.metrics = "Score based on: sufficient length (200+ chars), presence of key "
                     "terms (benefit, example, conclusion, important, because), and "
                     "use of numbered or bulleted structure.";
    config.max_iterations = 5;
    config.target_score = 90;
    config.plateau_iters = 3;
    config.eval_fn = eval_fn;
    config.eval_ctx = NULL;
    config.progress_fn = on_progress;
    config.progress_ctx = NULL;

    printf("Running evolution loop...\n\n");

    adam_evolve_result_t result = adam_evolve(settings, &config);

    if (result.status == ADAM_OK) {
        printf("\n--- Evolution Complete ---\n\n");

        const char *reason;
        switch (result.stop_reason) {
            case ADAM_EVOLVE_STOP_SCORE:     reason = "target score reached"; break;
            case ADAM_EVOLVE_STOP_PLATEAU:   reason = "plateau (no improvement)"; break;
            case ADAM_EVOLVE_STOP_MAX_ITERS: reason = "max iterations"; break;
            case ADAM_EVOLVE_STOP_ABORTED:   reason = "aborted"; break;
            default:                         reason = "error"; break;
        }
        printf("Stop reason: %s\n", reason);
        printf("Best score: %d (iteration %d)\n\n", result.best_score, result.best_iteration);

        if (result.best_output) {
            printf("Best output:\n%s\n\n", result.best_output);
        }
        if (result.strategy) {
            printf("Final strategy:\n%s\n\n", result.strategy);
        }

        printf("  [%d iterations, %d in / %d out tokens, $%.4f, %.1f ms]\n",
               result.attempt_total, result.total_input_tokens,
               result.total_output_tokens, result.total_cost_usd, result.elapsed_ms);
    } else {
        fprintf(stderr, "Evolution failed: %s\n", adam_status_string(result.status));
    }

    adam_evolve_result_free(&result);
    adam_settings_destroy(settings);
    free(api_key);
    adam_cleanup();
    return 0;
}
