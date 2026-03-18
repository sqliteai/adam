//
//  adam_evolve.c
//  Adam — Self-evolving agent loop
//
//  Created by Marco Bambini on 17/03/26.
//

#include "adam.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Portable high-resolution timing
#ifdef _WIN32
  #include <windows.h>
  static double adam_time_ms(void) {
      LARGE_INTEGER freq, now;
      QueryPerformanceFrequency(&freq);
      QueryPerformanceCounter(&now);
      return (double)now.QuadPart / (double)freq.QuadPart * 1000.0;
  }
#elif defined(__EMSCRIPTEN__)
  #include <emscripten.h>
  static double adam_time_ms(void) {
      return emscripten_get_now();
  }
#else
  #include <time.h>
  static double adam_time_ms(void) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
  }
#endif

// ============================================================================
// MARK: - Config Defaults
// ============================================================================

adam_evolve_config_t adam_evolve_config_defaults(void) {
    adam_evolve_config_t c;
    memset(&c, 0, sizeof(c));
    c.max_iterations  = 10;
    c.target_score    = 95;
    c.plateau_iters   = 3;
    return c;
}

// ============================================================================
// MARK: - Result Free
// ============================================================================

void adam_evolve_result_free(adam_evolve_result_t *r) {
    if (!r) return;
    free(r->best_output);
    free(r->strategy);
    free(r->insights);
    for (int i = 0; i < r->attempt_count; i++)
        free(r->attempts[i].output);
    memset(r, 0, sizeof(*r));
}

// ============================================================================
// MARK: - Internal: Prompt Builders
// ============================================================================

// Build a dynamic string with snprintf, returning a malloc'd buffer.
// Returns NULL on OOM.
static char *build_attempt_prompt(const char *task, const char *strategy,
                                   const char *metrics, const char *insights,
                                   const adam_evolve_attempt_t *attempts,
                                   int attempt_count, int iteration) {
    // Estimate size
    size_t est = 2048;
    est += strlen(task);
    est += strategy ? strlen(strategy) : 0;
    est += metrics ? strlen(metrics) : 0;
    est += insights ? strlen(insights) : 0;
    for (int i = 0; i < attempt_count; i++)
        est += attempts[i].output ? strlen(attempts[i].output) + 64 : 64;

    char *buf = malloc(est);
    if (!buf) return NULL;

    size_t pos = 0;
    pos += (size_t)snprintf(buf + pos, est - pos,
        "You are a self-improving agent on iteration #%d.\n\n"
        "## TASK (immutable)\n%s\n\n"
        "## CURRENT STRATEGY\n%s\n\n",
        iteration + 1,
        task,
        strategy ? strategy : "No strategy yet.");

    if (metrics) {
        pos += (size_t)snprintf(buf + pos, est - pos,
            "## SCORING CRITERIA\n%s\n\n", metrics);
    }

    if (insights && strlen(insights) > 0) {
        pos += (size_t)snprintf(buf + pos, est - pos,
            "## INSIGHTS (accumulated lessons)\n%s\n\n", insights);
    }

    if (attempt_count > 0) {
        pos += (size_t)snprintf(buf + pos, est - pos,
            "## PREVIOUS ATTEMPTS\n");
        for (int i = 0; i < attempt_count; i++) {
            const char *out = attempts[i].output ? attempts[i].output : "(empty)";
            // Truncate long outputs in the summary
            size_t out_len = strlen(out);
            if (out_len > 200) {
                pos += (size_t)snprintf(buf + pos, est - pos,
                    "- Iteration %d (score %d): %.200s...\n",
                    attempts[i].iteration + 1, attempts[i].score, out);
            } else {
                pos += (size_t)snprintf(buf + pos, est - pos,
                    "- Iteration %d (score %d): %s\n",
                    attempts[i].iteration + 1, attempts[i].score, out);
            }
        }
        pos += (size_t)snprintf(buf + pos, est - pos, "\n");
    }

    pos += (size_t)snprintf(buf + pos, est - pos,
        "Produce your best attempt to accomplish the TASK.\n"
        "Improve on all previous attempts using the strategy and insights.\n"
        "Output ONLY the attempt, no commentary.");

    buf[pos] = '\0';
    return buf;
}

static char *build_refine_prompt(const char *task, const char *old_strategy,
                                  const char *output, int old_score,
                                  int new_score, const char *metrics) {
    size_t est = 2048;
    est += strlen(task);
    est += old_strategy ? strlen(old_strategy) : 0;
    est += output ? strlen(output) : 0;
    est += metrics ? strlen(metrics) : 0;

    char *buf = malloc(est);
    if (!buf) return NULL;

    snprintf(buf, est,
        "The score improved from %d to %d.\n\n"
        "## TASK\n%s\n\n"
        "## OLD STRATEGY\n%s\n\n"
        "## ATTEMPT THAT SCORED %d\n%s\n\n"
        "%s%s%s"
        "Refine the strategy to build on what worked.\n"
        "Output ONLY the updated strategy, no commentary.",
        old_score, new_score,
        task,
        old_strategy ? old_strategy : "(none)",
        new_score,
        output ? output : "(empty)",
        metrics ? "## METRICS\n" : "",
        metrics ? metrics : "",
        metrics ? "\n\n" : "");

    return buf;
}

static char *build_insight_prompt(const char *task, const char *output,
                                   int score, const char *insights,
                                   const char *metrics, int iteration) {
    size_t est = 2048;
    est += strlen(task);
    est += output ? strlen(output) : 0;
    est += insights ? strlen(insights) : 0;
    est += metrics ? strlen(metrics) : 0;

    char *buf = malloc(est);
    if (!buf) return NULL;

    snprintf(buf, est,
        "Iteration %d scored %d/100.\n\n"
        "## TASK\n%s\n\n"
        "## ATTEMPT\n%s\n\n"
        "## CURRENT INSIGHTS\n%s\n\n"
        "%s%s%s"
        "Update the insights with any new lessons learned.\n"
        "Include: what works, what doesn't, and open questions.\n"
        "Output ONLY the complete updated insights.",
        iteration + 1, score,
        task,
        output ? output : "(empty)",
        insights ? insights : "No insights yet.",
        metrics ? "## METRICS\n" : "",
        metrics ? metrics : "",
        metrics ? "\n\n" : "");

    return buf;
}

// ============================================================================
// MARK: - Internal: Ring Buffer
// ============================================================================

static void ring_push(adam_evolve_result_t *r, const char *output,
                       int score, int iteration) {
    if (r->attempt_count < ADAM_EVOLVE_MAX_ATTEMPTS) {
        // Still filling up
        int idx = r->attempt_count;
        r->attempts[idx].output = output ? strdup(output) : NULL;
        r->attempts[idx].score = score;
        r->attempts[idx].iteration = iteration;
        r->attempt_count++;
    } else {
        // Overwrite oldest (index 0, shift everything left)
        free(r->attempts[0].output);
        memmove(&r->attempts[0], &r->attempts[1],
                (ADAM_EVOLVE_MAX_ATTEMPTS - 1) * sizeof(adam_evolve_attempt_t));
        int idx = ADAM_EVOLVE_MAX_ATTEMPTS - 1;
        r->attempts[idx].output = output ? strdup(output) : NULL;
        r->attempts[idx].score = score;
        r->attempts[idx].iteration = iteration;
    }
}

// ============================================================================
// MARK: - Internal: Helper to accumulate run stats
// ============================================================================

static void accum_stats(adam_evolve_result_t *r, const adam_run_result_t *run) {
    r->total_input_tokens  += run->input_tokens;
    r->total_output_tokens += run->output_tokens;
    r->total_cost_usd      += run->cost_usd;
}

// ============================================================================
// MARK: - Evolution Loop
// ============================================================================

adam_evolve_result_t adam_evolve(adam_settings_t *settings,
                                 const adam_evolve_config_t *config) {
    adam_evolve_result_t result;
    memset(&result, 0, sizeof(result));
    result.best_score = -1;
    result.best_iteration = -1;

    // Validate inputs
    if (!settings || !config) {
        result.status = ADAM_ERR_INVALID_PARAM;
        return result;
    }
    if (!config->task || !config->eval_fn) {
        result.status = ADAM_ERR_INVALID_PARAM;
        return result;
    }

    int max_iter = config->max_iterations > 0 ? config->max_iterations : 10;
    int target   = config->target_score > 0 ? config->target_score : 95;
    int plateau  = config->plateau_iters > 0 ? config->plateau_iters : 3;

    // Initialize mutable state
    char *strategy = strdup(config->initial_strategy
                            ? config->initial_strategy
                            : "No strategy yet.");
    char *insights = strdup("No insights yet.");
    if (!strategy || !insights) {
        free(strategy);
        free(insights);
        result.status = ADAM_ERR_ALLOC;
        return result;
    }

    // Record start time
    double t_start = adam_time_ms();

    adam_history_t *history = adam_history_create();
    if (!history) {
        free(strategy);
        free(insights);
        result.status = ADAM_ERR_ALLOC;
        return result;
    }

    int iters_since_improvement = 0;

    for (int i = 0; i < max_iter; i++) {
        // Check abort
        if (settings->abort_flag) {
            result.stop_reason = ADAM_EVOLVE_STOP_ABORTED;
            result.status = ADAM_ERR_ABORTED;
            break;
        }

        // 1. Build attempt prompt
        char *prompt = build_attempt_prompt(
            config->task, strategy, config->metrics, insights,
            result.attempts, result.attempt_count, i);
        if (!prompt) {
            result.status = ADAM_ERR_ALLOC;
            result.stop_reason = ADAM_EVOLVE_STOP_ERROR;
            break;
        }

        // 2. Run attempt (fresh history each time)
        adam_history_clear(history);
        adam_run_result_t run = adam_run(settings, history, prompt);
        free(prompt);

        if (run.status != ADAM_OK) {
            result.status = run.status;
            result.stop_reason = ADAM_EVOLVE_STOP_ERROR;
            adam_run_result_free(&run);
            break;
        }

        accum_stats(&result, &run);

        char *output = run.final_response ? strdup(run.final_response) : NULL;
        adam_run_result_free(&run);

        // 3. Evaluate
        int score = config->eval_fn(config->eval_ctx,
                                     output ? output : "", i);
        if (score < 0) {
            free(output);
            result.status = ADAM_ERR_PROVIDER;
            result.stop_reason = ADAM_EVOLVE_STOP_ERROR;
            break;
        }

        // 4. Store in ring buffer
        ring_push(&result, output, score, i);
        result.attempt_total = i + 1;

        // 5. Progress callback
        if (config->progress_fn) {
            config->progress_fn(config->progress_ctx, i, score,
                                result.best_score, output ? output : "");
        }

        // 6. Check improvement
        if (score > result.best_score) {
            int old_best = result.best_score;
            result.best_score = score;
            result.best_iteration = i;
            free(result.best_output);
            result.best_output = output ? strdup(output) : NULL;
            iters_since_improvement = 0;

            // Refine strategy via LLM
            char *refine = build_refine_prompt(
                config->task, strategy, output, old_best, score,
                config->metrics);
            if (refine) {
                adam_history_clear(history);
                adam_run_result_t ref = adam_run(settings, history, refine);
                free(refine);
                if (ref.status == ADAM_OK && ref.final_response) {
                    char *new_strat = strdup(ref.final_response);
                    if (new_strat) { free(strategy); strategy = new_strat; }
                }
                accum_stats(&result, &ref);
                adam_run_result_free(&ref);
            }
        } else {
            iters_since_improvement++;
        }

        // 7. Extract insights (always)
        char *ins_prompt = build_insight_prompt(
            config->task, output, score, insights, config->metrics, i);
        if (ins_prompt) {
            adam_history_clear(history);
            adam_run_result_t ins = adam_run(settings, history, ins_prompt);
            free(ins_prompt);
            if (ins.status == ADAM_OK && ins.final_response) {
                char *new_ins = strdup(ins.final_response);
                if (new_ins) { free(insights); insights = new_ins; }
            }
            accum_stats(&result, &ins);
            adam_run_result_free(&ins);
        }

        free(output);

        // 8. Check stop conditions
        if (result.best_score >= target) {
            result.status = ADAM_OK;
            result.stop_reason = ADAM_EVOLVE_STOP_SCORE;
            result.attempt_total = i + 1;
            break;
        }
        if (iters_since_improvement >= plateau) {
            result.status = ADAM_OK;
            result.stop_reason = ADAM_EVOLVE_STOP_PLATEAU;
            result.attempt_total = i + 1;
            break;
        }

        // Completed all iterations
        if (i == max_iter - 1) {
            result.status = ADAM_OK;
            result.stop_reason = ADAM_EVOLVE_STOP_MAX_ITERS;
            result.attempt_total = max_iter;
        }
    }

    // Transfer ownership of strategy and insights to result
    result.strategy = strategy;
    result.insights = insights;

    // Elapsed time
    result.elapsed_ms = adam_time_ms() - t_start;

    adam_history_destroy(history);
    return result;
}
