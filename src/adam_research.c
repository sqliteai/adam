//
//  adam_research.c
//  Adam — Autonomous research mode
//
//  The research loop runs the agent in a persistent conversation,
//  using tools to gather information across multiple iterations.
//  Findings are accumulated and synthesized into a final report.
//
//  Created by Marco Bambini on 17/03/26.
//

#include "adam.h"
#define JSMN_STATIC
#include "jsmn.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Portable high-resolution timing
#ifdef _WIN32
  #include <windows.h>
  static double adam_research_time_ms(void) {
      LARGE_INTEGER freq, now;
      QueryPerformanceFrequency(&freq);
      QueryPerformanceCounter(&now);
      return (double)now.QuadPart / (double)freq.QuadPart * 1000.0;
  }
#elif defined(__EMSCRIPTEN__)
  #include <emscripten.h>
  static double adam_research_time_ms(void) {
      return emscripten_get_now();
  }
#else
  #include <time.h>
  static double adam_research_time_ms(void) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
  }
#endif

// ============================================================================
// MARK: - Config Defaults
// ============================================================================

adam_research_config_t adam_research_config_defaults(void) {
    adam_research_config_t c;
    memset(&c, 0, sizeof(c));
    c.max_iterations = 5;
    return c;
}

// ============================================================================
// MARK: - Result Free
// ============================================================================

void adam_research_result_free(adam_research_result_t *r) {
    if (!r) return;
    free(r->report);
    for (size_t i = 0; i < r->finding_count; i++) {
        free(r->findings[i].content);
        free(r->findings[i].source);
    }
    free(r->findings);
    memset(r, 0, sizeof(*r));
}

// ============================================================================
// MARK: - Internal: Finding Management
// ============================================================================

static adam_status_t findings_push(adam_research_result_t *r,
                                    const char *content, const char *source,
                                    int iteration) {
    size_t new_count = r->finding_count + 1;
    adam_research_finding_t *arr = realloc(r->findings,
        new_count * sizeof(adam_research_finding_t));
    if (!arr) return ADAM_ERR_ALLOC;
    r->findings = arr;

    adam_research_finding_t *f = &r->findings[r->finding_count];
    f->content = content ? strdup(content) : NULL;
    if (content && !f->content) return ADAM_ERR_ALLOC;
    f->source = source ? strdup(source) : NULL;
    f->iteration = iteration;
    r->finding_count = new_count;
    return ADAM_OK;
}

// ============================================================================
// MARK: - Internal: Parse Findings from Response
// ============================================================================

// Extract findings from LLM response. Looks for lines starting with
// "FINDING:" and optional "SOURCE:" on the next line.
static void parse_findings(adam_research_result_t *r, const char *response,
                            int iteration) {
    if (!response) return;

    const char *p = response;
    int max_findings = 100;
    while ((p = strstr(p, "FINDING:")) != NULL && max_findings-- > 0) {
        p += 8; // skip "FINDING:"
        while (*p == ' ') p++;

        // Find end of finding (next FINDING:, SOURCE:, or end)
        const char *end = p;
        while (*end && *end != '\n') end++;
        if (end == p) { p = end; continue; }

        // Extract content
        size_t len = (size_t)(end - p);
        char *content = malloc(len + 1);
        if (!content) continue;
        memcpy(content, p, len);
        content[len] = '\0';

        // Check for SOURCE: on next line
        char *source = NULL;
        const char *next = end;
        while (*next == '\n' || *next == '\r') next++;
        if (strncmp(next, "SOURCE:", 7) == 0) {
            next += 7;
            while (*next == ' ') next++;
            const char *src_end = next;
            while (*src_end && *src_end != '\n') src_end++;
            if (src_end > next) {
                source = malloc((size_t)(src_end - next) + 1);
                if (source) {
                    memcpy(source, next, (size_t)(src_end - next));
                    source[src_end - next] = '\0';
                }
            }
        }

        findings_push(r, content, source, iteration);
        free(content);
        free(source);
        p = end;
    }
}

// ============================================================================
// MARK: - Internal: Build System Prompt
// ============================================================================

static char *build_research_system(const char *question,
                                    const char *instructions) {
    size_t est = 2048;
    est += strlen(question);
    est += instructions ? strlen(instructions) : 0;

    char *buf = malloc(est);
    if (!buf) return NULL;

    size_t pos = 0;
    pos += (size_t)snprintf(buf + pos, est - pos,
        "You are an autonomous research agent. Your task is to thoroughly "
        "investigate the following question using all available tools.\n\n"
        "## Research Question\n%s\n\n", question);

    if (instructions) {
        pos += (size_t)snprintf(buf + pos, est - pos,
            "## Additional Instructions\n%s\n\n", instructions);
    }

    pos += (size_t)snprintf(buf + pos, est - pos,
        "## Research Protocol\n"
        "1. Use available tools to search for information.\n"
        "2. For each important fact you discover, output it as:\n"
        "   FINDING: <the fact>\n"
        "   SOURCE: <where you found it>\n"
        "3. After gathering information, identify gaps and search further.\n"
        "4. When your research is thorough and complete, begin your final "
        "message with RESEARCH_COMPLETE and provide a comprehensive report.\n"
        "5. Never fabricate information. If a tool search returns no results, "
        "note it and try a different approach.\n");

    buf[pos] = '\0';
    return buf;
}

// ============================================================================
// MARK: - Internal: Build Continuation Prompt
// ============================================================================

static char *build_continue_prompt(const adam_research_finding_t *findings,
                                    size_t finding_count, int iteration) {
    size_t est = 1024;
    for (size_t i = 0; i < finding_count; i++)
        est += findings[i].content ? strlen(findings[i].content) + 64 : 64;

    char *buf = malloc(est);
    if (!buf) return NULL;

    size_t pos = 0;

    if (finding_count > 0) {
        pos += (size_t)snprintf(buf + pos, est - pos,
            "Research iteration %d. Findings so far (%zu):\n",
            iteration + 1, finding_count);
        for (size_t i = 0; i < finding_count; i++) {
            const char *c = findings[i].content ? findings[i].content : "";
            pos += (size_t)snprintf(buf + pos, est - pos,
                "- %s%s%s\n", c,
                findings[i].source ? " [" : "",
                findings[i].source ? findings[i].source : "");
            if (findings[i].source)
                pos += (size_t)snprintf(buf + pos, est - pos, "]");
        }
        pos += (size_t)snprintf(buf + pos, est - pos,
            "\nIdentify gaps in your research and search for more information. "
            "Use tools to fill those gaps. "
            "If research is complete, say RESEARCH_COMPLETE and provide "
            "your final comprehensive report.");
    } else {
        pos += (size_t)snprintf(buf + pos, est - pos,
            "Begin your research. Use available tools to search for information "
            "about the research question. Report each finding with "
            "FINDING: and SOURCE: tags.");
    }

    buf[pos] = '\0';
    return buf;
}

// ============================================================================
// MARK: - Internal: Build Report Prompt
// ============================================================================

static char *build_report_prompt(const adam_research_finding_t *findings,
                                  size_t finding_count,
                                  const char *question) {
    size_t est = 2048;
    est += strlen(question);
    for (size_t i = 0; i < finding_count; i++)
        est += findings[i].content ? strlen(findings[i].content) + 128 : 128;

    char *buf = malloc(est);
    if (!buf) return NULL;

    size_t pos = 0;
    pos += (size_t)snprintf(buf + pos, est - pos,
        "Synthesize all research findings into a comprehensive report "
        "answering: %s\n\n"
        "## All Findings\n", question);

    for (size_t i = 0; i < finding_count; i++) {
        const char *c = findings[i].content ? findings[i].content : "";
        if (findings[i].source) {
            pos += (size_t)snprintf(buf + pos, est - pos,
                "- %s [Source: %s]\n", c, findings[i].source);
        } else {
            pos += (size_t)snprintf(buf + pos, est - pos, "- %s\n", c);
        }
    }

    pos += (size_t)snprintf(buf + pos, est - pos,
        "\nWrite a well-organized report with citations. "
        "Do NOT include FINDING: or SOURCE: tags in the report.");

    buf[pos] = '\0';
    return buf;
}

// ============================================================================
// MARK: - Internal: Accumulate Stats
// ============================================================================

static void research_accum(adam_research_result_t *r,
                            const adam_run_result_t *run) {
    r->total_input_tokens  += run->input_tokens;
    r->total_output_tokens += run->output_tokens;
    r->total_cost_usd      += run->cost_usd;
}

// ============================================================================
// MARK: - Research Loop
// ============================================================================

adam_research_result_t adam_research(adam_settings_t *settings,
                                     const adam_research_config_t *config) {
    adam_research_result_t result;
    memset(&result, 0, sizeof(result));

    // Validate
    if (!settings || !config) {
        result.status = ADAM_ERR_INVALID_PARAM;
        return result;
    }
    if (!config->question) {
        result.status = ADAM_ERR_INVALID_PARAM;
        return result;
    }

    int max_iter = config->max_iterations > 0 ? config->max_iterations : 5;

    // Record start time
    double t_start = adam_research_time_ms();

    // Build research system prompt and inject it into settings
    char *sys = build_research_system(config->question, config->instructions);
    if (!sys) {
        result.status = ADAM_ERR_ALLOC;
        return result;
    }

    // Save and override identity/instructions
    const char *saved_identity = settings->identity;
    const char *saved_instructions = settings->instructions;
    settings->identity = sys;
    settings->instructions = NULL;

    adam_history_t *history = adam_history_create();
    if (!history) {
        free(sys);
        settings->identity = saved_identity;
        settings->instructions = saved_instructions;
        result.status = ADAM_ERR_ALLOC;
        return result;
    }

    int complete = 0;

    for (int i = 0; i < max_iter && !complete; i++) {
        // Check abort
        if (settings->abort_flag) {
            result.status = ADAM_ERR_ABORTED;
            result.stop_reason = ADAM_RESEARCH_STOP_ABORTED;
            break;
        }

        // Build iteration prompt
        char *prompt = NULL;
        if (i == 0) {
            prompt = build_continue_prompt(NULL, 0, 0);
        } else {
            prompt = build_continue_prompt(result.findings,
                                            result.finding_count, i);
        }
        if (!prompt) {
            result.status = ADAM_ERR_ALLOC;
            result.stop_reason = ADAM_RESEARCH_STOP_ERROR;
            break;
        }

        // Run agent with tools
        adam_run_result_t run = adam_run(settings, history, prompt);
        free(prompt);

        if (run.status != ADAM_OK) {
            result.status = run.status;
            result.stop_reason = ADAM_RESEARCH_STOP_ERROR;
            adam_run_result_free(&run);
            break;
        }

        research_accum(&result, &run);
        result.total_iterations = i + 1;

        // Parse findings from response
        if (run.final_response) {
            parse_findings(&result, run.final_response, i);
        }

        // Progress callback
        if (config->on_progress) {
            char status[64];
            snprintf(status, sizeof(status), "iteration %d: %zu findings",
                     i + 1, result.finding_count);
            config->on_progress(config->progress_ctx, i, status);
        }

        // Check completeness
        if (run.final_response &&
            strstr(run.final_response, "RESEARCH_COMPLETE") != NULL) {
            // Agent declared research complete — use its response as report
            const char *report_start = strstr(run.final_response,
                                               "RESEARCH_COMPLETE");
            report_start += strlen("RESEARCH_COMPLETE");
            while (*report_start == '\n' || *report_start == '\r'
                   || *report_start == ' ')
                report_start++;
            if (*report_start) {
                result.report = strdup(report_start);
            } else {
                // RESEARCH_COMPLETE with no trailing text — use full response
                result.report = strdup(run.final_response);
            }
            result.status = ADAM_OK;
            result.stop_reason = ADAM_RESEARCH_STOP_COMPLETE;
            complete = 1;
        } else if (config->is_complete) {
            // User callback decides
            if (config->is_complete(config->complete_ctx,
                                     run.final_response ? run.final_response : "",
                                     i)) {
                result.status = ADAM_OK;
                result.stop_reason = ADAM_RESEARCH_STOP_CALLBACK;
                complete = 1;
            }
        }

        adam_run_result_free(&run);

        // If last iteration and not complete, mark as max_iters
        if (i == max_iter - 1 && !complete) {
            result.status = ADAM_OK;
            result.stop_reason = ADAM_RESEARCH_STOP_MAX_ITERS;
        }
    }

    // If we have findings but no report, ask LLM to synthesize one
    if (!result.report && result.finding_count > 0
        && result.status == ADAM_OK) {
        char *rpt_prompt = build_report_prompt(result.findings,
                                                result.finding_count,
                                                config->question);
        if (rpt_prompt) {
            adam_history_clear(history);
            adam_run_result_t rpt = adam_run(settings, history, rpt_prompt);
            free(rpt_prompt);
            if (rpt.status == ADAM_OK && rpt.final_response) {
                result.report = strdup(rpt.final_response);
            }
            research_accum(&result, &rpt);
            adam_run_result_free(&rpt);
        }
    }

    // If still no report, use the last response as fallback
    if (!result.report && history->count > 0) {
        for (size_t i = history->count; i > 0; i--) {
            if (history->items[i - 1].role == ADAM_ROLE_ASSISTANT
                && history->items[i - 1].content) {
                result.report = strdup(history->items[i - 1].content);
                break;
            }
        }
    }

    // Elapsed time
    result.elapsed_ms = adam_research_time_ms() - t_start;

    // Restore settings
    settings->identity = saved_identity;
    settings->instructions = saved_instructions;
    free(sys);
    adam_history_destroy(history);
    return result;
}

// ============================================================================
// MARK: - Built-in Tool: research
// ============================================================================

// jsmn helpers (same pattern as adam_memory.c)

static int jtok_eq(const char *json, const jsmntok_t *tok, const char *s) {
    size_t slen = strlen(s);
    return (tok->type == JSMN_STRING
         && (size_t)(tok->end - tok->start) == slen
         && memcmp(json + tok->start, s, slen) == 0);
}

static char *jtok_strdup(const char *json, const jsmntok_t *tok) {
    size_t len = (size_t)(tok->end - tok->start);
    char *s = malloc(len + 1);
    if (!s) return NULL;
    memcpy(s, json + tok->start, len);
    s[len] = '\0';
    return s;
}

adam_tool_result_t adam_tool_research(arena_t *arena, void *ctx,
                                      const char *args_json, size_t args_len) {
    adam_tool_result_t res = { .for_llm = NULL, .for_user = NULL, .success = 0 };
    adam_settings_t *s = (adam_settings_t *)ctx;
    if (!s || !args_json) {
        res.for_llm = arena_strdup(arena, "Error: settings not configured");
        return res;
    }

    // Parse JSON: {"question":"...","instructions":"...","max_iterations":5}
    jsmntok_t tokens[32];
    jsmn_parser parser;
    jsmn_init(&parser);
    int ntok = jsmn_parse(&parser, args_json, args_len, tokens, 32);
    if (ntok < 1) {
        res.for_llm = arena_strdup(arena, "Error: invalid JSON");
        return res;
    }

    char *question = NULL;
    char *instructions = NULL;
    int max_iterations = 0;

    for (int i = 1; i < ntok - 1; i++) {
        if (jtok_eq(args_json, &tokens[i], "question") &&
            tokens[i + 1].type == JSMN_STRING) {
            question = jtok_strdup(args_json, &tokens[i + 1]);
            i++;
        } else if (jtok_eq(args_json, &tokens[i], "instructions") &&
                   tokens[i + 1].type == JSMN_STRING) {
            instructions = jtok_strdup(args_json, &tokens[i + 1]);
            i++;
        } else if (jtok_eq(args_json, &tokens[i], "max_iterations") &&
                   tokens[i + 1].type == JSMN_PRIMITIVE) {
            max_iterations = atoi(args_json + tokens[i + 1].start);
            i++;
        }
    }

    if (!question) {
        free(instructions);
        res.for_llm = arena_strdup(arena, "Error: 'question' is required");
        return res;
    }

    // Run research
    adam_research_config_t cfg = adam_research_config_defaults();
    cfg.question = question;
    cfg.instructions = instructions;
    if (max_iterations > 0) cfg.max_iterations = max_iterations;

    adam_research_result_t rr = adam_research(s, &cfg);

    free(question);
    free(instructions);

    if (rr.status != ADAM_OK && rr.status != ADAM_ERR_ABORTED) {
        const char *msg = rr.report ? rr.report : "Research failed";
        res.for_llm = arena_strdup(arena, msg);
        adam_research_result_free(&rr);
        return res;
    }

    // Build result for LLM: report + findings summary
    size_t buf_size = 4096;
    if (rr.report) buf_size += strlen(rr.report);
    for (size_t i = 0; i < rr.finding_count; i++)
        buf_size += rr.findings[i].content ? strlen(rr.findings[i].content) + 64 : 64;

    char *buf = arena_alloc(arena, buf_size);
    if (!buf) {
        res.for_llm = arena_strdup(arena, rr.report ? rr.report : "Research complete.");
        adam_research_result_free(&rr);
        res.success = 1;
        return res;
    }

    size_t pos = 0;
    if (rr.report) {
        pos += (size_t)snprintf(buf + pos, buf_size - pos,
            "## Research Report\n\n%s\n\n", rr.report);
    }

    if (rr.finding_count > 0) {
        pos += (size_t)snprintf(buf + pos, buf_size - pos,
            "## Key Findings (%zu)\n\n", rr.finding_count);
        for (size_t i = 0; i < rr.finding_count && pos < buf_size - 128; i++) {
            const char *c = rr.findings[i].content ? rr.findings[i].content : "";
            if (rr.findings[i].source) {
                pos += (size_t)snprintf(buf + pos, buf_size - pos,
                    "- %s [%s]\n", c, rr.findings[i].source);
            } else {
                pos += (size_t)snprintf(buf + pos, buf_size - pos,
                    "- %s\n", c);
            }
        }
    }

    pos += (size_t)snprintf(buf + pos, buf_size - pos,
        "\n(%d iterations, %d findings)",
        rr.total_iterations, (int)rr.finding_count);

    buf[pos] = '\0';
    res.for_llm = buf;
    res.for_user = arena_strdup(arena, "Researching...");
    res.success = 1;

    adam_research_result_free(&rr);
    return res;
}
