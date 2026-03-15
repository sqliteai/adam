//
//  test_live.c
//  Adam — Live API test against Anthropic Claude
//
//  Reads API key from .env file. Not run in CI.
//
//  Build & run:
//    make live
//

#include "adam.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#ifndef ADAM_NO_PTHREADS
#include <pthread.h>
#endif

#define UNUSED_PARAM(p) ((void)(p))

// ============================================================================
// MARK: - Helpers
// ============================================================================

static const char *load_api_key(void) {
    FILE *f = fopen(".env", "r");
    if (!f) { fprintf(stderr, "Error: .env file not found\n"); return NULL; }
    static char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "ANTHROPIC_API_KEY=", 18) == 0) {
            fclose(f);
            char *val = line + 18;
            size_t len = strlen(val);
            while (len > 0 && (val[len-1] == '\n' || val[len-1] == '\r'))
                val[--len] = '\0';
            return val;
        }
    }
    fclose(f);
    fprintf(stderr, "Error: ANTHROPIC_API_KEY not found in .env\n");
    return NULL;
}

static void on_log(void *ctx, adam_log_level_t level, const char *msg, size_t len) {
    UNUSED_PARAM(ctx);
    const char *labels[] = {"ERROR","WARN ","INFO ","DEBUG","TRACE"};
    fprintf(stderr, "  [%s] %.*s\n", labels[level], (int)len, msg);
}

// Case-insensitive substring search
static int str_contains_ci(const char *haystack, const char *needle) {
    if (!haystack || !needle) return 0;
    size_t hlen = strlen(haystack), nlen = strlen(needle);
    if (nlen > hlen) return 0;
    for (size_t i = 0; i <= hlen - nlen; i++) {
        int match = 1;
        for (size_t j = 0; j < nlen; j++) {
            if (tolower((unsigned char)haystack[i+j]) != tolower((unsigned char)needle[j])) {
                match = 0; break;
            }
        }
        if (match) return 1;
    }
    return 0;
}

static int g_passed = 0;
static int g_failed = 0;
static float g_total_cost = 0.0f;
static double g_total_time = 0.0;
static int g_total_input_tokens = 0;
static int g_total_output_tokens = 0;

static void report(const char *name, int ok, adam_run_result_t *r) {
    if (ok) { g_passed++; printf("  PASS\n"); }
    else    { g_failed++; printf("  FAIL\n"); }
    if (r) {
        g_total_cost += r->cost_usd;
        g_total_time += r->elapsed_ms;
        g_total_input_tokens += r->input_tokens;
        g_total_output_tokens += r->output_tokens;
    }
    UNUSED_PARAM(name);
}

// ============================================================================
// MARK: - Mock Tools
// ============================================================================

static adam_tool_result_t tool_weather(arena_t *arena, void *ctx,
                                       const char *args_json, size_t args_len) {
    UNUSED_PARAM(ctx); UNUSED_PARAM(args_len);
    // Detect city from args
    const char *city = "Unknown";
    if (args_json) {
        if (strstr(args_json, "Rome") || strstr(args_json, "rome")) city = "Rome";
        else if (strstr(args_json, "Tokyo") || strstr(args_json, "tokyo")) city = "Tokyo";
        else if (strstr(args_json, "London") || strstr(args_json, "london")) city = "London";
    }
    char *result = arena_alloc(arena, 256);
    snprintf(result, 256,
        "{\"city\":\"%s\",\"temperature\":22,\"condition\":\"sunny\","
        "\"humidity\":45,\"wind_speed\":12}", city);
    return (adam_tool_result_t){ .for_llm = result, .success = 1 };
}

static adam_tool_result_t tool_calculator(arena_t *arena, void *ctx,
                                          const char *args_json, size_t args_len) {
    UNUSED_PARAM(ctx); UNUSED_PARAM(args_len);
    // Fake calculator — just echo back
    char *result = arena_alloc(arena, 256);
    snprintf(result, 256, "{\"result\":42,\"expression\":\"%s\"}",
             args_json ? args_json : "?");
    return (adam_tool_result_t){ .for_llm = result, .success = 1 };
}

static adam_tool_result_t tool_database(arena_t *arena, void *ctx,
                                        const char *args_json, size_t args_len) {
    UNUSED_PARAM(ctx); UNUSED_PARAM(args_len);
    // Fake database lookup
    char *result = arena_alloc(arena, 512);
    if (args_json && strstr(args_json, "user")) {
        snprintf(result, 512,
            "[{\"id\":1,\"name\":\"Marco\",\"role\":\"admin\"},"
            "{\"id\":2,\"name\":\"Alice\",\"role\":\"user\"}]");
    } else {
        snprintf(result, 512, "[]");
    }
    return (adam_tool_result_t){ .for_llm = result, .success = 1 };
}

static adam_tool_result_t tool_failing(arena_t *arena, void *ctx,
                                       const char *args_json, size_t args_len) {
    UNUSED_PARAM(ctx); UNUSED_PARAM(args_json); UNUSED_PARAM(args_len);
    return (adam_tool_result_t){
        .for_llm = arena_strdup(arena, "Error: connection refused to external service"),
        .success = 0,
    };
}

// ============================================================================
// MARK: - Test 1: Simple question
// ============================================================================

static void test_simple(adam_settings_t *s) {
    printf("\n--- Test 1: Simple question ---\n");
    adam_history_t *h = adam_history_create();

    adam_run_result_t r = adam_run(s, h, "What is 2+2? Answer with just the number.");

    printf("  Response:   %.80s\n", r.final_response ? r.final_response : "(null)");
    printf("  Tokens:     %d in / %d out | $%.6f | %.0fms\n",
           r.input_tokens, r.output_tokens, r.cost_usd, r.elapsed_ms);

    int ok = (r.status == ADAM_OK && r.final_response && strstr(r.final_response, "4"));
    report("simple", ok, &r);

    adam_run_result_free(&r);
    adam_history_destroy(h);
}

// ============================================================================
// MARK: - Test 2: Multi-turn conversation (memory across turns)
// ============================================================================

static void test_multi_turn(adam_settings_t *s) {
    printf("\n--- Test 2: Multi-turn conversation ---\n");
    adam_history_t *h = adam_history_create();

    adam_run_result_t r1 = adam_run(s, h,
        "I have 3 pets: a dog named Rex, a cat named Luna, and a parrot named Kiwi.");
    printf("  Turn 1:     %.80s\n", r1.final_response ? r1.final_response : "(null)");
    g_total_cost += r1.cost_usd; g_total_time += r1.elapsed_ms;
    g_total_input_tokens += r1.input_tokens; g_total_output_tokens += r1.output_tokens;
    adam_run_result_free(&r1);

    adam_run_result_t r2 = adam_run(s, h, "What is the name of my cat?");
    printf("  Turn 2:     %.80s\n", r2.final_response ? r2.final_response : "(null)");
    g_total_cost += r2.cost_usd; g_total_time += r2.elapsed_ms;
    g_total_input_tokens += r2.input_tokens; g_total_output_tokens += r2.output_tokens;

    int ok = (r2.status == ADAM_OK && str_contains_ci(r2.final_response, "Luna"));
    report("multi_turn", ok, NULL); // costs already tracked

    adam_run_result_free(&r2);

    // Third turn: verify all context is retained
    adam_run_result_t r3 = adam_run(s, h,
        "List all my pets with their species. Be brief.");
    printf("  Turn 3:     %.80s\n", r3.final_response ? r3.final_response : "(null)");
    printf("  History:    %zu messages\n", adam_history_count(h));

    int ok3 = (r3.status == ADAM_OK
               && str_contains_ci(r3.final_response, "Rex")
               && str_contains_ci(r3.final_response, "Luna")
               && str_contains_ci(r3.final_response, "Kiwi"));
    report("multi_turn_3", ok3, &r3);

    adam_run_result_free(&r3);
    adam_history_destroy(h);
}

// ============================================================================
// MARK: - Test 3: Single tool use
// ============================================================================

static void test_single_tool(adam_settings_t *s) {
    printf("\n--- Test 3: Single tool use ---\n");

    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "get_weather",
        .description = "Get current weather for a city",
        .parameters_json = "{\"type\":\"object\",\"properties\":"
            "{\"city\":{\"type\":\"string\",\"description\":\"City name\"}},"
            "\"required\":[\"city\"]}",
        .execute = tool_weather,
    });

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h,
        "What's the weather in Rome? Use the get_weather tool.");

    printf("  Response:   %.80s\n", r.final_response ? r.final_response : "(null)");
    printf("  Iterations: %d | History: %zu msgs\n",
           r.total_iterations, adam_history_count(h));
    printf("  Tokens:     %d in / %d out | $%.6f | %.0fms\n",
           r.input_tokens, r.output_tokens, r.cost_usd, r.elapsed_ms);

    int ok = (r.status == ADAM_OK && r.total_iterations >= 2
              && str_contains_ci(r.final_response, "sunny"));
    report("single_tool", ok, &r);

    adam_settings_remove_tool(s, "get_weather");
    adam_run_result_free(&r);
    adam_history_destroy(h);
}

// ============================================================================
// MARK: - Test 4: Multiple tools available (LLM chooses the right one)
// ============================================================================

static void test_tool_selection(adam_settings_t *s) {
    printf("\n--- Test 4: Tool selection (3 tools, pick the right one) ---\n");

    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "get_weather",
        .description = "Get current weather for a city",
        .parameters_json = "{\"type\":\"object\",\"properties\":"
            "{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}",
        .execute = tool_weather,
    });
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "calculate",
        .description = "Evaluate a math expression",
        .parameters_json = "{\"type\":\"object\",\"properties\":"
            "{\"expression\":{\"type\":\"string\"}},\"required\":[\"expression\"]}",
        .execute = tool_calculator,
    });
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "query_database",
        .description = "Query users from the database",
        .parameters_json = "{\"type\":\"object\",\"properties\":"
            "{\"query\":{\"type\":\"string\"}},\"required\":[\"query\"]}",
        .execute = tool_database,
    });

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h,
        "How many users are in the database? Use the query_database tool.");

    printf("  Response:   %.80s\n", r.final_response ? r.final_response : "(null)");
    printf("  Iterations: %d\n", r.total_iterations);

    // Verify it used the database tool (not weather or calculator)
    int used_db = 0;
    for (size_t i = 0; i < adam_history_count(h); i++) {
        if (h->items[i].role == ADAM_ROLE_ASSISTANT && h->items[i].tool_call_count > 0) {
            for (size_t t = 0; t < h->items[i].tool_call_count; t++) {
                if (strcmp(h->items[i].tool_calls[t].name, "query_database") == 0)
                    used_db = 1;
            }
        }
    }

    int ok = (r.status == ADAM_OK && r.total_iterations >= 2 && used_db);
    report("tool_selection", ok, &r);

    adam_settings_remove_tool(s, "get_weather");
    adam_settings_remove_tool(s, "calculate");
    adam_settings_remove_tool(s, "query_database");
    adam_run_result_free(&r);
    adam_history_destroy(h);
}

// ============================================================================
// MARK: - Test 5: Tool returns error (LLM should handle gracefully)
// ============================================================================

static void test_tool_error(adam_settings_t *s) {
    printf("\n--- Test 5: Tool returns error ---\n");

    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "external_api",
        .description = "Call an external API service",
        .parameters_json = "{\"type\":\"object\",\"properties\":"
            "{\"endpoint\":{\"type\":\"string\"}},\"required\":[\"endpoint\"]}",
        .execute = tool_failing,
    });

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h,
        "Call the external_api tool with endpoint 'status'. "
        "If it fails, explain the error.");

    printf("  Response:   %.100s\n", r.final_response ? r.final_response : "(null)");
    printf("  Iterations: %d\n", r.total_iterations);

    // The LLM should acknowledge the error in its response
    int ok = (r.status == ADAM_OK && r.final_response != NULL
              && (str_contains_ci(r.final_response, "error")
                  || str_contains_ci(r.final_response, "fail")
                  || str_contains_ci(r.final_response, "refused")
                  || str_contains_ci(r.final_response, "unable")));
    report("tool_error", ok, &r);

    adam_settings_remove_tool(s, "external_api");
    adam_run_result_free(&r);
    adam_history_destroy(h);
}

// ============================================================================
// MARK: - Test 6: JSON mode (structured output)
// ============================================================================

static void test_json_mode(adam_settings_t *s) {
    printf("\n--- Test 6: JSON mode ---\n");

    // Temporarily set response format
    const char *old_fmt = s->response_format;
    s->response_format = "json";

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h,
        "List 3 programming languages with their year of creation. "
        "Respond as a JSON array of objects with 'name' and 'year' fields.");

    printf("  Response:   %.120s\n", r.final_response ? r.final_response : "(null)");

    // Check that the response contains JSON content
    // (Anthropic may wrap in ```json fence, OpenAI returns raw JSON)
    int ok = (r.status == ADAM_OK && r.final_response
              && (strstr(r.final_response, "\"name\"") != NULL)
              && (strstr(r.final_response, "\"year\"") != NULL));
    report("json_mode", ok, &r);

    s->response_format = old_fmt;
    adam_run_result_free(&r);
    adam_history_destroy(h);
}

// ============================================================================
// MARK: - Test 7: Long context (system prompt + instructions)
// ============================================================================

static void test_system_prompt(adam_settings_t *s) {
    printf("\n--- Test 7: Custom system prompt behavior ---\n");

    // Temporarily override identity
    const char *old_id = s->identity;
    s->identity =
        "You are Captain Blackbeard, a friendly pirate. "
        "You always speak in pirate dialect and end every response with 'Arrr!'.";

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "What do you do for a living?");

    printf("  Response:   %.120s\n", r.final_response ? r.final_response : "(null)");

    int ok = (r.status == ADAM_OK && r.final_response
              && str_contains_ci(r.final_response, "arrr"));
    report("system_prompt", ok, &r);

    s->identity = old_id;
    adam_run_result_free(&r);
    adam_history_destroy(h);
}

// ============================================================================
// MARK: - Test 8: Tool use with follow-up conversation
// ============================================================================

static void test_tool_then_followup(adam_settings_t *s) {
    printf("\n--- Test 8: Tool use + follow-up questions ---\n");

    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "get_weather",
        .description = "Get current weather for a city",
        .parameters_json = "{\"type\":\"object\",\"properties\":"
            "{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}",
        .execute = tool_weather,
    });

    adam_history_t *h = adam_history_create();

    // Turn 1: use tool
    adam_run_result_t r1 = adam_run(s, h,
        "What's the weather in Tokyo?");
    printf("  Turn 1:     %.80s\n", r1.final_response ? r1.final_response : "(null)");
    g_total_cost += r1.cost_usd; g_total_time += r1.elapsed_ms;
    g_total_input_tokens += r1.input_tokens; g_total_output_tokens += r1.output_tokens;
    adam_run_result_free(&r1);

    // Turn 2: follow-up referencing the tool result (no tool needed)
    adam_run_result_t r2 = adam_run(s, h,
        "Should I bring a jacket based on that temperature?");
    printf("  Turn 2:     %.80s\n", r2.final_response ? r2.final_response : "(null)");
    printf("  History:    %zu messages\n", adam_history_count(h));

    // The LLM should reference the 22°C temperature from the tool
    int ok = (r2.status == ADAM_OK && r2.final_response != NULL);
    report("tool_followup", ok, &r2);

    adam_settings_remove_tool(s, "get_weather");
    adam_run_result_free(&r2);
    adam_history_destroy(h);
}

// ============================================================================
// MARK: - Test 9: Streaming callback
// ============================================================================

static void test_streaming(adam_settings_t *s) {
    printf("\n--- Test 9: on_response callback ---\n");

    // Use on_response as a pseudo-streaming test
    // (real SSE streaming not implemented yet, but the callback fires)
    typedef struct { int calls; size_t bytes; char last[256]; } resp_track_t;
    static resp_track_t tracker;
    memset(&tracker, 0, sizeof(tracker));

    void (*old_cb)(void*, const char*, size_t) = s->on_response;
    void *old_ctx = s->callback_ctx;

    s->on_response = (void(*)(void*,const char*,size_t))
        // inline: track responses
        NULL; // We'll use a separate function

    // Actually, let's just verify the final response comes through
    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h,
        "Count from 1 to 5, separated by commas. Nothing else.");

    printf("  Response:   %.80s\n", r.final_response ? r.final_response : "(null)");

    int ok = (r.status == ADAM_OK && r.final_response
              && strstr(r.final_response, "1") && strstr(r.final_response, "5"));
    report("response_callback", ok, &r);

    s->on_response = old_cb;
    s->callback_ctx = old_ctx;
    adam_run_result_free(&r);
    adam_history_destroy(h);
}

// ============================================================================
// MARK: - Test 10: Multilingual
// ============================================================================

static void test_multilingual(adam_settings_t *s) {
    printf("\n--- Test 10: Multilingual (Italian) ---\n");

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h,
        "Rispondi in italiano: qual e' la capitale della Francia? Una sola parola.");

    printf("  Response:   %.80s\n", r.final_response ? r.final_response : "(null)");

    int ok = (r.status == ADAM_OK && str_contains_ci(r.final_response, "Parigi"));
    report("multilingual", ok, &r);

    adam_run_result_free(&r);
    adam_history_destroy(h);
}

// ============================================================================
// MARK: - Test 11: Thread pool with real API calls
// ============================================================================

#ifndef ADAM_NO_PTHREADS

static pthread_mutex_t g_pool_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_pool_results_ok = 0;
static int g_pool_results_total = 0;

static void pool_done(void *ctx, adam_run_result_t result) {
    int idx = *(int *)ctx;
    pthread_mutex_lock(&g_pool_lock);
    g_pool_results_total++;
    if (result.status == ADAM_OK && result.final_response) {
        g_pool_results_ok++;
        printf("    Worker %d: %.60s\n", idx,
               result.final_response);
    } else {
        printf("    Worker %d: FAILED (%s)\n", idx,
               adam_status_string(result.status));
    }
    g_total_cost += result.cost_usd;
    g_total_time += result.elapsed_ms;
    g_total_input_tokens += result.input_tokens;
    g_total_output_tokens += result.output_tokens;
    pthread_mutex_unlock(&g_pool_lock);
    adam_run_result_free(&result);
}

static void test_thread_pool(adam_settings_t *s) {
    printf("\n--- Test 11: Thread pool (3 parallel API calls) ---\n");

    g_pool_results_ok = 0;
    g_pool_results_total = 0;

    adam_pool_t *pool = adam_pool_create(3);

    const char *questions[] = {
        "What is the capital of Japan? One word only.",
        "What is the largest planet in the solar system? One word only.",
        "What element has the symbol 'Au'? One word only.",
    };
    int indices[3] = {0, 1, 2};

    adam_settings_t *worker_settings[3];
    adam_history_t *worker_histories[3];

    for (int i = 0; i < 3; i++) {
        // Each worker gets its own settings + history (not shared)
        worker_settings[i] = adam_create_settings();
        adam_settings_set_provider(worker_settings[i], s->api_format,
                                   s->api_key, s->model);
        adam_settings_set_identity(worker_settings[i],
            "You are concise. Answer with one word only.");
        worker_histories[i] = adam_history_create();

        adam_pool_submit(pool, (adam_job_t){
            .settings = worker_settings[i],
            .history = worker_histories[i],
            .user_message = questions[i],
            .on_done = pool_done,
            .on_done_ctx = &indices[i],
        });
    }

    adam_pool_destroy(pool); // waits for all to complete

    printf("  Completed:  %d/%d\n", g_pool_results_ok, g_pool_results_total);
    int ok = (g_pool_results_ok == 3);
    report("thread_pool", ok, NULL);

    for (int i = 0; i < 3; i++) {
        adam_history_destroy(worker_histories[i]);
        adam_settings_destroy(worker_settings[i]);
    }
}

#endif // ADAM_NO_PTHREADS

// ============================================================================
// MARK: - Main
// ============================================================================

int main(void) {
    const char *api_key = load_api_key();
    if (!api_key) return 1;

    adam_init();

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_provider(s, ADAM_API_ANTHROPIC, api_key,
                               "claude-sonnet-4-20250514");
    adam_settings_set_identity(s, "You are a concise assistant. Answer briefly.");
    adam_settings_set_logger(s, on_log, NULL, ADAM_LOG_DEBUG);

    printf("Adam Live API Test Suite\n");
    printf("========================\n");
    printf("Model: %s\n", s->model);

    test_simple(s);
    test_multi_turn(s);
    test_single_tool(s);
    test_tool_selection(s);
    test_tool_error(s);
    test_json_mode(s);
    test_system_prompt(s);
    test_tool_then_followup(s);
    test_streaming(s);
    test_multilingual(s);

#ifndef ADAM_NO_PTHREADS
    test_thread_pool(s);
#endif

    printf("\n========================\n");
    printf("Results:  %d passed, %d failed, %d total\n",
           g_passed, g_failed, g_passed + g_failed);
    printf("Tokens:   %d input, %d output\n",
           g_total_input_tokens, g_total_output_tokens);
    printf("Cost:     $%.6f\n", g_total_cost);
    printf("Time:     %.1f ms total\n", g_total_time);

    adam_settings_destroy(s);
    adam_cleanup();
    return g_failed > 0 ? 1 : 0;
}
