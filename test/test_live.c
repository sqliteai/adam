//
//  test_live.c
//  Adam — Live API test against Anthropic Claude
//
//  Reads API key from .env file. Not run in CI.
//
//  Build:
//    make live
//
//  Run:
//    ./test_live
//

#include "adam.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Load API key from .env file
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

// Logger
static void on_log(void *ctx, adam_log_level_t level, const char *msg, size_t len) {
    (void)ctx;
    const char *labels[] = {"ERROR","WARN ","INFO ","DEBUG","TRACE"};
    fprintf(stderr, "  [%s] %.*s\n", labels[level], (int)len, msg);
}

// ============================================================================
// Test 1: Simple question
// ============================================================================

static int test_simple(adam_settings_t *s) {
    printf("\n--- Test 1: Simple question ---\n");

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "What is 2+2? Answer with just the number.");

    printf("  Status:     %s\n", adam_status_string(r.status));
    printf("  Response:   %s\n", r.final_response ? r.final_response : "(null)");
    printf("  Iterations: %d\n", r.total_iterations);
    printf("  Tokens:     %d in, %d out\n", r.input_tokens, r.output_tokens);
    printf("  Cost:       $%.6f\n", r.cost_usd);
    printf("  Time:       %.1f ms\n", r.elapsed_ms);

    int ok = (r.status == ADAM_OK && r.final_response != NULL);
    if (ok) printf("  PASS\n"); else printf("  FAIL\n");

    adam_run_result_free(&r);
    adam_history_destroy(h);
    return ok;
}

// ============================================================================
// Test 2: Multi-turn conversation
// ============================================================================

static int test_multi_turn(adam_settings_t *s) {
    printf("\n--- Test 2: Multi-turn conversation ---\n");

    adam_history_t *h = adam_history_create();

    adam_run_result_t r1 = adam_run(s, h, "My name is Marco. Remember it.");
    printf("  Turn 1:     %s\n", r1.final_response ? r1.final_response : "(null)");
    adam_run_result_free(&r1);

    adam_run_result_t r2 = adam_run(s, h, "What is my name?");
    printf("  Turn 2:     %s\n", r2.final_response ? r2.final_response : "(null)");
    printf("  History:    %zu messages\n", adam_history_count(h));

    int ok = (r2.status == ADAM_OK && r2.final_response
              && strstr(r2.final_response, "Marco") != NULL);
    if (ok) printf("  PASS\n"); else printf("  FAIL\n");

    adam_run_result_free(&r2);
    adam_history_destroy(h);
    return ok;
}

// ============================================================================
// Test 3: Tool use
// ============================================================================

static adam_tool_result_t weather_tool(arena_t *arena, void *ctx,
                                       const char *args_json, size_t args_len) {
    (void)ctx; (void)args_json; (void)args_len;
    return (adam_tool_result_t){
        .for_llm = arena_strdup(arena,
            "{\"temperature\":22,\"condition\":\"sunny\",\"city\":\"Rome\"}"),
        .success = 1,
    };
}

static int test_tool_use(adam_settings_t *s) {
    printf("\n--- Test 3: Tool use ---\n");

    // Add a weather tool
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "get_weather",
        .description = "Get the current weather for a city",
        .parameters_json = "{\"type\":\"object\",\"properties\":"
            "{\"city\":{\"type\":\"string\",\"description\":\"City name\"}},"
            "\"required\":[\"city\"]}",
        .execute = weather_tool,
    });

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h,
        "What's the weather in Rome? Use the get_weather tool.");

    printf("  Status:     %s\n", adam_status_string(r.status));
    printf("  Response:   %s\n", r.final_response ? r.final_response : "(null)");
    printf("  Iterations: %d\n", r.total_iterations);
    printf("  History:    %zu messages\n", adam_history_count(h));
    printf("  Tokens:     %d in, %d out\n", r.input_tokens, r.output_tokens);
    printf("  Cost:       $%.6f\n", r.cost_usd);
    printf("  Time:       %.1f ms\n", r.elapsed_ms);

    // Should have used the tool (iterations > 1)
    int ok = (r.status == ADAM_OK && r.total_iterations > 1
              && r.final_response != NULL);
    if (ok) printf("  PASS\n"); else printf("  FAIL\n");

    // Remove tool for subsequent tests
    adam_settings_remove_tool(s, "get_weather");

    adam_run_result_free(&r);
    adam_history_destroy(h);
    return ok;
}

// ============================================================================
// Main
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

    printf("Adam Live API Test\n");
    printf("==================\n");
    printf("Model: %s\n", s->model);

    int passed = 0, total = 3;
    passed += test_simple(s);
    passed += test_multi_turn(s);
    passed += test_tool_use(s);

    printf("\n==================\n");
    printf("Results: %d/%d passed\n", passed, total);

    adam_settings_destroy(s);
    adam_cleanup();
    return (passed == total) ? 0 : 1;
}
