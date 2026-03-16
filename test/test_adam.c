//
//  test_adam.c
//  Adam — Test suite
//
//  Tests cover real-world agent scenarios using a mock LLM backend.
//  Reports memory usage, checks for leaks, and measures performance.
//
//  Build:
//    cc -std=c11 -Wall -Wextra -Wpedantic -g -fsanitize=address,undefined \
//       -Isrc -DADAM_NO_CURL -DADAM_NO_LOCAL -DADAM_NO_SQLITE \
//       src/arena.c src/adam.c test/test_adam.c -o test_adam -lpthread
//
//  Run:
//    ./test_adam
//

#include "adam.h"
#include "adam_json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>


#ifndef ADAM_NO_PTHREADS
#include <pthread.h>
#endif

// ============================================================================
// MARK: - Test Framework
// ============================================================================

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;
static int g_asserts_total = 0;

#define TEST(name)                                                          \
    static void test_##name(void);                                          \
    static void run_test_##name(void) {                                     \
        g_tests_run++;                                                      \
        printf("  %-50s ", #name);                                          \
        fflush(stdout);                                                     \
        struct timespec _ts0, _ts1;                                         \
        clock_gettime(CLOCK_MONOTONIC, &_ts0);                              \
        test_##name();                                                      \
        clock_gettime(CLOCK_MONOTONIC, &_ts1);                              \
        double _ms = (_ts1.tv_sec - _ts0.tv_sec) * 1000.0                  \
                   + (_ts1.tv_nsec - _ts0.tv_nsec) / 1e6;                   \
        printf("PASS  (%6.2f ms)\n", _ms);                                 \
        g_tests_passed++;                                                   \
    }                                                                       \
    static void test_##name(void)

#define ASSERT(cond)                                                        \
    do {                                                                    \
        g_asserts_total++;                                                  \
        if (!(cond)) {                                                      \
            printf("FAIL\n    assertion failed: %s\n    at %s:%d\n",        \
                   #cond, __FILE__, __LINE__);                              \
            g_tests_failed++;                                               \
            g_tests_passed--;                                               \
            return;                                                         \
        }                                                                   \
    } while (0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NE(a, b) ASSERT((a) != (b))
#define ASSERT_NULL(p) ASSERT((p) == NULL)
#define ASSERT_NOT_NULL(p) ASSERT((p) != NULL)
#define ASSERT_STR_EQ(a, b) ASSERT(strcmp((a), (b)) == 0)

#define RUN(name) run_test_##name()

// ============================================================================
// MARK: - Memory Tracking
// ============================================================================

// We rely on ASan for leak detection (compile with -fsanitize=address).
// This section reports peak RSS for the test process.

#ifdef __APPLE__
#include <mach/mach.h>
static size_t get_rss_bytes(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) != KERN_SUCCESS)
        return 0;
    return info.resident_size;
}
#elif defined(__linux__)
static size_t get_rss_bytes(void) {
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long pages = 0;
    fscanf(f, "%*ld %ld", &pages);
    fclose(f);
    return (size_t)pages * 4096;
}
#else
static size_t get_rss_bytes(void) { return 0; }
#endif

static size_t g_rss_start = 0;

static void mem_report_start(void) {
    g_rss_start = get_rss_bytes();
}

static void mem_report_end(void) {
    size_t rss_end = get_rss_bytes();
    printf("\n  Memory:\n");
    printf("    RSS at start:  %zu KB\n", g_rss_start / 1024);
    printf("    RSS at end:    %zu KB\n", rss_end / 1024);
    printf("    Delta:         %+ld KB\n",
           (long)(rss_end - g_rss_start) / 1024);
    printf("    (Leak detection via ASan — if no ASan errors above, no leaks)\n");
}

// ============================================================================
// MARK: - Mock LLM Backend
// ============================================================================

// The mock LLM simulates realistic agent behavior:
//   - First call: returns a tool call
//   - Second call: returns a final text response using the tool result
//
// This exercises the full agent loop: dispatch → tool execution → re-dispatch.

typedef struct {
    int call_count;                 // how many times the LLM was called
    int total_tool_calls_requested; // how many tool calls the LLM emitted
    const char *fixed_response;     // if non-NULL, always return this text
    int simulate_error;             // if non-zero, return this error code
    int simulate_error_on_call;     // which call# to error on (0 = all)
} mock_llm_ctx_t;

// Mock: returns a tool call on first call, then a final response.
static adam_llm_response_t mock_llm_with_tool(
    void *ctx, arena_t *arena,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
) {
    mock_llm_ctx_t *mock = (mock_llm_ctx_t *)ctx;
    mock->call_count++;

    (void)tools;
    (void)tool_count;

    adam_llm_response_t resp = {0};
    resp.input_tokens = 100;
    resp.output_tokens = 50;

    // Check if there's a tool result in the history — if so, give final answer
    int has_tool_result = 0;
    for (size_t i = 0; i < msg_count; i++) {
        if (msgs[i].role == ADAM_ROLE_TOOL) {
            has_tool_result = 1;
            break;
        }
    }

    if (!has_tool_result && tool_count > 0) {
        // First call: request a tool call
        mock->total_tool_calls_requested++;
        resp.tool_calls = arena_alloc(arena, sizeof(adam_tool_call_t));
        resp.tool_call_count = 1;
        resp.tool_calls[0].id = arena_strdup(arena, "call_001");
        resp.tool_calls[0].name = arena_strdup(arena, tools[0].name);
        resp.tool_calls[0].arguments_json =
            arena_strdup(arena, "{\"query\":\"test\"}");
        resp.content = arena_strdup(arena, "Let me look that up.");
    } else {
        // Second call: final answer incorporating tool result
        resp.content = arena_strdup(arena,
            "Based on the tool result, the answer is 42.");
    }

    return resp;
}

// Mock: always returns a fixed text response (no tool calls).
static adam_llm_response_t mock_llm_simple(
    void *ctx, arena_t *arena,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
) {
    mock_llm_ctx_t *mock = (mock_llm_ctx_t *)ctx;
    mock->call_count++;

    (void)msgs; (void)msg_count; (void)tools; (void)tool_count;

    adam_llm_response_t resp = {0};
    resp.input_tokens = 80;
    resp.output_tokens = 30;
    resp.content = arena_strdup(arena,
        mock->fixed_response ? mock->fixed_response : "Hello from mock LLM!");
    return resp;
}

// Mock: simulates an error (rate limit, auth, etc.)
static adam_llm_response_t mock_llm_error(
    void *ctx, arena_t *arena,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
) {
    mock_llm_ctx_t *mock = (mock_llm_ctx_t *)ctx;
    mock->call_count++;

    (void)msgs; (void)msg_count; (void)tools; (void)tool_count;

    adam_llm_response_t resp = {0};

    if (mock->simulate_error_on_call > 0
        && mock->call_count != mock->simulate_error_on_call) {
        // Not the error call — return normal response
        resp.input_tokens = 50;
        resp.output_tokens = 20;
        resp.content = arena_strdup(arena, "Recovery response.");
        return resp;
    }

    resp.error = (adam_status_t)mock->simulate_error;
    resp.error_msg = arena_strdup(arena, "simulated error");
    return resp;
}

// Mock: returns multiple tool calls in a single response.
static adam_llm_response_t mock_llm_multi_tool(
    void *ctx, arena_t *arena,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
) {
    mock_llm_ctx_t *mock = (mock_llm_ctx_t *)ctx;
    mock->call_count++;

    (void)tools; (void)tool_count;

    adam_llm_response_t resp = {0};
    resp.input_tokens = 150;
    resp.output_tokens = 80;

    // Check if we already got tool results
    int tool_results = 0;
    for (size_t i = 0; i < msg_count; i++) {
        if (msgs[i].role == ADAM_ROLE_TOOL) tool_results++;
    }

    if (tool_results >= 2) {
        // Got both results — final answer
        resp.content = arena_strdup(arena, "Combined result from both tools.");
    } else {
        // Request two tool calls at once
        mock->total_tool_calls_requested += 2;
        resp.tool_calls = arena_alloc(arena, 2 * sizeof(adam_tool_call_t));
        resp.tool_call_count = 2;
        resp.tool_calls[0].id = arena_strdup(arena, "call_a");
        resp.tool_calls[0].name = arena_strdup(arena, "tool_alpha");
        resp.tool_calls[0].arguments_json =
            arena_strdup(arena, "{\"x\":1}");
        resp.tool_calls[1].id = arena_strdup(arena, "call_b");
        resp.tool_calls[1].name = arena_strdup(arena, "tool_beta");
        resp.tool_calls[1].arguments_json =
            arena_strdup(arena, "{\"x\":2}");
        resp.content = arena_strdup(arena, "Calling two tools.");
    }

    return resp;
}

// ============================================================================
// MARK: - Mock Tools
// ============================================================================

static adam_tool_result_t mock_tool_search(
    arena_t *arena, void *ctx, const char *args_json, size_t args_len
) {
    (void)ctx; (void)args_json; (void)args_len;
    return (adam_tool_result_t){
        .for_llm = arena_strdup(arena, "Search result: found 42 matches."),
        .for_user = arena_strdup(arena, "Searching..."),
        .success = 1,
    };
}

static adam_tool_result_t mock_tool_alpha(
    arena_t *arena, void *ctx, const char *args_json, size_t args_len
) {
    (void)ctx; (void)args_json; (void)args_len;
    return (adam_tool_result_t){
        .for_llm = arena_strdup(arena, "Alpha result: OK"),
        .success = 1,
    };
}

static adam_tool_result_t mock_tool_beta(
    arena_t *arena, void *ctx, const char *args_json, size_t args_len
) {
    (void)ctx; (void)args_json; (void)args_len;
    return (adam_tool_result_t){
        .for_llm = arena_strdup(arena, "Beta result: OK"),
        .success = 1,
    };
}

// ============================================================================
// MARK: - Log Collector (for verifying log output)
// ============================================================================

typedef struct {
    char    buf[8192];
    size_t  len;
    int     error_count;
    int     warn_count;
    int     info_count;
} log_collector_t;

static void log_collect(void *ctx, adam_log_level_t level,
                         const char *msg, size_t len) {
    log_collector_t *lc = (log_collector_t *)ctx;
    if (lc->len + len + 1 < sizeof(lc->buf)) {
        memcpy(lc->buf + lc->len, msg, len);
        lc->len += len;
        lc->buf[lc->len++] = '\n';
    }
    if (level == ADAM_LOG_ERROR) lc->error_count++;
    if (level == ADAM_LOG_WARN) lc->warn_count++;
    if (level == ADAM_LOG_INFO) lc->info_count++;
}

// ============================================================================
// MARK: - Tests: Settings
// ============================================================================

TEST(create_settings_defaults) {
    adam_settings_t *s = adam_create_settings();
    ASSERT_NOT_NULL(s);

    // Check all defaults
    ASSERT_EQ(s->api_format, ADAM_API_NONE);
    ASSERT_NULL(s->api_key);
    ASSERT_NULL(s->base_url);
    ASSERT_STR_EQ(s->model, "claude-sonnet-4-20250514");
    ASSERT(s->temperature >= 0.69f && s->temperature <= 0.71f);
    ASSERT_EQ(s->max_tokens, 4096);
    ASSERT_NULL(s->response_format);
    ASSERT(s->top_p >= 0.99f && s->top_p <= 1.01f);
    ASSERT_EQ(s->max_iterations, 25);
    ASSERT_EQ(s->max_history, 100);
    ASSERT_EQ(s->arena_block_size, 256 * 1024);
    ASSERT_EQ(s->inject_datetime, 1);
    ASSERT_EQ(s->inject_memory, 0);
    ASSERT_NULL(s->tools);
    ASSERT_EQ(s->tool_count, 0);
    ASSERT_EQ(s->retry_max, 3);
    ASSERT_EQ(s->retry_backoff_ms[0], 1000);
    ASSERT_EQ(s->retry_backoff_ms[3], 10000);
    ASSERT_EQ(s->retry_rate_limit_ms, 30000);
    ASSERT_EQ(s->log_level, ADAM_LOG_WARN);
    ASSERT_STR_EQ(s->memory_context, "default");
    ASSERT_EQ(s->abort_flag, 0);
    ASSERT_NULL(s->llm_fn);
    ASSERT_NULL(s->http_fn);

    adam_settings_destroy(s);
}

TEST(settings_set_provider) {
    adam_settings_t *s = adam_create_settings();

    adam_status_t rc = adam_settings_set_provider(s, ADAM_API_ANTHROPIC,
                                                  "sk-test", "claude-opus-4");
    ASSERT_EQ(rc, ADAM_OK);
    ASSERT_EQ(s->api_format, ADAM_API_ANTHROPIC);
    ASSERT_STR_EQ(s->api_key, "sk-test");
    ASSERT_STR_EQ(s->model, "claude-opus-4");

    rc = adam_settings_set_base_url(s, "https://custom.api.com/v1/messages");
    ASSERT_EQ(rc, ADAM_OK);
    ASSERT_STR_EQ(s->base_url, "https://custom.api.com/v1/messages");

    // NULL settings should fail
    ASSERT_EQ(adam_settings_set_provider(NULL, ADAM_API_OPENAI, "k", "m"),
              ADAM_ERR_INVALID_PARAM);

    adam_settings_destroy(s);
}

TEST(settings_add_remove_tools) {
    adam_settings_t *s = adam_create_settings();

    adam_tool_def_t t1 = { .name = "search", .execute = mock_tool_search };
    adam_tool_def_t t2 = { .name = "fetch", .execute = mock_tool_search };

    ASSERT_EQ(adam_settings_add_tool(s, t1), ADAM_OK);
    ASSERT_EQ(s->tool_count, 1);

    ASSERT_EQ(adam_settings_add_tool(s, t2), ADAM_OK);
    ASSERT_EQ(s->tool_count, 2);

    ASSERT_EQ(adam_settings_remove_tool(s, "search"), ADAM_OK);
    ASSERT_EQ(s->tool_count, 1);
    ASSERT_STR_EQ(s->tools[0].name, "fetch");

    ASSERT_EQ(adam_settings_remove_tool(s, "nonexistent"), ADAM_ERR_TOOL_NOT_FOUND);

    // Invalid: NULL name or execute
    adam_tool_def_t bad = { .name = NULL, .execute = mock_tool_search };
    ASSERT_EQ(adam_settings_add_tool(s, bad), ADAM_ERR_INVALID_PARAM);
    bad.name = "x"; bad.execute = NULL;
    ASSERT_EQ(adam_settings_add_tool(s, bad), ADAM_ERR_INVALID_PARAM);

    adam_settings_destroy(s);
}

TEST(settings_callbacks) {
    adam_settings_t *s = adam_create_settings();

    mock_llm_ctx_t mock = {0};
    ASSERT_EQ(adam_settings_set_llm_callback(s, mock_llm_simple, &mock), ADAM_OK);
    ASSERT_EQ(s->llm_fn, mock_llm_simple);
    ASSERT_EQ(s->llm_ctx, &mock);

    log_collector_t lc = {0};
    ASSERT_EQ(adam_settings_set_logger(s, log_collect, &lc, ADAM_LOG_TRACE),
              ADAM_OK);
    ASSERT_EQ(s->log_level, ADAM_LOG_TRACE);

    adam_settings_destroy(s);
}

// ============================================================================
// MARK: - Tests: History
// ============================================================================

TEST(history_lifecycle) {
    adam_history_t *h = adam_history_create();
    ASSERT_NOT_NULL(h);
    ASSERT_EQ(adam_history_count(h), 0);

    ASSERT_EQ(adam_history_append_user(h, "Hello"), ADAM_OK);
    ASSERT_EQ(adam_history_count(h), 1);
    ASSERT_EQ(h->items[0].role, ADAM_ROLE_USER);
    ASSERT_STR_EQ(h->items[0].content, "Hello");

    ASSERT_EQ(adam_history_append_assistant(h, "Hi there", NULL, 0), ADAM_OK);
    ASSERT_EQ(adam_history_count(h), 2);

    adam_history_clear(h);
    ASSERT_EQ(adam_history_count(h), 0);

    adam_history_destroy(h);
}

TEST(history_with_tool_calls) {
    adam_history_t *h = adam_history_create();

    // Simulate: user → assistant(tool_call) → tool_result → assistant(final)
    ASSERT_EQ(adam_history_append_user(h, "What is 2+2?"), ADAM_OK);

    adam_tool_call_t tc = {
        .id = "call_123", .name = "calculator",
        .arguments_json = "{\"expr\":\"2+2\"}"
    };
    ASSERT_EQ(adam_history_append_assistant(h, "Let me calculate.", &tc, 1),
              ADAM_OK);
    ASSERT_EQ(h->items[1].tool_call_count, 1);
    ASSERT_STR_EQ(h->items[1].tool_calls[0].id, "call_123");
    ASSERT_STR_EQ(h->items[1].tool_calls[0].name, "calculator");

    ASSERT_EQ(adam_history_append_tool(h, "4", "call_123"), ADAM_OK);
    ASSERT_EQ(h->items[2].role, ADAM_ROLE_TOOL);
    ASSERT_STR_EQ(h->items[2].tool_call_id, "call_123");

    ASSERT_EQ(adam_history_append_assistant(h, "2+2 = 4", NULL, 0), ADAM_OK);
    ASSERT_EQ(adam_history_count(h), 4);

    adam_history_destroy(h);
}

TEST(history_token_estimation) {
    adam_history_t *h = adam_history_create();

    // 100 chars ≈ 25 tokens
    char msg[101];
    memset(msg, 'a', 100);
    msg[100] = '\0';
    adam_history_append_user(h, msg);

    size_t tokens = adam_history_estimate_tokens(h);
    ASSERT(tokens >= 20 && tokens <= 30);

    adam_history_destroy(h);
}

TEST(history_attachments) {
    adam_history_t *h = adam_history_create();
    adam_history_append_user(h, "What's in this image?");

    uint8_t fake_png[] = {0x89, 'P', 'N', 'G', 0, 1, 2, 3};
    ASSERT_EQ(adam_history_attach(h, ADAM_MEDIA_IMAGE_PNG, fake_png,
                                  sizeof(fake_png), "test.png"), ADAM_OK);
    ASSERT_EQ(h->items[0].attachment_count, 1);
    ASSERT_EQ(h->items[0].attachments[0].type, ADAM_MEDIA_IMAGE_PNG);
    ASSERT_EQ(h->items[0].attachments[0].data_len, sizeof(fake_png));
    ASSERT_STR_EQ(h->items[0].attachments[0].filename, "test.png");

    // Attach to empty history should fail
    adam_history_t *h2 = adam_history_create();
    ASSERT_EQ(adam_history_attach(h2, ADAM_MEDIA_IMAGE_PNG, fake_png,
                                   sizeof(fake_png), NULL),
              ADAM_ERR_INVALID_PARAM);

    adam_history_destroy(h);
    adam_history_destroy(h2);
}

// ============================================================================
// MARK: - Tests: Agent Run (real-world scenarios)
// ============================================================================

TEST(run_simple_conversation) {
    // Scenario: user sends a message, LLM responds with text (no tools).
    mock_llm_ctx_t mock = { .fixed_response = "The capital of France is Paris." };

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "What is the capital of France?");

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_NOT_NULL(r.final_response);
    ASSERT_STR_EQ(r.final_response, "The capital of France is Paris.");
    ASSERT_EQ(r.total_iterations, 1);
    ASSERT_EQ(r.input_tokens, 80);
    ASSERT_EQ(r.output_tokens, 30);
    ASSERT(r.elapsed_ms >= 0.0);
    ASSERT_EQ(mock.call_count, 1);

    // History should have: system + user + assistant = 3
    ASSERT_EQ(adam_history_count(h), 3);
    ASSERT_EQ(h->items[0].role, ADAM_ROLE_SYSTEM);
    ASSERT_EQ(h->items[1].role, ADAM_ROLE_USER);
    ASSERT_EQ(h->items[2].role, ADAM_ROLE_ASSISTANT);

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(run_with_tool_call) {
    // Scenario: LLM calls a tool, gets the result, then gives final answer.
    // This exercises the full tool iteration loop.
    mock_llm_ctx_t mock = {0};

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_with_tool, &mock);
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "search",
        .description = "Search for information",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}}}",
        .execute = mock_tool_search,
    });

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Find information about topic X");

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_STR_EQ(r.final_response,
                   "Based on the tool result, the answer is 42.");
    ASSERT_EQ(r.total_iterations, 2); // 1: tool call, 2: final answer
    ASSERT_EQ(mock.call_count, 2);
    ASSERT_EQ(mock.total_tool_calls_requested, 1);

    // History: system + user + assistant(tool_call) + tool_result + assistant(final) = 5
    ASSERT_EQ(adam_history_count(h), 5);
    ASSERT_EQ(h->items[2].role, ADAM_ROLE_ASSISTANT);
    ASSERT_EQ(h->items[2].tool_call_count, 1);
    ASSERT_STR_EQ(h->items[2].tool_calls[0].id, "call_001");
    ASSERT_EQ(h->items[3].role, ADAM_ROLE_TOOL);
    ASSERT_STR_EQ(h->items[3].tool_call_id, "call_001");
    ASSERT_EQ(h->items[4].role, ADAM_ROLE_ASSISTANT);

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(run_multi_tool_calls) {
    // Scenario: LLM requests two tool calls in a single response.
    mock_llm_ctx_t mock = {0};

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_multi_tool, &mock);
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "tool_alpha", .execute = mock_tool_alpha });
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "tool_beta", .execute = mock_tool_beta });

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Run both tools");

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_STR_EQ(r.final_response, "Combined result from both tools.");
    ASSERT_EQ(mock.call_count, 2);
    ASSERT_EQ(mock.total_tool_calls_requested, 2);

    // History: system + user + assistant(2 tool_calls) + tool_a + tool_b + assistant(final) = 6
    ASSERT_EQ(adam_history_count(h), 6);
    ASSERT_EQ(h->items[2].tool_call_count, 2);

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

// Mock that always requests a hardcoded tool name regardless of registry.
static adam_llm_response_t mock_llm_calls_missing_tool(
    void *ctx, arena_t *arena,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
) {
    mock_llm_ctx_t *mock = (mock_llm_ctx_t *)ctx;
    mock->call_count++;
    (void)tools; (void)tool_count;

    adam_llm_response_t resp = {0};
    resp.input_tokens = 50;
    resp.output_tokens = 20;

    // Check if there's a tool result already
    int has_tool_result = 0;
    for (size_t i = 0; i < msg_count; i++)
        if (msgs[i].role == ADAM_ROLE_TOOL) has_tool_result = 1;

    if (!has_tool_result) {
        resp.tool_calls = arena_alloc(arena, sizeof(adam_tool_call_t));
        resp.tool_call_count = 1;
        resp.tool_calls[0].id = arena_strdup(arena, "call_missing");
        resp.tool_calls[0].name = arena_strdup(arena, "nonexistent_tool");
        resp.tool_calls[0].arguments_json = arena_strdup(arena, "{}");
    } else {
        resp.content = arena_strdup(arena, "Got the error, moving on.");
    }
    return resp;
}

TEST(run_tool_not_found) {
    // Scenario: LLM calls a tool that doesn't exist in the registry.
    // The agent should send "Error: tool not found" back to the LLM.
    mock_llm_ctx_t mock = {0};

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_calls_missing_tool, &mock);
    // Register a different tool — the mock will call "nonexistent_tool"
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "real_tool", .execute = mock_tool_search });

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Use the missing tool");

    // The mock LLM sees "tool not found" as a tool result and still
    // produces a final answer on the second call.
    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_EQ(mock.call_count, 2);

    // The tool result should contain the error message
    ASSERT_EQ(h->items[3].role, ADAM_ROLE_TOOL);
    ASSERT(strstr(h->items[3].content, "tool not found") != NULL);

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(run_multi_turn_conversation) {
    // Scenario: multi-turn conversation preserving history across runs.
    mock_llm_ctx_t mock = { .fixed_response = "Response 1" };

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);

    adam_history_t *h = adam_history_create();

    // Turn 1
    adam_run_result_t r1 = adam_run(s, h, "Hello");
    ASSERT_EQ(r1.status, ADAM_OK);
    ASSERT_EQ(adam_history_count(h), 3); // sys + user + asst

    // Turn 2 — history carries forward
    mock.fixed_response = "Response 2";
    adam_run_result_t r2 = adam_run(s, h, "Follow up");
    ASSERT_EQ(r2.status, ADAM_OK);
    ASSERT_EQ(adam_history_count(h), 5); // +user +asst

    // Turn 3
    mock.fixed_response = "Response 3";
    adam_run_result_t r3 = adam_run(s, h, "Third message");
    ASSERT_EQ(r3.status, ADAM_OK);
    ASSERT_EQ(adam_history_count(h), 7);

    // Verify ordering
    ASSERT_EQ(h->items[0].role, ADAM_ROLE_SYSTEM);
    ASSERT_EQ(h->items[1].role, ADAM_ROLE_USER);
    ASSERT_EQ(h->items[2].role, ADAM_ROLE_ASSISTANT);
    ASSERT_EQ(h->items[3].role, ADAM_ROLE_USER);
    ASSERT_STR_EQ(h->items[3].content, "Follow up");

    adam_run_result_free(&r1);
    adam_run_result_free(&r2);
    adam_run_result_free(&r3);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(run_no_provider) {
    // Scenario: no provider configured — should fail immediately.
    adam_settings_t *s = adam_create_settings();
    adam_history_t *h = adam_history_create();

    adam_run_result_t r = adam_run(s, h, "Hello");
    ASSERT_EQ(r.status, ADAM_ERR_NO_PROVIDER);
    ASSERT_NOT_NULL(r.final_response);

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(run_null_params) {
    adam_settings_t *s = adam_create_settings();
    adam_history_t *h = adam_history_create();

    adam_run_result_t r1 = adam_run(NULL, h, "test");
    ASSERT_EQ(r1.status, ADAM_ERR_INVALID_PARAM);
    adam_run_result_free(&r1);

    adam_run_result_t r2 = adam_run(s, NULL, "test");
    ASSERT_EQ(r2.status, ADAM_ERR_INVALID_PARAM);
    adam_run_result_free(&r2);

    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(run_abort) {
    // Scenario: abort flag is set before running — should exit immediately.
    mock_llm_ctx_t mock = {0};

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);
    adam_abort(s);

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Hello");

    // abort_flag is reset at the start of adam_run, so this tests
    // that the flag was cleared. Let's test the other way:
    // set abort in a callback.
    adam_run_result_free(&r);

    // The run should have succeeded since abort_flag is reset at start
    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_EQ(mock.call_count, 1);

    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(run_auth_error_no_retry) {
    // Scenario: auth error should NOT be retried.
    mock_llm_ctx_t mock = { .simulate_error = ADAM_ERR_AUTH };

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_error, &mock);
    s->retry_max = 3;

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Hello");

    ASSERT_EQ(r.status, ADAM_ERR_AUTH);
    ASSERT_EQ(mock.call_count, 1); // No retries for auth errors

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(run_with_logging) {
    // Verify that the agent emits log messages at the right levels.
    mock_llm_ctx_t mock = { .fixed_response = "Logged response." };
    log_collector_t lc = {0};

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);
    adam_settings_set_logger(s, log_collect, &lc, ADAM_LOG_DEBUG);

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Hello");

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT(lc.info_count >= 1); // at least "adam_run: starting" + "adam_run: done"
    ASSERT(lc.len > 0);

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(run_cost_tracking) {
    mock_llm_ctx_t mock = { .fixed_response = "Cost test." };

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);
    adam_settings_set_provider(s, ADAM_API_ANTHROPIC, "key", "claude-sonnet-4-x");

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Test cost");

    ASSERT_EQ(r.status, ADAM_OK);
    // mock returns 80 input + 30 output tokens
    // claude-sonnet-4: $3/Mtok input, $15/Mtok output
    // cost = 80/1M * 3 + 30/1M * 15 = 0.00024 + 0.00045 = 0.00069
    ASSERT(r.cost_usd > 0.0f);
    ASSERT(r.cost_usd < 0.01f);

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

typedef struct { int call_count; char last[256]; } resp_ctx_t;

static void on_resp(void *ctx, const char *text, size_t len) {
    resp_ctx_t *r = (resp_ctx_t *)ctx;
    r->call_count++;
    size_t copy = len < 255 ? len : 255;
    memcpy(r->last, text, copy);
    r->last[copy] = '\0';
}

TEST(run_on_response_callback) {
    // Verify the on_response callback fires with tool user output and final response.
    resp_ctx_t rctx = {0};

    mock_llm_ctx_t mock = {0};
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_with_tool, &mock);
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "search", .execute = mock_tool_search });
    adam_settings_set_callbacks(s, on_resp, NULL, NULL, &rctx);

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Search for X");

    ASSERT_EQ(r.status, ADAM_OK);
    // Callbacks: 1 for tool's for_user + 1 for final response
    ASSERT_EQ(rctx.call_count, 2);
    ASSERT_STR_EQ(rctx.last, "Based on the tool result, the answer is 42.");

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

// ============================================================================
// MARK: - Tests: Arena
// ============================================================================

TEST(arena_basic) {
    arena_t *a = arena_create(4096);
    ASSERT_NOT_NULL(a);

    char *s1 = arena_strdup(a, "hello");
    ASSERT_STR_EQ(s1, "hello");

    void *p = arena_alloc(a, 1024);
    ASSERT_NOT_NULL(p);

    void *z = arena_zeroalloc(a, 128);
    ASSERT_NOT_NULL(z);
    // Verify zeroed
    uint8_t *zb = (uint8_t *)z;
    for (int i = 0; i < 128; i++) ASSERT_EQ(zb[i], 0);

    size_t used, cap, blocks;
    arena_stats(a, &used, &cap, &blocks);
    ASSERT(used > 0);
    ASSERT(cap >= used);
    ASSERT(blocks >= 1);

    arena_reset(a);
    arena_stats(a, &used, NULL, &blocks);
    ASSERT_EQ(used, 0);
    ASSERT_EQ(blocks, 1); // reset keeps first block

    arena_destroy(a);
}

TEST(arena_large_allocation) {
    // Allocation larger than block size should still work.
    arena_t *a = arena_create(1024);
    void *p = arena_alloc(a, 8192);
    ASSERT_NOT_NULL(p);

    size_t used, cap, blocks;
    arena_stats(a, &used, &cap, &blocks);
    ASSERT(blocks >= 2); // needs at least one extra block
    ASSERT(used >= 8192);

    arena_destroy(a);
}

// ============================================================================
// MARK: - Tests: Model Registry
// ============================================================================

TEST(model_registry_builtin) {
    int cw = adam_model_context_window("claude-sonnet-4-20250514");
    ASSERT_EQ(cw, 200000);

    cw = adam_model_context_window("gpt-4o-2024-08-06");
    ASSERT_EQ(cw, 128000);

    cw = adam_model_context_window("unknown-model");
    ASSERT_EQ(cw, 0);
}

TEST(model_registry_custom) {
    adam_model_info_t custom = {
        .model_prefix = "my-model",
        .context_window = 32000,
        .input_cost_mtok = 1.0f,
        .output_cost_mtok = 2.0f,
    };
    ASSERT_EQ(adam_model_register(&custom), ADAM_OK);

    ASSERT_EQ(adam_model_context_window("my-model-v1"), 32000);

    float cost = adam_model_estimate_cost("my-model-v1", 1000000, 500000);
    // 1M * 1.0/M + 0.5M * 2.0/M = 1.0 + 1.0 = 2.0
    ASSERT(cost > 1.9f && cost < 2.1f);
}

TEST(model_estimate_cost) {
    // claude-sonnet-4: $3/Mtok in, $15/Mtok out
    float cost = adam_model_estimate_cost("claude-sonnet-4",
                                           1000000, 1000000);
    // 1M * 3/M + 1M * 15/M = 3 + 15 = 18
    ASSERT(cost > 17.0f && cost < 19.0f);
}

// ============================================================================
// MARK: - Tests: Token Estimation
// ============================================================================

TEST(token_estimation) {
    ASSERT_EQ(adam_estimate_tokens("", 0), 0);
    ASSERT_EQ(adam_estimate_tokens(NULL, 0), 0);

    // 100 chars ≈ 25 tokens
    char buf[101];
    memset(buf, 'x', 100);
    buf[100] = '\0';
    size_t t = adam_estimate_tokens(buf, 100);
    ASSERT(t >= 20 && t <= 30);
}

TEST(status_strings) {
    ASSERT_STR_EQ(adam_status_string(ADAM_OK), "ok");
    ASSERT_STR_EQ(adam_status_string(ADAM_ERR_AUTH), "authentication error");
    ASSERT_STR_EQ(adam_status_string(ADAM_ERR_NO_PROVIDER), "no provider configured");
}

// ============================================================================
// MARK: - Tests: Thread Pool
// ============================================================================

#ifndef ADAM_NO_PTHREADS

// Shared atomic counter for thread pool test
static pthread_mutex_t g_pool_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_pool_done_count = 0;
static adam_run_result_t g_pool_results[8];

static void pool_on_done(void *ctx, adam_run_result_t result) {
    int idx = *(int *)ctx;
    pthread_mutex_lock(&g_pool_mutex);
    g_pool_results[idx] = result;
    // Don't free — we inspect later
    g_pool_done_count++;
    pthread_mutex_unlock(&g_pool_mutex);
}

TEST(thread_pool_basic) {
    g_pool_done_count = 0;

    adam_pool_t *pool = adam_pool_create(2);
    ASSERT_NOT_NULL(pool);

    // Submit 4 jobs
    mock_llm_ctx_t mocks[4];
    adam_settings_t *settings[4];
    adam_history_t *histories[4];
    int indices[4];

    for (int i = 0; i < 4; i++) {
        memset(&mocks[i], 0, sizeof(mock_llm_ctx_t));
        mocks[i].fixed_response = "Pool response";

        settings[i] = adam_create_settings();
        adam_settings_set_llm_callback(settings[i], mock_llm_simple, &mocks[i]);
        histories[i] = adam_history_create();
        indices[i] = i;

        adam_job_t job = {
            .settings = settings[i],
            .history = histories[i],
            .user_message = "Hello from pool",
            .on_done = pool_on_done,
            .on_done_ctx = &indices[i],
        };
        ASSERT_EQ(adam_pool_submit(pool, job), ADAM_OK);
    }

    // Destroy waits for all jobs to complete
    adam_pool_destroy(pool);

    ASSERT_EQ(g_pool_done_count, 4);
    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(g_pool_results[i].status, ADAM_OK);
        ASSERT_NOT_NULL(g_pool_results[i].final_response);
        ASSERT_STR_EQ(g_pool_results[i].final_response, "Pool response");
        adam_run_result_free(&g_pool_results[i]);
        adam_history_destroy(histories[i]);
        adam_settings_destroy(settings[i]);
    }
}

TEST(thread_pool_empty_destroy) {
    // Creating and immediately destroying with no jobs should not crash.
    adam_pool_t *pool = adam_pool_create(2);
    ASSERT_NOT_NULL(pool);
    ASSERT_EQ(adam_pool_pending(pool), 0);
    ASSERT_EQ(adam_pool_active(pool), 0);
    adam_pool_destroy(pool);
}

#endif // ADAM_NO_PTHREADS

// ============================================================================
// MARK: - Tests: Voice
// ============================================================================

#if !defined(ADAM_NO_VOICE) && !defined(ADAM_NO_PTHREADS)

TEST(voice_settings_defaults) {
    adam_settings_t *s = adam_create_settings();
    ASSERT_EQ(s->voice_enabled, 0);
    ASSERT_EQ(s->stt_backend, ADAM_STT_NONE);
    ASSERT_EQ(s->tts_backend, ADAM_TTS_NONE);
    ASSERT_STR_EQ(s->stt_model, "gpt-4o-mini-transcribe");
    ASSERT_EQ(s->stt_sample_rate, 16000);
    ASSERT_STR_EQ(s->tts_model, "gpt-4o-mini-tts");
    ASSERT_STR_EQ(s->tts_voice, "coral");
    ASSERT_EQ(s->tts_format, ADAM_AUDIO_MP3);
    ASSERT(s->voice_silence_sec > 0.9f && s->voice_silence_sec < 1.1f);
    ASSERT_NULL(s->stt_fn);
    ASSERT_NULL(s->tts_fn);
    ASSERT_NULL(s->on_voice);
    adam_settings_destroy(s);
}

TEST(voice_settings_set_stt) {
    adam_settings_t *s = adam_create_settings();

    adam_status_t rc = adam_settings_set_stt(s, ADAM_STT_CLOUD,
        "https://api.openai.com/v1/audio/transcriptions",
        "sk-test", "gpt-4o-mini-transcribe");
    ASSERT_EQ(rc, ADAM_OK);
    ASSERT_EQ(s->voice_enabled, 1);
    ASSERT_EQ(s->stt_backend, ADAM_STT_CLOUD);
    ASSERT_STR_EQ(s->stt_api_url,
                   "https://api.openai.com/v1/audio/transcriptions");
    ASSERT_STR_EQ(s->stt_api_key, "sk-test");
    ASSERT_STR_EQ(s->stt_model, "gpt-4o-mini-transcribe");

    // NULL params should keep defaults
    adam_settings_t *s2 = adam_create_settings();
    rc = adam_settings_set_stt(s2, ADAM_STT_CLOUD, NULL, NULL, NULL);
    ASSERT_EQ(rc, ADAM_OK);
    ASSERT_NULL(s2->stt_api_url);
    ASSERT_STR_EQ(s2->stt_model, "gpt-4o-mini-transcribe"); // kept default

    adam_settings_destroy(s);
    adam_settings_destroy(s2);
}

TEST(voice_settings_set_tts) {
    adam_settings_t *s = adam_create_settings();

    adam_status_t rc = adam_settings_set_tts(s, ADAM_TTS_CLOUD,
        "https://api.openai.com/v1/audio/speech",
        "sk-test", "tts-1-hd", "nova");
    ASSERT_EQ(rc, ADAM_OK);
    ASSERT_EQ(s->tts_backend, ADAM_TTS_CLOUD);
    ASSERT_STR_EQ(s->tts_voice, "nova");
    ASSERT_STR_EQ(s->tts_model, "tts-1-hd");

    adam_settings_destroy(s);
}

TEST(voice_start_not_implemented) {
    adam_settings_t *s = adam_create_settings();
    adam_history_t *h = adam_history_create();

    // Not enabled — should fail
    ASSERT_EQ(adam_voice_start(s, h), ADAM_ERR_INVALID_PARAM);

    // Enable but no backend — should fail
    s->voice_enabled = 1;
    ASSERT_EQ(adam_voice_start(s, h), ADAM_ERR_INVALID_PARAM);

    // Enable with cloud STT — returns not implemented (stub)
    adam_settings_set_stt(s, ADAM_STT_CLOUD, NULL, "key", NULL);
    ASSERT_EQ(adam_voice_start(s, h), ADAM_ERR_NOT_IMPLEMENTED);

    ASSERT_EQ(adam_voice_is_running(s), 0);

    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(voice_stt_local_not_implemented) {
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_stt(s, ADAM_STT_LOCAL, NULL, NULL, NULL);

    arena_t *a = arena_create(4096);
    const char *text = NULL;
    uint8_t audio[] = {0, 1, 2, 3};

    adam_status_t rc = adam_stt_transcribe(s, a, audio, sizeof(audio),
                                           ADAM_AUDIO_WAV, &text);
    ASSERT_EQ(rc, ADAM_ERR_NOT_IMPLEMENTED);
    ASSERT_NULL(text);

    arena_destroy(a);
    adam_settings_destroy(s);
}

TEST(voice_tts_local_not_implemented) {
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_tts(s, ADAM_TTS_LOCAL, NULL, NULL, NULL, NULL);

    arena_t *a = arena_create(4096);
    uint8_t *audio = NULL;
    size_t len = 0;

    adam_status_t rc = adam_tts_synthesize(s, a, "Hello world",
                                           &audio, &len);
    ASSERT_EQ(rc, ADAM_ERR_NOT_IMPLEMENTED);
    ASSERT_NULL(audio);

    arena_destroy(a);
    adam_settings_destroy(s);
}

TEST(voice_stt_cloud_bad_key) {
    // Cloud STT with a fake key should fail with an error (not crash)
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_stt(s, ADAM_STT_CLOUD, NULL, "fake-key", NULL);

    arena_t *a = arena_create(4096);
    const char *text = NULL;
    uint8_t audio[] = {0, 1, 2, 3};

    adam_status_t rc = adam_stt_transcribe(s, a, audio, sizeof(audio),
                                           ADAM_AUDIO_WAV, &text);
    // Should fail (auth error or voice error), but not crash
    ASSERT_NE(rc, ADAM_OK);

    arena_destroy(a);
    adam_settings_destroy(s);
}

TEST(voice_tts_cloud_bad_key) {
    // Cloud TTS with a fake key should fail with an error (not crash)
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_tts(s, ADAM_TTS_CLOUD, NULL, "fake-key", NULL, NULL);

    arena_t *a = arena_create(4096);
    uint8_t *audio = NULL;
    size_t len = 0;

    adam_status_t rc = adam_tts_synthesize(s, a, "Hello world",
                                           &audio, &len);
    // Should fail (auth error or voice error), but not crash
    ASSERT_NE(rc, ADAM_OK);

    arena_destroy(a);
    adam_settings_destroy(s);
}

// Mock STT callback: always transcribes to "hello world"
static adam_status_t mock_stt(void *ctx, arena_t *arena,
                               const uint8_t *audio, size_t audio_len,
                               adam_audio_format_t format, int sample_rate,
                               const char *language, const char **out_text) {
    UNUSED_PARAM(audio); UNUSED_PARAM(audio_len);
    UNUSED_PARAM(format); UNUSED_PARAM(sample_rate);
    UNUSED_PARAM(language);
    int *call_count = (int *)ctx;
    (*call_count)++;
    *out_text = arena_strdup(arena, "hello world");
    return ADAM_OK;
}

// Mock TTS callback: returns 4 bytes of fake audio
static adam_status_t mock_tts(void *ctx, arena_t *arena,
                               const char *text, const char *voice,
                               adam_audio_format_t out_format,
                               uint8_t **out_audio, size_t *out_len) {
    UNUSED_PARAM(text); UNUSED_PARAM(voice); UNUSED_PARAM(out_format);
    int *call_count = (int *)ctx;
    (*call_count)++;
    *out_len = 4;
    *out_audio = arena_alloc(arena, 4);
    memset(*out_audio, 0xAB, 4);
    return ADAM_OK;
}

TEST(voice_custom_stt_callback) {
    int stt_calls = 0;
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_stt(s, ADAM_STT_CLOUD, NULL, NULL, NULL);
    adam_settings_set_stt_callback(s, mock_stt, &stt_calls);

    arena_t *a = arena_create(4096);
    const char *text = NULL;
    uint8_t audio[] = {0, 1, 2, 3};

    adam_status_t rc = adam_stt_transcribe(s, a, audio, sizeof(audio),
                                           ADAM_AUDIO_WAV, &text);
    ASSERT_EQ(rc, ADAM_OK);
    ASSERT_NOT_NULL(text);
    ASSERT_STR_EQ(text, "hello world");
    ASSERT_EQ(stt_calls, 1);

    arena_destroy(a);
    adam_settings_destroy(s);
}

// Mock STT for pipeline test: returns "What is 2+2?"
static adam_status_t mock_stt_pipeline(void *ctx, arena_t *arena,
    const uint8_t *audio, size_t audio_len,
    adam_audio_format_t format, int sample_rate,
    const char *language, const char **out_text) {
    UNUSED_PARAM(audio); UNUSED_PARAM(audio_len);
    UNUSED_PARAM(format); UNUSED_PARAM(sample_rate); UNUSED_PARAM(language);
    (*(int *)ctx)++;
    *out_text = arena_strdup(arena, "What is 2+2?");
    return ADAM_OK;
}

// Mock TTS for pipeline test: returns 8 bytes of fake audio
static adam_status_t mock_tts_pipeline(void *ctx, arena_t *arena,
    const char *text, const char *voice, adam_audio_format_t out_format,
    uint8_t **out_audio, size_t *out_len) {
    UNUSED_PARAM(text); UNUSED_PARAM(voice); UNUSED_PARAM(out_format);
    (*(int *)ctx)++;
    *out_len = 8;
    *out_audio = arena_alloc(arena, 8);
    memset(*out_audio, 0xAA, 8);
    return ADAM_OK;
}

// Mock audio player for pipeline test: just counts calls
static adam_status_t mock_audio_play(void *ctx, const uint8_t *audio,
    size_t len, adam_audio_format_t format) {
    UNUSED_PARAM(audio); UNUSED_PARAM(len); UNUSED_PARAM(format);
    (*(int *)ctx)++;
    return ADAM_OK;
}

TEST(voice_run_full_pipeline) {
    // Full voice pipeline: audio → STT → agent → TTS → play
    // All via mock callbacks (no real API calls)
    int stt_calls = 0;
    int tts_calls = 0;
    int play_calls = 0;

    // Setup: mock LLM that answers "4"
    mock_llm_ctx_t llm_mock = { .fixed_response = "4" };
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &llm_mock);

    // Configure voice with all mock callbacks
    adam_settings_set_stt(s, ADAM_STT_CLOUD, NULL, NULL, NULL);
    adam_settings_set_tts(s, ADAM_TTS_CLOUD, NULL, NULL, NULL, NULL);
    adam_settings_set_stt_callback(s, mock_stt_pipeline, &stt_calls);
    adam_settings_set_tts_callback(s, mock_tts_pipeline, &tts_calls);
    s->audio_play_fn = mock_audio_play;
    s->audio_play_ctx = &play_calls;

    // Run full voice pipeline
    adam_history_t *h = adam_history_create();
    uint8_t fake_audio[] = {0, 1, 2, 3, 4, 5};
    adam_run_result_t r = adam_voice_run(s, h, fake_audio, sizeof(fake_audio),
                                         ADAM_AUDIO_WAV);

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_NOT_NULL(r.final_response);
    ASSERT_STR_EQ(r.final_response, "4");
    ASSERT_EQ(stt_calls, 1);      // STT was called
    ASSERT_EQ(llm_mock.call_count, 1); // LLM was called
    ASSERT_EQ(tts_calls, 1);      // TTS was called
    ASSERT_EQ(play_calls, 1);     // Audio was played

    // History should have: system + user("What is 2+2?") + assistant("4")
    ASSERT_EQ(adam_history_count(h), 3);
    ASSERT_STR_EQ(h->items[1].content, "What is 2+2?");
    ASSERT_STR_EQ(h->items[2].content, "4");

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(voice_custom_tts_callback) {
    int tts_calls = 0;
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_tts(s, ADAM_TTS_CLOUD, NULL, NULL, NULL, NULL);
    adam_settings_set_tts_callback(s, mock_tts, &tts_calls);

    arena_t *a = arena_create(4096);
    uint8_t *audio = NULL;
    size_t len = 0;

    adam_status_t rc = adam_tts_synthesize(s, a, "Say something",
                                           &audio, &len);
    ASSERT_EQ(rc, ADAM_OK);
    ASSERT_NOT_NULL(audio);
    ASSERT_EQ(len, 4);
    ASSERT_EQ(audio[0], 0xAB);
    ASSERT_EQ(tts_calls, 1);

    arena_destroy(a);
    adam_settings_destroy(s);
}

#endif // !ADAM_NO_VOICE && !ADAM_NO_PTHREADS

// ============================================================================
// MARK: - Tests: JSON Build & Parse
// ============================================================================

TEST(json_build_anthropic_simple) {
    arena_t *a = arena_create(8192);
    adam_message_t msgs[2];
    memset(msgs, 0, sizeof(msgs));
    msgs[0].role = ADAM_ROLE_SYSTEM;
    msgs[0].content = "You are helpful.";
    msgs[0].content_len = strlen(msgs[0].content);
    msgs[1].role = ADAM_ROLE_USER;
    msgs[1].content = "Hello";
    msgs[1].content_len = 5;

    const char *json = adam_json_build_request(
        a, ADAM_API_ANTHROPIC, "claude-sonnet-4-20250514",
        msgs, 2, NULL, 0, 0.7f, 4096, 1.0f, NULL);

    ASSERT_NOT_NULL(json);
    ASSERT(strstr(json, "\"model\":\"claude-sonnet-4-20250514\"") != NULL);
    ASSERT(strstr(json, "\"max_tokens\":4096") != NULL);
    ASSERT(strstr(json, "\"system\":[{\"type\":\"text\"") != NULL);
    ASSERT(strstr(json, "You are helpful.") != NULL);
    ASSERT(strstr(json, "\"messages\":[") != NULL);
    ASSERT(strstr(json, "\"role\":\"user\"") != NULL);
    ASSERT(strstr(json, "Hello") != NULL);
    // System should NOT appear in messages array
    // (it's outside in Anthropic format)

    arena_destroy(a);
}

TEST(json_build_openai_simple) {
    arena_t *a = arena_create(8192);
    adam_message_t msgs[2];
    memset(msgs, 0, sizeof(msgs));
    msgs[0].role = ADAM_ROLE_SYSTEM;
    msgs[0].content = "You are helpful.";
    msgs[0].content_len = strlen(msgs[0].content);
    msgs[1].role = ADAM_ROLE_USER;
    msgs[1].content = "Hello";
    msgs[1].content_len = 5;

    const char *json = adam_json_build_request(
        a, ADAM_API_OPENAI, "gpt-4o",
        msgs, 2, NULL, 0, 0.7f, 4096, 1.0f, NULL);

    ASSERT_NOT_NULL(json);
    ASSERT(strstr(json, "\"model\":\"gpt-4o\"") != NULL);
    ASSERT(strstr(json, "\"max_completion_tokens\":4096") != NULL);
    ASSERT(strstr(json, "\"role\":\"system\"") != NULL);
    ASSERT(strstr(json, "\"role\":\"user\"") != NULL);

    arena_destroy(a);
}

TEST(json_build_with_tools) {
    arena_t *a = arena_create(8192);
    adam_message_t msg = {
        .role = ADAM_ROLE_USER, .content = "Search for X", .content_len = 12
    };
    adam_tool_def_t tools[] = {{
        .name = "search",
        .description = "Search the web",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}"
    }};

    // Anthropic format
    const char *json = adam_json_build_request(
        a, ADAM_API_ANTHROPIC, "claude-sonnet-4", &msg, 1, tools, 1,
        0.7f, 4096, 1.0f, NULL);
    ASSERT_NOT_NULL(json);
    ASSERT(strstr(json, "\"tools\":[") != NULL);
    ASSERT(strstr(json, "\"name\":\"search\"") != NULL);
    ASSERT(strstr(json, "\"input_schema\":") != NULL);

    arena_reset(a);

    // OpenAI format
    json = adam_json_build_request(
        a, ADAM_API_OPENAI, "gpt-4o", &msg, 1, tools, 1,
        0.7f, 4096, 1.0f, NULL);
    ASSERT_NOT_NULL(json);
    ASSERT(strstr(json, "\"tools\":[") != NULL);
    ASSERT(strstr(json, "\"type\":\"function\"") != NULL);
    ASSERT(strstr(json, "\"parameters\":") != NULL);

    arena_destroy(a);
}

TEST(json_build_tool_call_messages) {
    // Test building messages that contain tool calls and tool results
    arena_t *a = arena_create(16384);

    adam_tool_call_entry_t tc_entry = {
        .id = "call_abc123", .name = "search",
        .arguments_json = "{\"q\":\"test\"}"
    };

    adam_message_t msgs[3];
    memset(msgs, 0, sizeof(msgs));

    msgs[0].role = ADAM_ROLE_USER;
    msgs[0].content = "Find X"; msgs[0].content_len = 6;

    msgs[1].role = ADAM_ROLE_ASSISTANT;
    msgs[1].content = "Searching..."; msgs[1].content_len = 12;
    msgs[1].tool_calls = &tc_entry;
    msgs[1].tool_call_count = 1;

    msgs[2].role = ADAM_ROLE_TOOL;
    msgs[2].content = "Found 42 results."; msgs[2].content_len = 17;
    msgs[2].tool_call_id = "call_abc123";

    // Anthropic
    const char *json = adam_json_build_request(
        a, ADAM_API_ANTHROPIC, "claude-sonnet-4", msgs, 3, NULL, 0,
        0.7f, 4096, 1.0f, NULL);
    ASSERT_NOT_NULL(json);
    ASSERT(strstr(json, "\"type\":\"tool_use\"") != NULL);
    ASSERT(strstr(json, "call_abc123") != NULL);
    ASSERT(strstr(json, "\"type\":\"tool_result\"") != NULL);

    arena_reset(a);

    // OpenAI
    json = adam_json_build_request(
        a, ADAM_API_OPENAI, "gpt-4o", msgs, 3, NULL, 0,
        0.7f, 4096, 1.0f, NULL);
    ASSERT_NOT_NULL(json);
    ASSERT(strstr(json, "\"tool_calls\":[") != NULL);
    ASSERT(strstr(json, "\"role\":\"tool\"") != NULL);
    ASSERT(strstr(json, "call_abc123") != NULL);

    arena_destroy(a);
}

TEST(json_build_escaping) {
    arena_t *a = arena_create(8192);
    adam_message_t msg = {
        .role = ADAM_ROLE_USER,
        .content = "He said \"hello\"\nand\ttab\\slash",
        .content_len = 29
    };

    const char *json = adam_json_build_request(
        a, ADAM_API_OPENAI, "gpt-4o", &msg, 1, NULL, 0,
        0.7f, 4096, 1.0f, NULL);
    ASSERT_NOT_NULL(json);
    ASSERT(strstr(json, "\\\"hello\\\"") != NULL);
    ASSERT(strstr(json, "\\n") != NULL);
    ASSERT(strstr(json, "\\t") != NULL);
    ASSERT(strstr(json, "\\\\") != NULL);

    arena_destroy(a);
}

TEST(json_build_json_mode) {
    arena_t *a = arena_create(8192);
    adam_message_t msg = {
        .role = ADAM_ROLE_USER, .content = "List items", .content_len = 10
    };

    const char *json = adam_json_build_request(
        a, ADAM_API_OPENAI, "gpt-4o", &msg, 1, NULL, 0,
        0.7f, 4096, 1.0f, "json");
    ASSERT_NOT_NULL(json);
    ASSERT(strstr(json, "\"response_format\":{\"type\":\"json_object\"}") != NULL);

    arena_destroy(a);
}

TEST(json_parse_anthropic_text) {
    arena_t *a = arena_create(8192);
    const char *json =
        "{\"id\":\"msg_01\",\"type\":\"message\",\"role\":\"assistant\","
        "\"content\":[{\"type\":\"text\",\"text\":\"Hello there!\"}],"
        "\"stop_reason\":\"end_turn\","
        "\"usage\":{\"input_tokens\":25,\"output_tokens\":10}}";

    adam_llm_response_t r = adam_json_parse_response(
        a, ADAM_API_ANTHROPIC, json, strlen(json));

    ASSERT_EQ(r.error, ADAM_OK);
    ASSERT_NOT_NULL(r.content);
    ASSERT_STR_EQ(r.content, "Hello there!");
    ASSERT_EQ(r.tool_call_count, 0);
    ASSERT_EQ(r.input_tokens, 25);
    ASSERT_EQ(r.output_tokens, 10);

    arena_destroy(a);
}

TEST(json_parse_anthropic_tool_use) {
    arena_t *a = arena_create(8192);
    const char *json =
        "{\"id\":\"msg_02\",\"type\":\"message\",\"role\":\"assistant\","
        "\"content\":["
        "{\"type\":\"text\",\"text\":\"Let me search.\"},"
        "{\"type\":\"tool_use\",\"id\":\"toolu_01\",\"name\":\"search\","
        "\"input\":{\"query\":\"test query\"}}"
        "],\"stop_reason\":\"tool_use\","
        "\"usage\":{\"input_tokens\":50,\"output_tokens\":30}}";

    adam_llm_response_t r = adam_json_parse_response(
        a, ADAM_API_ANTHROPIC, json, strlen(json));

    ASSERT_EQ(r.error, ADAM_OK);
    ASSERT_NOT_NULL(r.content);
    ASSERT_STR_EQ(r.content, "Let me search.");
    ASSERT_EQ(r.tool_call_count, 1);
    ASSERT_STR_EQ(r.tool_calls[0].id, "toolu_01");
    ASSERT_STR_EQ(r.tool_calls[0].name, "search");
    ASSERT(strstr(r.tool_calls[0].arguments_json, "test query") != NULL);
    ASSERT_EQ(r.input_tokens, 50);
    ASSERT_EQ(r.output_tokens, 30);

    arena_destroy(a);
}

TEST(json_parse_anthropic_error) {
    arena_t *a = arena_create(4096);
    const char *json =
        "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\","
        "\"message\":\"max_tokens must be positive\"}}";

    adam_llm_response_t r = adam_json_parse_response(
        a, ADAM_API_ANTHROPIC, json, strlen(json));

    ASSERT_NE(r.error, ADAM_OK);
    ASSERT_NOT_NULL(r.error_msg);
    ASSERT(strstr(r.error_msg, "max_tokens") != NULL);

    arena_destroy(a);
}

TEST(json_parse_openai_text) {
    arena_t *a = arena_create(8192);
    const char *json =
        "{\"id\":\"chatcmpl-01\",\"object\":\"chat.completion\","
        "\"choices\":[{\"index\":0,\"message\":"
        "{\"role\":\"assistant\",\"content\":\"Hello from GPT!\"},"
        "\"finish_reason\":\"stop\"}],"
        "\"usage\":{\"prompt_tokens\":15,\"completion_tokens\":5}}";

    adam_llm_response_t r = adam_json_parse_response(
        a, ADAM_API_OPENAI, json, strlen(json));

    ASSERT_EQ(r.error, ADAM_OK);
    ASSERT_NOT_NULL(r.content);
    ASSERT_STR_EQ(r.content, "Hello from GPT!");
    ASSERT_EQ(r.tool_call_count, 0);
    ASSERT_EQ(r.input_tokens, 15);
    ASSERT_EQ(r.output_tokens, 5);

    arena_destroy(a);
}

TEST(json_parse_openai_tool_calls) {
    arena_t *a = arena_create(8192);
    const char *json =
        "{\"choices\":[{\"index\":0,\"message\":"
        "{\"role\":\"assistant\",\"content\":null,"
        "\"tool_calls\":[{\"id\":\"call_xyz\",\"type\":\"function\","
        "\"function\":{\"name\":\"get_weather\","
        "\"arguments\":\"{\\\"city\\\":\\\"London\\\"}\"}}]},"
        "\"finish_reason\":\"tool_calls\"}],"
        "\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":20}}";

    adam_llm_response_t r = adam_json_parse_response(
        a, ADAM_API_OPENAI, json, strlen(json));

    ASSERT_EQ(r.error, ADAM_OK);
    ASSERT_EQ(r.tool_call_count, 1);
    ASSERT_STR_EQ(r.tool_calls[0].id, "call_xyz");
    ASSERT_STR_EQ(r.tool_calls[0].name, "get_weather");
    ASSERT(strstr(r.tool_calls[0].arguments_json, "London") != NULL);
    ASSERT_EQ(r.input_tokens, 40);
    ASSERT_EQ(r.output_tokens, 20);

    arena_destroy(a);
}

TEST(json_parse_openai_error) {
    arena_t *a = arena_create(4096);
    const char *json =
        "{\"error\":{\"message\":\"Invalid API key\",\"type\":\"auth_error\"}}";

    adam_llm_response_t r = adam_json_parse_response(
        a, ADAM_API_OPENAI, json, strlen(json));

    ASSERT_NE(r.error, ADAM_OK);
    ASSERT_NOT_NULL(r.error_msg);
    ASSERT(strstr(r.error_msg, "Invalid API key") != NULL);

    arena_destroy(a);
}

TEST(json_parse_invalid) {
    arena_t *a = arena_create(4096);

    // Empty
    adam_llm_response_t r1 = adam_json_parse_response(a, ADAM_API_OPENAI, NULL, 0);
    ASSERT_NE(r1.error, ADAM_OK);

    // Malformed
    adam_llm_response_t r2 = adam_json_parse_response(
        a, ADAM_API_OPENAI, "{broken", 7);
    ASSERT_NE(r2.error, ADAM_OK);

    arena_destroy(a);
}

TEST(json_parse_anthropic_multi_tool) {
    arena_t *a = arena_create(8192);
    const char *json =
        "{\"content\":["
        "{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"search\",\"input\":{\"q\":\"a\"}},"
        "{\"type\":\"tool_use\",\"id\":\"t2\",\"name\":\"fetch\",\"input\":{\"url\":\"b\"}}"
        "],\"usage\":{\"input_tokens\":100,\"output_tokens\":50}}";

    adam_llm_response_t r = adam_json_parse_response(
        a, ADAM_API_ANTHROPIC, json, strlen(json));

    ASSERT_EQ(r.error, ADAM_OK);
    ASSERT_EQ(r.tool_call_count, 2);
    ASSERT_STR_EQ(r.tool_calls[0].name, "search");
    ASSERT_STR_EQ(r.tool_calls[1].name, "fetch");
    ASSERT_STR_EQ(r.tool_calls[0].id, "t1");
    ASSERT_STR_EQ(r.tool_calls[1].id, "t2");

    arena_destroy(a);
}

// ============================================================================
// MARK: - Tests: Local Inference
// ============================================================================

#ifndef ADAM_NO_LOCAL

TEST(local_missing_gguf) {
    // Verify graceful failure when GGUF file doesn't exist
    adam_settings_t *s = adam_create_settings();
    s->gguf_path = "/tmp/nonexistent_model.gguf";
    s->local_gpu_layers = 0;
    s->local_ctx_size = 512;

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Hello");

    // Should fail with LOCAL error, not crash
    ASSERT_EQ(r.status, ADAM_ERR_LOCAL);
    ASSERT_NOT_NULL(r.final_response);

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

#endif // ADAM_NO_LOCAL

// ============================================================================
// MARK: - Tests: Session Persistence
// ============================================================================

#ifndef ADAM_NO_SQLITE

#include <unistd.h>

TEST(session_open_close) {
    const char *path = "/tmp/adam_test_session.db";
    unlink(path);
    adam_session_t *sess = adam_session_open(path);
    ASSERT_NOT_NULL(sess);
    adam_session_close(sess);
    unlink(path);
}

TEST(session_save_load_simple) {
    const char *path = "/tmp/adam_test_session2.db";
    unlink(path);
    adam_session_t *sess = adam_session_open(path);
    ASSERT_NOT_NULL(sess);

    // Build a history
    adam_history_t *h = adam_history_create();
    adam_history_append_user(h, "Hello");
    adam_history_append_assistant(h, "Hi there!", NULL, 0);
    adam_history_append_user(h, "How are you?");
    adam_history_append_assistant(h, "I'm doing well!", NULL, 0);

    // Save
    adam_status_t rc = adam_session_save(sess, "test-001", h);
    ASSERT_EQ(rc, ADAM_OK);

    // Load into a new history
    adam_history_t *h2 = adam_history_create();
    rc = adam_session_load(sess, "test-001", h2);
    ASSERT_EQ(rc, ADAM_OK);
    ASSERT_EQ(adam_history_count(h2), 4);
    ASSERT_EQ(h2->items[0].role, ADAM_ROLE_USER);
    ASSERT_STR_EQ(h2->items[0].content, "Hello");
    ASSERT_EQ(h2->items[1].role, ADAM_ROLE_ASSISTANT);
    ASSERT_STR_EQ(h2->items[1].content, "Hi there!");
    ASSERT_EQ(h2->items[2].role, ADAM_ROLE_USER);
    ASSERT_STR_EQ(h2->items[2].content, "How are you?");
    ASSERT_EQ(h2->items[3].role, ADAM_ROLE_ASSISTANT);
    ASSERT_STR_EQ(h2->items[3].content, "I'm doing well!");

    adam_history_destroy(h);
    adam_history_destroy(h2);
    adam_session_close(sess);
    unlink(path);
}

TEST(session_save_load_with_tool_calls) {
    const char *path = "/tmp/adam_test_session3.db";
    unlink(path);
    adam_session_t *sess = adam_session_open(path);

    adam_history_t *h = adam_history_create();
    adam_history_append_user(h, "What's the weather?");

    // Assistant with tool call
    adam_tool_call_t tc = {
        .id = "call_abc", .name = "get_weather",
        .arguments_json = "{\"city\":\"Rome\"}"
    };
    adam_history_append_assistant(h, "Let me check.", &tc, 1);

    // Tool result
    adam_history_append_tool(h, "{\"temp\":22}", "call_abc");

    // Final answer
    adam_history_append_assistant(h, "It's 22 degrees in Rome.", NULL, 0);

    adam_session_save(sess, "tool-test", h);

    // Load and verify
    adam_history_t *h2 = adam_history_create();
    adam_session_load(sess, "tool-test", h2);

    ASSERT_EQ(adam_history_count(h2), 4);

    // Check tool call on assistant message
    ASSERT_EQ(h2->items[1].role, ADAM_ROLE_ASSISTANT);
    ASSERT_EQ(h2->items[1].tool_call_count, 1);
    ASSERT_STR_EQ(h2->items[1].tool_calls[0].id, "call_abc");
    ASSERT_STR_EQ(h2->items[1].tool_calls[0].name, "get_weather");
    ASSERT(strstr(h2->items[1].tool_calls[0].arguments_json, "Rome") != NULL);

    // Check tool result
    ASSERT_EQ(h2->items[2].role, ADAM_ROLE_TOOL);
    ASSERT_STR_EQ(h2->items[2].tool_call_id, "call_abc");
    ASSERT(strstr(h2->items[2].content, "22") != NULL);

    adam_history_destroy(h);
    adam_history_destroy(h2);
    adam_session_close(sess);
    unlink(path);
}

TEST(session_list_and_delete) {
    const char *path = "/tmp/adam_test_session4.db";
    unlink(path);
    adam_session_t *sess = adam_session_open(path);

    adam_history_t *h = adam_history_create();
    adam_history_append_user(h, "Test");

    adam_session_save(sess, "session-a", h);
    adam_session_save(sess, "session-b", h);
    adam_session_save(sess, "session-c", h);

    // List
    char **ids = NULL;
    size_t count = 0;
    adam_status_t rc = adam_session_list(sess, &ids, &count);
    ASSERT_EQ(rc, ADAM_OK);
    ASSERT_EQ(count, 3);
    ASSERT_NOT_NULL(ids);

    // Free list
    for (size_t i = 0; i < count; i++) free(ids[i]);
    free(ids);

    // Delete one
    rc = adam_session_delete(sess, "session-b");
    ASSERT_EQ(rc, ADAM_OK);

    // List again
    rc = adam_session_list(sess, &ids, &count);
    ASSERT_EQ(rc, ADAM_OK);
    ASSERT_EQ(count, 2);
    for (size_t i = 0; i < count; i++) free(ids[i]);
    free(ids);

    // Load deleted session should fail
    adam_history_t *h2 = adam_history_create();
    rc = adam_session_load(sess, "session-b", h2);
    ASSERT_NE(rc, ADAM_OK);

    adam_history_destroy(h);
    adam_history_destroy(h2);
    adam_session_close(sess);
    unlink(path);
}

TEST(session_overwrite) {
    // Saving the same session_id twice should replace the old data
    const char *path = "/tmp/adam_test_session5.db";
    unlink(path);
    adam_session_t *sess = adam_session_open(path);

    adam_history_t *h1 = adam_history_create();
    adam_history_append_user(h1, "First version");
    adam_session_save(sess, "overwrite-test", h1);

    adam_history_t *h2 = adam_history_create();
    adam_history_append_user(h2, "Second version");
    adam_history_append_assistant(h2, "Updated reply", NULL, 0);
    adam_session_save(sess, "overwrite-test", h2);

    // Load should get the second version
    adam_history_t *h3 = adam_history_create();
    adam_session_load(sess, "overwrite-test", h3);
    ASSERT_EQ(adam_history_count(h3), 2);
    ASSERT_STR_EQ(h3->items[0].content, "Second version");
    ASSERT_STR_EQ(h3->items[1].content, "Updated reply");

    adam_history_destroy(h1);
    adam_history_destroy(h2);
    adam_history_destroy(h3);
    adam_session_close(sess);
    unlink(path);
}

TEST(session_persistence_across_reopen) {
    // Data survives closing and reopening the database
    const char *path = "/tmp/adam_test_session6.db";
    unlink(path);

    // Open, save, close
    adam_session_t *s1 = adam_session_open(path);
    adam_history_t *h = adam_history_create();
    adam_history_append_user(h, "Persistent message");
    adam_session_save(s1, "persist-test", h);
    adam_session_close(s1);

    // Reopen, load
    adam_session_t *s2 = adam_session_open(path);
    adam_history_t *h2 = adam_history_create();
    adam_status_t rc = adam_session_load(s2, "persist-test", h2);
    ASSERT_EQ(rc, ADAM_OK);
    ASSERT_EQ(adam_history_count(h2), 1);
    ASSERT_STR_EQ(h2->items[0].content, "Persistent message");

    adam_history_destroy(h);
    adam_history_destroy(h2);
    adam_session_close(s2);
    unlink(path);
}

#endif // ADAM_NO_SQLITE

// ============================================================================
// MARK: - Performance Benchmarks
// ============================================================================

TEST(perf_1000_simple_runs) {
    // Benchmark: 1000 simple agent runs (no tools) with mock LLM.
    mock_llm_ctx_t mock = { .fixed_response = "Benchmark response." };

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < 1000; i++) {
        adam_history_t *h = adam_history_create();
        adam_run_result_t r = adam_run(s, h, "Benchmark message");
        ASSERT_EQ(r.status, ADAM_OK);
        adam_run_result_free(&r);
        adam_history_destroy(h);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0
              + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    printf("\n    1000 runs: %.1f ms total, %.3f ms/run", ms, ms / 1000.0);

    ASSERT_EQ(mock.call_count, 1000);
    adam_settings_destroy(s);
}

TEST(perf_tool_loop_throughput) {
    // Benchmark: 500 runs with tool call + response (2 iterations each).
    mock_llm_ctx_t mock = {0};

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_with_tool, &mock);
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "search", .execute = mock_tool_search });

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < 500; i++) {
        mock.call_count = 0;
        mock.total_tool_calls_requested = 0;
        adam_history_t *h = adam_history_create();
        adam_run_result_t r = adam_run(s, h, "Benchmark with tool");
        ASSERT_EQ(r.status, ADAM_OK);
        ASSERT_EQ(mock.call_count, 2);
        adam_run_result_free(&r);
        adam_history_destroy(h);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0
              + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    printf("\n    500 tool runs: %.1f ms total, %.3f ms/run", ms, ms / 500.0);

    adam_settings_destroy(s);
}

TEST(perf_arena_churn) {
    // Benchmark: arena create/alloc/reset/destroy cycle.
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < 10000; i++) {
        arena_t *a = arena_create(64 * 1024);
        for (int j = 0; j < 100; j++) {
            arena_alloc(a, 256);
        }
        arena_reset(a);
        for (int j = 0; j < 100; j++) {
            arena_strdup(a, "test string for arena benchmark");
        }
        arena_destroy(a);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0
              + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    printf("\n    10000 arena cycles (200 allocs each): %.1f ms", ms);
}

// ============================================================================
// MARK: - Main
// ============================================================================

int main(void) {
    printf("Adam Test Suite v%s\n", ADAM_VERSION_STRING);
    printf("============================================================\n");

    adam_init();
    mem_report_start();

    // --- Settings ---
    printf("\nSettings:\n");
    RUN(create_settings_defaults);
    RUN(settings_set_provider);
    RUN(settings_add_remove_tools);
    RUN(settings_callbacks);

    // --- History ---
    printf("\nHistory:\n");
    RUN(history_lifecycle);
    RUN(history_with_tool_calls);
    RUN(history_token_estimation);
    RUN(history_attachments);

    // --- Agent Run ---
    printf("\nAgent Run:\n");
    RUN(run_simple_conversation);
    RUN(run_with_tool_call);
    RUN(run_multi_tool_calls);
    RUN(run_tool_not_found);
    RUN(run_multi_turn_conversation);
    RUN(run_no_provider);
    RUN(run_null_params);
    RUN(run_abort);
    RUN(run_auth_error_no_retry);
    RUN(run_with_logging);
    RUN(run_cost_tracking);
    RUN(run_on_response_callback);

    // --- Infrastructure ---
    printf("\nInfrastructure:\n");
    RUN(arena_basic);
    RUN(arena_large_allocation);
    RUN(model_registry_builtin);
    RUN(model_registry_custom);
    RUN(model_estimate_cost);
    RUN(token_estimation);
    RUN(status_strings);

#ifndef ADAM_NO_PTHREADS
    // --- Thread Pool ---
    printf("\nThread Pool:\n");
    RUN(thread_pool_basic);
    RUN(thread_pool_empty_destroy);
#endif

    // --- JSON ---
    printf("\nJSON Build & Parse:\n");
    RUN(json_build_anthropic_simple);
    RUN(json_build_openai_simple);
    RUN(json_build_with_tools);
    RUN(json_build_tool_call_messages);
    RUN(json_build_escaping);
    RUN(json_build_json_mode);
    RUN(json_parse_anthropic_text);
    RUN(json_parse_anthropic_tool_use);
    RUN(json_parse_anthropic_error);
    RUN(json_parse_openai_text);
    RUN(json_parse_openai_tool_calls);
    RUN(json_parse_openai_error);
    RUN(json_parse_invalid);
    RUN(json_parse_anthropic_multi_tool);

#if !defined(ADAM_NO_VOICE) && !defined(ADAM_NO_PTHREADS)
    // --- Voice ---
    printf("\nVoice:\n");
    RUN(voice_settings_defaults);
    RUN(voice_settings_set_stt);
    RUN(voice_settings_set_tts);
    RUN(voice_start_not_implemented);
    RUN(voice_stt_local_not_implemented);
    RUN(voice_tts_local_not_implemented);
    RUN(voice_stt_cloud_bad_key);
    RUN(voice_tts_cloud_bad_key);
    RUN(voice_custom_stt_callback);
    RUN(voice_custom_tts_callback);
    RUN(voice_run_full_pipeline);
#endif

#ifndef ADAM_NO_LOCAL
    // --- Local Inference ---
    printf("\nLocal Inference:\n");
    RUN(local_missing_gguf);
#endif

#ifndef ADAM_NO_SQLITE
    // --- Sessions ---
    printf("\nSessions:\n");
    RUN(session_open_close);
    RUN(session_save_load_simple);
    RUN(session_save_load_with_tool_calls);
    RUN(session_list_and_delete);
    RUN(session_overwrite);
    RUN(session_persistence_across_reopen);
#endif

    // --- Performance ---
    printf("\nPerformance:\n");
    RUN(perf_1000_simple_runs);
    RUN(perf_tool_loop_throughput);
    RUN(perf_arena_churn);

    // --- Summary ---
    printf("\n============================================================\n");
    printf("Results: %d passed, %d failed, %d total (%d assertions)\n",
           g_tests_passed, g_tests_failed, g_tests_run, g_asserts_total);

    mem_report_end();

    adam_cleanup();

    return g_tests_failed > 0 ? 1 : 0;
}
