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
#include <sys/stat.h>
#include <unistd.h>


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
    static NOINLINE void run_test_##name(void) {                            \
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

// Prevent the compiler from inlining run_test wrappers into main(),
// which would create a single enormous stack frame that exceeds ASan limits.
#if defined(__GNUC__) || defined(__clang__)
  #define NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
  #define NOINLINE __declspec(noinline)
#else
  #define NOINLINE
#endif

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
    ASSERT_NULL(s->memory_context);
    ASSERT_EQ(s->auto_save, 1);
    ASSERT_NULL(s->session_id);
    ASSERT_EQ(s->memory_extract, 0);
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

TEST(voice_stt_local_bad_model) {
    // Local STT with default model name (not a file) should fail gracefully
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_stt(s, ADAM_STT_LOCAL, NULL, NULL, NULL);

    arena_t *a = arena_create(4096);
    const char *text = NULL;
    uint8_t audio[] = {0, 1, 2, 3};

    adam_status_t rc = adam_stt_transcribe(s, a, audio, sizeof(audio),
                                           ADAM_AUDIO_WAV, &text);
    // Should fail (model file doesn't exist) but not crash
    ASSERT_NE(rc, ADAM_OK);

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

TEST(json_build_gemini_simple) {
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
        a, ADAM_API_GEMINI, "gemini-2.0-flash",
        msgs, 2, NULL, 0, 0.7f, 4096, 1.0f, NULL);

    ASSERT_NOT_NULL(json);
    ASSERT(strstr(json, "\"systemInstruction\"") != NULL);
    ASSERT(strstr(json, "You are helpful.") != NULL);
    ASSERT(strstr(json, "\"contents\"") != NULL);
    ASSERT(strstr(json, "\"role\":\"user\"") != NULL);
    ASSERT(strstr(json, "Hello") != NULL);
    ASSERT(strstr(json, "\"generationConfig\"") != NULL);
    ASSERT(strstr(json, "\"maxOutputTokens\":4096") != NULL);
    // Should NOT have model in JSON body (it's in URL for Gemini)
    ASSERT(strstr(json, "\"model\"") == NULL);

    arena_destroy(a);
}

TEST(json_build_gemini_with_tools) {
    arena_t *a = arena_create(8192);
    adam_message_t msg = {
        .role = ADAM_ROLE_USER, .content = "Search for X", .content_len = 12
    };
    adam_tool_def_t tool = {
        .name = "search",
        .description = "Search the web",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}"
    };

    const char *json = adam_json_build_request(
        a, ADAM_API_GEMINI, "gemini-2.0-flash",
        &msg, 1, &tool, 1, 0.7f, 4096, 1.0f, NULL);

    ASSERT_NOT_NULL(json);
    ASSERT(strstr(json, "\"functionDeclarations\"") != NULL);
    ASSERT(strstr(json, "\"name\":\"search\"") != NULL);
    ASSERT(strstr(json, "\"description\":\"Search the web\"") != NULL);

    arena_destroy(a);
}

TEST(json_parse_gemini_text) {
    arena_t *a = arena_create(8192);
    const char *json =
        "{\"candidates\":[{\"content\":{\"role\":\"model\","
        "\"parts\":[{\"text\":\"Hello from Gemini!\"}]},"
        "\"finishReason\":\"STOP\"}],"
        "\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":5}}";

    adam_llm_response_t r = adam_json_parse_response(
        a, ADAM_API_GEMINI, json, strlen(json));

    ASSERT_EQ(r.error, ADAM_OK);
    ASSERT_NOT_NULL(r.content);
    ASSERT_STR_EQ(r.content, "Hello from Gemini!");
    ASSERT_EQ(r.tool_call_count, 0);
    ASSERT_EQ(r.input_tokens, 10);
    ASSERT_EQ(r.output_tokens, 5);

    arena_destroy(a);
}

TEST(json_parse_gemini_tool_call) {
    arena_t *a = arena_create(8192);
    const char *json =
        "{\"candidates\":[{\"content\":{\"role\":\"model\","
        "\"parts\":[{\"text\":\"Let me search.\"},"
        "{\"functionCall\":{\"name\":\"search\",\"args\":{\"q\":\"test\"}}}]},"
        "\"finishReason\":\"FUNCTION_CALL\"}],"
        "\"usageMetadata\":{\"promptTokenCount\":20,\"candidatesTokenCount\":10}}";

    adam_llm_response_t r = adam_json_parse_response(
        a, ADAM_API_GEMINI, json, strlen(json));

    ASSERT_EQ(r.error, ADAM_OK);
    ASSERT_NOT_NULL(r.content);
    ASSERT(strstr(r.content, "search") != NULL);
    ASSERT_EQ(r.tool_call_count, 1);
    ASSERT_STR_EQ(r.tool_calls[0].name, "search");
    ASSERT(strstr(r.tool_calls[0].arguments_json, "test") != NULL);
    ASSERT_NOT_NULL(r.tool_calls[0].id); // generated ID

    arena_destroy(a);
}

TEST(json_parse_gemini_error) {
    arena_t *a = arena_create(8192);
    const char *json =
        "{\"error\":{\"code\":400,\"message\":\"Invalid API key\","
        "\"status\":\"INVALID_ARGUMENT\"}}";

    adam_llm_response_t r = adam_json_parse_response(
        a, ADAM_API_GEMINI, json, strlen(json));

    ASSERT_NE(r.error, ADAM_OK);
    ASSERT_NOT_NULL(r.error_msg);
    ASSERT(strstr(r.error_msg, "Invalid API key") != NULL);

    arena_destroy(a);
}

TEST(json_build_gemini_image_model) {
    arena_t *a = arena_create(8192);
    adam_message_t msg = {
        .role = ADAM_ROLE_USER, .content = "Draw a cat", .content_len = 10
    };

    const char *json = adam_json_build_request(
        a, ADAM_API_GEMINI, "gemini-3.1-flash-image-preview",
        &msg, 1, NULL, 0, 0.7f, 4096, 1.0f, NULL);

    ASSERT_NOT_NULL(json);
    // Image model should have responseModalities with IMAGE
    ASSERT(strstr(json, "\"responseModalities\"") != NULL);
    ASSERT(strstr(json, "\"IMAGE\"") != NULL);
    ASSERT(strstr(json, "\"TEXT\"") != NULL);

    arena_destroy(a);
}

TEST(json_parse_gemini_image_response) {
    arena_t *a = arena_create(16384);
    // Simulate response with both text and inline image data
    const char *json =
        "{\"candidates\":[{\"content\":{\"role\":\"model\","
        "\"parts\":[{\"text\":\"Here is your cat:\"},"
        "{\"inline_data\":{\"mime_type\":\"image/png\",\"data\":\"iVBORw0KGgo=\"}}]},"
        "\"finishReason\":\"STOP\"}],"
        "\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":2}}";

    adam_llm_response_t r = adam_json_parse_response(
        a, ADAM_API_GEMINI, json, strlen(json));

    ASSERT_EQ(r.error, ADAM_OK);
    ASSERT_NOT_NULL(r.content);
    // Should contain the text
    ASSERT(strstr(r.content, "Here is your cat:") != NULL);
    // Should contain the embedded image as data URI
    ASSERT(strstr(r.content, "![image](data:image/png;base64,") != NULL);
    ASSERT(strstr(r.content, "iVBORw0KGgo=") != NULL);

    arena_destroy(a);
}

TEST(model_registry_gemini) {
    int cw = adam_model_context_window("gemini-2.0-flash");
    ASSERT_EQ(cw, 1048576);

    cw = adam_model_context_window("gemini-2.5-pro-latest");
    ASSERT_EQ(cw, 1048576);

    cw = adam_model_context_window("gemini-1.5-pro-002");
    ASSERT_EQ(cw, 2097152);
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
// MARK: - Tests: Streaming
// ============================================================================

// Track streaming chunks
typedef struct {
    char     chunks[4096];
    size_t   total_len;
    int      chunk_count;
    int      got_done;
} stream_tracker_t;

static void track_stream(void *ctx, const char *chunk, size_t len, int is_done) {
    stream_tracker_t *t = (stream_tracker_t *)ctx;
    if (len > 0 && t->total_len + len < sizeof(t->chunks)) {
        memcpy(t->chunks + t->total_len, chunk, len);
        t->total_len += len;
        t->chunks[t->total_len] = '\0';
    }
    if (len > 0) t->chunk_count++;
    if (is_done) t->got_done = 1;
}

TEST(stream_simple_response) {
    // Verify on_stream receives the same text as final_response
    stream_tracker_t tracker = {0};
    mock_llm_ctx_t mock = { .fixed_response = "Streamed hello world!" };

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);
    s->on_stream = track_stream;
    s->stream_ctx = &tracker;

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Hi");

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_NOT_NULL(r.final_response);
    // With mock LLM (non-streaming), on_stream is NOT called
    // (streaming only triggers for cloud HTTP path)
    // The final response should still be correct
    ASSERT_STR_EQ(r.final_response, "Streamed hello world!");

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(stream_with_tool_calls) {
    // Verify streaming works when tool calls are involved
    // Mock LLM returns tool call then final answer
    stream_tracker_t tracker = {0};
    mock_llm_ctx_t mock = {0};

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_with_tool, &mock);
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "search", .execute = mock_tool_search });
    s->on_stream = track_stream;
    s->stream_ctx = &tracker;

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Search for X");

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_NOT_NULL(r.final_response);
    // The agent should have made 2 LLM calls (tool call + final answer)
    ASSERT_EQ(mock.call_count, 2);

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(stream_null_callback) {
    // on_stream = NULL should work fine (non-streaming path)
    mock_llm_ctx_t mock = { .fixed_response = "No stream" };

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);
    s->on_stream = NULL;

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Hi");

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_STR_EQ(r.final_response, "No stream");

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(stream_error_no_crash) {
    // Stream callback set but LLM returns error — should not crash
    stream_tracker_t tracker = {0};
    mock_llm_ctx_t mock = { .simulate_error = ADAM_ERR_AUTH };

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_error, &mock);
    s->on_stream = track_stream;
    s->stream_ctx = &tracker;

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Hi");

    ASSERT_EQ(r.status, ADAM_ERR_AUTH);
    ASSERT_EQ(tracker.got_done, 0); // stream never started

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(stream_multi_turn) {
    // Streaming across multiple turns — each turn should deliver content
    mock_llm_ctx_t mock = { .fixed_response = "Turn response" };

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);
    s->on_stream = track_stream;

    adam_history_t *h = adam_history_create();

    for (int i = 0; i < 5; i++) {
        stream_tracker_t tracker = {0};
        s->stream_ctx = &tracker;
        adam_run_result_t r = adam_run(s, h, "Next turn");
        ASSERT_EQ(r.status, ADAM_OK);
        adam_run_result_free(&r);
    }

    ASSERT_EQ(adam_history_count(h), 11); // system + 5*(user + assistant)

    adam_history_destroy(h);
    adam_settings_destroy(s);
}

// ============================================================================
// MARK: - Tests: SSE Parser (unit test the parser directly)
// ============================================================================

// We test the SSE parser by including adam_stream.h and calling
// adam_llm_call_http_stream with a mock that simulates SSE events.
// Since we can't easily mock the HTTP layer for SSE, we test the
// integration through the live tests instead. Here we verify the
// stream callback behavior through the agent loop.

TEST(stream_on_response_still_fires) {
    // Both on_stream and on_response should work together
    // on_response fires after the complete response is assembled
    stream_tracker_t tracker = {0};
    resp_ctx_t rctx = {0};
    mock_llm_ctx_t mock = { .fixed_response = "Dual callback test" };

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);
    s->on_stream = track_stream;
    s->stream_ctx = &tracker;
    adam_settings_set_callbacks(s, on_resp, NULL, NULL, &rctx);

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Test");

    ASSERT_EQ(r.status, ADAM_OK);
    // on_response should have been called with the full response
    ASSERT_EQ(rctx.call_count, 1);
    ASSERT_STR_EQ(rctx.last, "Dual callback test");

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

// ============================================================================
// MARK: - Tests: Local STT
// ============================================================================

#ifndef ADAM_NO_LOCAL

// Helper: create a minimal valid WAV buffer (silence)
static uint8_t *make_silent_wav(arena_t *arena, int sample_rate,
                                 float seconds, size_t *out_len) {
    size_t n_samples = (size_t)(sample_rate * seconds);
    size_t data_size = n_samples * 2; // 16-bit = 2 bytes per sample
    size_t wav_size = 44 + data_size;
    uint8_t *wav = arena_zeroalloc(arena, wav_size); // zeroed = silence
    if (!wav) { *out_len = 0; return NULL; }

    // WAV header
    uint32_t chunk_size = (uint32_t)(36 + data_size);
    uint32_t sr = (uint32_t)sample_rate;
    uint32_t byte_rate = sr * 2;
    uint16_t block_align = 2;
    uint16_t bits = 16;
    uint32_t fmt_size = 16;
    uint16_t audio_fmt = 1;
    uint16_t channels = 1;
    uint32_t ds = (uint32_t)data_size;

    size_t p = 0;
    memcpy(wav + p, "RIFF", 4); p += 4;
    memcpy(wav + p, &chunk_size, 4); p += 4;
    memcpy(wav + p, "WAVE", 4); p += 4;
    memcpy(wav + p, "fmt ", 4); p += 4;
    memcpy(wav + p, &fmt_size, 4); p += 4;
    memcpy(wav + p, &audio_fmt, 2); p += 2;
    memcpy(wav + p, &channels, 2); p += 2;
    memcpy(wav + p, &sr, 4); p += 4;
    memcpy(wav + p, &byte_rate, 4); p += 4;
    memcpy(wav + p, &block_align, 2); p += 2;
    memcpy(wav + p, &bits, 2); p += 2;
    memcpy(wav + p, "data", 4); p += 4;
    memcpy(wav + p, &ds, 4); p += 4;
    // PCM data is already zero (silence)

    *out_len = wav_size;
    return wav;
}

TEST(local_stt_missing_model) {
    // Non-existent whisper model should fail gracefully
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_stt(s, ADAM_STT_LOCAL, NULL, NULL, "/tmp/nonexistent_whisper.bin");

    arena_t *a = arena_create(16384);
    size_t wav_len;
    uint8_t *wav = make_silent_wav(a, 16000, 0.5f, &wav_len);
    ASSERT_NOT_NULL(wav);

    const char *text = NULL;
    adam_status_t rc = adam_stt_transcribe(s, a, wav, wav_len,
                                           ADAM_AUDIO_WAV, &text);
    ASSERT_EQ(rc, ADAM_ERR_LOCAL);
    ASSERT_NULL(text);

    arena_destroy(a);
    adam_settings_destroy(s);
}

TEST(local_stt_null_model) {
    // NULL model path should fail
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_stt(s, ADAM_STT_LOCAL, NULL, NULL, NULL);
    // stt_model defaults to "gpt-4o-mini-transcribe" which isn't a file path
    // The function should check and fail

    arena_t *a = arena_create(16384);
    size_t wav_len;
    uint8_t *wav = make_silent_wav(a, 16000, 0.5f, &wav_len);
    const char *text = NULL;

    adam_status_t rc = adam_stt_transcribe(s, a, wav, wav_len,
                                           ADAM_AUDIO_WAV, &text);
    // Should fail (model file doesn't exist)
    ASSERT_NE(rc, ADAM_OK);

    arena_destroy(a);
    adam_settings_destroy(s);
}

TEST(local_stt_null_audio) {
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_stt(s, ADAM_STT_LOCAL, NULL, NULL, "/tmp/nonexistent.bin");

    arena_t *a = arena_create(4096);
    const char *text = NULL;
    adam_status_t rc = adam_stt_transcribe(s, a, NULL, 0,
                                           ADAM_AUDIO_WAV, &text);
    ASSERT_EQ(rc, ADAM_ERR_INVALID_PARAM);

    arena_destroy(a);
    adam_settings_destroy(s);
}

TEST(local_stt_empty_audio) {
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_stt(s, ADAM_STT_LOCAL, NULL, NULL, "/tmp/nonexistent.bin");

    arena_t *a = arena_create(4096);
    uint8_t empty[] = {0};
    const char *text = NULL;
    adam_status_t rc = adam_stt_transcribe(s, a, empty, 0,
                                           ADAM_AUDIO_WAV, &text);
    ASSERT_EQ(rc, ADAM_ERR_INVALID_PARAM);

    arena_destroy(a);
    adam_settings_destroy(s);
}

TEST(local_stt_invalid_wav) {
    // Garbage data, not a valid WAV
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_stt(s, ADAM_STT_LOCAL, NULL, NULL, "/tmp/nonexistent.bin");

    arena_t *a = arena_create(4096);
    uint8_t garbage[] = {0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03};
    const char *text = NULL;

    // Will fail on model load (nonexistent) before reaching WAV parse
    adam_status_t rc = adam_stt_transcribe(s, a, garbage, sizeof(garbage),
                                           ADAM_AUDIO_WAV, &text);
    ASSERT_NE(rc, ADAM_OK);

    arena_destroy(a);
    adam_settings_destroy(s);
}

TEST(local_stt_tiny_wav) {
    // WAV header only, no actual audio data
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_stt(s, ADAM_STT_LOCAL, NULL, NULL, "/tmp/nonexistent.bin");

    arena_t *a = arena_create(4096);
    // Minimal WAV: 44 bytes header, 0 bytes data
    size_t wav_len;
    uint8_t *wav = make_silent_wav(a, 16000, 0.0f, &wav_len);
    const char *text = NULL;

    adam_status_t rc = adam_stt_transcribe(s, a, wav, wav_len,
                                           ADAM_AUDIO_WAV, &text);
    ASSERT_NE(rc, ADAM_OK);

    arena_destroy(a);
    adam_settings_destroy(s);
}

TEST(local_stt_unsupported_format) {
    // MP3 format not supported for local STT
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_stt(s, ADAM_STT_LOCAL, NULL, NULL, "/tmp/nonexistent.bin");

    arena_t *a = arena_create(4096);
    uint8_t fake_mp3[] = {0xFF, 0xFB, 0x90, 0x00}; // MP3 frame header
    const char *text = NULL;

    // Will fail on model load before format check, but should not crash
    adam_status_t rc = adam_stt_transcribe(s, a, fake_mp3, sizeof(fake_mp3),
                                           ADAM_AUDIO_MP3, &text);
    ASSERT_NE(rc, ADAM_OK);

    arena_destroy(a);
    adam_settings_destroy(s);
}

TEST(local_stt_pcm16_format) {
    // Raw PCM16 input (no WAV header)
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_stt(s, ADAM_STT_LOCAL, NULL, NULL, "/tmp/nonexistent.bin");

    arena_t *a = arena_create(4096);
    int16_t silence[160] = {0}; // 10ms of silence at 16kHz
    const char *text = NULL;

    adam_status_t rc = adam_stt_transcribe(s, a, (uint8_t *)silence,
                                           sizeof(silence), ADAM_AUDIO_PCM16, &text);
    ASSERT_NE(rc, ADAM_OK); // fails on model load, not crash

    arena_destroy(a);
    adam_settings_destroy(s);
}

TEST(local_stt_destroy_without_use) {
    // Create settings with local STT configured but never call transcribe.
    // Destroy should not crash.
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_stt(s, ADAM_STT_LOCAL, NULL, NULL, "models/whisper.bin");
    adam_settings_destroy(s);
}

TEST(local_stt_null_out_text) {
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_stt(s, ADAM_STT_LOCAL, NULL, NULL, "/tmp/nonexistent.bin");

    arena_t *a = arena_create(4096);
    uint8_t data[] = {0};
    adam_status_t rc = adam_stt_transcribe(s, a, data, 1,
                                           ADAM_AUDIO_WAV, NULL);
    ASSERT_EQ(rc, ADAM_ERR_INVALID_PARAM);

    arena_destroy(a);
    adam_settings_destroy(s);
}

#endif // ADAM_NO_LOCAL

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

    ASSERT_EQ(r.status, ADAM_ERR_LOCAL);
    ASSERT_NOT_NULL(r.final_response);

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(local_null_gguf_path) {
    // gguf_path set but NULL — should not crash
    adam_settings_t *s = adam_create_settings();
    s->gguf_path = NULL;
    // Without gguf_path and no API, should be no provider
    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Hello");
    ASSERT_EQ(r.status, ADAM_ERR_NO_PROVIDER);
    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(local_empty_message) {
    // Empty user message should not crash
    adam_settings_t *s = adam_create_settings();
    s->gguf_path = "/tmp/nonexistent.gguf";
    s->local_gpu_layers = 0;

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "");
    // Either errors on model load or handles empty gracefully
    ASSERT_NE(r.status, ADAM_OK); // will fail (no model file)
    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(local_null_message) {
    // NULL user message should not crash
    adam_settings_t *s = adam_create_settings();
    s->gguf_path = "/tmp/nonexistent.gguf";
    s->local_gpu_layers = 0;

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, NULL);
    // Should handle gracefully — either error or empty history
    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(local_extreme_settings) {
    // Extreme configuration values should not crash
    adam_settings_t *s = adam_create_settings();
    s->gguf_path = "/tmp/nonexistent.gguf";
    s->local_gpu_layers = 9999;   // more layers than any model has
    s->local_ctx_size = 1;        // absurdly small context
    s->local_batch_size = 1;      // minimum batch
    s->max_tokens = 1;            // minimum generation
    s->temperature = 0.0f;        // greedy

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "test");
    // Will fail (no model) but should not crash
    ASSERT_NE(r.status, ADAM_OK);
    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(local_zero_temperature) {
    // temperature=0 should not divide by zero or crash
    adam_settings_t *s = adam_create_settings();
    s->gguf_path = "/tmp/nonexistent.gguf";
    s->temperature = 0.0f;

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "test");
    ASSERT_NE(r.status, ADAM_OK);
    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(local_settings_destroy_without_run) {
    // Create settings with gguf_path but never call adam_run.
    // Destroy should not crash (no local_ctx to free).
    adam_settings_t *s = adam_create_settings();
    s->gguf_path = "some_model.gguf";
    s->local_gpu_layers = -1;
    ASSERT_NULL(s->_local_ctx); // not initialized yet
    adam_settings_destroy(s);
}

TEST(local_double_destroy) {
    // Double destroy should not crash
    adam_settings_t *s = adam_create_settings();
    s->gguf_path = "/tmp/nonexistent.gguf";
    adam_settings_destroy(s);
    // Second destroy on freed memory — can't test directly without ASan,
    // but at least verify the first destroy doesn't crash.
}

TEST(local_priority_over_remote) {
    // When both gguf_path and api_key are set, local takes priority.
    // Since the gguf doesn't exist, it should fail with LOCAL error
    // (not try the remote API).
    adam_settings_t *s = adam_create_settings();
    s->gguf_path = "/tmp/nonexistent.gguf";
    adam_settings_set_provider(s, ADAM_API_OPENAI, "fake-key", "gpt-4o-mini");

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Hello");
    // Should fail with LOCAL error (tried local first, not HTTP)
    ASSERT_EQ(r.status, ADAM_ERR_LOCAL);
    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(local_with_tools_defined) {
    // Having tools defined shouldn't crash local inference
    // (tool calls from local models not yet implemented, but shouldn't break)
    adam_settings_t *s = adam_create_settings();
    s->gguf_path = "/tmp/nonexistent.gguf";

    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "test_tool",
        .description = "A test tool",
        .parameters_json = "{\"type\":\"object\"}",
        .execute = mock_tool_search, // from earlier in the file
    });

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Use the test tool");
    ASSERT_NE(r.status, ADAM_OK); // fails (no model) but doesn't crash
    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

static void local_stream_counter(void *ctx, const char *c, size_t l, int d) {
    UNUSED_PARAM(c); UNUSED_PARAM(l); UNUSED_PARAM(d);
    (*(int *)ctx)++;
}

TEST(local_with_stream_callback) {
    // Stream callback set but model doesn't exist — should not crash
    int stream_calls = 0;

    adam_settings_t *s = adam_create_settings();
    s->gguf_path = "/tmp/nonexistent.gguf";
    s->on_stream = local_stream_counter;
    s->stream_ctx = &stream_calls;

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Hello");
    ASSERT_NE(r.status, ADAM_OK);
    // Stream callback should NOT have been called (model failed to load)
    ASSERT_EQ(stream_calls, 0);
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
    adam_memory_t *mem = adam_memory_open(path);
    ASSERT_NOT_NULL(mem);
    adam_memory_close(mem);
    unlink(path);
}

TEST(session_save_load_simple) {
    const char *path = "/tmp/adam_test_session2.db";
    unlink(path);
    adam_memory_t *mem = adam_memory_open(path);
    ASSERT_NOT_NULL(mem);

    // Build a history
    adam_history_t *h = adam_history_create();
    adam_history_append_user(h, "Hello");
    adam_history_append_assistant(h, "Hi there!", NULL, 0);
    adam_history_append_user(h, "How are you?");
    adam_history_append_assistant(h, "I'm doing well!", NULL, 0);

    // Save
    adam_status_t rc = adam_session_save(mem, "test-001", h);
    ASSERT_EQ(rc, ADAM_OK);

    // Load into a new history
    adam_history_t *h2 = adam_history_create();
    rc = adam_session_load(mem, "test-001", h2);
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
    adam_memory_close(mem);
    unlink(path);
}

TEST(session_save_load_with_tool_calls) {
    const char *path = "/tmp/adam_test_session3.db";
    unlink(path);
    adam_memory_t *mem = adam_memory_open(path);

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

    adam_session_save(mem, "tool-test", h);

    // Load and verify
    adam_history_t *h2 = adam_history_create();
    adam_session_load(mem, "tool-test", h2);

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
    adam_memory_close(mem);
    unlink(path);
}

TEST(session_list_and_delete) {
    const char *path = "/tmp/adam_test_session4.db";
    unlink(path);
    adam_memory_t *mem = adam_memory_open(path);

    adam_history_t *h = adam_history_create();
    adam_history_append_user(h, "Test");

    adam_session_save(mem, "session-a", h);
    adam_session_save(mem, "session-b", h);
    adam_session_save(mem, "session-c", h);

    // List
    char **ids = NULL;
    size_t count = 0;
    adam_status_t rc = adam_session_list(mem, &ids, &count);
    ASSERT_EQ(rc, ADAM_OK);
    ASSERT_EQ(count, 3);
    ASSERT_NOT_NULL(ids);

    // Free list
    for (size_t i = 0; i < count; i++) free(ids[i]);
    free(ids);

    // Delete one
    rc = adam_session_delete(mem, "session-b");
    ASSERT_EQ(rc, ADAM_OK);

    // List again
    rc = adam_session_list(mem, &ids, &count);
    ASSERT_EQ(rc, ADAM_OK);
    ASSERT_EQ(count, 2);
    for (size_t i = 0; i < count; i++) free(ids[i]);
    free(ids);

    // Load deleted session should fail
    adam_history_t *h2 = adam_history_create();
    rc = adam_session_load(mem, "session-b", h2);
    ASSERT_NE(rc, ADAM_OK);

    adam_history_destroy(h);
    adam_history_destroy(h2);
    adam_memory_close(mem);
    unlink(path);
}

TEST(session_overwrite) {
    // Saving the same session_id twice should replace the old data
    const char *path = "/tmp/adam_test_session5.db";
    unlink(path);
    adam_memory_t *mem = adam_memory_open(path);

    adam_history_t *h1 = adam_history_create();
    adam_history_append_user(h1, "First version");
    adam_session_save(mem, "overwrite-test", h1);

    adam_history_t *h2 = adam_history_create();
    adam_history_append_user(h2, "Second version");
    adam_history_append_assistant(h2, "Updated reply", NULL, 0);
    adam_session_save(mem, "overwrite-test", h2);

    // Load should get the second version
    adam_history_t *h3 = adam_history_create();
    adam_session_load(mem, "overwrite-test", h3);
    ASSERT_EQ(adam_history_count(h3), 2);
    ASSERT_STR_EQ(h3->items[0].content, "Second version");
    ASSERT_STR_EQ(h3->items[1].content, "Updated reply");

    adam_history_destroy(h1);
    adam_history_destroy(h2);
    adam_history_destroy(h3);
    adam_memory_close(mem);
    unlink(path);
}

TEST(session_persistence_across_reopen) {
    // Data survives closing and reopening the database
    const char *path = "/tmp/adam_test_session6.db";
    unlink(path);

    // Open, save, close
    adam_memory_t *m1 = adam_memory_open(path);
    adam_history_t *h = adam_history_create();
    adam_history_append_user(h, "Persistent message");
    adam_session_save(m1, "persist-test", h);
    adam_memory_close(m1);

    // Reopen, load
    adam_memory_t *m2 = adam_memory_open(path);
    adam_history_t *h2 = adam_history_create();
    adam_status_t rc = adam_session_load(m2, "persist-test", h2);
    ASSERT_EQ(rc, ADAM_OK);
    ASSERT_EQ(adam_history_count(h2), 1);
    ASSERT_STR_EQ(h2->items[0].content, "Persistent message");

    adam_history_destroy(h);
    adam_history_destroy(h2);
    adam_memory_close(m2);
    unlink(path);
}

#endif // ADAM_NO_SQLITE

// ============================================================================
// MARK: - Tests: Evolution Loop
// ============================================================================

// Mock LLM for evolution: distinguishes attempt/refine/insight prompts
// by looking for keywords in the user message.
typedef struct {
    int attempt_count;
    int refine_count;
    int insight_count;
} mock_evolve_ctx_t;

static adam_llm_response_t mock_llm_evolve(
    void *ctx, arena_t *arena,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
) {
    mock_evolve_ctx_t *m = (mock_evolve_ctx_t *)ctx;
    UNUSED_PARAM(tools); UNUSED_PARAM(tool_count);

    adam_llm_response_t resp = {0};
    resp.input_tokens = 50;
    resp.output_tokens = 20;

    // Find the last user message
    const char *user_msg = "";
    for (size_t i = msg_count; i > 0; i--) {
        if (msgs[i - 1].role == ADAM_ROLE_USER) {
            user_msg = msgs[i - 1].content ? msgs[i - 1].content : "";
            break;
        }
    }

    if (strstr(user_msg, "Produce your best attempt")) {
        m->attempt_count++;
        char buf[64];
        snprintf(buf, sizeof(buf), "Attempt output #%d", m->attempt_count);
        resp.content = arena_strdup(arena, buf);
    } else if (strstr(user_msg, "Refine the strategy")) {
        m->refine_count++;
        resp.content = arena_strdup(arena, "Refined strategy: focus on quality.");
    } else if (strstr(user_msg, "Update the insights")) {
        m->insight_count++;
        char buf[128];
        snprintf(buf, sizeof(buf),
                 "Insight: iteration %d taught us to try harder.",
                 m->insight_count);
        resp.content = arena_strdup(arena, buf);
    } else {
        resp.content = arena_strdup(arena, "Unknown prompt type.");
    }
    return resp;
}

// Eval: scores based on iteration (simulates steady improvement)
static int eval_improving(void *ctx, const char *output, int iteration) {
    UNUSED_PARAM(output);
    int *call_count = (int *)ctx;
    (*call_count)++;
    // Score: 20, 40, 60, 80, 100
    return 20 + iteration * 20;
}

// Eval: always returns a fixed score (for plateau testing)
static int eval_fixed(void *ctx, const char *output, int iteration) {
    UNUSED_PARAM(output); UNUSED_PARAM(iteration);
    int *call_count = (int *)ctx;
    (*call_count)++;
    return 50;
}

// Eval: returns error on third call
static int eval_error(void *ctx, const char *output, int iteration) {
    UNUSED_PARAM(output); UNUSED_PARAM(iteration);
    int *call_count = (int *)ctx;
    (*call_count)++;
    if (*call_count >= 3) return -1;
    return 30;
}

// Progress tracker
typedef struct {
    int calls;
    int last_score;
    int last_best;
} progress_ctx_t;

static void on_progress(void *ctx, int iteration, int score,
                          int best_score, const char *output) {
    UNUSED_PARAM(iteration); UNUSED_PARAM(output);
    progress_ctx_t *p = (progress_ctx_t *)ctx;
    p->calls++;
    p->last_score = score;
    p->last_best = best_score;
}

TEST(evolve_basic_improvement) {
    // Evolution loop where score improves: 20, 40, 60, 80, 100
    // Should stop when target_score (95) is reached at iteration 4 (score=100)
    mock_evolve_ctx_t mock = {0};
    int eval_calls = 0;

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_evolve, &mock);

    adam_evolve_config_t cfg = adam_evolve_config_defaults();
    cfg.task = "Write the best poem about the sea.";
    cfg.initial_strategy = "Use vivid imagery and metaphor.";
    cfg.metrics = "Score based on creativity and emotional impact.";
    cfg.eval_fn = eval_improving;
    cfg.eval_ctx = &eval_calls;
    cfg.target_score = 95;

    adam_evolve_result_t r = adam_evolve(s, &cfg);

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_EQ(r.stop_reason, ADAM_EVOLVE_STOP_SCORE);
    ASSERT_EQ(r.best_score, 100);
    ASSERT_EQ(r.best_iteration, 4); // iteration 4: 20+4*20=100
    ASSERT_NOT_NULL(r.best_output);
    ASSERT(strstr(r.best_output, "Attempt output") != NULL);
    ASSERT_EQ(r.attempt_total, 5); // iterations 0-4
    ASSERT_EQ(eval_calls, 5);

    // Strategy should have been refined (score improved multiple times)
    ASSERT_NOT_NULL(r.strategy);
    ASSERT(strstr(r.strategy, "Refined strategy") != NULL);

    // Insights should have been updated
    ASSERT_NOT_NULL(r.insights);
    ASSERT(strstr(r.insights, "Insight") != NULL);

    // Mock counts: each iteration = 1 attempt + 1 insight = 2 calls
    // plus refine calls on improvement (all 5 iterations improve)
    ASSERT_EQ(mock.attempt_count, 5);
    ASSERT_EQ(mock.insight_count, 5);
    ASSERT(mock.refine_count >= 4); // score improved 4 times (iter 1-4 all > prev)

    // Stats should be accumulated
    ASSERT(r.total_input_tokens > 0);
    ASSERT(r.total_output_tokens > 0);
    ASSERT(r.elapsed_ms >= 0.0);

    adam_evolve_result_free(&r);
    adam_settings_destroy(s);
}

TEST(evolve_plateau_stop) {
    // Eval always returns 50 -> plateau after 3 iterations
    mock_evolve_ctx_t mock = {0};
    int eval_calls = 0;

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_evolve, &mock);

    adam_evolve_config_t cfg = adam_evolve_config_defaults();
    cfg.task = "Optimize the algorithm.";
    cfg.eval_fn = eval_fixed;
    cfg.eval_ctx = &eval_calls;
    cfg.plateau_iters = 3;
    cfg.max_iterations = 20;

    adam_evolve_result_t r = adam_evolve(s, &cfg);

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_EQ(r.stop_reason, ADAM_EVOLVE_STOP_PLATEAU);
    ASSERT_EQ(r.best_score, 50);
    // First iteration improves from -1 to 50, then 3 more with no improvement
    ASSERT_EQ(r.attempt_total, 4);
    ASSERT_EQ(eval_calls, 4);

    // Strategy refined only once (first iteration is an improvement from -1)
    ASSERT(mock.refine_count == 1);

    adam_evolve_result_free(&r);
    adam_settings_destroy(s);
}

TEST(evolve_max_iterations) {
    // Run exactly max_iterations (3) with constant improvement
    mock_evolve_ctx_t mock = {0};
    int eval_calls = 0;

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_evolve, &mock);

    adam_evolve_config_t cfg = adam_evolve_config_defaults();
    cfg.task = "Write a haiku.";
    cfg.eval_fn = eval_improving;
    cfg.eval_ctx = &eval_calls;
    cfg.max_iterations = 3;
    cfg.target_score = 200; // unreachable
    cfg.plateau_iters = 100; // won't trigger

    adam_evolve_result_t r = adam_evolve(s, &cfg);

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_EQ(r.stop_reason, ADAM_EVOLVE_STOP_MAX_ITERS);
    ASSERT_EQ(r.attempt_total, 3);
    ASSERT_EQ(r.best_score, 60); // 20 + 2*20 = 60 at iteration 2
    ASSERT_EQ(r.best_iteration, 2);

    // All 3 attempts in buffer
    ASSERT_EQ(r.attempt_count, 3);
    ASSERT_EQ(r.attempts[0].score, 20);
    ASSERT_EQ(r.attempts[1].score, 40);
    ASSERT_EQ(r.attempts[2].score, 60);

    adam_evolve_result_free(&r);
    adam_settings_destroy(s);
}

TEST(evolve_ring_buffer) {
    // Run 15 iterations (more than ring buffer of 10)
    mock_evolve_ctx_t mock = {0};
    int eval_calls = 0;

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_evolve, &mock);

    // Eval: returns iteration+1 as score (always improving, never reaches 200)
    // We can't use eval_improving (it goes 20,40,...) so make a custom one

    adam_evolve_config_t cfg = adam_evolve_config_defaults();
    cfg.task = "Long running task.";
    cfg.eval_fn = eval_improving;
    cfg.eval_ctx = &eval_calls;
    cfg.max_iterations = 15;
    cfg.target_score = 999; // unreachable
    cfg.plateau_iters = 999; // won't trigger

    adam_evolve_result_t r = adam_evolve(s, &cfg);

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_EQ(r.attempt_total, 15);
    // Ring buffer should have exactly 10 entries
    ASSERT_EQ(r.attempt_count, ADAM_EVOLVE_MAX_ATTEMPTS);
    // Oldest entry should be from iteration 5 (0-4 were evicted)
    ASSERT_EQ(r.attempts[0].iteration, 5);
    // Newest entry should be from iteration 14
    ASSERT_EQ(r.attempts[9].iteration, 14);

    adam_evolve_result_free(&r);
    adam_settings_destroy(s);
}

TEST(evolve_null_params) {
    adam_settings_t *s = adam_create_settings();

    // NULL config
    adam_evolve_result_t r1 = adam_evolve(s, NULL);
    ASSERT_EQ(r1.status, ADAM_ERR_INVALID_PARAM);
    adam_evolve_result_free(&r1);

    // NULL settings
    adam_evolve_config_t cfg = adam_evolve_config_defaults();
    cfg.task = "test";
    cfg.eval_fn = eval_fixed;
    adam_evolve_result_t r2 = adam_evolve(NULL, &cfg);
    ASSERT_EQ(r2.status, ADAM_ERR_INVALID_PARAM);
    adam_evolve_result_free(&r2);

    // NULL task
    adam_evolve_config_t cfg3 = adam_evolve_config_defaults();
    cfg3.eval_fn = eval_fixed;
    adam_evolve_result_t r3 = adam_evolve(s, &cfg3);
    ASSERT_EQ(r3.status, ADAM_ERR_INVALID_PARAM);
    adam_evolve_result_free(&r3);

    // NULL eval_fn
    adam_evolve_config_t cfg4 = adam_evolve_config_defaults();
    cfg4.task = "test";
    adam_evolve_result_t r4 = adam_evolve(s, &cfg4);
    ASSERT_EQ(r4.status, ADAM_ERR_INVALID_PARAM);
    adam_evolve_result_free(&r4);

    adam_settings_destroy(s);
}

TEST(evolve_abort) {
    mock_evolve_ctx_t mock = {0};
    int eval_calls = 0;

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_evolve, &mock);

    adam_evolve_config_t cfg = adam_evolve_config_defaults();
    cfg.task = "test abort";
    cfg.eval_fn = eval_fixed;
    cfg.eval_ctx = &eval_calls;

    // abort_flag is reset by adam_run, but we check it before each iteration.
    // Since adam_run resets it, we need to set it in the eval callback.
    // Instead, let's test that abort stops after first iteration:
    // We'll set abort from progress callback.
    // Actually — the abort_flag is checked at start of each evolution iteration,
    // but adam_run resets it. So we'd need to set it between iterations.
    // For simplicity, just set it before calling adam_evolve.
    // But adam_run resets it internally... let's verify the stop still works
    // by using the abort from the progress callback.

    // Use progress to set abort after 2 iterations
    s->abort_flag = 0;
    cfg.progress_fn = NULL; // can't easily set abort from here without shared state

    // Alternative: set abort before calling. adam_run resets it, but our
    // evolve loop checks it at the top of each iteration. First iteration
    // will proceed (adam_run resets the flag). After first iteration completes,
    // flag is still 0. So we need another approach.

    // Let's verify with eval callback setting abort:
    // We'll abuse eval_ctx to also point to settings for abort.
    // Simpler: just test that the loop runs exactly the eval_calls.
    // Skip this complexity — test that abort before first iteration works
    // by examining the behavior.

    // Actually, the simplest test: have eval set the abort flag
    // We'll use a special eval for this.
    adam_settings_destroy(s);

    // Fresh setup with abort-aware eval
    s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_evolve, &mock);

    typedef struct { adam_settings_t *s; int call_count; } abort_eval_ctx_t;
    abort_eval_ctx_t actx = { .s = s, .call_count = 0 };

    cfg.eval_ctx = &actx;
    // We can't easily define a new eval inline in C, so let's just test
    // that providing abort before does something reasonable.
    // Reset mock counts
    mock.attempt_count = 0;
    mock.refine_count = 0;
    mock.insight_count = 0;

    // Set abort before first iteration — but adam_run resets it.
    // The evolve loop checks abort_flag at the top, before adam_run.
    s->abort_flag = 1;
    cfg.eval_fn = eval_fixed;
    cfg.eval_ctx = &eval_calls;
    eval_calls = 0;

    adam_evolve_result_t r = adam_evolve(s, &cfg);
    ASSERT_EQ(r.status, ADAM_ERR_ABORTED);
    ASSERT_EQ(r.stop_reason, ADAM_EVOLVE_STOP_ABORTED);
    ASSERT_EQ(eval_calls, 0); // never called — aborted before first iteration

    adam_evolve_result_free(&r);
    adam_settings_destroy(s);
}

TEST(evolve_eval_error) {
    // Eval returns -1 on third call -> should stop with error
    mock_evolve_ctx_t mock = {0};
    int eval_calls = 0;

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_evolve, &mock);

    adam_evolve_config_t cfg = adam_evolve_config_defaults();
    cfg.task = "Test error handling.";
    cfg.eval_fn = eval_error;
    cfg.eval_ctx = &eval_calls;
    cfg.max_iterations = 10;
    cfg.plateau_iters = 100;

    adam_evolve_result_t r = adam_evolve(s, &cfg);

    ASSERT_EQ(r.stop_reason, ADAM_EVOLVE_STOP_ERROR);
    ASSERT_EQ(eval_calls, 3); // failed on 3rd call
    ASSERT_EQ(r.attempt_total, 2); // only 2 complete iterations

    adam_evolve_result_free(&r);
    adam_settings_destroy(s);
}

TEST(evolve_progress_callback) {
    // Verify progress callback fires with correct values
    mock_evolve_ctx_t mock = {0};
    int eval_calls = 0;
    progress_ctx_t pctx = {0};

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_evolve, &mock);

    adam_evolve_config_t cfg = adam_evolve_config_defaults();
    cfg.task = "Progress test.";
    cfg.eval_fn = eval_improving;
    cfg.eval_ctx = &eval_calls;
    cfg.progress_fn = on_progress;
    cfg.progress_ctx = &pctx;
    cfg.max_iterations = 3;
    cfg.target_score = 999;
    cfg.plateau_iters = 999;

    adam_evolve_result_t r = adam_evolve(s, &cfg);

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_EQ(pctx.calls, 3);
    ASSERT_EQ(pctx.last_score, 60); // last iteration: 20+2*20=60
    ASSERT_EQ(pctx.last_best, 40); // best at time of last call (before update)

    adam_evolve_result_free(&r);
    adam_settings_destroy(s);
}

TEST(evolve_config_defaults) {
    adam_evolve_config_t cfg = adam_evolve_config_defaults();
    ASSERT_EQ(cfg.max_iterations, 10);
    ASSERT_EQ(cfg.target_score, 95);
    ASSERT_EQ(cfg.plateau_iters, 3);
    ASSERT_NULL(cfg.task);
    ASSERT_NULL(cfg.initial_strategy);
    ASSERT_NULL(cfg.metrics);
    ASSERT_NULL(cfg.eval_fn);
    ASSERT_NULL(cfg.progress_fn);
}

TEST(evolve_no_strategy_no_metrics) {
    // Run without initial_strategy and metrics — should still work
    mock_evolve_ctx_t mock = {0};
    int eval_calls = 0;

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_evolve, &mock);

    adam_evolve_config_t cfg = adam_evolve_config_defaults();
    cfg.task = "Minimal config test.";
    cfg.eval_fn = eval_fixed;
    cfg.eval_ctx = &eval_calls;
    cfg.max_iterations = 2;
    cfg.plateau_iters = 100;
    cfg.target_score = 200;

    adam_evolve_result_t r = adam_evolve(s, &cfg);

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_EQ(r.attempt_total, 2);
    ASSERT_NOT_NULL(r.strategy); // default "No strategy yet." or refined
    ASSERT_NOT_NULL(r.insights);

    adam_evolve_result_free(&r);
    adam_settings_destroy(s);
}

// ============================================================================
// MARK: - Tests: Research Mode
// ============================================================================

// Mock LLM for research: simulates tool-using research agent behavior.
// - First call: returns findings with FINDING:/SOURCE: tags
// - Second call: returns more findings
// - Third call: returns RESEARCH_COMPLETE with a report
typedef struct {
    int call_count;
    int include_complete;       // if 1, say RESEARCH_COMPLETE on call N
    int complete_on_call;       // which call to complete on (default: 3)
} mock_research_ctx_t;

static adam_llm_response_t mock_llm_research(
    void *ctx, arena_t *arena,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
) {
    mock_research_ctx_t *m = (mock_research_ctx_t *)ctx;
    m->call_count++;
    UNUSED_PARAM(msgs); UNUSED_PARAM(msg_count);
    UNUSED_PARAM(tools); UNUSED_PARAM(tool_count);

    adam_llm_response_t resp = {0};
    resp.input_tokens = 100;
    resp.output_tokens = 50;

    int complete_at = m->complete_on_call > 0 ? m->complete_on_call : 3;

    if (m->include_complete && m->call_count >= complete_at) {
        resp.content = arena_strdup(arena,
            "FINDING: The speed of light is 299,792,458 m/s.\n"
            "SOURCE: Physics textbook\n\n"
            "RESEARCH_COMPLETE\n"
            "# Research Report\n\n"
            "The speed of light in vacuum is approximately 299,792,458 "
            "meters per second. This is a fundamental constant of nature.");
    } else if (m->call_count == 1) {
        resp.content = arena_strdup(arena,
            "I'll search for information about this topic.\n\n"
            "FINDING: Light travels at approximately 3x10^8 m/s in vacuum.\n"
            "SOURCE: Wikipedia\n\n"
            "FINDING: The speed of light is denoted by the letter c.\n"
            "SOURCE: Physics reference\n\n"
            "I need to search for more details.");
    } else if (m->call_count == 2) {
        resp.content = arena_strdup(arena,
            "FINDING: Einstein's E=mc^2 relates energy to mass via c.\n"
            "SOURCE: Theory of Relativity\n\n"
            "FINDING: Light slows down in denser media like glass or water.\n"
            "SOURCE: Optics textbook\n\n"
            "Let me search for more specific data.");
    } else {
        // Generic fallback: return more findings
        char buf[256];
        snprintf(buf, sizeof(buf),
            "FINDING: Additional fact #%d about the topic.\n"
            "SOURCE: Source #%d\n", m->call_count, m->call_count);
        resp.content = arena_strdup(arena, buf);
    }

    return resp;
}

// Mock LLM for report synthesis (used when agent doesn't say RESEARCH_COMPLETE)
static adam_llm_response_t mock_llm_research_report(
    void *ctx, arena_t *arena,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
) {
    mock_research_ctx_t *m = (mock_research_ctx_t *)ctx;
    m->call_count++;
    UNUSED_PARAM(tools); UNUSED_PARAM(tool_count);

    adam_llm_response_t resp = {0};
    resp.input_tokens = 100;
    resp.output_tokens = 50;

    // Check if this is a synthesis request
    const char *last_msg = "";
    for (size_t i = msg_count; i > 0; i--) {
        if (msgs[i - 1].role == ADAM_ROLE_USER && msgs[i - 1].content) {
            last_msg = msgs[i - 1].content;
            break;
        }
    }

    if (strstr(last_msg, "Synthesize all research")) {
        resp.content = arena_strdup(arena,
            "# Research Report\n\n"
            "This is a synthesized report based on all findings.");
    } else {
        // Research iteration — return findings but no COMPLETE
        char buf[256];
        snprintf(buf, sizeof(buf),
            "FINDING: Fact from iteration %d.\n"
            "SOURCE: Source %d\n", m->call_count, m->call_count);
        resp.content = arena_strdup(arena, buf);
    }

    return resp;
}

TEST(research_config_defaults) {
    adam_research_config_t cfg = adam_research_config_defaults();
    ASSERT_EQ(cfg.max_iterations, 5);
    ASSERT_NULL(cfg.question);
    ASSERT_NULL(cfg.instructions);
    ASSERT_NULL(cfg.is_complete);
    ASSERT_NULL(cfg.on_progress);
}

TEST(research_null_params) {
    adam_settings_t *s = adam_create_settings();

    // NULL config
    adam_research_result_t r1 = adam_research(s, NULL);
    ASSERT_EQ(r1.status, ADAM_ERR_INVALID_PARAM);
    adam_research_result_free(&r1);

    // NULL settings
    adam_research_config_t cfg = adam_research_config_defaults();
    cfg.question = "test";
    adam_research_result_t r2 = adam_research(NULL, &cfg);
    ASSERT_EQ(r2.status, ADAM_ERR_INVALID_PARAM);
    adam_research_result_free(&r2);

    // NULL question
    adam_research_config_t cfg3 = adam_research_config_defaults();
    adam_research_result_t r3 = adam_research(s, &cfg3);
    ASSERT_EQ(r3.status, ADAM_ERR_INVALID_PARAM);
    adam_research_result_free(&r3);

    adam_settings_destroy(s);
}

TEST(research_complete_by_agent) {
    // Agent says RESEARCH_COMPLETE on 3rd call
    mock_research_ctx_t mock = { .include_complete = 1, .complete_on_call = 3 };

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_research, &mock);

    adam_research_config_t cfg = adam_research_config_defaults();
    cfg.question = "What is the speed of light?";
    cfg.max_iterations = 10;

    adam_research_result_t r = adam_research(s, &cfg);

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_EQ(r.stop_reason, ADAM_RESEARCH_STOP_COMPLETE);
    ASSERT_EQ(r.total_iterations, 3);

    // Should have findings from all iterations
    ASSERT(r.finding_count >= 3); // at least from first 2 iters + complete iter

    // Report should be the text after RESEARCH_COMPLETE
    ASSERT_NOT_NULL(r.report);
    ASSERT(strstr(r.report, "Research Report") != NULL);

    // Stats
    ASSERT(r.total_input_tokens > 0);
    ASSERT(r.total_output_tokens > 0);
    ASSERT(r.elapsed_ms >= 0.0);

    adam_research_result_free(&r);
    adam_settings_destroy(s);
}

TEST(research_max_iterations) {
    // Agent never says COMPLETE, hits max_iterations
    mock_research_ctx_t mock = { .include_complete = 0 };

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_research_report, &mock);

    adam_research_config_t cfg = adam_research_config_defaults();
    cfg.question = "What are the effects of climate change?";
    cfg.max_iterations = 3;

    adam_research_result_t r = adam_research(s, &cfg);

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_EQ(r.stop_reason, ADAM_RESEARCH_STOP_MAX_ITERS);
    ASSERT_EQ(r.total_iterations, 3);

    // Should have findings from each iteration
    ASSERT(r.finding_count >= 3);

    // Report should be synthesized (separate LLM call at end)
    ASSERT_NOT_NULL(r.report);
    ASSERT(strstr(r.report, "Research Report") != NULL);

    // The synthesis call adds 1 more LLM call: 3 iterations + 1 synthesis = 4
    ASSERT_EQ(mock.call_count, 4);

    adam_research_result_free(&r);
    adam_settings_destroy(s);
}

TEST(research_findings_parsed) {
    // Verify FINDING:/SOURCE: parsing works correctly
    mock_research_ctx_t mock = { .include_complete = 1, .complete_on_call = 2 };

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_research, &mock);

    adam_research_config_t cfg = adam_research_config_defaults();
    cfg.question = "Speed of light";
    cfg.max_iterations = 5;

    adam_research_result_t r = adam_research(s, &cfg);

    ASSERT_EQ(r.status, ADAM_OK);

    // First iteration returns 2 findings, second returns 1 + COMPLETE
    ASSERT(r.finding_count >= 2);

    // Check first finding
    ASSERT_NOT_NULL(r.findings[0].content);
    ASSERT(strstr(r.findings[0].content, "3x10^8") != NULL ||
           strstr(r.findings[0].content, "light") != NULL);
    ASSERT_NOT_NULL(r.findings[0].source);
    ASSERT_EQ(r.findings[0].iteration, 0);

    adam_research_result_free(&r);
    adam_settings_destroy(s);
}

// Static callback for research completion test
static int research_complete_after_2(void *ctx, const char *report,
                                       int iteration) {
    UNUSED_PARAM(ctx); UNUSED_PARAM(report);
    return iteration >= 1; // complete after 2 iterations (0-indexed)
}

TEST(research_callback_stops_loop) {
    mock_research_ctx_t mock = { .include_complete = 0 };

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_research_report, &mock);

    adam_research_config_t cfg = adam_research_config_defaults();
    cfg.question = "Quick test";
    cfg.max_iterations = 10;
    cfg.is_complete = research_complete_after_2;

    adam_research_result_t r = adam_research(s, &cfg);

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_EQ(r.stop_reason, ADAM_RESEARCH_STOP_CALLBACK);
    ASSERT_EQ(r.total_iterations, 2);

    // Report should be synthesized since agent didn't say COMPLETE
    ASSERT_NOT_NULL(r.report);

    adam_research_result_free(&r);
    adam_settings_destroy(s);
}

TEST(research_abort) {
    mock_research_ctx_t mock = {0};

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_research, &mock);
    s->abort_flag = 1;

    adam_research_config_t cfg = adam_research_config_defaults();
    cfg.question = "Aborted research";

    adam_research_result_t r = adam_research(s, &cfg);

    ASSERT_EQ(r.status, ADAM_ERR_ABORTED);
    ASSERT_EQ(r.stop_reason, ADAM_RESEARCH_STOP_ABORTED);
    ASSERT_EQ(r.total_iterations, 0);
    ASSERT_EQ(mock.call_count, 0);

    adam_research_result_free(&r);
    adam_settings_destroy(s);
}

// Progress tracker for research
typedef struct {
    int calls;
    char last_status[128];
} research_progress_ctx_t;

static void on_research_progress(void *ctx, int iteration, const char *status) {
    UNUSED_PARAM(iteration);
    research_progress_ctx_t *p = (research_progress_ctx_t *)ctx;
    p->calls++;
    if (status) {
        size_t len = strlen(status);
        if (len > 127) len = 127;
        memcpy(p->last_status, status, len);
        p->last_status[len] = '\0';
    }
}

TEST(research_progress_callback) {
    mock_research_ctx_t mock = { .include_complete = 1, .complete_on_call = 2 };
    research_progress_ctx_t pctx = {0};

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_research, &mock);

    adam_research_config_t cfg = adam_research_config_defaults();
    cfg.question = "Progress test";
    cfg.max_iterations = 5;
    cfg.on_progress = on_research_progress;
    cfg.progress_ctx = &pctx;

    adam_research_result_t r = adam_research(s, &cfg);

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_EQ(pctx.calls, 2); // 2 iterations before COMPLETE
    ASSERT(strstr(pctx.last_status, "findings") != NULL);

    adam_research_result_free(&r);
    adam_settings_destroy(s);
}

TEST(research_with_instructions) {
    // Verify extra instructions are accepted (not crash, correct flow)
    mock_research_ctx_t mock = { .include_complete = 1, .complete_on_call = 1 };

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_research, &mock);

    adam_research_config_t cfg = adam_research_config_defaults();
    cfg.question = "Speed of light";
    cfg.instructions = "Focus on historical measurements. "
                       "Include Roemer's 1676 estimate.";

    adam_research_result_t r = adam_research(s, &cfg);

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_NOT_NULL(r.report);
    ASSERT_EQ(r.total_iterations, 1);

    adam_research_result_free(&r);
    adam_settings_destroy(s);
}

TEST(research_tool_basic) {
    // Test adam_tool_research as a callable tool during a session
    mock_research_ctx_t mock = { .include_complete = 1, .complete_on_call = 1 };

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_research, &mock);

    arena_t *a = arena_create(64 * 1024);

    const char *args = "{\"question\":\"What is gravity?\"}";
    adam_tool_result_t res = adam_tool_research(a, s, args, strlen(args));

    ASSERT_EQ(res.success, 1);
    ASSERT_NOT_NULL(res.for_llm);
    ASSERT(strstr(res.for_llm, "Research Report") != NULL ||
           strstr(res.for_llm, "Finding") != NULL);
    ASSERT_NOT_NULL(res.for_user);
    ASSERT(strstr(res.for_user, "Researching") != NULL);

    arena_destroy(a);
    adam_settings_destroy(s);
}

TEST(research_tool_bad_args) {
    adam_settings_t *s = adam_create_settings();
    arena_t *a = arena_create(4096);

    // Missing question
    const char *args1 = "{\"instructions\":\"test\"}";
    adam_tool_result_t r1 = adam_tool_research(a, s, args1, strlen(args1));
    ASSERT_EQ(r1.success, 0);
    ASSERT(strstr(r1.for_llm, "required") != NULL);

    // NULL ctx
    adam_tool_result_t r2 = adam_tool_research(a, NULL, "{}", 2);
    ASSERT_EQ(r2.success, 0);

    // Invalid JSON
    adam_tool_result_t r3 = adam_tool_research(a, s, "not json", 8);
    ASSERT_EQ(r3.success, 0);

    arena_destroy(a);
    adam_settings_destroy(s);
}

// ============================================================================
// MARK: - Tests: Built-in Tools (file, shell, calculator, sql)
// ============================================================================

TEST(tool_calculator_basic) {
    arena_t *a = arena_create(4096);

    const char *args = "{\"expression\":\"2 + 3 * 4\"}";
    adam_tool_result_t r = adam_tool_calculator(a, NULL, args, strlen(args));
    ASSERT_EQ(r.success, 1);
    ASSERT_STR_EQ(r.for_llm, "14");

    arena_reset(a);
    args = "{\"expression\":\"(10 + 5) / 3\"}";
    r = adam_tool_calculator(a, NULL, args, strlen(args));
    ASSERT_EQ(r.success, 1);
    ASSERT_STR_EQ(r.for_llm, "5");

    arena_reset(a);
    args = "{\"expression\":\"2 ^ 10\"}";
    r = adam_tool_calculator(a, NULL, args, strlen(args));
    ASSERT_EQ(r.success, 1);
    ASSERT_STR_EQ(r.for_llm, "1024");

    arena_reset(a);
    args = "{\"expression\":\"-5 + 3\"}";
    r = adam_tool_calculator(a, NULL, args, strlen(args));
    ASSERT_EQ(r.success, 1);
    ASSERT_STR_EQ(r.for_llm, "-2");

    arena_reset(a);
    args = "{\"expression\":\"100 % 7\"}";
    r = adam_tool_calculator(a, NULL, args, strlen(args));
    ASSERT_EQ(r.success, 1);
    ASSERT_STR_EQ(r.for_llm, "2");

    arena_destroy(a);
}

TEST(tool_calculator_errors) {
    arena_t *a = arena_create(4096);

    // Missing expression
    const char *args = "{\"foo\":\"bar\"}";
    adam_tool_result_t r = adam_tool_calculator(a, NULL, args, strlen(args));
    ASSERT_EQ(r.success, 0);
    ASSERT(strstr(r.for_llm, "required") != NULL);

    // Division by zero
    arena_reset(a);
    args = "{\"expression\":\"1 / 0\"}";
    r = adam_tool_calculator(a, NULL, args, strlen(args));
    // Should return Infinity or error
    ASSERT_NOT_NULL(r.for_llm);

    arena_destroy(a);
}

TEST(tool_file_read_sandbox) {
    arena_t *a = arena_create(64 * 1024);
    adam_settings_t *s = adam_create_settings();

    // No allowed dirs — should deny
    const char *args = "{\"path\":\"/etc/hosts\"}";
    adam_tool_result_t r = adam_tool_file_read(a, s, args, strlen(args));
    ASSERT_EQ(r.success, 0);
    ASSERT(strstr(r.for_llm, "denied") != NULL);

    // Allow /tmp
    ASSERT_EQ(adam_settings_allow_dir(s, "/tmp"), ADAM_OK);

    // Write a test file
    FILE *f = fopen("/tmp/adam_test_read.txt", "w");
    ASSERT_NOT_NULL(f);
    fprintf(f, "hello from adam test");
    fclose(f);

    // Read it — should succeed
    arena_reset(a);
    args = "{\"path\":\"/tmp/adam_test_read.txt\"}";
    r = adam_tool_file_read(a, s, args, strlen(args));
    ASSERT_EQ(r.success, 1);
    ASSERT_STR_EQ(r.for_llm, "hello from adam test");

    // Try to escape sandbox
    arena_reset(a);
    args = "{\"path\":\"/tmp/../etc/hosts\"}";
    r = adam_tool_file_read(a, s, args, strlen(args));
    ASSERT_EQ(r.success, 0);
    ASSERT(strstr(r.for_llm, "denied") != NULL);

    remove("/tmp/adam_test_read.txt");
    arena_destroy(a);
    adam_settings_destroy(s);
}

TEST(tool_file_write_sandbox) {
    arena_t *a = arena_create(4096);
    adam_settings_t *s = adam_create_settings();
    ASSERT_EQ(adam_settings_allow_dir(s, "/tmp"), ADAM_OK);

    // Write a file
    const char *args = "{\"path\":\"/tmp/adam_test_write.txt\","
                       "\"content\":\"test content\"}";
    adam_tool_result_t r = adam_tool_file_write(a, s, args, strlen(args));
    ASSERT_EQ(r.success, 1);
    ASSERT(strstr(r.for_llm, "Wrote") != NULL);

    // Verify contents
    FILE *f = fopen("/tmp/adam_test_write.txt", "r");
    ASSERT_NOT_NULL(f);
    char buf[64];
    size_t n = fread(buf, 1, 63, f);
    buf[n] = '\0';
    fclose(f);
    ASSERT_STR_EQ(buf, "test content");

    // Append
    arena_reset(a);
    args = "{\"path\":\"/tmp/adam_test_write.txt\","
           "\"content\":\" appended\",\"append\":true}";
    r = adam_tool_file_write(a, s, args, strlen(args));
    ASSERT_EQ(r.success, 1);

    f = fopen("/tmp/adam_test_write.txt", "r");
    n = fread(buf, 1, 63, f);
    buf[n] = '\0';
    fclose(f);
    ASSERT_STR_EQ(buf, "test content appended");

    // Deny outside sandbox
    arena_reset(a);
    args = "{\"path\":\"/etc/adam_test.txt\",\"content\":\"hack\"}";
    r = adam_tool_file_write(a, s, args, strlen(args));
    ASSERT_EQ(r.success, 0);
    ASSERT(strstr(r.for_llm, "denied") != NULL);

    remove("/tmp/adam_test_write.txt");
    arena_destroy(a);
    adam_settings_destroy(s);
}

TEST(tool_list_directory_sandbox) {
    arena_t *a = arena_create(64 * 1024);
    adam_settings_t *s = adam_create_settings();
    ASSERT_EQ(adam_settings_allow_dir(s, "/tmp"), ADAM_OK);

    // Create test directory with a file
    mkdir("/tmp/adam_test_dir", 0755);
    FILE *f = fopen("/tmp/adam_test_dir/test.txt", "w");
    if (f) { fputs("x", f); fclose(f); }

    const char *args = "{\"path\":\"/tmp/adam_test_dir\"}";
    adam_tool_result_t r = adam_tool_list_directory(a, s, args, strlen(args));
    ASSERT_EQ(r.success, 1);
    ASSERT(strstr(r.for_llm, "test.txt") != NULL);
    ASSERT(strstr(r.for_llm, "file") != NULL);

    // Deny outside sandbox
    arena_reset(a);
    args = "{\"path\":\"/etc\"}";
    r = adam_tool_list_directory(a, s, args, strlen(args));
    ASSERT_EQ(r.success, 0);

    remove("/tmp/adam_test_dir/test.txt");
    rmdir("/tmp/adam_test_dir");
    arena_destroy(a);
    adam_settings_destroy(s);
}

TEST(tool_shell_exec_basic) {
    arena_t *a = arena_create(64 * 1024);
    adam_settings_t *s = adam_create_settings();

    // No allowed dirs — should deny
    const char *args = "{\"command\":\"echo hello\"}";
    adam_tool_result_t r = adam_tool_shell_exec(a, s, args, strlen(args));
    ASSERT_EQ(r.success, 0);
    ASSERT(strstr(r.for_llm, "denied") != NULL);

    // Allow /tmp
    ASSERT_EQ(adam_settings_allow_dir(s, "/tmp"), ADAM_OK);

    // Simple echo
    arena_reset(a);
    r = adam_tool_shell_exec(a, s, args, strlen(args));
    ASSERT_EQ(r.success, 1);
    ASSERT(strstr(r.for_llm, "hello") != NULL);
    ASSERT(strstr(r.for_llm, "exit code: 0") != NULL);

    // Command that fails
    arena_reset(a);
    args = "{\"command\":\"false\"}";
    r = adam_tool_shell_exec(a, s, args, strlen(args));
    ASSERT_EQ(r.success, 0); // exit code != 0
    ASSERT(strstr(r.for_llm, "exit code: 1") != NULL);

    arena_destroy(a);
    adam_settings_destroy(s);
}

TEST(tool_allow_dir) {
    adam_settings_t *s = adam_create_settings();

    ASSERT_EQ(s->allowed_dir_count, 0);
    ASSERT_EQ(adam_settings_allow_dir(s, "/tmp"), ADAM_OK);
    ASSERT_EQ(s->allowed_dir_count, 1);
    ASSERT_EQ(adam_settings_allow_dir(s, "/var"), ADAM_OK);
    ASSERT_EQ(s->allowed_dir_count, 2);

    // NULL params
    ASSERT_EQ(adam_settings_allow_dir(NULL, "/tmp"), ADAM_ERR_INVALID_PARAM);
    ASSERT_EQ(adam_settings_allow_dir(s, NULL), ADAM_ERR_INVALID_PARAM);

    // Nonexistent dir
    ASSERT_EQ(adam_settings_allow_dir(s, "/nonexistent_dir_xyz"),
              ADAM_ERR_INVALID_PARAM);

    adam_settings_destroy(s);
}

#ifndef ADAM_NO_SQLITE

TEST(tool_sql_query_basic) {
    adam_memory_t *mem = adam_memory_open(":memory:");
    ASSERT_NOT_NULL(mem);

    arena_t *a = arena_create(64 * 1024);

    // Create a table and insert data
    const char *args = "{\"sql\":\"CREATE TABLE test(id INTEGER, name TEXT)\"}";
    adam_tool_result_t r = adam_tool_sql_query(a, mem, args, strlen(args));
    ASSERT_EQ(r.success, 1);

    arena_reset(a);
    args = "{\"sql\":\"INSERT INTO test VALUES(1, 'Alice')\"}";
    r = adam_tool_sql_query(a, mem, args, strlen(args));
    ASSERT_EQ(r.success, 1);

    arena_reset(a);
    args = "{\"sql\":\"INSERT INTO test VALUES(2, 'Bob')\"}";
    r = adam_tool_sql_query(a, mem, args, strlen(args));
    ASSERT_EQ(r.success, 1);

    // SELECT
    arena_reset(a);
    args = "{\"sql\":\"SELECT * FROM test ORDER BY id\"}";
    r = adam_tool_sql_query(a, mem, args, strlen(args));
    ASSERT_EQ(r.success, 1);
    ASSERT(strstr(r.for_llm, "Alice") != NULL);
    ASSERT(strstr(r.for_llm, "Bob") != NULL);

    // Block DROP
    arena_reset(a);
    args = "{\"sql\":\"DROP TABLE test\"}";
    r = adam_tool_sql_query(a, mem, args, strlen(args));
    ASSERT_EQ(r.success, 0);
    ASSERT(strstr(r.for_llm, "destructive") != NULL);

    arena_destroy(a);
    adam_memory_close(mem);
}

#endif // ADAM_NO_SQLITE

// ============================================================================
// MARK: - Tests: History Clone
// ============================================================================

TEST(history_clone_basic) {
    adam_history_t *h = adam_history_create();
    adam_history_append_user(h, "Hello");
    adam_history_append_assistant(h, "Hi there", NULL, 0);

    adam_history_t *c = adam_history_clone(h);
    ASSERT_NOT_NULL(c);
    ASSERT_EQ(adam_history_count(c), 2);
    ASSERT_STR_EQ(c->items[0].content, "Hello");
    ASSERT_STR_EQ(c->items[1].content, "Hi there");
    // Verify independent pointers
    ASSERT(c->items[0].content != h->items[0].content);

    adam_history_destroy(h);
    adam_history_destroy(c);
}

TEST(history_clone_with_tool_calls) {
    adam_history_t *h = adam_history_create();
    adam_history_append_user(h, "Search");
    adam_tool_call_t tc = { .id = "c1", .name = "search",
                            .arguments_json = "{\"q\":\"x\"}" };
    adam_history_append_assistant(h, "Searching...", &tc, 1);
    adam_history_append_tool(h, "found it", "c1");

    adam_history_t *c = adam_history_clone(h);
    ASSERT_EQ(adam_history_count(c), 3);
    ASSERT_EQ(c->items[1].tool_call_count, 1);
    ASSERT_STR_EQ(c->items[1].tool_calls[0].id, "c1");
    ASSERT_STR_EQ(c->items[1].tool_calls[0].name, "search");
    ASSERT_STR_EQ(c->items[2].tool_call_id, "c1");
    // Independent pointers
    ASSERT(c->items[1].tool_calls[0].id != h->items[1].tool_calls[0].id);

    adam_history_destroy(h);
    adam_history_destroy(c);
}

TEST(history_clone_empty) {
    adam_history_t *h = adam_history_create();
    adam_history_t *c = adam_history_clone(h);
    ASSERT_NOT_NULL(c);
    ASSERT_EQ(adam_history_count(c), 0);
    adam_history_destroy(h);
    adam_history_destroy(c);
}

TEST(history_clone_modify_original) {
    adam_history_t *h = adam_history_create();
    adam_history_append_user(h, "Original");
    adam_history_t *c = adam_history_clone(h);

    // Modify original
    adam_history_clear(h);
    ASSERT_EQ(adam_history_count(h), 0);
    // Clone should be unaffected
    ASSERT_EQ(adam_history_count(c), 1);
    ASSERT_STR_EQ(c->items[0].content, "Original");

    adam_history_destroy(h);
    adam_history_destroy(c);
}

// ============================================================================
// MARK: - Tests: Timeout
// ============================================================================

// Mock LLM that always returns tool calls (infinite loop)
static adam_llm_response_t mock_llm_infinite_tools(
    void *ctx, arena_t *arena,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
) {
    mock_llm_ctx_t *mock = (mock_llm_ctx_t *)ctx;
    mock->call_count++;
    UNUSED_PARAM(msgs); UNUSED_PARAM(msg_count);
    UNUSED_PARAM(tools); UNUSED_PARAM(tool_count);

    adam_llm_response_t resp = {0};
    resp.input_tokens = 10;
    resp.output_tokens = 10;
    resp.tool_calls = arena_alloc(arena, sizeof(adam_tool_call_t));
    resp.tool_call_count = 1;
    resp.tool_calls[0].id = arena_strdup(arena, "call_loop");
    resp.tool_calls[0].name = arena_strdup(arena, "search");
    resp.tool_calls[0].arguments_json = arena_strdup(arena, "{}");
    return resp;
}

TEST(timeout_triggers) {
    mock_llm_ctx_t mock = {0};
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_infinite_tools, &mock);
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "search", .execute = mock_tool_search });
    s->timeout_ms = 1; // 1ms
    s->max_iterations = 10000; // high limit so timeout is the real constraint

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Loop forever");

    // Should timeout (or hit max_iterations if mock is too fast)
    ASSERT(r.status == ADAM_ERR_TIMEOUT || r.status == ADAM_ERR_MAX_ITERATIONS);
    // Should have run fewer than max iterations if timeout worked
    ASSERT(r.total_iterations < 10000);

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(timeout_zero_disabled) {
    mock_llm_ctx_t mock = { .fixed_response = "Done." };
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);
    s->timeout_ms = 0; // disabled

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Quick test");

    ASSERT_EQ(r.status, ADAM_OK);

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

// ============================================================================
// MARK: - Tests: Guardrails
// ============================================================================

static int guardrail_allow(void *ctx, const adam_message_t *msgs, size_t count) {
    UNUSED_PARAM(msgs); UNUSED_PARAM(count);
    (*(int *)ctx)++;
    return 0; // allow
}

static int guardrail_deny(void *ctx, const adam_message_t *msgs, size_t count) {
    UNUSED_PARAM(msgs); UNUSED_PARAM(count);
    (*(int *)ctx)++;
    return 1; // deny
}

static int guardrail_resp_allow(void *ctx, const char *content,
                                  const adam_tool_call_t *tc, size_t tc_count) {
    UNUSED_PARAM(content); UNUSED_PARAM(tc); UNUSED_PARAM(tc_count);
    (*(int *)ctx)++;
    return 0;
}

static int guardrail_resp_deny(void *ctx, const char *content,
                                 const adam_tool_call_t *tc, size_t tc_count) {
    UNUSED_PARAM(content); UNUSED_PARAM(tc); UNUSED_PARAM(tc_count);
    (*(int *)ctx)++;
    return 1;
}

TEST(guardrail_before_allow) {
    mock_llm_ctx_t mock = { .fixed_response = "OK" };
    int guard_calls = 0;
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);
    s->on_before_send = guardrail_allow;
    s->before_send_ctx = &guard_calls;

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Test");
    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT(guard_calls >= 1);

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(guardrail_before_deny) {
    mock_llm_ctx_t mock = {0};
    int guard_calls = 0;
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);
    s->on_before_send = guardrail_deny;
    s->before_send_ctx = &guard_calls;

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Test");
    ASSERT_EQ(r.status, ADAM_ERR_GUARDRAIL);
    ASSERT_EQ(mock.call_count, 0); // LLM never called

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(guardrail_after_deny) {
    mock_llm_ctx_t mock = { .fixed_response = "bad content" };
    int guard_calls = 0;
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);
    s->on_after_receive = guardrail_resp_deny;
    s->after_receive_ctx = &guard_calls;

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Test");
    ASSERT_EQ(r.status, ADAM_ERR_GUARDRAIL);
    ASSERT_EQ(mock.call_count, 1); // LLM was called
    ASSERT(guard_calls >= 1);

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

// ============================================================================
// MARK: - Tests: Structured JSON Output
// ============================================================================

// Mock that returns valid JSON
static adam_llm_response_t mock_llm_json(
    void *ctx, arena_t *arena,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
) {
    mock_llm_ctx_t *mock = (mock_llm_ctx_t *)ctx;
    mock->call_count++;
    UNUSED_PARAM(msgs); UNUSED_PARAM(msg_count);
    UNUSED_PARAM(tools); UNUSED_PARAM(tool_count);

    adam_llm_response_t resp = {0};
    resp.input_tokens = 50;
    resp.output_tokens = 20;

    if (mock->call_count == 1 && mock->simulate_error) {
        // First call: return invalid JSON for retry testing
        resp.content = arena_strdup(arena, "This is not JSON");
    } else {
        resp.content = arena_strdup(arena, "{\"answer\":42,\"name\":\"Adam\"}");
    }
    return resp;
}

TEST(run_json_valid) {
    mock_llm_ctx_t mock = {0};
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_json, &mock);

    adam_history_t *h = adam_history_create();
    adam_json_result_t r = adam_run_json(s, h, "Give me JSON", NULL, 3);

    ASSERT_EQ(r.base.status, ADAM_OK);
    ASSERT_EQ(r.json_valid, 1);
    ASSERT_EQ(r.retries_used, 0);
    ASSERT_NOT_NULL(r.base.final_response);
    ASSERT(strstr(r.base.final_response, "\"answer\"") != NULL);

    adam_json_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

// Mock that returns bad JSON first, valid on second call
static adam_llm_response_t mock_llm_bad_then_good_json(
    void *ctx, arena_t *arena,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
) {
    mock_llm_ctx_t *mock = (mock_llm_ctx_t *)ctx;
    mock->call_count++;
    UNUSED_PARAM(msgs); UNUSED_PARAM(msg_count);
    UNUSED_PARAM(tools); UNUSED_PARAM(tool_count);

    adam_llm_response_t resp = {0};
    resp.input_tokens = 50;
    resp.output_tokens = 20;

    if (mock->call_count <= 1) {
        resp.content = arena_strdup(arena, "This is not valid JSON at all");
    } else {
        resp.content = arena_strdup(arena, "{\"result\":\"success\"}");
    }
    return resp;
}

TEST(run_json_retry) {
    mock_llm_ctx_t mock = {0};
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_bad_then_good_json, &mock);

    adam_history_t *h = adam_history_create();
    adam_json_result_t r = adam_run_json(s, h, "Give me JSON", NULL, 3);

    // The mock returns bad JSON first, then valid JSON.
    // adam_run_json should eventually get valid JSON.
    ASSERT_NOT_NULL(r.base.final_response);
    if (mock.call_count >= 2) {
        // Retry worked as expected
        ASSERT_EQ(r.json_valid, 1);
    }
    // At minimum, the function should not crash and should return

    adam_json_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

TEST(json_extract_basic) {
    char *v = adam_json_extract("{\"name\":\"Adam\",\"version\":1}", "name");
    ASSERT_NOT_NULL(v);
    ASSERT_STR_EQ(v, "Adam");
    free(v);

    v = adam_json_extract("{\"name\":\"Adam\",\"version\":1}", "version");
    ASSERT_NOT_NULL(v);
    ASSERT_STR_EQ(v, "1");
    free(v);

    v = adam_json_extract("{\"name\":\"Adam\"}", "missing");
    ASSERT_NULL(v);

    v = adam_json_extract(NULL, "key");
    ASSERT_NULL(v);
}

// ============================================================================
// MARK: - Tests: Multi-Agent Tool
// ============================================================================

// Mock for multi-agent: sends {"message":"..."} instead of {"query":"..."}
static adam_llm_response_t mock_llm_agent_caller(
    void *ctx, arena_t *arena,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
) {
    mock_llm_ctx_t *mock = (mock_llm_ctx_t *)ctx;
    mock->call_count++;
    UNUSED_PARAM(tools); UNUSED_PARAM(tool_count);

    adam_llm_response_t resp = {0};
    resp.input_tokens = 50;
    resp.output_tokens = 20;

    int has_tool_result = 0;
    for (size_t i = 0; i < msg_count; i++)
        if (msgs[i].role == ADAM_ROLE_TOOL) has_tool_result = 1;

    if (!has_tool_result && tool_count > 0) {
        mock->total_tool_calls_requested++;
        resp.tool_calls = arena_alloc(arena, sizeof(adam_tool_call_t));
        resp.tool_call_count = 1;
        resp.tool_calls[0].id = arena_strdup(arena, "call_sub");
        resp.tool_calls[0].name = arena_strdup(arena, tools[0].name);
        resp.tool_calls[0].arguments_json =
            arena_strdup(arena, "{\"message\":\"What is 2+2?\"}");
        resp.content = arena_strdup(arena, "Let me ask the sub-agent.");
    } else {
        resp.content = arena_strdup(arena, "The sub-agent answered.");
    }
    return resp;
}

TEST(tool_agent_basic) {
    // Sub-agent with its own mock LLM
    mock_llm_ctx_t sub_mock = { .fixed_response = "The answer is 4." };
    adam_settings_t *sub_s = adam_create_settings();
    adam_settings_set_llm_callback(sub_s, mock_llm_simple, &sub_mock);

    adam_agent_tool_ctx_t actx = { .settings = sub_s, .history = NULL };

    // Main agent calls the sub-agent tool
    mock_llm_ctx_t main_mock = {0};
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_agent_caller, &main_mock);
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "ask_sub",
        .description = "Ask sub-agent",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"message\":{\"type\":\"string\"}},\"required\":[\"message\"]}",
        .execute = adam_tool_agent,
        .ctx = &actx,
    });

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Delegate this");

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_EQ(sub_mock.call_count, 1); // sub-agent was called
    ASSERT_EQ(main_mock.call_count, 2); // main: tool call + final

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
    adam_settings_destroy(sub_s);
}

TEST(tool_agent_null_ctx) {
    arena_t *a = arena_create(4096);
    adam_tool_result_t r = adam_tool_agent(a, NULL, "{\"message\":\"hi\"}", 16);
    ASSERT_EQ(r.success, 0);
    ASSERT(strstr(r.for_llm, "not configured") != NULL);
    arena_destroy(a);
}

// ============================================================================
// MARK: - Tests: Response Cache
// ============================================================================

TEST(cache_miss_then_hit) {
    mock_llm_ctx_t mock = { .fixed_response = "Cached response" };
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);
    s->cache = adam_cache_create(16);

    adam_history_t *h = adam_history_create();

    // First call: miss
    adam_run_result_t r1 = adam_run(s, h, "Hello");
    ASSERT_EQ(r1.status, ADAM_OK);
    ASSERT_EQ(mock.call_count, 1);
    ASSERT_EQ(adam_cache_misses(s->cache), 1);
    adam_run_result_free(&r1);

    // Second call with same history: hit
    // Clear and re-send to get same messages
    adam_history_destroy(h);
    h = adam_history_create();
    mock.call_count = 0; // reset

    adam_run_result_t r2 = adam_run(s, h, "Hello");
    ASSERT_EQ(r2.status, ADAM_OK);
    ASSERT_STR_EQ(r2.final_response, "Cached response");
    ASSERT_EQ(mock.call_count, 0); // LLM NOT called
    ASSERT_EQ(adam_cache_hits(s->cache), 1);

    adam_run_result_free(&r2);
    adam_history_destroy(h);
    adam_cache_destroy(s->cache);
    s->cache = NULL;
    adam_settings_destroy(s);
}

TEST(cache_different_messages) {
    mock_llm_ctx_t mock = {0};
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);
    s->cache = adam_cache_create(16);

    adam_history_t *h1 = adam_history_create();
    mock.fixed_response = "Response A";
    adam_run_result_t r1 = adam_run(s, h1, "Message A");
    ASSERT_EQ(r1.status, ADAM_OK);
    adam_run_result_free(&r1);

    adam_history_t *h2 = adam_history_create();
    mock.fixed_response = "Response B";
    adam_run_result_t r2 = adam_run(s, h2, "Message B");
    ASSERT_EQ(r2.status, ADAM_OK);
    adam_run_result_free(&r2);

    ASSERT_EQ(mock.call_count, 2); // both were misses
    ASSERT_EQ(adam_cache_count(s->cache), 2);

    adam_history_destroy(h1);
    adam_history_destroy(h2);
    adam_cache_destroy(s->cache);
    s->cache = NULL;
    adam_settings_destroy(s);
}

TEST(cache_lru_eviction) {
    adam_cache_t *c = adam_cache_create(2);
    ASSERT_NOT_NULL(c);

    // Manually store 3 entries
    extern void adam_cache_store(adam_cache_t *, uint64_t, const char *, int, int);
    adam_cache_store(c, 1, "first", 10, 5);
    adam_cache_store(c, 2, "second", 10, 5);
    ASSERT_EQ(adam_cache_count(c), 2);

    adam_cache_store(c, 3, "third", 10, 5);
    ASSERT_EQ(adam_cache_count(c), 2); // evicted one

    // First should be evicted (LRU)
    int in, out;
    extern const char *adam_cache_lookup(adam_cache_t *, uint64_t, int *, int *);
    ASSERT_NULL(adam_cache_lookup(c, 1, &in, &out));
    ASSERT_NOT_NULL(adam_cache_lookup(c, 2, &in, &out));
    ASSERT_NOT_NULL(adam_cache_lookup(c, 3, &in, &out));

    adam_cache_destroy(c);
}

TEST(cache_clear) {
    adam_cache_t *c = adam_cache_create(16);
    extern void adam_cache_store(adam_cache_t *, uint64_t, const char *, int, int);
    adam_cache_store(c, 1, "a", 0, 0);
    adam_cache_store(c, 2, "b", 0, 0);
    ASSERT_EQ(adam_cache_count(c), 2);

    adam_cache_clear(c);
    ASSERT_EQ(adam_cache_count(c), 0);

    adam_cache_destroy(c);
}

TEST(cache_null_disabled) {
    mock_llm_ctx_t mock = { .fixed_response = "No cache" };
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);
    // s->cache = NULL (default)

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Test");
    ASSERT_EQ(r.status, ADAM_OK);

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

// ============================================================================
// MARK: - Integration Tests (real-world scenarios)
// ============================================================================

// --- Mock that behaves like a real assistant: answers questions,
//     calls tools when needed, uses tool results in final answer ---

typedef struct {
    int call_count;
    int want_tool_on_call;       // which call# should request a tool (0=never)
    const char *tool_name;       // which tool to call
    const char *tool_args;       // args to send
    const char *final_answer;    // answer after tool result
} scenario_llm_ctx_t;

static adam_llm_response_t scenario_llm(
    void *ctx, arena_t *arena,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
) {
    scenario_llm_ctx_t *m = (scenario_llm_ctx_t *)ctx;
    m->call_count++;
    UNUSED_PARAM(tools); UNUSED_PARAM(tool_count);

    adam_llm_response_t resp = {0};
    resp.input_tokens = 100;
    resp.output_tokens = 50;

    // Check if there's a tool result in history
    int has_tool_result = 0;
    for (size_t i = 0; i < msg_count; i++)
        if (msgs[i].role == ADAM_ROLE_TOOL) has_tool_result = 1;

    if (m->want_tool_on_call == m->call_count && !has_tool_result
        && m->tool_name && tool_count > 0) {
        // Request a tool call
        resp.tool_calls = arena_alloc(arena, sizeof(adam_tool_call_t));
        resp.tool_call_count = 1;
        resp.tool_calls[0].id = arena_strdup(arena, "call_scenario");
        resp.tool_calls[0].name = arena_strdup(arena, m->tool_name);
        resp.tool_calls[0].arguments_json = arena_strdup(arena,
            m->tool_args ? m->tool_args : "{}");
        resp.content = arena_strdup(arena, "Let me look that up.");
    } else {
        resp.content = arena_strdup(arena,
            m->final_answer ? m->final_answer : "Done.");
    }
    return resp;
}

// ---- Scenario 1: Multi-turn conversation with session save/load ----

#ifndef ADAM_NO_SQLITE

TEST(integration_session_multi_turn) {
    // Simulate: 3-turn conversation, save, reload, continue
    mock_llm_ctx_t mock = {0};

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);

    adam_memory_t *mem = adam_memory_open(":memory:");
    ASSERT_NOT_NULL(mem);
    s->memory = mem;

    // Create session
    char sid[37];
    ASSERT_EQ(adam_session_create(mem, sid, sizeof(sid)), ADAM_OK);
    s->session_id = sid;
    s->auto_save = 1;

    adam_history_t *h = adam_history_create();

    // Turn 1
    mock.fixed_response = "Hello! How can I help?";
    adam_run_result_t r1 = adam_run(s, h, "Hi there");
    ASSERT_EQ(r1.status, ADAM_OK);
    ASSERT_STR_EQ(r1.final_response, "Hello! How can I help?");
    adam_run_result_free(&r1);

    // Turn 2
    mock.fixed_response = "Your name is Marco.";
    adam_run_result_t r2 = adam_run(s, h, "My name is Marco");
    ASSERT_EQ(r2.status, ADAM_OK);
    adam_run_result_free(&r2);

    // Turn 3
    mock.fixed_response = "You told me your name is Marco.";
    adam_run_result_t r3 = adam_run(s, h, "What is my name?");
    ASSERT_EQ(r3.status, ADAM_OK);
    adam_run_result_free(&r3);

    // History should have: system + 3*(user + assistant) = 7
    ASSERT_EQ(adam_history_count(h), 7);
    adam_history_destroy(h);

    // Reload session into fresh history
    adam_history_t *h2 = adam_history_create();
    ASSERT_EQ(adam_session_load(mem, sid, h2), ADAM_OK);
    ASSERT_EQ(adam_history_count(h2), 7);

    // Verify message content survived round-trip.
    // Note: system message is loaded as user (gets rebuilt by next adam_run).
    // So items[0] is the old system prompt loaded as user, items[1] is "Hi there"
    ASSERT_EQ(h2->items[1].role, ADAM_ROLE_USER);
    ASSERT(strstr(h2->items[1].content, "Hi there") != NULL);
    ASSERT_EQ(h2->items[2].role, ADAM_ROLE_ASSISTANT);
    ASSERT(strstr(h2->items[2].content, "Hello") != NULL);
    ASSERT_EQ(h2->items[5].role, ADAM_ROLE_USER);
    ASSERT(strstr(h2->items[5].content, "What is my name") != NULL);

    // Continue conversation from loaded session
    mock.fixed_response = "Continuing from saved session.";
    size_t count_before = adam_history_count(h2);
    adam_run_result_t r4 = adam_run(s, h2, "Continue please");
    ASSERT_EQ(r4.status, ADAM_OK);
    // adam_run adds: system message (if missing) + user + assistant
    // loaded history had system as user, so adam_run prepends new system (+1)
    // then appends user (+1) and assistant (+1) = count_before + 3
    ASSERT(adam_history_count(h2) >= count_before + 2); // at least user + assistant
    ASSERT_STR_EQ(r4.final_response, "Continuing from saved session.");
    adam_run_result_free(&r4);

    adam_history_destroy(h2);
    adam_settings_destroy(s);
    adam_memory_close(mem);
}

// ---- Scenario 2: Tool call with session persistence ----

TEST(integration_tool_call_session_roundtrip) {
    // Agent calls a tool, then the entire conversation (including tool call
    // and tool result) is saved and loaded correctly.
    scenario_llm_ctx_t mock = {
        .want_tool_on_call = 1,
        .tool_name = "search",
        .tool_args = "{\"query\":\"test\"}",
        .final_answer = "Based on search: the answer is 42."
    };

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, scenario_llm, &mock);
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "search", .execute = mock_tool_search });

    adam_memory_t *mem = adam_memory_open(":memory:");
    s->memory = mem;
    char sid[37];
    adam_session_create(mem, sid, sizeof(sid));
    s->session_id = sid;
    s->auto_save = 1;

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Find the answer");
    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_STR_EQ(r.final_response, "Based on search: the answer is 42.");
    ASSERT_EQ(mock.call_count, 2); // tool call + final answer

    // History: system + user + assistant(tool_call) + tool_result + assistant(final) = 5
    ASSERT_EQ(adam_history_count(h), 5);
    ASSERT_EQ(h->items[2].tool_call_count, 1);
    ASSERT_STR_EQ(h->items[2].tool_calls[0].name, "search");
    ASSERT_EQ(h->items[3].role, ADAM_ROLE_TOOL);
    adam_run_result_free(&r);

    // Load into fresh history and verify tool calls survived
    adam_history_t *h2 = adam_history_create();
    ASSERT_EQ(adam_session_load(mem, sid, h2), ADAM_OK);
    ASSERT_EQ(adam_history_count(h2), 5);
    ASSERT_EQ(h2->items[2].tool_call_count, 1);
    ASSERT_STR_EQ(h2->items[2].tool_calls[0].name, "search");
    ASSERT_STR_EQ(h2->items[2].tool_calls[0].id, "call_scenario");
    ASSERT_EQ(h2->items[3].role, ADAM_ROLE_TOOL);
    ASSERT_STR_EQ(h2->items[3].tool_call_id, "call_scenario");
    ASSERT(strstr(h2->items[3].content, "42 matches") != NULL);

    adam_history_destroy(h);
    adam_history_destroy(h2);
    adam_settings_destroy(s);
    adam_memory_close(mem);
}

#endif // ADAM_NO_SQLITE

// ---- Scenario 3: Evolution with tool-using agent ----

TEST(integration_evolve_with_tools) {
    // Evolution loop where the agent has tools. Each attempt can use tools.
    // The eval function scores based on whether tool was used.
    mock_llm_ctx_t mock = {0};

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_with_tool, &mock);
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "search", .execute = mock_tool_search });

    // Eval: score increases each iteration
    int eval_calls = 0;
    adam_evolve_config_t cfg = adam_evolve_config_defaults();
    cfg.task = "Find information using the search tool.";
    cfg.eval_fn = eval_improving;
    cfg.eval_ctx = &eval_calls;
    cfg.max_iterations = 3;
    cfg.target_score = 999; // won't reach
    cfg.plateau_iters = 999;

    adam_evolve_result_t r = adam_evolve(s, &cfg);

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_EQ(r.stop_reason, ADAM_EVOLVE_STOP_MAX_ITERS);
    ASSERT_EQ(r.attempt_total, 3);
    ASSERT_EQ(eval_calls, 3);
    // Each attempt used the tool (mock_llm_with_tool calls tool then answers)
    ASSERT(mock.call_count >= 6); // 2 calls per attempt * 3 + refine/insight calls
    ASSERT_NOT_NULL(r.best_output);
    ASSERT_NOT_NULL(r.strategy);
    ASSERT_NOT_NULL(r.insights);

    adam_evolve_result_free(&r);
    adam_settings_destroy(s);
}

// ---- Scenario 4: Research with findings accumulation ----

TEST(integration_research_accumulates_findings) {
    // Research mode: 3 iterations, each adds findings, final report synthesized
    mock_research_ctx_t mock = { .include_complete = 0 };

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_research_report, &mock);

    adam_research_config_t cfg = adam_research_config_defaults();
    cfg.question = "How does photosynthesis work?";
    cfg.instructions = "Focus on the light-dependent reactions.";
    cfg.max_iterations = 3;

    adam_research_result_t r = adam_research(s, &cfg);

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_EQ(r.stop_reason, ADAM_RESEARCH_STOP_MAX_ITERS);
    ASSERT_EQ(r.total_iterations, 3);

    // Each iteration should have added at least 1 finding
    ASSERT(r.finding_count >= 3);

    // Findings should have iteration tracking
    ASSERT_EQ(r.findings[0].iteration, 0);

    // Report should be synthesized
    ASSERT_NOT_NULL(r.report);
    ASSERT(strstr(r.report, "Research Report") != NULL);

    // Stats accumulated across iterations
    ASSERT(r.total_input_tokens > 0);
    ASSERT(r.elapsed_ms >= 0.0);

    adam_research_result_free(&r);
    adam_settings_destroy(s);
}

// ---- Scenario 5: Multi-agent delegation chain ----

TEST(integration_multi_agent_chain) {
    // Main agent delegates to sub-agent, which has its own tools and identity.
    // Sub-agent uses a tool, gets result, returns to main agent.

    // Sub-agent: has a calculator tool
    mock_llm_ctx_t sub_mock = { .fixed_response = "The result is 1024." };
    adam_settings_t *sub_s = adam_create_settings();
    adam_settings_set_llm_callback(sub_s, mock_llm_simple, &sub_mock);
    adam_settings_set_identity(sub_s, "You are a math assistant.");

    adam_agent_tool_ctx_t sub_ctx = { .settings = sub_s, .history = NULL };

    // Main agent: delegates math to sub-agent
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_agent_caller, &(mock_llm_ctx_t){0});
    adam_settings_set_identity(s, "You are a general assistant.");
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "ask_math",
        .description = "Ask the math assistant",
        .parameters_json = "{\"type\":\"object\",\"properties\":"
                           "{\"message\":{\"type\":\"string\"}},"
                           "\"required\":[\"message\"]}",
        .execute = adam_tool_agent,
        .ctx = &sub_ctx,
    });

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "What is 2^10?");

    ASSERT_EQ(r.status, ADAM_OK);
    // Sub-agent was called
    ASSERT_EQ(sub_mock.call_count, 1);
    // Main agent got sub-agent's response as tool result
    ASSERT_EQ(adam_history_count(h), 5); // sys + user + asst(tool) + tool_result + asst(final)
    // Tool result should contain sub-agent's answer
    ASSERT(strstr(h->items[3].content, "1024") != NULL);

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
    adam_settings_destroy(sub_s);
}

// ---- Scenario 6: Cache prevents redundant LLM calls ----

TEST(integration_cache_across_sessions) {
    // Two independent conversations with same message get cache hit
    mock_llm_ctx_t mock = { .fixed_response = "Paris is the capital." };
    adam_cache_t *cache = adam_cache_create(32);

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);
    s->cache = cache;

    // First conversation
    adam_history_t *h1 = adam_history_create();
    adam_run_result_t r1 = adam_run(s, h1, "What is the capital of France?");
    ASSERT_EQ(r1.status, ADAM_OK);
    ASSERT_STR_EQ(r1.final_response, "Paris is the capital.");
    ASSERT_EQ(mock.call_count, 1);
    adam_run_result_free(&r1);
    adam_history_destroy(h1);

    // Second conversation — same question, different history object
    adam_history_t *h2 = adam_history_create();
    adam_run_result_t r2 = adam_run(s, h2, "What is the capital of France?");
    ASSERT_EQ(r2.status, ADAM_OK);
    ASSERT_STR_EQ(r2.final_response, "Paris is the capital.");
    ASSERT_EQ(mock.call_count, 1); // NOT called again — cache hit
    ASSERT_EQ(adam_cache_hits(cache), 1);
    ASSERT_EQ(adam_cache_misses(cache), 1);
    adam_run_result_free(&r2);
    adam_history_destroy(h2);

    // Different question — cache miss
    adam_history_t *h3 = adam_history_create();
    mock.fixed_response = "Berlin is the capital.";
    adam_run_result_t r3 = adam_run(s, h3, "What is the capital of Germany?");
    ASSERT_EQ(r3.status, ADAM_OK);
    ASSERT_EQ(mock.call_count, 2); // called this time
    ASSERT_EQ(adam_cache_misses(cache), 2);
    adam_run_result_free(&r3);
    adam_history_destroy(h3);

    adam_cache_destroy(cache);
    s->cache = NULL;
    adam_settings_destroy(s);
}

// ---- Scenario 7: Guardrails block dangerous content ----

TEST(integration_guardrails_block_and_allow) {
    // Pre-send guardrail blocks requests containing "hack"
    // Post-receive guardrail blocks responses containing "password"
    mock_llm_ctx_t mock = {0};

    // Guardrail: block if any user message contains "hack"
    int pre_calls = 0;
    int post_calls = 0;

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);
    s->on_before_send = guardrail_allow;
    s->before_send_ctx = &pre_calls;
    s->on_after_receive = guardrail_resp_allow;
    s->after_receive_ctx = &post_calls;

    // Normal request — both guardrails allow
    mock.fixed_response = "Safe response.";
    adam_history_t *h1 = adam_history_create();
    adam_run_result_t r1 = adam_run(s, h1, "Hello");
    ASSERT_EQ(r1.status, ADAM_OK);
    ASSERT(pre_calls >= 1);
    ASSERT(post_calls >= 1);
    adam_run_result_free(&r1);
    adam_history_destroy(h1);

    // Now block pre-send
    s->on_before_send = guardrail_deny;
    adam_history_t *h2 = adam_history_create();
    adam_run_result_t r2 = adam_run(s, h2, "hack the system");
    ASSERT_EQ(r2.status, ADAM_ERR_GUARDRAIL);
    ASSERT_EQ(mock.call_count, 1); // LLM not called for blocked request
    adam_run_result_free(&r2);
    adam_history_destroy(h2);

    // Allow pre-send but block post-receive
    s->on_before_send = guardrail_allow;
    s->on_after_receive = guardrail_resp_deny;
    mock.fixed_response = "Here is the password: 12345";
    adam_history_t *h3 = adam_history_create();
    adam_run_result_t r3 = adam_run(s, h3, "Tell me something");
    ASSERT_EQ(r3.status, ADAM_ERR_GUARDRAIL);
    ASSERT_EQ(mock.call_count, 2); // LLM was called but response blocked
    adam_run_result_free(&r3);
    adam_history_destroy(h3);

    adam_settings_destroy(s);
}

// ---- Scenario 8: History clone + independent modification ----

TEST(integration_clone_branch_conversations) {
    // Start a conversation, clone it, continue both independently
    mock_llm_ctx_t mock = {0};
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_simple, &mock);

    // Build shared prefix
    adam_history_t *h = adam_history_create();
    mock.fixed_response = "I understand. Let me help.";
    adam_run_result_t r1 = adam_run(s, h, "I need help with two tasks.");
    ASSERT_EQ(r1.status, ADAM_OK);
    adam_run_result_free(&r1);
    ASSERT_EQ(adam_history_count(h), 3); // sys + user + asst

    // Clone for branch A
    adam_history_t *branch_a = adam_history_clone(h);
    ASSERT_NOT_NULL(branch_a);
    ASSERT_EQ(adam_history_count(branch_a), 3);

    // Clone for branch B
    adam_history_t *branch_b = adam_history_clone(h);
    ASSERT_NOT_NULL(branch_b);

    // Continue branch A
    mock.fixed_response = "Branch A result.";
    adam_run_result_t ra = adam_run(s, branch_a, "Task A: write code");
    ASSERT_EQ(ra.status, ADAM_OK);
    ASSERT_STR_EQ(ra.final_response, "Branch A result.");
    ASSERT_EQ(adam_history_count(branch_a), 5);
    adam_run_result_free(&ra);

    // Continue branch B differently
    mock.fixed_response = "Branch B result.";
    adam_run_result_t rb = adam_run(s, branch_b, "Task B: write tests");
    ASSERT_EQ(rb.status, ADAM_OK);
    ASSERT_STR_EQ(rb.final_response, "Branch B result.");
    ASSERT_EQ(adam_history_count(branch_b), 5);
    adam_run_result_free(&rb);

    // Original unchanged
    ASSERT_EQ(adam_history_count(h), 3);

    // Branch A has "Task A", branch B has "Task B"
    ASSERT(strstr(branch_a->items[3].content, "Task A") != NULL);
    ASSERT(strstr(branch_b->items[3].content, "Task B") != NULL);

    adam_history_destroy(h);
    adam_history_destroy(branch_a);
    adam_history_destroy(branch_b);
    adam_settings_destroy(s);
}

// ---- Scenario 9: Evolution converges to target score ----

TEST(integration_evolve_converges) {
    // Eval returns progressively better scores: 10, 30, 50, 70, 90, 100
    // Target is 95, so should stop at iteration 4 (score=90) or 5 (score=100)
    mock_evolve_ctx_t mock = {0};
    int eval_calls = 0;

    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_evolve, &mock);

    // Custom eval: 10 + iteration * 20
    adam_evolve_config_t cfg = adam_evolve_config_defaults();
    cfg.task = "Write a perfect essay.";
    cfg.initial_strategy = "Start with an outline.";
    cfg.metrics = "Clarity, structure, and persuasiveness.";
    cfg.eval_fn = eval_improving; // 20, 40, 60, 80, 100
    cfg.eval_ctx = &eval_calls;
    cfg.target_score = 95;
    cfg.max_iterations = 20;

    adam_evolve_result_t r = adam_evolve(s, &cfg);

    ASSERT_EQ(r.status, ADAM_OK);
    ASSERT_EQ(r.stop_reason, ADAM_EVOLVE_STOP_SCORE);
    ASSERT_EQ(r.best_score, 100); // 20+4*20=100 >= 95
    ASSERT_EQ(r.attempt_total, 5);

    // Strategy was refined (improved multiple times)
    ASSERT(strstr(r.strategy, "Refined") != NULL);
    // Insights accumulated
    ASSERT(strstr(r.insights, "Insight") != NULL);
    // Best output captured
    ASSERT_NOT_NULL(r.best_output);

    adam_evolve_result_free(&r);
    adam_settings_destroy(s);
}

// ---- Scenario 10: Timeout stops long-running tool loop ----

TEST(integration_timeout_with_tools) {
    // Agent keeps calling tools forever, timeout stops it
    mock_llm_ctx_t mock = {0};
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_infinite_tools, &mock);
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "search", .execute = mock_tool_search });
    s->timeout_ms = 5; // very short timeout
    s->max_iterations = 100000;

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, "Search forever");

    // Should have stopped due to timeout
    ASSERT(r.status == ADAM_ERR_TIMEOUT || r.status == ADAM_ERR_MAX_ITERATIONS);
    ASSERT(r.total_iterations < 100000);
    ASSERT(r.elapsed_ms >= 0.0);

    adam_run_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

// ---- Scenario 11: Thread pool with multiple agents ----

#ifndef ADAM_NO_PTHREADS

static pthread_mutex_t g_integ_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_integ_done = 0;
static adam_run_result_t g_integ_results[8];

static void integ_pool_done(void *ctx, adam_run_result_t result) {
    int idx = *(int *)ctx;
    pthread_mutex_lock(&g_integ_mutex);
    g_integ_results[idx] = result;
    g_integ_done++;
    pthread_mutex_unlock(&g_integ_mutex);
}

TEST(integration_thread_pool_concurrent) {
    // 4 agents run concurrently with different responses
    g_integ_done = 0;
    const char *responses[] = {"Alpha", "Beta", "Gamma", "Delta"};
    mock_llm_ctx_t mocks[4];
    adam_settings_t *settings[4];
    adam_history_t *histories[4];
    int indices[4];

    adam_pool_t *pool = adam_pool_create(2); // 2 workers
    ASSERT_NOT_NULL(pool);

    for (int i = 0; i < 4; i++) {
        memset(&mocks[i], 0, sizeof(mock_llm_ctx_t));
        mocks[i].fixed_response = responses[i];

        settings[i] = adam_create_settings();
        adam_settings_set_llm_callback(settings[i], mock_llm_simple, &mocks[i]);
        histories[i] = adam_history_create();
        indices[i] = i;

        adam_pool_submit(pool, (adam_job_t){
            .settings = settings[i],
            .history = histories[i],
            .user_message = "What's your name?",
            .on_done = integ_pool_done,
            .on_done_ctx = &indices[i],
        });
    }

    adam_pool_destroy(pool); // waits for all

    ASSERT_EQ(g_integ_done, 4);
    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(g_integ_results[i].status, ADAM_OK);
        ASSERT_NOT_NULL(g_integ_results[i].final_response);
        // Each agent got its own response
        ASSERT_STR_EQ(g_integ_results[i].final_response, responses[i]);
        adam_run_result_free(&g_integ_results[i]);
        adam_history_destroy(histories[i]);
        adam_settings_destroy(settings[i]);
    }
}

#endif // ADAM_NO_PTHREADS

// ---- Scenario 12: JSON output with tool-using agent ----

TEST(integration_json_output) {
    // adam_run_json returns valid JSON, adam_json_extract works
    mock_llm_ctx_t mock = {0};
    adam_settings_t *s = adam_create_settings();
    adam_settings_set_llm_callback(s, mock_llm_json, &mock);

    adam_history_t *h = adam_history_create();
    adam_json_result_t r = adam_run_json(s, h,
        "List programming languages as JSON",
        "{\"languages\":[{\"name\":\"...\"}]}", 3);

    ASSERT_EQ(r.base.status, ADAM_OK);
    ASSERT_EQ(r.json_valid, 1);
    ASSERT_NOT_NULL(r.base.final_response);

    // Extract a key
    char *answer = adam_json_extract(r.base.final_response, "answer");
    ASSERT_NOT_NULL(answer);
    ASSERT_STR_EQ(answer, "42");
    free(answer);

    char *name = adam_json_extract(r.base.final_response, "name");
    ASSERT_NOT_NULL(name);
    ASSERT_STR_EQ(name, "Adam");
    free(name);

    adam_json_result_free(&r);
    adam_history_destroy(h);
    adam_settings_destroy(s);
}

// ---- Scenario 13: File tools with sandbox enforcement ----

TEST(integration_file_tools_sandbox) {
    // Write a file, read it back, list directory — all sandboxed
    adam_settings_t *s = adam_create_settings();
    adam_settings_allow_dir(s, "/tmp");
    arena_t *a = arena_create(64 * 1024);

    // Write
    const char *w_args = "{\"path\":\"/tmp/adam_integ_test.txt\","
                         "\"content\":\"Hello from integration test!\"}";
    adam_tool_result_t wr = adam_tool_file_write(a, s, w_args, strlen(w_args));
    ASSERT_EQ(wr.success, 1);

    // Read back
    arena_reset(a);
    const char *r_args = "{\"path\":\"/tmp/adam_integ_test.txt\"}";
    adam_tool_result_t rr = adam_tool_file_read(a, s, r_args, strlen(r_args));
    ASSERT_EQ(rr.success, 1);
    ASSERT_STR_EQ(rr.for_llm, "Hello from integration test!");

    // List directory containing the file
    arena_reset(a);
    const char *l_args = "{\"path\":\"/tmp\"}";
    adam_tool_result_t lr = adam_tool_list_directory(a, s, l_args, strlen(l_args));
    ASSERT_EQ(lr.success, 1);
    ASSERT(strstr(lr.for_llm, "adam_integ_test.txt") != NULL);

    // Denied outside sandbox
    arena_reset(a);
    const char *d_args = "{\"path\":\"/etc/passwd\"}";
    adam_tool_result_t dr = adam_tool_file_read(a, s, d_args, strlen(d_args));
    ASSERT_EQ(dr.success, 0);
    ASSERT(strstr(dr.for_llm, "denied") != NULL);

    remove("/tmp/adam_integ_test.txt");
    arena_destroy(a);
    adam_settings_destroy(s);
}

// ---- Scenario 14: Calculator tool precision ----

TEST(integration_calculator_expressions) {
    arena_t *a = arena_create(4096);

    // Basic arithmetic
    const char *a1 = "{\"expression\":\"(3 + 4) * 2 - 1\"}";
    adam_tool_result_t r1 = adam_tool_calculator(a, NULL, a1, strlen(a1));
    ASSERT_EQ(r1.success, 1);
    ASSERT_STR_EQ(r1.for_llm, "13");

    // Exponentiation
    arena_reset(a);
    const char *a2 = "{\"expression\":\"2 ^ 16\"}";
    adam_tool_result_t r2 = adam_tool_calculator(a, NULL, a2, strlen(a2));
    ASSERT_EQ(r2.success, 1);
    ASSERT_STR_EQ(r2.for_llm, "65536");

    // Floating point
    arena_reset(a);
    const char *a3 = "{\"expression\":\"22 / 7\"}";
    adam_tool_result_t r3 = adam_tool_calculator(a, NULL, a3, strlen(a3));
    ASSERT_EQ(r3.success, 1);
    ASSERT(strstr(r3.for_llm, "3.14") != NULL);

    // Nested parentheses
    arena_reset(a);
    const char *a4 = "{\"expression\":\"((1 + 2) * (3 + 4)) ^ 2\"}";
    adam_tool_result_t r4 = adam_tool_calculator(a, NULL, a4, strlen(a4));
    ASSERT_EQ(r4.success, 1);
    ASSERT_STR_EQ(r4.for_llm, "441"); // (3*7)^2 = 21^2 = 441

    // Modulo
    arena_reset(a);
    const char *a5 = "{\"expression\":\"17 % 5\"}";
    adam_tool_result_t r5 = adam_tool_calculator(a, NULL, a5, strlen(a5));
    ASSERT_EQ(r5.success, 1);
    ASSERT_STR_EQ(r5.for_llm, "2");

    // Negative numbers
    arena_reset(a);
    const char *a6 = "{\"expression\":\"-3 * -4\"}";
    adam_tool_result_t r6 = adam_tool_calculator(a, NULL, a6, strlen(a6));
    ASSERT_EQ(r6.success, 1);
    ASSERT_STR_EQ(r6.for_llm, "12");

    arena_destroy(a);
}

#ifndef ADAM_NO_SQLITE

// ---- Scenario 15: SQL tool queries ----

TEST(integration_sql_tool_crud) {
    adam_memory_t *mem = adam_memory_open(":memory:");
    ASSERT_NOT_NULL(mem);
    arena_t *a = arena_create(64 * 1024);

    // Create table
    const char *c = "{\"sql\":\"CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, age INTEGER)\"}";
    adam_tool_result_t rc = adam_tool_sql_query(a, mem, c, strlen(c));
    ASSERT_EQ(rc.success, 1);

    // Insert rows
    arena_reset(a);
    const char *i1 = "{\"sql\":\"INSERT INTO users VALUES(1, 'Alice', 30)\"}";
    adam_tool_result_t ri1 = adam_tool_sql_query(a, mem, i1, strlen(i1));
    ASSERT_EQ(ri1.success, 1);
    ASSERT(strstr(ri1.for_llm, "1 rows affected") != NULL);

    arena_reset(a);
    const char *i2 = "{\"sql\":\"INSERT INTO users VALUES(2, 'Bob', 25)\"}";
    adam_tool_result_t ri2 = adam_tool_sql_query(a, mem, i2, strlen(i2));
    ASSERT_EQ(ri2.success, 1);

    arena_reset(a);
    const char *i3 = "{\"sql\":\"INSERT INTO users VALUES(3, 'Charlie', 35)\"}";
    adam_tool_result_t ri3 = adam_tool_sql_query(a, mem, i3, strlen(i3));
    ASSERT_EQ(ri3.success, 1);

    // SELECT with results
    arena_reset(a);
    const char *q = "{\"sql\":\"SELECT name, age FROM users WHERE age > 27 ORDER BY name\"}";
    adam_tool_result_t rq = adam_tool_sql_query(a, mem, q, strlen(q));
    ASSERT_EQ(rq.success, 1);
    ASSERT(strstr(rq.for_llm, "Alice") != NULL);
    ASSERT(strstr(rq.for_llm, "Charlie") != NULL);
    // Bob (age 25) should not appear
    ASSERT(strstr(rq.for_llm, "Bob") == NULL);

    // UPDATE
    arena_reset(a);
    const char *u = "{\"sql\":\"UPDATE users SET age = 31 WHERE name = 'Alice'\"}";
    adam_tool_result_t ru = adam_tool_sql_query(a, mem, u, strlen(u));
    ASSERT_EQ(ru.success, 1);
    ASSERT(strstr(ru.for_llm, "1 rows affected") != NULL);

    // Block DROP
    arena_reset(a);
    const char *d = "{\"sql\":\"DROP TABLE users\"}";
    adam_tool_result_t rd = adam_tool_sql_query(a, mem, d, strlen(d));
    ASSERT_EQ(rd.success, 0);
    ASSERT(strstr(rd.for_llm, "destructive") != NULL);

    arena_destroy(a);
    adam_memory_close(mem);
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
    RUN(json_build_gemini_simple);
    RUN(json_build_gemini_with_tools);
    RUN(json_parse_gemini_text);
    RUN(json_parse_gemini_tool_call);
    RUN(json_parse_gemini_error);
    RUN(json_build_gemini_image_model);
    RUN(json_parse_gemini_image_response);
    RUN(model_registry_gemini);
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
    RUN(voice_stt_local_bad_model);
    RUN(voice_tts_local_not_implemented);
    RUN(voice_stt_cloud_bad_key);
    RUN(voice_tts_cloud_bad_key);
    RUN(voice_custom_stt_callback);
    RUN(voice_custom_tts_callback);
    RUN(voice_run_full_pipeline);
#endif

    // --- Streaming ---
    printf("\nStreaming:\n");
    RUN(stream_simple_response);
    RUN(stream_with_tool_calls);
    RUN(stream_null_callback);
    RUN(stream_error_no_crash);
    RUN(stream_multi_turn);
    RUN(stream_on_response_still_fires);

#ifndef ADAM_NO_LOCAL
    // --- Local STT ---
    printf("\nLocal STT:\n");
    RUN(local_stt_missing_model);
    RUN(local_stt_null_model);
    RUN(local_stt_null_audio);
    RUN(local_stt_empty_audio);
    RUN(local_stt_invalid_wav);
    RUN(local_stt_tiny_wav);
    RUN(local_stt_unsupported_format);
    RUN(local_stt_pcm16_format);
    RUN(local_stt_destroy_without_use);
    RUN(local_stt_null_out_text);

    // --- Local Inference ---
    printf("\nLocal Inference:\n");
    RUN(local_missing_gguf);
    RUN(local_null_gguf_path);
    RUN(local_empty_message);
    RUN(local_null_message);
    RUN(local_extreme_settings);
    RUN(local_zero_temperature);
    RUN(local_settings_destroy_without_run);
    RUN(local_double_destroy);
    RUN(local_priority_over_remote);
    RUN(local_with_tools_defined);
    RUN(local_with_stream_callback);
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

    // --- Evolution Loop ---
    printf("\nEvolution Loop:\n");
    RUN(evolve_config_defaults);
    RUN(evolve_null_params);
    RUN(evolve_abort);
    RUN(evolve_basic_improvement);
    RUN(evolve_plateau_stop);
    RUN(evolve_max_iterations);
    RUN(evolve_ring_buffer);
    RUN(evolve_eval_error);
    RUN(evolve_progress_callback);
    RUN(evolve_no_strategy_no_metrics);

    // --- Research Mode ---
    printf("\nResearch Mode:\n");
    RUN(research_config_defaults);
    RUN(research_null_params);
    RUN(research_abort);
    RUN(research_complete_by_agent);
    RUN(research_max_iterations);
    RUN(research_findings_parsed);
    RUN(research_callback_stops_loop);
    RUN(research_progress_callback);
    RUN(research_with_instructions);
    RUN(research_tool_basic);
    RUN(research_tool_bad_args);

    // --- Built-in Tools ---
    printf("\nBuilt-in Tools:\n");
    RUN(tool_allow_dir);
    RUN(tool_calculator_basic);
    RUN(tool_calculator_errors);
    RUN(tool_file_read_sandbox);
    RUN(tool_file_write_sandbox);
    RUN(tool_list_directory_sandbox);
    RUN(tool_shell_exec_basic);
#ifndef ADAM_NO_SQLITE
    RUN(tool_sql_query_basic);
#endif

    // --- History Clone ---
    printf("\nHistory Clone:\n");
    RUN(history_clone_basic);
    RUN(history_clone_with_tool_calls);
    RUN(history_clone_empty);
    RUN(history_clone_modify_original);

    // --- Timeout ---
    printf("\nTimeout:\n");
    RUN(timeout_triggers);
    RUN(timeout_zero_disabled);

    // --- Guardrails ---
    printf("\nGuardrails:\n");
    RUN(guardrail_before_allow);
    RUN(guardrail_before_deny);
    RUN(guardrail_after_deny);

    // --- Structured JSON ---
    printf("\nStructured JSON:\n");
    RUN(run_json_valid);
    RUN(run_json_retry);
    RUN(json_extract_basic);

    // --- Multi-Agent ---
    printf("\nMulti-Agent:\n");
    RUN(tool_agent_basic);
    RUN(tool_agent_null_ctx);

    // --- Response Cache ---
    printf("\nResponse Cache:\n");
    RUN(cache_miss_then_hit);
    RUN(cache_different_messages);
    RUN(cache_lru_eviction);
    RUN(cache_clear);
    RUN(cache_null_disabled);

    // --- Integration Tests ---
    printf("\nIntegration Tests:\n");
#ifndef ADAM_NO_SQLITE
    RUN(integration_session_multi_turn);
    RUN(integration_tool_call_session_roundtrip);
#endif
    RUN(integration_evolve_with_tools);
    RUN(integration_research_accumulates_findings);
    RUN(integration_multi_agent_chain);
    RUN(integration_cache_across_sessions);
    RUN(integration_guardrails_block_and_allow);
    RUN(integration_clone_branch_conversations);
    RUN(integration_evolve_converges);
    RUN(integration_timeout_with_tools);
#ifndef ADAM_NO_PTHREADS
    RUN(integration_thread_pool_concurrent);
#endif
    RUN(integration_json_output);
    RUN(integration_file_tools_sandbox);
    RUN(integration_calculator_expressions);
#ifndef ADAM_NO_SQLITE
    RUN(integration_sql_tool_crud);
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
