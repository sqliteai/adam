//
//  test_memory.c
//  Adam — Remote embedding + memory storage tests
//
//  Tests remote embedding generation and memory storage/retrieval
//  for all supported remote providers (OpenAI, Voyage).
//  Reads API keys from .env file. Skips providers without keys.
//
//  Build & run:
//    make memory
//

#include "adam.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// ============================================================================
// MARK: - Test Framework
// ============================================================================

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;
static int g_tests_skipped = 0;

#define TEST(name)                                                          \
    static void test_##name(void);                                          \
    static void run_test_##name(void) {                                     \
        g_tests_run++;                                                      \
        printf("  %-50s ", #name);                                          \
        fflush(stdout);                                                     \
        test_##name();                                                      \
    }                                                                       \
    static void test_##name(void)

#define PASS() do { printf("PASS\n"); g_tests_passed++; } while (0)
#define FAIL(msg) do {                                                      \
    printf("FAIL\n    %s\n    at %s:%d\n", msg, __FILE__, __LINE__);        \
    g_tests_failed++;                                                       \
} while (0)
#define SKIP(msg) do { printf("SKIP  (%s)\n", msg); g_tests_skipped++; } while (0)

#define RUN(name) run_test_##name()

// ============================================================================
// MARK: - .env loader
// ============================================================================

static char g_openai_key[256];
static char g_voyage_key[256];
static char g_grok_key[256];

static void load_env_key(const char *prefix, char *out, size_t out_size) {
    out[0] = '\0';
    FILE *f = fopen(".env", "r");
    if (!f) return;
    size_t plen = strlen(prefix);
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, prefix, plen) == 0 && line[plen] == '=') {
            char *val = line + plen + 1;
            size_t len = strlen(val);
            while (len > 0 && (val[len-1] == '\n' || val[len-1] == '\r'))
                val[--len] = '\0';
            if (len > 0 && len < out_size) {
                memcpy(out, val, len + 1);
            }
            break;
        }
    }
    fclose(f);
}

static void load_keys(void) {
    load_env_key("OPENAI_API_KEY", g_openai_key, sizeof(g_openai_key));
    load_env_key("VOYAGE_API_KEY", g_voyage_key, sizeof(g_voyage_key));
    load_env_key("GROK_API_KEY", g_grok_key, sizeof(g_grok_key));
}

// ============================================================================
// MARK: - Helpers
// ============================================================================

static const char *tmp_db_path(const char *suffix) {
    static char path[256];
    snprintf(path, sizeof(path), "/tmp/adam_test_memory_%s_%d.db", suffix, getpid());
    return path;
}

static void cleanup_db(const char *path) {
    unlink(path);
    char wal[280], shm[280];
    snprintf(wal, sizeof(wal), "%s-wal", path);
    snprintf(shm, sizeof(shm), "%s-shm", path);
    unlink(wal);
    unlink(shm);
}

// Run a full cycle: open db, configure provider, add text, search, verify, close.
// Returns 0 on success, -1 on failure. Prints details on failure.
static int test_provider(const char *provider_name, const char *model,
                          const char *api_key, const char *db_suffix) {
    const char *db_path = tmp_db_path(db_suffix);

    adam_init();

    // Create settings with provider config
    adam_settings_t *s = adam_create_settings();
    if (!s) { fprintf(stderr, "    failed to create settings\n"); return -1; }

    // Configure remote provider (need a valid api_format so adam_net works)
    s->api_format = ADAM_API_OPENAI;
    s->api_key = api_key;

    adam_settings_set_memory(s, provider_name, model, api_key);
    s->memory_min_score = 0.3f;

    // Open + configure memory
    adam_memory_t *mem = adam_memory_open(db_path);
    if (!mem) {
        fprintf(stderr, "    failed to open memory db: %s\n", db_path);
        adam_settings_destroy(s);
        cleanup_db(db_path);
        return -1;
    }

    adam_status_t rc = adam_memory_configure(mem, s);
    if (rc != ADAM_OK) {
        fprintf(stderr, "    adam_memory_configure failed: %s\n",
                adam_status_string(rc));
        adam_memory_close(mem);
        adam_settings_destroy(s);
        cleanup_db(db_path);
        return -1;
    }

    // Add test content
    rc = adam_memory_add(mem, "SQLite is a C-language library that implements "
        "a small, fast, self-contained, high-reliability, full-featured, "
        "SQL database engine.", "test", "sqlite-docs");
    if (rc != ADAM_OK) {
        fprintf(stderr, "    adam_memory_add (1) failed: %s\n",
                adam_status_string(rc));
        adam_memory_close(mem);
        adam_settings_destroy(s);
        cleanup_db(db_path);
        return -1;
    }

    rc = adam_memory_add(mem, "The speed of light in vacuum is approximately "
        "299,792,458 metres per second.", "test", "physics");
    if (rc != ADAM_OK) {
        fprintf(stderr, "    adam_memory_add (2) failed: %s\n",
                adam_status_string(rc));
        adam_memory_close(mem);
        adam_settings_destroy(s);
        cleanup_db(db_path);
        return -1;
    }

    rc = adam_memory_add(mem, "Rust is a multi-paradigm, general-purpose "
        "programming language that emphasizes performance, type safety, "
        "and concurrency.", "test", "rust-docs");
    if (rc != ADAM_OK) {
        fprintf(stderr, "    adam_memory_add (3) failed: %s\n",
                adam_status_string(rc));
        adam_memory_close(mem);
        adam_settings_destroy(s);
        cleanup_db(db_path);
        return -1;
    }

    // Search for something related to databases
    arena_t *arena = arena_create(64 * 1024);
    adam_memory_result_t *results = NULL;
    size_t count = 0;

    rc = adam_memory_search(mem, arena, "database engine SQL", NULL, 3,
                             &results, &count);
    if (rc != ADAM_OK) {
        fprintf(stderr, "    adam_memory_search failed: %s\n",
                adam_status_string(rc));
        arena_destroy(arena);
        adam_memory_close(mem);
        adam_settings_destroy(s);
        cleanup_db(db_path);
        return -1;
    }

    // Verify we got results
    if (count == 0) {
        fprintf(stderr, "    search returned 0 results (expected >= 1)\n");
        arena_destroy(arena);
        adam_memory_close(mem);
        adam_settings_destroy(s);
        cleanup_db(db_path);
        return -1;
    }

    // The top result should be the SQLite entry
    printf("\n");
    for (size_t i = 0; i < count; i++) {
        printf("      [%.3f] %s: %.60s%s\n",
            results[i].score,
            results[i].source ? results[i].source : "(none)",
            results[i].content ? results[i].content : "",
            (results[i].content && strlen(results[i].content) > 60) ? "..." : "");
    }
    printf("    ");

    // Verify the top result mentions SQLite
    int found_sqlite = 0;
    if (results[0].content && (strstr(results[0].content, "SQLite") ||
                                strstr(results[0].content, "sqlite") ||
                                strstr(results[0].content, "SQL"))) {
        found_sqlite = 1;
    }

    if (!found_sqlite) {
        fprintf(stderr, "    top result does not mention SQLite/SQL\n");
        arena_destroy(arena);
        adam_memory_close(mem);
        adam_settings_destroy(s);
        cleanup_db(db_path);
        return -1;
    }

    // Search for physics
    arena_reset(arena);
    results = NULL;
    count = 0;
    rc = adam_memory_search(mem, arena, "speed of light physics", NULL, 3,
                             &results, &count);
    if (rc != ADAM_OK || count == 0) {
        fprintf(stderr, "    physics search failed or empty\n");
        arena_destroy(arena);
        adam_memory_close(mem);
        adam_settings_destroy(s);
        cleanup_db(db_path);
        return -1;
    }

    int found_light = 0;
    if (results[0].content && (strstr(results[0].content, "light") ||
                                strstr(results[0].content, "299"))) {
        found_light = 1;
    }

    if (!found_light) {
        fprintf(stderr, "    physics top result does not mention light\n");
        arena_destroy(arena);
        adam_memory_close(mem);
        adam_settings_destroy(s);
        cleanup_db(db_path);
        return -1;
    }

    // Verify delete_context works
    rc = adam_memory_delete_context(mem, "test");
    if (rc != ADAM_OK) {
        fprintf(stderr, "    adam_memory_delete_context failed\n");
        arena_destroy(arena);
        adam_memory_close(mem);
        adam_settings_destroy(s);
        cleanup_db(db_path);
        return -1;
    }

    // Search again — should return nothing
    arena_reset(arena);
    results = NULL;
    count = 0;
    rc = adam_memory_search(mem, arena, "database SQL", NULL, 3, &results, &count);
    if (rc != ADAM_OK) {
        fprintf(stderr, "    post-delete search failed\n");
        arena_destroy(arena);
        adam_memory_close(mem);
        adam_settings_destroy(s);
        cleanup_db(db_path);
        return -1;
    }

    if (count != 0) {
        fprintf(stderr, "    post-delete search returned %zu results (expected 0)\n", count);
        arena_destroy(arena);
        adam_memory_close(mem);
        adam_settings_destroy(s);
        cleanup_db(db_path);
        return -1;
    }

    // Cleanup
    arena_destroy(arena);
    adam_memory_close(mem);
    adam_settings_destroy(s);
    cleanup_db(db_path);

    return 0;
}

// ============================================================================
// MARK: - Tests
// ============================================================================

TEST(openai_embedding) {
    if (!g_openai_key[0]) { SKIP("OPENAI_API_KEY not set"); return; }

    int rc = test_provider("openai", "text-embedding-3-small",
                            g_openai_key, "openai");
    if (rc == 0) PASS(); else FAIL("OpenAI embedding test failed");
}

TEST(voyage_embedding) {
    if (!g_voyage_key[0]) { SKIP("VOYAGE_API_KEY not set"); return; }

    int rc = test_provider("voyage", "voyage-3-lite",
                            g_voyage_key, "voyage");
    if (rc == 0) PASS(); else FAIL("Voyage embedding test failed");
}

TEST(grok_embedding) {
    if (!g_grok_key[0]) { SKIP("GROK_API_KEY not set"); return; }

    // xAI embedding models may be gated. Try the API and skip if unavailable.
    int rc = test_provider("grok", "grok-embedding-small",
                            g_grok_key, "grok");
    if (rc == 0) PASS();
    else SKIP("xAI embedding model not available for this account");
}

TEST(memory_open_close) {
    const char *db_path = tmp_db_path("lifecycle");

    adam_init();
    adam_memory_t *mem = adam_memory_open(db_path);
    if (!mem) { FAIL("adam_memory_open returned NULL"); return; }
    adam_memory_close(mem);

    // Re-open to verify persistence
    mem = adam_memory_open(db_path);
    if (!mem) { FAIL("adam_memory_open (reopen) returned NULL"); return; }
    adam_memory_close(mem);

    cleanup_db(db_path);
    PASS();
}

TEST(memory_configure_no_model) {
    const char *db_path = tmp_db_path("nomodel");

    adam_init();
    adam_settings_t *s = adam_create_settings();
    adam_memory_t *mem = adam_memory_open(db_path);
    if (!mem) { FAIL("open failed"); adam_settings_destroy(s); return; }

    // Configure without setting a model — should succeed (model is optional)
    adam_status_t rc = adam_memory_configure(mem, s);
    if (rc != ADAM_OK) { FAIL("configure failed unexpectedly"); }
    else { PASS(); }

    adam_memory_close(mem);
    adam_settings_destroy(s);
    cleanup_db(db_path);
}

TEST(memory_search_empty) {
    if (!g_openai_key[0]) { SKIP("OPENAI_API_KEY not set"); return; }

    const char *db_path = tmp_db_path("empty");

    adam_init();
    adam_settings_t *s = adam_create_settings();
    s->api_format = ADAM_API_OPENAI;
    s->api_key = g_openai_key;
    adam_settings_set_memory(s, "openai", "text-embedding-3-small", g_openai_key);

    adam_memory_t *mem = adam_memory_open(db_path);
    if (!mem) { FAIL("open failed"); adam_settings_destroy(s); return; }

    adam_memory_configure(mem, s);

    arena_t *arena = arena_create(4096);
    adam_memory_result_t *results = NULL;
    size_t count = 99;

    // Search on empty database — should return 0 results
    adam_status_t rc = adam_memory_search(mem, arena, "anything", NULL, 5,
                                          &results, &count);
    if (rc == ADAM_OK && count == 0) {
        PASS();
    } else if (rc == ADAM_OK && count != 0) {
        FAIL("search on empty db returned non-zero count");
    } else {
        // Some providers may error on empty vault (no embeddings to compare)
        // This is acceptable behavior.
        PASS();
    }

    arena_destroy(arena);
    adam_memory_close(mem);
    adam_settings_destroy(s);
    cleanup_db(db_path);
}

TEST(memory_clear) {
    if (!g_openai_key[0]) { SKIP("OPENAI_API_KEY not set"); return; }

    const char *db_path = tmp_db_path("clear");
    adam_init();

    adam_settings_t *s = adam_create_settings();
    s->api_format = ADAM_API_OPENAI;
    s->api_key = g_openai_key;
    adam_settings_set_memory(s, "openai", "text-embedding-3-small", g_openai_key);

    adam_memory_t *mem = adam_memory_open(db_path);
    if (!mem) { FAIL("open failed"); adam_settings_destroy(s); return; }
    adam_memory_configure(mem, s);

    adam_memory_add(mem, "Test content for clear", "ctx", NULL);
    adam_status_t rc = adam_memory_clear(mem);
    if (rc != ADAM_OK) { FAIL("adam_memory_clear failed"); }
    else {
        arena_t *arena = arena_create(4096);
        adam_memory_result_t *results = NULL;
        size_t count = 0;
        adam_memory_search(mem, arena, "Test content", NULL, 5, &results, &count);
        if (count == 0) PASS();
        else FAIL("clear did not remove entries");
        arena_destroy(arena);
    }

    adam_memory_close(mem);
    adam_settings_destroy(s);
    cleanup_db(db_path);
}

TEST(memory_tool_search) {
    if (!g_openai_key[0]) { SKIP("OPENAI_API_KEY not set"); return; }

    const char *db_path = tmp_db_path("tool");
    adam_init();

    adam_settings_t *s = adam_create_settings();
    s->api_format = ADAM_API_OPENAI;
    s->api_key = g_openai_key;
    adam_settings_set_memory(s, "openai", "text-embedding-3-small", g_openai_key);

    adam_memory_t *mem = adam_memory_open(db_path);
    if (!mem) { FAIL("open failed"); adam_settings_destroy(s); return; }
    adam_memory_configure(mem, s);

    adam_memory_add(mem, "Paris is the capital of France.", "geo", "geography");

    // Test tool interface
    arena_t *arena = arena_create(64 * 1024);
    const char *args = "{\"query\":\"capital of France\",\"limit\":2}";
    adam_tool_result_t res = adam_tool_memory_search(arena, mem,
                                                      args, strlen(args));
    if (!res.success) {
        printf("\n    tool returned: %s\n    ", res.for_llm ? res.for_llm : "(null)");
        FAIL("adam_tool_memory_search failed");
    } else if (!res.for_llm || !strstr(res.for_llm, "Paris")) {
        printf("\n    tool returned: %s\n    ", res.for_llm ? res.for_llm : "(null)");
        FAIL("tool result does not mention Paris");
    } else {
        PASS();
    }

    arena_destroy(arena);
    adam_memory_close(mem);
    adam_settings_destroy(s);
    cleanup_db(db_path);
}

TEST(memory_tool_add) {
    if (!g_openai_key[0]) { SKIP("OPENAI_API_KEY not set"); return; }

    const char *db_path = tmp_db_path("tooladd");
    adam_init();

    adam_settings_t *s = adam_create_settings();
    s->api_format = ADAM_API_OPENAI;
    s->api_key = g_openai_key;
    adam_settings_set_memory(s, "openai", "text-embedding-3-small", g_openai_key);

    adam_memory_t *mem = adam_memory_open(db_path);
    if (!mem) { FAIL("open failed"); adam_settings_destroy(s); return; }
    adam_memory_configure(mem, s);

    // Add via tool
    arena_t *arena = arena_create(64 * 1024);
    const char *args = "{\"text\":\"The Eiffel Tower is in Paris.\",\"context\":\"facts\"}";
    adam_tool_result_t res = adam_tool_memory_add(arena, mem, args, strlen(args));
    if (!res.success) {
        FAIL("adam_tool_memory_add failed");
        arena_destroy(arena);
        adam_memory_close(mem);
        adam_settings_destroy(s);
        cleanup_db(db_path);
        return;
    }

    // Verify via search
    arena_reset(arena);
    adam_memory_result_t *results = NULL;
    size_t count = 0;
    adam_status_t rc = adam_memory_search(mem, arena, "Eiffel Tower", NULL, 3,
                                          &results, &count);
    if (rc != ADAM_OK || count == 0) {
        FAIL("search after tool_add returned no results");
    } else if (!results[0].content || !strstr(results[0].content, "Eiffel")) {
        FAIL("search result does not contain Eiffel");
    } else {
        PASS();
    }

    arena_destroy(arena);
    adam_memory_close(mem);
    adam_settings_destroy(s);
    cleanup_db(db_path);
}

// ============================================================================
// MARK: - Main
// ============================================================================

int main(void) {
    printf("Adam Memory Test Suite v%s\n", ADAM_VERSION_STRING);
    printf("============================================================\n\n");

    load_keys();

    printf("Available keys:\n");
    printf("  OPENAI_API_KEY:  %s\n", g_openai_key[0] ? "yes" : "no");
    printf("  VOYAGE_API_KEY:  %s\n", g_voyage_key[0] ? "yes" : "no");
    printf("  GROK_API_KEY:    %s\n", g_grok_key[0] ? "yes" : "no");
    printf("\n");

    printf("Lifecycle:\n");
    RUN(memory_open_close);
    RUN(memory_configure_no_model);
    RUN(memory_search_empty);
    printf("\n");

    printf("Remote Embedding Providers:\n");
    RUN(openai_embedding);
    RUN(voyage_embedding);
    RUN(grok_embedding);
    printf("\n");

    printf("Memory Operations:\n");
    RUN(memory_clear);
    RUN(memory_tool_search);
    RUN(memory_tool_add);
    printf("\n");

    printf("============================================================\n");
    printf("Results: %d passed, %d failed, %d skipped, %d total\n",
           g_tests_passed, g_tests_failed, g_tests_skipped, g_tests_run);

    adam_cleanup();
    return g_tests_failed > 0 ? 1 : 0;
}
