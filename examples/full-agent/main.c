//
//  full-agent/main.c
//  All Adam features combined in a single agent
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

// --- Guardrails ---

static int check_input(void *ctx, const adam_message_t *msgs, size_t msg_count) {
    (void)ctx;
    for (size_t i = 0; i < msg_count; i++) {
        if (msgs[i].role == ADAM_ROLE_USER && msgs[i].content &&
            strstr(msgs[i].content, "DROP TABLE") != NULL) {
            return 1;  // block SQL injection attempts
        }
    }
    return 0;
}

static int check_output(void *ctx, const char *content,
                         const adam_tool_call_t *tool_calls, size_t tool_call_count) {
    (void)ctx;
    (void)tool_calls;
    (void)tool_call_count;
    if (content && strstr(content, "SECRET") != NULL)
        return 1;  // block responses leaking secrets
    return 0;
}

// --- Streaming callback ---

static void on_stream(void *ctx, const char *chunk, size_t len, int is_done) {
    (void)ctx;
    if (len > 0)
        fwrite(chunk, 1, len, stdout);
    if (is_done)
        printf("\n");
}

// --- Logging callback ---

static void on_log(void *ctx, adam_log_level_t level, const char *msg, size_t len) {
    (void)ctx;
    const char *labels[] = {"ERROR", "WARN", "INFO", "DEBUG", "TRACE"};
    fprintf(stderr, "[%s] %.*s\n", labels[level], (int)len, msg);
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

    // --- Settings ---
    adam_settings_t *settings = adam_create_settings();

    // Provider
    adam_settings_set_provider(settings, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");

    // Identity and instructions
    adam_settings_set_identity(settings, "You are Atlas, a capable AI assistant with access to files, "
                              "a calculator, memory, and a SQL database.");
    adam_settings_set_instructions(settings, "Be concise and precise. Use tools when they help answer "
                                  "the user's question. Always verify file operations succeeded.");

    // --- Memory ---
    adam_memory_t *memory = adam_memory_open("/tmp/adam_full_agent.db");
    if (!memory) {
        fprintf(stderr, "Failed to open memory database\n");
        adam_settings_destroy(settings);
        free(api_key);
        adam_cleanup();
        return 1;
    }
    settings->memory = memory;

    // --- Session ---
    char session_id[64];
    adam_session_create(memory, session_id, sizeof(session_id));
    settings->session_id = session_id;
    settings->auto_save  = 1;

    // --- Response cache ---
    adam_cache_t *cache = adam_cache_create(128);
    settings->cache = cache;

    // --- Guardrails ---
    settings->on_before_send    = check_input;
    settings->before_send_ctx   = NULL;
    settings->on_after_receive  = check_output;
    settings->after_receive_ctx = NULL;

    // --- Filesystem sandbox ---
    adam_settings_allow_dir(settings, "/tmp");

    // --- Streaming ---
    adam_settings_set_stream(settings, on_stream, NULL);

    // --- Logging ---
    adam_settings_set_logger(settings, on_log, NULL, ADAM_LOG_INFO);

    // --- Register tools ---

    adam_settings_add_tool(settings, (adam_tool_def_t){
        .name            = "file_read",
        .description     = "Read a file",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}",
        .execute         = adam_tool_file_read,
        .ctx             = settings
    });

    adam_settings_add_tool(settings, (adam_tool_def_t){
        .name            = "file_write",
        .description     = "Write a file",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"content\"]}",
        .execute         = adam_tool_file_write,
        .ctx             = settings
    });

    adam_settings_add_tool(settings, (adam_tool_def_t){
        .name            = "shell_exec",
        .description     = "Run a shell command",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\"}},\"required\":[\"command\"]}",
        .execute         = adam_tool_shell_exec,
        .ctx             = settings
    });

    adam_settings_add_tool(settings, (adam_tool_def_t){
        .name            = "calculator",
        .description     = "Evaluate a math expression",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"expression\":{\"type\":\"string\"}},\"required\":[\"expression\"]}",
        .execute         = adam_tool_calculator,
        .ctx             = NULL
    });

    adam_settings_add_tool(settings, (adam_tool_def_t){
        .name            = "memory_search",
        .description     = "Search long-term memory",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"},\"limit\":{\"type\":\"integer\"}},\"required\":[\"query\"]}",
        .execute         = adam_tool_memory_search,
        .ctx             = memory
    });

    adam_settings_add_tool(settings, (adam_tool_def_t){
        .name            = "sql_query",
        .description     = "Execute a SQL query against the database",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"sql\":{\"type\":\"string\"}},\"required\":[\"sql\"]}",
        .execute         = adam_tool_sql_query,
        .ctx             = memory
    });

    // --- Run ---
    adam_history_t *history = adam_history_create();

    printf("User: Write 'Hello from the full agent!' to /tmp/full_agent_test.txt, "
           "then read it back and calculate 42 * 17.\n\n");
    printf("Assistant: ");

    adam_run_result_t result = adam_run(settings, history,
        "Write 'Hello from the full agent!' to /tmp/full_agent_test.txt, "
        "then read it back and calculate 42 * 17.");

    if (result.status != ADAM_OK) {
        fprintf(stderr, "\nRun failed: %s\n", adam_status_string(result.status));
    } else {
        printf("\n  [%d iterations, %d in / %d out tokens, $%.4f, %.1f ms]\n",
               result.total_iterations, result.input_tokens,
               result.output_tokens, result.cost_usd, result.elapsed_ms);
        printf("  Session: %s\n", session_id);
        printf("  Cache: %zu hits, %zu misses\n",
               adam_cache_hits(cache), adam_cache_misses(cache));
    }

    // --- Cleanup ---
    adam_run_result_free(&result);
    adam_history_destroy(history);
    adam_cache_destroy(cache);
    adam_memory_close(memory);
    adam_settings_destroy(settings);
    free(api_key);
    adam_cleanup();
    return 0;
}
