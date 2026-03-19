//
//  sqlite-query/main.c
//  Natural language queries against any SQLite database
//

#include "adam.h"
#ifndef ADAM_NO_SQLITE
#include "sqlite3.h"
#endif
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

// Stream callback: prints each token as it arrives
static void on_stream(void *ctx, const char *chunk, size_t len, int is_done) {
    (void)ctx;
    if (len > 0) {
        fwrite(chunk, 1, len, stdout);
        fflush(stdout);
    }
    if (is_done)
        printf("\n");
}

#ifndef ADAM_NO_SQLITE

extern sqlite3 *adam_memory_db(adam_memory_t *mem);

// Instructions string — must outlive settings (stores pointer, not copy)
static char *g_instructions = NULL;

// Read the database schema and row counts, returning a malloc'd string.
static char *read_db_schema(adam_memory_t *db_mem) {
    sqlite3 *db = adam_memory_db(db_mem);
    if (!db) return NULL;

    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;

    #define SCHEMA_APPEND(...) do { \
        int n = snprintf(buf + len, cap - len, __VA_ARGS__); \
        if (n > 0) { \
            if (len + (size_t)n >= cap) { \
                cap *= 2; \
                char *nb = realloc(buf, cap); \
                if (!nb) { free(buf); return NULL; } \
                buf = nb; \
                snprintf(buf + len, cap - len, __VA_ARGS__); \
            } \
            len += (size_t)n; \
        } \
    } while(0)

    // Query sqlite_master for all schema objects (exclude internal adam tables)
    sqlite3_stmt *vm = NULL;
    int rc = sqlite3_prepare_v2(db,
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE sql IS NOT NULL "
        "AND name NOT LIKE 'dbmem_%' "
        "AND name NOT LIKE '_sqliteai_%' "
        "AND name NOT LIKE 'sqlite_%' "
        "AND name NOT IN ('sessions', 'messages') "
        "ORDER BY type, name", -1, &vm, NULL);
    if (rc != SQLITE_OK) { free(buf); return NULL; }

    const char *prev_type = "";
    while (sqlite3_step(vm) == SQLITE_ROW) {
        const char *type = (const char *)sqlite3_column_text(vm, 0);
        const char *sql  = (const char *)sqlite3_column_text(vm, 2);
        if (!type || !sql) continue;

        if (strcmp(type, prev_type) != 0) {
            SCHEMA_APPEND("\n-- %ss:\n", type);
            prev_type = type;
        }
        SCHEMA_APPEND("%s;\n", sql);
    }
    sqlite3_finalize(vm);

    // Add row counts for each user table
    rc = sqlite3_prepare_v2(db,
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' "
        "AND name NOT LIKE 'dbmem_%' "
        "AND name NOT LIKE '_sqliteai_%' "
        "AND name NOT IN ('sessions', 'messages') "
        "ORDER BY name", -1, &vm, NULL);
    if (rc == SQLITE_OK) {
        SCHEMA_APPEND("\n-- Row counts:\n");
        while (sqlite3_step(vm) == SQLITE_ROW) {
            const char *tname = (const char *)sqlite3_column_text(vm, 0);
            if (!tname) continue;
            char count_sql[256];
            snprintf(count_sql, sizeof(count_sql),
                     "SELECT COUNT(*) FROM \"%s\"", tname);
            sqlite3_stmt *cnt;
            if (sqlite3_prepare_v2(db, count_sql, -1, &cnt, NULL) == SQLITE_OK) {
                if (sqlite3_step(cnt) == SQLITE_ROW)
                    SCHEMA_APPEND("--   %s: %d rows\n",
                                  tname, sqlite3_column_int(cnt, 0));
                sqlite3_finalize(cnt);
            }
        }
        sqlite3_finalize(vm);
    }

    #undef SCHEMA_APPEND
    buf[len] = '\0';
    return buf;
}

#endif // ADAM_NO_SQLITE

int main(int argc, char *argv[]) {
#ifdef ADAM_NO_SQLITE
    (void)argc; (void)argv;
    fprintf(stderr, "This example requires SQLite support (built without ADAM_NO_SQLITE).\n");
    return 1;
#else
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <database.db>\n", argv[0]);
        return 1;
    }
    const char *db_path = argv[1];

    // --- Init ---
    adam_status_t status = adam_init();
    if (status != ADAM_OK) {
        fprintf(stderr, "adam_init failed: %s\n", adam_status_string(status));
        return 1;
    }

    // --- Auto-detect provider from .env ---
    char *api_key = NULL;
    adam_api_format_t api_format = ADAM_API_NONE;
    const char *model = NULL;

    api_key = env_load("ANTHROPIC_API_KEY");
    if (api_key) {
        api_format = ADAM_API_ANTHROPIC;
        model = "claude-sonnet-4-20250514";
    }
    if (!api_key) {
        api_key = env_load("GEMINI_API_KEY");
        if (api_key) {
            api_format = ADAM_API_GEMINI;
            model = "gemini-2.5-flash";
        }
    }
    if (!api_key) {
        api_key = env_load("OPENAI_API_KEY");
        if (api_key) {
            api_format = ADAM_API_OPENAI;
            model = "gpt-4o";
        }
    }
    if (!api_key) {
        fprintf(stderr, "No API key found. Set ANTHROPIC_API_KEY, GEMINI_API_KEY, "
                        "or OPENAI_API_KEY in .env\n");
        adam_cleanup();
        return 1;
    }

    // --- Settings ---
    adam_settings_t *settings = adam_create_settings();
    adam_settings_set_provider(settings, api_format, api_key, model);

    // --- Open the database ---
    adam_memory_t *db_mem = adam_memory_open(db_path);
    if (!db_mem) {
        fprintf(stderr, "Failed to open database: %s\n", db_path);
        adam_settings_destroy(settings);
        free(api_key);
        adam_cleanup();
        return 1;
    }

    // --- Read schema and inject as instructions ---
    char *schema = read_db_schema(db_mem);
    if (schema) {
        size_t instr_len = strlen(schema) + 256;
        g_instructions = malloc(instr_len);
        if (g_instructions) {
            snprintf(g_instructions, instr_len,
                "You have access to a SQLite database. Use the sql_query tool "
                "to query it. Here is the database schema:\n%s", schema);
            adam_settings_set_instructions(settings, g_instructions);
        }
    }

    // --- Register sql_query tool ---
    adam_settings_add_tool(settings, (adam_tool_def_t){
        .name = "sql_query",
        .description = "Execute a SQL query against the connected SQLite database",
        .parameters_json = "{\"type\":\"object\",\"properties\":"
                           "{\"sql\":{\"type\":\"string\",\"description\":\"SQL query to execute\"}},"
                           "\"required\":[\"sql\"]}",
        .execute = adam_tool_sql_query,
        .ctx = db_mem
    });

    // --- Enable streaming ---
    adam_settings_set_stream(settings, on_stream, NULL);

    // --- Print banner ---
    printf("SQLite Query Agent\n");
    printf("Database: %s\n", db_path);
    if (schema) {
        printf("\nSchema:\n%s\n", schema);
        free(schema);
    } else {
        printf("\n(no user tables found)\n\n");
    }
    printf("Ask questions about your data in natural language. Type 'quit' to exit.\n\n");

    // --- Interactive chat loop ---
    adam_history_t *history = adam_history_create();
    char input[4096];

    for (;;) {
        printf("You: ");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin))
            break;

        // Strip trailing newline
        size_t ilen = strlen(input);
        while (ilen > 0 && (input[ilen-1] == '\n' || input[ilen-1] == '\r'))
            input[--ilen] = '\0';

        if (ilen == 0)
            continue;
        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0)
            break;

        printf("\nAssistant: ");
        fflush(stdout);

        adam_run_result_t result = adam_run(settings, history, input);

        if (result.status != ADAM_OK) {
            printf("\nError: %s\n", adam_status_string(result.status));
        } else {
            // Streaming callback already printed the response; show stats
            printf("  [%d iterations, %d in / %d out tokens, %.1f ms]\n",
                   result.total_iterations,
                   result.input_tokens, result.output_tokens,
                   result.elapsed_ms);
        }
        printf("\n");

        adam_run_result_free(&result);
    }

    // --- Cleanup ---
    adam_history_destroy(history);
    adam_memory_close(db_mem);
    adam_settings_destroy(settings);
    free(g_instructions);
    free(api_key);
    adam_cleanup();
    return 0;
#endif // ADAM_NO_SQLITE
}
