//
//  adam_ext_chat.c
//  Core chat functions — adam(), adam_ask(), adam_sql()
//

#include "adam_ext.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// MARK: - SQL tool for the agent (runs on the same database)
// ============================================================================

static adam_tool_result_t ext_sql_tool(arena_t *arena, void *tool_ctx,
                                        const char *args_json, size_t args_len) {
    adam_ext_ctx_t *ctx = (adam_ext_ctx_t *)tool_ctx;
    adam_tool_result_t res = { .for_llm = NULL, .for_user = NULL, .success = 0 };

    if (!args_json) {
        res.for_llm = arena_strdup(arena, "Error: no arguments");
        return res;
    }

    // Extract "sql" field from JSON (simple parser)
    const char *sql_key = "\"sql\"";
    const char *p = strstr(args_json, sql_key);
    if (!p) {
        res.for_llm = arena_strdup(arena, "Error: 'sql' field required");
        return res;
    }
    p += strlen(sql_key);
    while (*p == ' ' || *p == ':') p++;
    if (*p != '"') {
        res.for_llm = arena_strdup(arena, "Error: 'sql' must be a string");
        return res;
    }
    p++; // skip opening quote

    // Extract SQL string (handle escapes)
    char *sql = malloc(args_len + 1);
    if (!sql) {
        res.for_llm = arena_strdup(arena, "Error: allocation failed");
        return res;
    }
    size_t j = 0;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) {
            p++;
            switch (*p) {
            case 'n':  sql[j++] = '\n'; break;
            case 't':  sql[j++] = '\t'; break;
            case '\\': sql[j++] = '\\'; break;
            case '"':  sql[j++] = '"';  break;
            default:   sql[j++] = *p;   break;
            }
        } else {
            sql[j++] = *p;
        }
        p++;
    }
    sql[j] = '\0';

    // Block destructive SQL
    char upper[16] = {0};
    for (int i = 0; i < 15 && sql[i]; i++)
        upper[i] = (sql[i] >= 'a' && sql[i] <= 'z')
                    ? (char)(sql[i] - 32) : sql[i];
    if (strncmp(upper, "DROP", 4) == 0 || strncmp(upper, "TRUNCATE", 8) == 0) {
        free(sql);
        res.for_llm = arena_strdup(arena, "Error: destructive SQL not allowed");
        return res;
    }

    // Execute query
    adam_ext_result_t qr = {0};
    int rc = ctx->db.exec_query(ctx->db.db_ctx, sql, &qr);
    free(sql);

    if (rc != 0) {
        res.for_llm = arena_strdup(arena, "Error: query execution failed");
        return res;
    }

    // Format results as tab-separated text
    size_t buf_size = 8192;
    char *buf = arena_alloc(arena, buf_size);
    if (!buf) {
        adam_ext_result_free(&qr);
        res.for_llm = arena_strdup(arena, "Error: allocation failed");
        return res;
    }

    size_t pos = 0;
    // Column headers
    for (int c = 0; c < qr.n_cols && pos < buf_size - 64; c++) {
        if (c > 0) buf[pos++] = '\t';
        const char *name = qr.col_names ? qr.col_names[c] : "?";
        pos += (size_t)snprintf(buf + pos, buf_size - pos, "%s", name ? name : "?");
    }
    if (qr.n_cols > 0 && pos < buf_size - 1) buf[pos++] = '\n';

    // Rows (max 100)
    int max_rows = qr.n_rows < 100 ? qr.n_rows : 100;
    for (int r = 0; r < max_rows && pos < buf_size - 64; r++) {
        for (int c = 0; c < qr.n_cols; c++) {
            if (c > 0) buf[pos++] = '\t';
            const char *val = qr.values[r * qr.n_cols + c];
            pos += (size_t)snprintf(buf + pos, buf_size - pos, "%s",
                                     val ? val : "NULL");
        }
        buf[pos++] = '\n';
    }
    if (qr.n_rows > 100)
        pos += (size_t)snprintf(buf + pos, buf_size - pos,
                                 "... (%d more rows)\n", qr.n_rows - 100);
    buf[pos] = '\0';

    adam_ext_result_free(&qr);
    res.for_llm = buf;
    res.success = 1;
    return res;
}

// ============================================================================
// MARK: - Ensure settings are ready
// ============================================================================

int adam_ext_ensure_settings(adam_ext_ctx_t *ctx,
                              char *err_buf, size_t err_buf_size) {
    // Always re-read config from DB (settings may have been set in a
    // previous SQL call within the same connection)
    if (adam_ext_config_apply(ctx) != 0 || !ctx->settings) {
        snprintf(err_buf, err_buf_size,
            "adam not configured. Use: SELECT adam_config('provider','anthropic'); "
            "SELECT adam_config('api_key','sk-...');");
        return -1;
    }

    // Check that a provider is set
    if (ctx->settings->api_format == ADAM_API_NONE
#ifndef ADAM_NO_LOCAL
        && !ctx->settings->gguf_path
#endif
    ) {
        snprintf(err_buf, err_buf_size,
            "No LLM provider configured. Set 'provider' and 'api_key' via adam_config.");
        return -1;
    }

    return 0;
}

// ============================================================================
// MARK: - adam() — stateless single-shot chat
// ============================================================================

char *adam_ext_chat(adam_ext_ctx_t *ctx, const char *message,
                     char *err_buf, size_t err_buf_size) {
    if (adam_ext_ensure_settings(ctx, err_buf, err_buf_size) != 0)
        return NULL;

    adam_history_t *h = adam_history_create();
    if (!h) {
        snprintf(err_buf, err_buf_size, "Failed to create history");
        return NULL;
    }

    adam_run_result_t r = adam_run(ctx->settings, h, message);
    adam_history_destroy(h);

    if (r.status != ADAM_OK) {
        snprintf(err_buf, err_buf_size, "%s",
                 r.final_response ? r.final_response : adam_status_string(r.status));
        adam_run_result_free(&r);
        return NULL;
    }

    char *response = r.final_response;
    r.final_response = NULL;
    adam_run_result_free(&r);
    return response;
}

// ============================================================================
// MARK: - adam_ask() — stateful SQL-aware agent
// ============================================================================

char *adam_ext_ask(adam_ext_ctx_t *ctx, const char *message,
                    char *err_buf, size_t err_buf_size) {
    if (adam_ext_ensure_settings(ctx, err_buf, err_buf_size) != 0)
        return NULL;

    // Auto-create session if none exists
    if (!ctx->session_id) {
        char *id = adam_ext_session_create(ctx);
        if (!id) {
            snprintf(err_buf, err_buf_size, "Failed to create session");
            return NULL;
        }
        free(id); // session_id is stored in ctx
    }

    if (!ctx->history)
        ctx->history = adam_history_create();

    // Register SQL tool if not already done
    if (ctx->settings->tool_count == 0) {
        // Read schema and inject into instructions
        char *schema = adam_ext_read_schema(ctx);
        if (schema) {
            size_t instr_len = strlen(schema) + 512;
            char *instr = malloc(instr_len);
            if (instr) {
                snprintf(instr, instr_len,
                    "You have access to a SQL database. Use the sql_query tool "
                    "to query it. Always use the tool for data questions.\n\n"
                    "Database schema:\n%s", schema);
                adam_settings_set_instructions(ctx->settings, instr);
                // Don't free instr — settings stores the pointer
            }
            free(schema);
        }

        adam_settings_add_tool(ctx->settings, (adam_tool_def_t){
            .name = "sql_query",
            .description = "Execute a SQL query against the connected database. "
                           "Returns results as tab-separated text.",
            .parameters_json = "{\"type\":\"object\",\"properties\":"
                "{\"sql\":{\"type\":\"string\",\"description\":\"SQL query to execute\"}},"
                "\"required\":[\"sql\"]}",
            .execute = ext_sql_tool,
            .ctx = ctx
        });
    }

    // Run the agent
    adam_run_result_t r = adam_run(ctx->settings, ctx->history, message);

    // Auto-save session
    adam_ext_session_save(ctx);

    if (r.status != ADAM_OK) {
        snprintf(err_buf, err_buf_size, "%s",
                 r.final_response ? r.final_response : adam_status_string(r.status));
        adam_run_result_free(&r);
        return NULL;
    }

    char *response = r.final_response;
    r.final_response = NULL;
    adam_run_result_free(&r);
    return response;
}

// ============================================================================
// MARK: - adam_sql() — generate SQL without executing
// ============================================================================

char *adam_ext_sql(adam_ext_ctx_t *ctx, const char *question,
                    char *err_buf, size_t err_buf_size) {
    if (adam_ext_ensure_settings(ctx, err_buf, err_buf_size) != 0)
        return NULL;

    // Read schema
    char *schema = adam_ext_read_schema(ctx);
    if (!schema) {
        snprintf(err_buf, err_buf_size, "Failed to read database schema");
        return NULL;
    }

    // Build prompt
    size_t prompt_len = strlen(schema) + strlen(question) + 512;
    char *prompt = malloc(prompt_len);
    if (!prompt) {
        free(schema);
        snprintf(err_buf, err_buf_size, "Allocation failed");
        return NULL;
    }
    snprintf(prompt, prompt_len,
        "Given this database schema:\n%s\n\n"
        "Write a SQL query to answer: %s\n\n"
        "Return ONLY the SQL query, no explanation.",
        schema, question);
    free(schema);

    // One-shot call with a dedicated identity
    adam_settings_t *s = adam_create_settings();
    if (!s) { free(prompt); snprintf(err_buf, err_buf_size, "Alloc failed"); return NULL; }

    // Copy provider settings
    s->api_format = ctx->settings->api_format;
    s->api_key = ctx->settings->api_key;
    s->model = ctx->settings->model;
    s->base_url = ctx->settings->base_url;
    s->temperature = 0.0f; // deterministic SQL generation

    adam_settings_set_identity(s,
        "You are a SQL expert. Return only valid SQL. No markdown, no explanation.");

    adam_history_t *h = adam_history_create();
    adam_run_result_t r = adam_run(s, h, prompt);
    free(prompt);
    adam_history_destroy(h);
    adam_settings_destroy(s);

    if (r.status != ADAM_OK) {
        snprintf(err_buf, err_buf_size, "%s",
                 r.final_response ? r.final_response : adam_status_string(r.status));
        adam_run_result_free(&r);
        return NULL;
    }

    char *response = r.final_response;
    r.final_response = NULL;
    adam_run_result_free(&r);
    return response;
}
