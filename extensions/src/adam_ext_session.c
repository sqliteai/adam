//
//  adam_ext_session.c
//  Session management — create, load, save, clear
//

#include "adam_ext.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

int adam_ext_session_init_tables(adam_ext_ctx_t *ctx) {
    int rc = ctx->db.exec_stmt(ctx->db.db_ctx,
        "CREATE TABLE IF NOT EXISTS _adam_sessions "
        "(id TEXT PRIMARY KEY, created_at INTEGER DEFAULT (strftime('%s','now')))");
    if (rc != 0) return rc;

    rc = ctx->db.exec_stmt(ctx->db.db_ctx,
        "CREATE TABLE IF NOT EXISTS _adam_messages "
        "(session_id TEXT NOT NULL, idx INTEGER NOT NULL, "
        "role INTEGER NOT NULL, content TEXT, "
        "PRIMARY KEY (session_id, idx))");
    return rc;
}

// Simple UUID-like ID generator (not cryptographic, good enough for session IDs)
static void generate_id(char *buf, size_t buf_size) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t ms = (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
    snprintf(buf, buf_size, "%08x-%04x-%04x-%04x-%012llx",
             (unsigned)(ms >> 32),
             (unsigned)((ms >> 16) & 0xFFFF),
             (unsigned)(0x7000 | ((ms >> 4) & 0x0FFF)),
             (unsigned)(0x8000 | (rand() & 0x3FFF)),
             (unsigned long long)(ms ^ (uint64_t)rand() << 16 ^ (uint64_t)rand()));
}

char *adam_ext_session_create(adam_ext_ctx_t *ctx) {
    // Clear previous session
    if (ctx->session_id) {
        adam_ext_session_save(ctx);
        free(ctx->session_id);
        ctx->session_id = NULL;
    }
    if (ctx->history)
        adam_history_clear(ctx->history);
    else
        ctx->history = adam_history_create();

    char id[64];
    generate_id(id, sizeof(id));

    char sql[256];
    snprintf(sql, sizeof(sql),
        "INSERT INTO _adam_sessions(id) VALUES('%s')", id);
    if (ctx->db.exec_stmt(ctx->db.db_ctx, sql) != 0)
        return NULL;

    ctx->session_id = strdup(id);
    return strdup(id);
}

const char *adam_ext_session_get(adam_ext_ctx_t *ctx) {
    return ctx->session_id;
}

int adam_ext_session_clear(adam_ext_ctx_t *ctx) {
    if (ctx->session_id) {
        char sql[256];
        snprintf(sql, sizeof(sql),
            "DELETE FROM _adam_messages WHERE session_id='%s'",
            ctx->session_id);
        ctx->db.exec_stmt(ctx->db.db_ctx, sql);

        free(ctx->session_id);
        ctx->session_id = NULL;
    }
    if (ctx->history)
        adam_history_clear(ctx->history);
    return 0;
}

int adam_ext_session_save(adam_ext_ctx_t *ctx) {
    if (!ctx->session_id || !ctx->history) return 0;

    // Delete old messages
    char sql[512];
    snprintf(sql, sizeof(sql),
        "DELETE FROM _adam_messages WHERE session_id='%s'",
        ctx->session_id);
    ctx->db.exec_stmt(ctx->db.db_ctx, sql);

    // Insert current history
    for (size_t i = 0; i < ctx->history->count; i++) {
        const adam_message_t *m = &ctx->history->items[i];
        const char *content = m->content ? m->content : "";

        // Escape content for SQL
        size_t clen = strlen(content);
        char *escaped = malloc(clen * 2 + 1);
        if (!escaped) continue;
        size_t j = 0;
        for (size_t k = 0; k < clen; k++) {
            if (content[k] == '\'') escaped[j++] = '\'';
            escaped[j++] = content[k];
        }
        escaped[j] = '\0';

        size_t sql_len = j + 256;
        char *ins = malloc(sql_len);
        if (ins) {
            snprintf(ins, sql_len,
                "INSERT INTO _adam_messages(session_id,idx,role,content) "
                "VALUES('%s',%zu,%d,'%s')",
                ctx->session_id, i, (int)m->role, escaped);
            ctx->db.exec_stmt(ctx->db.db_ctx, ins);
            free(ins);
        }
        free(escaped);
    }
    return 0;
}

int adam_ext_session_load(adam_ext_ctx_t *ctx, const char *session_id) {
    if (!ctx->history)
        ctx->history = adam_history_create();
    else
        adam_history_clear(ctx->history);

    free(ctx->session_id);
    ctx->session_id = strdup(session_id);

    char sql[256];
    snprintf(sql, sizeof(sql),
        "SELECT role, content FROM _adam_messages "
        "WHERE session_id='%s' ORDER BY idx", session_id);

    adam_ext_result_t result = {0};
    if (ctx->db.exec_query(ctx->db.db_ctx, sql, &result) != 0)
        return -1;

    for (int i = 0; i < result.n_rows; i++) {
        int role = atoi(result.values[i * result.n_cols]);
        const char *content = result.values[i * result.n_cols + 1];
        if (!content) content = "";

        switch (role) {
        case 1: adam_history_append_user(ctx->history, content); break;
        case 2: adam_history_append_assistant(ctx->history, content, NULL, 0); break;
        default: break;
        }
    }
    adam_ext_result_free(&result);
    return 0;
}
