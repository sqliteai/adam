//
//  adam_ext_schema.c
//  Read database schema for injection into agent system prompt
//

#include "adam_ext.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

char *adam_ext_read_schema(adam_ext_ctx_t *ctx) {
    // Use the db abstraction's read_schema if available
    if (ctx->db.read_schema)
        return ctx->db.read_schema(ctx->db.db_ctx);

    // Fallback: generic SQL approach (works for SQLite, may need adaptation for PG)
    adam_ext_result_t result = {0};
    int rc = ctx->db.exec_query(ctx->db.db_ctx,
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE sql IS NOT NULL "
        "AND name NOT LIKE '_adam_%' "
        "AND name NOT LIKE 'sqlite_%' "
        "AND name NOT LIKE 'dbmem_%' "
        "AND name NOT LIKE '_sqliteai_%' "
        "ORDER BY type, name", &result);

    if (rc != 0) return NULL;

    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    if (!buf) { adam_ext_result_free(&result); return NULL; }

    #define APPEND(...) do { \
        int n = snprintf(buf + len, cap - len, __VA_ARGS__); \
        if (n > 0) { \
            if (len + (size_t)n >= cap) { \
                cap *= 2; buf = realloc(buf, cap); \
                if (!buf) { adam_ext_result_free(&result); return NULL; } \
                snprintf(buf + len, cap - len, __VA_ARGS__); \
            } \
            len += (size_t)n; \
        } \
    } while(0)

    for (int i = 0; i < result.n_rows; i++) {
        const char *sql = result.values[i * result.n_cols + 2];
        if (sql) APPEND("%s;\n", sql);
    }
    adam_ext_result_free(&result);

    // Row counts for user tables
    adam_ext_result_t tables = {0};
    rc = ctx->db.exec_query(ctx->db.db_ctx,
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE '_adam_%' AND name NOT LIKE 'sqlite_%' "
        "AND name NOT LIKE 'dbmem_%' AND name NOT LIKE '_sqliteai_%' "
        "ORDER BY name", &tables);

    if (rc == 0 && tables.n_rows > 0) {
        APPEND("\n-- Row counts:\n");
        for (int i = 0; i < tables.n_rows; i++) {
            const char *tname = tables.values[i];
            if (!tname) continue;
            char count_sql[256];
            snprintf(count_sql, sizeof(count_sql),
                     "SELECT COUNT(*) FROM \"%s\"", tname);
            adam_ext_result_t cnt = {0};
            if (ctx->db.exec_query(ctx->db.db_ctx, count_sql, &cnt) == 0
                && cnt.n_rows > 0 && cnt.values[0]) {
                APPEND("--   %s: %s rows\n", tname, cnt.values[0]);
            }
            adam_ext_result_free(&cnt);
        }
    }
    adam_ext_result_free(&tables);

    #undef APPEND
    buf[len] = '\0';
    return buf;
}
