//
//  pg_adam.c
//  Adam PostgreSQL Extension — register SQL functions
//
//  CREATE EXTENSION adam;
//  SELECT adam_config('provider', 'anthropic');
//  SELECT adam_config('api_key', 'sk-ant-...');
//  SELECT adam_ask('How many users are there?');
//

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "executor/spi.h"
#include "miscadmin.h"

PG_MODULE_MAGIC;

#include "adam_ext.h"
#include <string.h>
#include <stdlib.h>

// ============================================================================
// MARK: - PostgreSQL database abstraction implementation
// ============================================================================

// PG uses SPI (Server Programming Interface) for SQL from extensions.
// SPI_connect/SPI_finish bracket each query session.

static int pg_exec_query(void *db_ctx, const char *sql,
                          adam_ext_result_t *out) {
    (void)db_ctx;
    int rc;

    if (SPI_connect() != SPI_OK_CONNECT) return -1;

    rc = SPI_execute(sql, true, 0); // read-only = true
    if (rc != SPI_OK_SELECT || SPI_tuptable == NULL) {
        SPI_finish();
        // Might be a non-SELECT statement that returned rows
        return -1;
    }

    int n_cols = SPI_tuptable->tupdesc->natts;
    int n_rows = (int)SPI_processed;
    out->n_cols = n_cols;
    out->n_rows = n_rows;

    // Column names
    out->col_names = calloc((size_t)n_cols, sizeof(char *));
    if (out->col_names) {
        for (int i = 0; i < n_cols; i++) {
            const char *name = SPI_fname(SPI_tuptable->tupdesc, i + 1);
            out->col_names[i] = name ? strdup(name) : strdup("?");
        }
    }

    // Values
    out->values = calloc((size_t)(n_rows * n_cols), sizeof(char *));
    if (out->values) {
        for (int r = 0; r < n_rows; r++) {
            HeapTuple tuple = SPI_tuptable->vals[r];
            for (int c = 0; c < n_cols; c++) {
                char *val = SPI_getvalue(tuple, SPI_tuptable->tupdesc, c + 1);
                out->values[r * n_cols + c] = val ? strdup(val) : NULL;
                if (val) pfree(val);
            }
        }
    }

    SPI_finish();
    return 0;
}

static int pg_exec_stmt(void *db_ctx, const char *sql) {
    (void)db_ctx;

    if (SPI_connect() != SPI_OK_CONNECT) return -1;

    int rc = SPI_execute(sql, false, 0); // read-only = false
    SPI_finish();

    return (rc >= 0) ? 0 : -1;
}

static char *pg_read_schema(void *db_ctx) {
    (void)db_ctx;

    adam_ext_result_t result = {0};
    int rc;

    if (SPI_connect() != SPI_OK_CONNECT) return NULL;

    // Get table definitions from information_schema
    rc = SPI_execute(
        "SELECT table_name, column_name, data_type, is_nullable "
        "FROM information_schema.columns "
        "WHERE table_schema = 'public' "
        "AND table_name NOT LIKE '_adam_%' "
        "ORDER BY table_name, ordinal_position", true, 0);

    if (rc != SPI_OK_SELECT || SPI_tuptable == NULL) {
        SPI_finish();
        return NULL;
    }

    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    if (!buf) { SPI_finish(); return NULL; }

    #define APPEND(...) do { \
        int n = snprintf(buf + len, cap - len, __VA_ARGS__); \
        if (n > 0) { \
            if (len + (size_t)n >= cap) { \
                cap *= 2; buf = realloc(buf, cap); \
                if (!buf) { SPI_finish(); return NULL; } \
                snprintf(buf + len, cap - len, __VA_ARGS__); \
            } \
            len += (size_t)n; \
        } \
    } while(0)

    const char *prev_table = "";
    int n_rows = (int)SPI_processed;

    for (int r = 0; r < n_rows; r++) {
        HeapTuple tuple = SPI_tuptable->vals[r];
        char *tname = SPI_getvalue(tuple, SPI_tuptable->tupdesc, 1);
        char *cname = SPI_getvalue(tuple, SPI_tuptable->tupdesc, 2);
        char *dtype = SPI_getvalue(tuple, SPI_tuptable->tupdesc, 3);
        char *nullable = SPI_getvalue(tuple, SPI_tuptable->tupdesc, 4);

        if (tname && strcmp(tname, prev_table) != 0) {
            if (len > 0) APPEND(");\n");
            APPEND("CREATE TABLE %s (\n", tname);
            prev_table = tname; // safe: SPI memory lives until SPI_finish
        } else if (len > 0) {
            APPEND(",\n");
        }

        APPEND("  %s %s", cname ? cname : "?", dtype ? dtype : "?");
        if (nullable && strcmp(nullable, "NO") == 0)
            APPEND(" NOT NULL");

        if (tname) pfree(tname);
        if (cname) pfree(cname);
        if (dtype) pfree(dtype);
        if (nullable) pfree(nullable);
    }
    if (len > 0) APPEND("\n);\n");

    // Row counts
    SPI_execute(
        "SELECT relname, n_live_tup FROM pg_stat_user_tables "
        "WHERE schemaname = 'public' AND relname NOT LIKE '_adam_%' "
        "ORDER BY relname", true, 0);

    if (SPI_processed > 0) {
        APPEND("\n-- Row counts:\n");
        for (uint64 r = 0; r < SPI_processed; r++) {
            HeapTuple tuple = SPI_tuptable->vals[r];
            char *tname = SPI_getvalue(tuple, SPI_tuptable->tupdesc, 1);
            char *cnt = SPI_getvalue(tuple, SPI_tuptable->tupdesc, 2);
            if (tname && cnt) APPEND("--   %s: %s rows\n", tname, cnt);
            if (tname) pfree(tname);
            if (cnt) pfree(cnt);
        }
    }

    #undef APPEND

    SPI_finish();
    buf[len] = '\0';
    return buf;
}

// ============================================================================
// MARK: - Per-session extension context
// ============================================================================

// Store context in a static variable (per-backend process in PG).
// Each PG backend (connection) gets its own process, so this is safe.
static adam_ext_ctx_t *g_ctx = NULL;

static adam_ext_ctx_t *get_ctx(void) {
    if (!g_ctx) {
        adam_ext_db_t db = {
            .exec_query  = pg_exec_query,
            .exec_stmt   = pg_exec_stmt,
            .read_schema = pg_read_schema,
            .db_ctx      = NULL  // PG uses SPI, no explicit db handle
        };
        g_ctx = adam_ext_ctx_create(db);
        if (g_ctx) {
            adam_ext_config_init(g_ctx);
            adam_ext_session_init_tables(g_ctx);
            adam_ext_config_apply(g_ctx);
        }
    }
    return g_ctx;
}

// ============================================================================
// MARK: - Helper: C string → PG text datum
// ============================================================================

static text *cstring_to_text_copy(const char *s) {
    size_t len = strlen(s);
    text *t = (text *)palloc(VARHDRSZ + len);
    SET_VARSIZE(t, VARHDRSZ + len);
    memcpy(VARDATA(t), s, len);
    return t;
}

// ============================================================================
// MARK: - SQL functions
// ============================================================================

PG_FUNCTION_INFO_V1(pg_adam_config);
Datum pg_adam_config(PG_FUNCTION_ARGS) {
    adam_ext_ctx_t *ctx = get_ctx();
    if (!ctx) ereport(ERROR, (errmsg("adam: failed to initialize")));

    text *key_t = PG_GETARG_TEXT_PP(0);
    text *val_t = PG_GETARG_TEXT_PP(1);
    char *key = text_to_cstring(key_t);
    char *val = text_to_cstring(val_t);

    if (adam_ext_config_set(ctx, key, val) != 0) {
        pfree(key); pfree(val);
        ereport(ERROR, (errmsg("adam: failed to save config")));
    }

    ctx->initialized = 0;
    adam_ext_config_apply(ctx);

    text *result = cstring_to_text_copy(val);
    pfree(key); pfree(val);
    PG_RETURN_TEXT_P(result);
}

PG_FUNCTION_INFO_V1(pg_adam);
Datum pg_adam(PG_FUNCTION_ARGS) {
    adam_ext_ctx_t *ctx = get_ctx();
    if (!ctx) ereport(ERROR, (errmsg("adam: failed to initialize")));

    text *msg_t = PG_GETARG_TEXT_PP(0);
    char *msg = text_to_cstring(msg_t);

    char err[512] = {0};
    char *response = adam_ext_chat(ctx, msg, err, sizeof(err));
    pfree(msg);

    if (!response)
        ereport(ERROR, (errmsg("adam: %s", err)));

    text *result = cstring_to_text_copy(response);
    free(response);
    PG_RETURN_TEXT_P(result);
}

PG_FUNCTION_INFO_V1(pg_adam_ask);
Datum pg_adam_ask(PG_FUNCTION_ARGS) {
    adam_ext_ctx_t *ctx = get_ctx();
    if (!ctx) ereport(ERROR, (errmsg("adam: failed to initialize")));

    text *msg_t = PG_GETARG_TEXT_PP(0);
    char *msg = text_to_cstring(msg_t);

    char err[512] = {0};
    char *response = adam_ext_ask(ctx, msg, err, sizeof(err));
    pfree(msg);

    if (!response)
        ereport(ERROR, (errmsg("adam: %s", err)));

    text *result = cstring_to_text_copy(response);
    free(response);
    PG_RETURN_TEXT_P(result);
}

PG_FUNCTION_INFO_V1(pg_adam_sql);
Datum pg_adam_sql(PG_FUNCTION_ARGS) {
    adam_ext_ctx_t *ctx = get_ctx();
    if (!ctx) ereport(ERROR, (errmsg("adam: failed to initialize")));

    text *q_t = PG_GETARG_TEXT_PP(0);
    char *question = text_to_cstring(q_t);

    char err[512] = {0};
    char *response = adam_ext_sql(ctx, question, err, sizeof(err));
    pfree(question);

    if (!response)
        ereport(ERROR, (errmsg("adam: %s", err)));

    text *result = cstring_to_text_copy(response);
    free(response);
    PG_RETURN_TEXT_P(result);
}

PG_FUNCTION_INFO_V1(pg_adam_create_session);
Datum pg_adam_create_session(PG_FUNCTION_ARGS) {
    adam_ext_ctx_t *ctx = get_ctx();
    if (!ctx) ereport(ERROR, (errmsg("adam: failed to initialize")));

    char *id = adam_ext_session_create(ctx);
    if (!id) ereport(ERROR, (errmsg("adam: failed to create session")));

    text *result = cstring_to_text_copy(id);
    free(id);
    PG_RETURN_TEXT_P(result);
}

PG_FUNCTION_INFO_V1(pg_adam_get_session);
Datum pg_adam_get_session(PG_FUNCTION_ARGS) {
    adam_ext_ctx_t *ctx = get_ctx();
    if (!ctx) PG_RETURN_NULL();

    const char *id = adam_ext_session_get(ctx);
    if (!id) PG_RETURN_NULL();

    PG_RETURN_TEXT_P(cstring_to_text_copy(id));
}

PG_FUNCTION_INFO_V1(pg_adam_clear_session);
Datum pg_adam_clear_session(PG_FUNCTION_ARGS) {
    adam_ext_ctx_t *ctx = get_ctx();
    if (!ctx) PG_RETURN_BOOL(false);

    adam_ext_session_clear(ctx);
    PG_RETURN_BOOL(true);
}
