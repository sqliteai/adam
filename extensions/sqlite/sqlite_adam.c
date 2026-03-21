//
//  sqlite_adam.c
//  Adam SQLite Extension — register SQL functions
//
//  .load adam
//  SELECT adam_config('provider', 'anthropic');
//  SELECT adam_config('api_key', 'sk-ant-...');
//  SELECT adam_ask('How many users are there?');
//

#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1

#include "adam_ext.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// ============================================================================
// MARK: - SQLite database abstraction implementation
// ============================================================================

static int sqlite_exec_query(void *db_ctx, const char *sql,
                              adam_ext_result_t *out) {
    sqlite3 *db = (sqlite3 *)db_ctx;
    sqlite3_stmt *vm = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) return -1;

    int n_cols = sqlite3_column_count(vm);
    out->n_cols = n_cols;

    // Column names
    out->col_names = calloc((size_t)n_cols, sizeof(char *));
    if (out->col_names) {
        for (int i = 0; i < n_cols; i++) {
            const char *name = sqlite3_column_name(vm, i);
            out->col_names[i] = name ? strdup(name) : strdup("?");
        }
    }

    // Collect rows
    size_t cap = 64, count = 0;
    out->values = calloc(cap * (size_t)n_cols, sizeof(char *));

    while (sqlite3_step(vm) == SQLITE_ROW) {
        if (count >= cap) {
            cap *= 2;
            char **nv = realloc(out->values, cap * (size_t)n_cols * sizeof(char *));
            if (!nv) break;
            out->values = nv;
        }
        for (int c = 0; c < n_cols; c++) {
            const char *val = (const char *)sqlite3_column_text(vm, c);
            out->values[count * (size_t)n_cols + (size_t)c] =
                val ? strdup(val) : NULL;
        }
        count++;
    }
    out->n_rows = (int)count;

    sqlite3_finalize(vm);
    return 0;
}

static int sqlite_exec_stmt(void *db_ctx, const char *sql) {
    sqlite3 *db = (sqlite3 *)db_ctx;
    char *err = NULL;
    int rc = sqlite3_exec(db, sql, NULL, NULL, &err);
    if (err) sqlite3_free(err);
    return rc == SQLITE_OK ? 0 : -1;
}

static char *sqlite_read_schema(void *db_ctx) {
    // Delegate to the shared schema reader via a temporary ctx
    // (a bit roundabout, but avoids duplicating the schema logic)
    adam_ext_db_t db = {
        .exec_query = sqlite_exec_query,
        .exec_stmt = sqlite_exec_stmt,
        .read_schema = NULL, // force fallback to generic implementation
        .db_ctx = db_ctx
    };
    adam_ext_ctx_t tmp = { .db = db };
    return adam_ext_read_schema(&tmp);
}

// ============================================================================
// MARK: - SQL function: adam_config(key, value)
// ============================================================================

static void fn_adam_config(sqlite3_context *context, int argc,
                            sqlite3_value **argv) {
    if (argc != 2) {
        sqlite3_result_error(context, "adam_config(key, value) requires 2 args", -1);
        return;
    }
    adam_ext_ctx_t *ctx = (adam_ext_ctx_t *)sqlite3_user_data(context);
    const char *key = (const char *)sqlite3_value_text(argv[0]);
    const char *val = (const char *)sqlite3_value_text(argv[1]);
    if (!key || !val) {
        sqlite3_result_error(context, "key and value must be TEXT", -1);
        return;
    }

    if (adam_ext_config_set(ctx, key, val) != 0) {
        sqlite3_result_error(context, "Failed to save config", -1);
        return;
    }

    // Re-apply config to settings
    ctx->initialized = 0;
    adam_ext_config_apply(ctx);

    sqlite3_result_text(context, val, -1, SQLITE_TRANSIENT);
}

// ============================================================================
// MARK: - SQL function: adam(message)
// ============================================================================

static void fn_adam(sqlite3_context *context, int argc, sqlite3_value **argv) {
    if (argc != 1) {
        sqlite3_result_error(context, "adam(message) requires 1 arg", -1);
        return;
    }
    adam_ext_ctx_t *ctx = (adam_ext_ctx_t *)sqlite3_user_data(context);
    const char *msg = (const char *)sqlite3_value_text(argv[0]);
    if (!msg) {
        sqlite3_result_error(context, "message must be TEXT", -1);
        return;
    }

    char err[512] = {0};
    char *response = adam_ext_chat(ctx, msg, err, sizeof(err));
    if (response) {
        sqlite3_result_text(context, response, -1, free);
    } else {
        sqlite3_result_error(context, err, -1);
    }
}

// ============================================================================
// MARK: - SQL function: adam_ask(message)
// ============================================================================

static void fn_adam_ask(sqlite3_context *context, int argc,
                         sqlite3_value **argv) {
    if (argc != 1) {
        sqlite3_result_error(context, "adam_ask(message) requires 1 arg", -1);
        return;
    }
    adam_ext_ctx_t *ctx = (adam_ext_ctx_t *)sqlite3_user_data(context);
    const char *msg = (const char *)sqlite3_value_text(argv[0]);
    if (!msg) {
        sqlite3_result_error(context, "message must be TEXT", -1);
        return;
    }

    char err[512] = {0};
    char *response = adam_ext_ask(ctx, msg, err, sizeof(err));
    if (response) {
        sqlite3_result_text(context, response, -1, free);
    } else {
        sqlite3_result_error(context, err, -1);
    }
}

// ============================================================================
// MARK: - SQL function: adam_sql(question)
// ============================================================================

static void fn_adam_sql(sqlite3_context *context, int argc,
                         sqlite3_value **argv) {
    if (argc != 1) {
        sqlite3_result_error(context, "adam_sql(question) requires 1 arg", -1);
        return;
    }
    adam_ext_ctx_t *ctx = (adam_ext_ctx_t *)sqlite3_user_data(context);
    const char *question = (const char *)sqlite3_value_text(argv[0]);
    if (!question) {
        sqlite3_result_error(context, "question must be TEXT", -1);
        return;
    }

    char err[512] = {0};
    char *response = adam_ext_sql(ctx, question, err, sizeof(err));
    if (response) {
        sqlite3_result_text(context, response, -1, free);
    } else {
        sqlite3_result_error(context, err, -1);
    }
}

// ============================================================================
// MARK: - SQL function: adam_create_session()
// ============================================================================

static void fn_adam_create_session(sqlite3_context *context, int argc,
                                    sqlite3_value **argv) {
    UNUSED_PARAM(argc); UNUSED_PARAM(argv);
    adam_ext_ctx_t *ctx = (adam_ext_ctx_t *)sqlite3_user_data(context);

    char *id = adam_ext_session_create(ctx);
    if (id) {
        sqlite3_result_text(context, id, -1, free);
    } else {
        sqlite3_result_error(context, "Failed to create session", -1);
    }
}

// ============================================================================
// MARK: - SQL function: adam_get_session()
// ============================================================================

static void fn_adam_get_session(sqlite3_context *context, int argc,
                                 sqlite3_value **argv) {
    UNUSED_PARAM(argc); UNUSED_PARAM(argv);
    adam_ext_ctx_t *ctx = (adam_ext_ctx_t *)sqlite3_user_data(context);

    const char *id = adam_ext_session_get(ctx);
    if (id) {
        sqlite3_result_text(context, id, -1, SQLITE_TRANSIENT);
    } else {
        sqlite3_result_null(context);
    }
}

// ============================================================================
// MARK: - SQL function: adam_clear_session()
// ============================================================================

static void fn_adam_clear_session(sqlite3_context *context, int argc,
                                   sqlite3_value **argv) {
    UNUSED_PARAM(argc); UNUSED_PARAM(argv);
    adam_ext_ctx_t *ctx = (adam_ext_ctx_t *)sqlite3_user_data(context);

    adam_ext_session_clear(ctx);
    sqlite3_result_int(context, 1);
}

// ============================================================================
// MARK: - SQL function: adam_version()
// ============================================================================

static void fn_adam_version(sqlite3_context *context, int argc,
                             sqlite3_value **argv) {
    UNUSED_PARAM(argc); UNUSED_PARAM(argv);
    sqlite3_result_text(context, ADAM_VERSION_STRING, -1, SQLITE_STATIC);
}

// ============================================================================
// MARK: - Extension entry point
// ============================================================================

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_adam_init(sqlite3 *db, char **pzErrMsg,
                       const sqlite3_api_routines *pApi) {
    SQLITE_EXTENSION_INIT2(pApi);
    (void)pzErrMsg;

    // Create database abstraction
    adam_ext_db_t ext_db = {
        .exec_query  = sqlite_exec_query,
        .exec_stmt   = sqlite_exec_stmt,
        .read_schema = sqlite_read_schema,
        .db_ctx      = db
    };

    // Create extension context
    adam_ext_ctx_t *ctx = adam_ext_ctx_create(ext_db);
    if (!ctx) return SQLITE_NOMEM;

    // Initialize tables
    adam_ext_config_init(ctx);
    adam_ext_session_init_tables(ctx);

    // Load existing config (if any)
    adam_ext_config_apply(ctx);

    // Register functions (destructor on first function frees ctx on db close)
    #define REG(name, func, narg) \
        sqlite3_create_function_v2(db, name, narg, SQLITE_UTF8, ctx, \
                                    func, NULL, NULL, NULL)

    REG("adam_config",         fn_adam_config,         2);
    REG("adam",                fn_adam,                 1);
    REG("adam_ask",            fn_adam_ask,             1);
    REG("adam_sql",            fn_adam_sql,             1);
    REG("adam_create_session", fn_adam_create_session,  0);
    REG("adam_get_session",    fn_adam_get_session,     0);
    REG("adam_version",        fn_adam_version,         0);

    // Register clear_session with destructor to free ctx on db close
    sqlite3_create_function_v2(db, "adam_clear_session", 0, SQLITE_UTF8,
        ctx, fn_adam_clear_session, NULL, NULL,
        (void(*)(void*))adam_ext_ctx_destroy);

    #undef REG

    return SQLITE_OK;
}
