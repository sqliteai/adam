//
//  adam_ext.h
//  Adam Database Extension — shared header for SQLite and PostgreSQL
//
//  Defines the extension context, database abstraction layer, and
//  shared function signatures used by both sqlite_adam.c and pg_adam.c.
//

#pragma once

#include "adam.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// MARK: - Database abstraction
// ============================================================================

// Query result: a simple 2D array of strings (rows × cols).
// Allocated by the caller's query function, freed by adam_ext_result_free.
typedef struct {
    char      **values;     // flat array: values[row * n_cols + col]
    int         n_rows;
    int         n_cols;
    char      **col_names;  // column names (n_cols entries)
} adam_ext_result_t;

void adam_ext_result_free(adam_ext_result_t *r);

// Database abstraction — implemented per-backend (SQLite / PostgreSQL).
// All strings returned in results are malloc'd and owned by the result.
typedef struct {
    // Execute a query and return results. Returns 0 on success.
    int (*exec_query)(void *db_ctx, const char *sql,
                      adam_ext_result_t *out_result);

    // Execute a statement with no result (INSERT/UPDATE/DELETE/CREATE).
    int (*exec_stmt)(void *db_ctx, const char *sql);

    // Read the database schema as a text description for the LLM.
    // Returns a malloc'd string (caller frees).
    char *(*read_schema)(void *db_ctx);

    // Opaque database context (sqlite3* or PGconn*).
    void *db_ctx;
} adam_ext_db_t;

// ============================================================================
// MARK: - Extension context (per-connection state)
// ============================================================================

typedef struct {
    adam_ext_db_t    db;            // database abstraction
    adam_settings_t *settings;      // LLM settings (created on first use)
    adam_history_t  *history;       // current session history
    char            *session_id;    // current session UUID (NULL if none)
    int              initialized;   // 1 after first adam_config call
    char            *schema_cache;  // cached schema text (invalidated on ask)
} adam_ext_ctx_t;

// ============================================================================
// MARK: - Shared functions (implemented in extensions/src/)
// ============================================================================

// --- Config ---
// Read/write config from/to _adam_config table.
// Creates the table if it doesn't exist.
int         adam_ext_config_init(adam_ext_ctx_t *ctx);
int         adam_ext_config_set(adam_ext_ctx_t *ctx,
                const char *key, const char *value);
char       *adam_ext_config_get(adam_ext_ctx_t *ctx, const char *key);
int         adam_ext_config_apply(adam_ext_ctx_t *ctx);

// --- Session ---
// Create/get/clear the current session.
// Session history is stored in _adam_sessions / _adam_messages tables.
int         adam_ext_session_init_tables(adam_ext_ctx_t *ctx);
char       *adam_ext_session_create(adam_ext_ctx_t *ctx);
const char *adam_ext_session_get(adam_ext_ctx_t *ctx);
int         adam_ext_session_clear(adam_ext_ctx_t *ctx);
int         adam_ext_session_save(adam_ext_ctx_t *ctx);
int         adam_ext_session_load(adam_ext_ctx_t *ctx, const char *session_id);

// --- Chat ---
// The core agent functions.
// All return malloc'd strings (caller frees) or NULL on error.
char       *adam_ext_chat(adam_ext_ctx_t *ctx, const char *message,
                char *err_buf, size_t err_buf_size);
char       *adam_ext_ask(adam_ext_ctx_t *ctx, const char *message,
                char *err_buf, size_t err_buf_size);
char       *adam_ext_sql(adam_ext_ctx_t *ctx, const char *question,
                char *err_buf, size_t err_buf_size);

// --- Schema ---
// Read the database schema, filtering out internal _adam_* tables.
char       *adam_ext_read_schema(adam_ext_ctx_t *ctx);

// --- Lifecycle ---
adam_ext_ctx_t *adam_ext_ctx_create(adam_ext_db_t db);
void            adam_ext_ctx_destroy(adam_ext_ctx_t *ctx);

// Ensure settings are initialized from config. Called lazily.
int             adam_ext_ensure_settings(adam_ext_ctx_t *ctx,
                    char *err_buf, size_t err_buf_size);

#ifdef __cplusplus
}
#endif
