//
//  adam_memory.c
//  Adam — Long-term memory via sqlite-memory + sqlite-vector
//
//  Created by Marco Bambini on 17/03/26.
//

#include "adam.h"

#ifndef ADAM_NO_SQLITE

// SQLITE_CORE must be defined before sqlite-memory.h / sqlite-vector.h
// so that sqlite3ext.h does not redefine sqlite3_* as extension API macros.
#define SQLITE_CORE
#include "sqlite3.h"
#include "sqlite-memory.h"
#include "sqlite-vector.h"
#define JSMN_STATIC
#include "jsmn.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

// ============================================================================
// MARK: - Struct
// ============================================================================

struct adam_memory_t {
    sqlite3 *db;
    int      markdown_output;
    char    *markdown_dir;
};

// Internal accessor for session code (adam_session.c) sharing the same db.
sqlite3 *adam_memory_db(adam_memory_t *mem) {
    return mem ? mem->db : NULL;
}

// ============================================================================
// MARK: - Internal helpers
// ============================================================================

static int exec_simple(sqlite3 *db, const char *sql) {
    return sqlite3_exec(db, sql, NULL, NULL, NULL);
}

static int exec_select_text1(sqlite3 *db, const char *sql, const char *arg1) {
    sqlite3_stmt *vm = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) return rc;
    rc = sqlite3_bind_text(vm, 1, arg1, -1, SQLITE_TRANSIENT);
    if (rc != SQLITE_OK) { sqlite3_finalize(vm); return rc; }
    rc = sqlite3_step(vm);
    sqlite3_finalize(vm);
    return (rc == SQLITE_ROW || rc == SQLITE_DONE) ? SQLITE_OK : rc;
}

static int exec_select_text2(sqlite3 *db, const char *sql, const char *arg1, const char *arg2) {
    sqlite3_stmt *vm = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) return rc;
    rc = sqlite3_bind_text(vm, 1, arg1, -1, SQLITE_TRANSIENT);
    if (rc != SQLITE_OK) { sqlite3_finalize(vm); return rc; }
    if (arg2) rc = sqlite3_bind_text(vm, 2, arg2, -1, SQLITE_TRANSIENT);
    else      rc = sqlite3_bind_null(vm, 2);
    if (rc != SQLITE_OK) { sqlite3_finalize(vm); return rc; }
    rc = sqlite3_step(vm);
    sqlite3_finalize(vm);
    return (rc == SQLITE_ROW || rc == SQLITE_DONE) ? SQLITE_OK : rc;
}

// ============================================================================
// MARK: - Open / Configure / Close
// ============================================================================

adam_memory_t *adam_memory_open(const char *db_path) {
    if (!db_path) return NULL;

    adam_memory_t *mem = calloc(1, sizeof(adam_memory_t));
    if (!mem) return NULL;

    int rc = sqlite3_open(db_path, &mem->db);
    if (rc != SQLITE_OK) {
        free(mem);
        return NULL;
    }

    // WAL mode for concurrent reads
    exec_simple(mem->db, "PRAGMA journal_mode=WAL;");
    exec_simple(mem->db, "PRAGMA synchronous=NORMAL;");

    // Load sqlite-vector first (required by sqlite-memory for search)
    rc = sqlite3_vector_init(mem->db, NULL, NULL);
    if (rc != SQLITE_OK) {
        sqlite3_close(mem->db);
        free(mem);
        return NULL;
    }

    // Load sqlite-memory
    rc = sqlite3_memory_init(mem->db, NULL, NULL);
    if (rc != SQLITE_OK) {
        sqlite3_close(mem->db);
        free(mem);
        return NULL;
    }

    // Create session tables (shared database)
    exec_simple(mem->db,
        "CREATE TABLE IF NOT EXISTS sessions("
        "  id TEXT PRIMARY KEY,"
        "  sync INTEGER DEFAULT 0,"
        "  created_at INTEGER DEFAULT (unixepoch()),"
        "  updated_at INTEGER DEFAULT (unixepoch())"
        ")");

    exec_simple(mem->db,
        "CREATE TABLE IF NOT EXISTS messages("
        "  session_id TEXT NOT NULL,"
        "  idx INTEGER NOT NULL,"
        "  role INTEGER NOT NULL,"
        "  content TEXT,"
        "  content_len INTEGER DEFAULT 0,"
        "  tool_call_id TEXT,"
        "  tool_calls_json TEXT,"
        "  created_at INTEGER DEFAULT (unixepoch()),"
        "  PRIMARY KEY (session_id, idx),"
        "  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE"
        ")");

    exec_simple(mem->db, "PRAGMA foreign_keys=ON");

    return mem;
}

// Implemented in adam_http.c — registers a custom embedding provider that
// uses adam_net_post_json() for HTTP, supporting Voyage, OpenAI, etc.
extern adam_status_t adam_embed_provider_register(sqlite3 *db,
                                                   adam_settings_t *s,
                                                   const char *provider_name);

adam_status_t adam_memory_configure(adam_memory_t *mem, const adam_settings_t *s) {
    if (!mem || !s) return ADAM_ERR_INVALID_PARAM;

    const char *provider = s->memory_embedding_provider
                           ? s->memory_embedding_provider : "local";
    const char *model = s->memory_embedding_model;

    // Set API key before model — memory_set_model triggers provider init
    // which may need the key.
    const char *api_key = s->memory_api_key ? s->memory_api_key : s->api_key;
    if (api_key) {
        int rc = exec_select_text1(mem->db,
            "SELECT memory_set_apikey(?1)", api_key);
        if (rc != SQLITE_OK) return ADAM_ERR_SQLITE;
    }

    // For non-local providers, register the adam HTTP-based embedding engine
    // before calling memory_set_model (which triggers the engine's init).
    if (strcmp(provider, "local") != 0) {
        adam_status_t rc = adam_embed_provider_register(
            mem->db, (adam_settings_t *)s, provider);
        if (rc != ADAM_OK) return rc;
    }

    // Set embedding model (triggers provider init)
    if (model) {
        int rc = exec_select_text2(mem->db,
            "SELECT memory_set_model(?1, ?2)", provider, model);
        if (rc != SQLITE_OK) return ADAM_ERR_SQLITE;
    }

    // Apply search options via memory_set_option
    if (s->memory_max_results > 0) {
        char sql[128];
        snprintf(sql, sizeof(sql),
            "SELECT memory_set_option('max_results', %d)", s->memory_max_results);
        exec_simple(mem->db, sql);
    }

    if (s->memory_min_score > 0.0f) {
        char sql[128];
        snprintf(sql, sizeof(sql),
            "SELECT memory_set_option('min_score', %.4f)",
            (double)s->memory_min_score);
        exec_simple(mem->db, sql);
    }

    // Cache markdown settings
    mem->markdown_output = s->memory_markdown_output;
    free(mem->markdown_dir);
    mem->markdown_dir = s->memory_dir ? strdup(s->memory_dir) : NULL;

    return ADAM_OK;
}

void adam_memory_close(adam_memory_t *mem) {
    if (!mem) return;
    if (mem->db) sqlite3_close(mem->db);
    free(mem->markdown_dir);
    free(mem);
}

// ============================================================================
// MARK: - Markdown output helper
// ============================================================================

static void write_markdown(const adam_memory_t *mem, const char *text,
                            const char *context, const char *source) {
    if (!mem->markdown_output || !mem->markdown_dir) return;

    // Generate filename from source or timestamp+hash
    char filename[512];
    if (source && source[0]) {
        // Sanitize source for filename: replace non-alnum with _
        char safe[256];
        size_t j = 0;
        for (size_t i = 0; source[i] && j < sizeof(safe) - 1; i++) {
            char c = source[i];
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '-' || c == '_')
                safe[j++] = c;
            else if (c == '/' || c == ' ' || c == '.')
                safe[j++] = '_';
        }
        safe[j] = '\0';
        snprintf(filename, sizeof(filename), "%s/%s.md",
                 mem->markdown_dir, safe);
    } else {
        time_t now = time(NULL);
        snprintf(filename, sizeof(filename), "%s/memory_%ld.md",
                 mem->markdown_dir, (long)now);
    }

    FILE *f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "---\n");
    if (context) fprintf(f, "context: %s\n", context);
    if (source) fprintf(f, "source: %s\n", source);
    time_t now = time(NULL);
    struct tm *tm = localtime(&now);
    if (tm) {
        fprintf(f, "date: %04d-%02d-%02d\n",
                tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday);
    }
    fprintf(f, "---\n\n%s\n", text);
    fclose(f);
}

// ============================================================================
// MARK: - Add
// ============================================================================

adam_status_t adam_memory_add(adam_memory_t *mem, const char *text,
                              const char *context, const char *source) {
    if (!mem || !text) return ADAM_ERR_INVALID_PARAM;

    int rc = exec_select_text2(mem->db,
        "SELECT memory_add_text(?1, ?2)", text, context);
    if (rc != SQLITE_OK) return ADAM_ERR_SQLITE;

    write_markdown(mem, text, context, source);
    return ADAM_OK;
}

adam_status_t adam_memory_add_file(adam_memory_t *mem, const char *path,
                                   const char *context) {
    if (!mem || !path) return ADAM_ERR_INVALID_PARAM;

    int rc = exec_select_text2(mem->db,
        "SELECT memory_add_file(?1, ?2)", path, context);
    return (rc == SQLITE_OK) ? ADAM_OK : ADAM_ERR_SQLITE;
}

adam_status_t adam_memory_add_directory(adam_memory_t *mem,
                                        const char *dir_path,
                                        const char *context) {
    if (!mem || !dir_path) return ADAM_ERR_INVALID_PARAM;

    int rc = exec_select_text2(mem->db,
        "SELECT memory_add_directory(?1, ?2)", dir_path, context);
    return (rc == SQLITE_OK) ? ADAM_OK : ADAM_ERR_SQLITE;
}

// ============================================================================
// MARK: - Search
// ============================================================================

adam_status_t adam_memory_search(adam_memory_t *mem, arena_t *arena,
                                 const char *query, const char *context,
                                 int limit,
                                 adam_memory_result_t **results,
                                 size_t *count) {
    if (!mem || !arena || !query || !results || !count)
        return ADAM_ERR_INVALID_PARAM;

    *results = NULL;
    *count = 0;

    if (limit <= 0) limit = 5;

    // Allocate result array from arena (max = limit entries)
    adam_memory_result_t *arr = arena_alloc(arena,
        (size_t)limit * sizeof(adam_memory_result_t));
    if (!arr) return ADAM_ERR_ALLOC;

    const char *sql = context
        ? "SELECT path, snippet, context, ranking "
          "FROM memory_search WHERE query = ?1 AND context = ?2 LIMIT ?3"
        : "SELECT path, snippet, context, ranking "
          "FROM memory_search WHERE query = ?1 LIMIT ?2";

    sqlite3_stmt *vm = NULL;
    int rc = sqlite3_prepare_v2(mem->db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) return ADAM_ERR_SQLITE;

    sqlite3_bind_text(vm, 1, query, -1, SQLITE_TRANSIENT);
    if (context) {
        sqlite3_bind_text(vm, 2, context, -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(vm, 3, limit);
    } else {
        sqlite3_bind_int(vm, 2, limit);
    }

    size_t n = 0;
    while (sqlite3_step(vm) == SQLITE_ROW && (int)n < limit) {
        const char *path    = (const char *)sqlite3_column_text(vm, 0);
        const char *snippet = (const char *)sqlite3_column_text(vm, 1);
        const char *ctx     = (const char *)sqlite3_column_text(vm, 2);
        double ranking      = sqlite3_column_double(vm, 3);

        arr[n].source  = path    ? arena_strdup(arena, path)    : NULL;
        arr[n].content = snippet ? arena_strdup(arena, snippet) : NULL;
        arr[n].context = ctx     ? arena_strdup(arena, ctx)     : NULL;
        arr[n].score   = (float)ranking;
        n++;
    }

    sqlite3_finalize(vm);

    *results = arr;
    *count = n;
    return ADAM_OK;
}

// ============================================================================
// MARK: - Delete / Clear
// ============================================================================

adam_status_t adam_memory_delete_context(adam_memory_t *mem, const char *ctx) {
    if (!mem || !ctx) return ADAM_ERR_INVALID_PARAM;

    int rc = exec_select_text1(mem->db,
        "SELECT memory_delete_context(?1)", ctx);
    return (rc == SQLITE_OK) ? ADAM_OK : ADAM_ERR_SQLITE;
}

adam_status_t adam_memory_clear(adam_memory_t *mem) {
    if (!mem) return ADAM_ERR_INVALID_PARAM;

    int rc = exec_simple(mem->db, "SELECT memory_clear()");
    return (rc == SQLITE_OK) ? ADAM_OK : ADAM_ERR_SQLITE;
}

// ============================================================================
// MARK: - Sync
// ============================================================================

adam_status_t adam_memory_sync(adam_memory_t *mem,
                               const char *alias, const char *context) {
    UNUSED_PARAM(mem);
    UNUSED_PARAM(alias);
    UNUSED_PARAM(context);
    return ADAM_ERR_NOT_IMPLEMENTED;
}

// ============================================================================
// MARK: - Built-in Tools
// ============================================================================

// jsmn helpers
static int tok_eq(const char *json, const jsmntok_t *tok, const char *s) {
    size_t slen = strlen(s);
    return (tok->type == JSMN_STRING
         && (size_t)(tok->end - tok->start) == slen
         && memcmp(json + tok->start, s, slen) == 0);
}

static const char *tok_arena_str(arena_t *arena, const char *json,
                                  const jsmntok_t *tok) {
    size_t len = (size_t)(tok->end - tok->start);
    char *s = arena_alloc(arena, len + 1);
    if (!s) return NULL;
    memcpy(s, json + tok->start, len);
    s[len] = '\0';
    return s;
}

adam_tool_result_t adam_tool_memory_search(arena_t *arena, void *ctx,
                                           const char *args_json,
                                           size_t args_len) {
    adam_tool_result_t res = { .for_llm = NULL, .for_user = NULL, .success = 0 };
    adam_memory_t *mem = (adam_memory_t *)ctx;
    if (!mem || !args_json) {
        res.for_llm = arena_strdup(arena, "Error: memory not configured");
        return res;
    }

    // Parse JSON: {"query":"...","context":"...","limit":5}
    jsmntok_t tokens[32];
    jsmn_parser parser;
    jsmn_init(&parser);
    int ntok = jsmn_parse(&parser, args_json, args_len, tokens, 32);
    if (ntok < 1) {
        res.for_llm = arena_strdup(arena, "Error: invalid JSON");
        return res;
    }

    const char *query = NULL;
    const char *context = NULL;
    int limit = 5;

    for (int i = 1; i < ntok - 1; i++) {
        if (tok_eq(args_json, &tokens[i], "query") &&
            tokens[i + 1].type == JSMN_STRING) {
            query = tok_arena_str(arena, args_json, &tokens[i + 1]);
            i++;
        } else if (tok_eq(args_json, &tokens[i], "context") &&
                   tokens[i + 1].type == JSMN_STRING) {
            context = tok_arena_str(arena, args_json, &tokens[i + 1]);
            i++;
        } else if (tok_eq(args_json, &tokens[i], "limit") &&
                   tokens[i + 1].type == JSMN_PRIMITIVE) {
            limit = atoi(args_json + tokens[i + 1].start);
            i++;
        }
    }

    if (!query) {
        res.for_llm = arena_strdup(arena, "Error: 'query' is required");
        return res;
    }

    adam_memory_result_t *results = NULL;
    size_t count = 0;
    adam_status_t rc = adam_memory_search(mem, arena, query, context, limit,
                                          &results, &count);
    if (rc != ADAM_OK) {
        res.for_llm = arena_strdup(arena, "Error: memory search failed");
        return res;
    }

    if (count == 0) {
        res.for_llm = arena_strdup(arena, "No relevant memories found.");
        res.success = 1;
        return res;
    }

    // Format results
    size_t buf_size = 4096;
    char *buf = arena_alloc(arena, buf_size);
    if (!buf) {
        res.for_llm = arena_strdup(arena, "Error: allocation failed");
        return res;
    }

    size_t pos = 0;
    pos += (size_t)snprintf(buf + pos, buf_size - pos,
        "Found %zu result(s):\n\n", count);

    for (size_t i = 0; i < count && pos < buf_size - 128; i++) {
        // Grow buffer if needed
        size_t need = (results[i].content ? strlen(results[i].content) : 0) + 128;
        if (pos + need >= buf_size) {
            size_t new_size = buf_size;
            while (new_size < pos + need + 1) new_size *= 2;
            char *new_buf = arena_alloc(arena, new_size);
            if (new_buf) {
                memcpy(new_buf, buf, pos);
                buf = new_buf;
                buf_size = new_size;
            }
        }
        pos += (size_t)snprintf(buf + pos, buf_size - pos,
            "[%.2f] %s\n%s\n\n",
            results[i].score,
            results[i].source ? results[i].source : "(text)",
            results[i].content ? results[i].content : "");
    }

    buf[pos] = '\0';
    res.for_llm = buf;
    res.success = 1;
    return res;
}

adam_tool_result_t adam_tool_memory_add(arena_t *arena, void *ctx,
                                        const char *args_json,
                                        size_t args_len) {
    adam_tool_result_t res = { .for_llm = NULL, .for_user = NULL, .success = 0 };
    adam_memory_t *mem = (adam_memory_t *)ctx;
    if (!mem || !args_json) {
        res.for_llm = arena_strdup(arena, "Error: memory not configured");
        return res;
    }

    // Parse JSON: {"text":"...","context":"...","source":"..."}
    jsmntok_t tokens[32];
    jsmn_parser parser;
    jsmn_init(&parser);
    int ntok = jsmn_parse(&parser, args_json, args_len, tokens, 32);
    if (ntok < 1) {
        res.for_llm = arena_strdup(arena, "Error: invalid JSON");
        return res;
    }

    const char *text = NULL;
    const char *context = NULL;
    const char *source = NULL;

    for (int i = 1; i < ntok - 1; i++) {
        if (tok_eq(args_json, &tokens[i], "text") &&
            tokens[i + 1].type == JSMN_STRING) {
            text = tok_arena_str(arena, args_json, &tokens[i + 1]);
            i++;
        } else if (tok_eq(args_json, &tokens[i], "context") &&
                   tokens[i + 1].type == JSMN_STRING) {
            context = tok_arena_str(arena, args_json, &tokens[i + 1]);
            i++;
        } else if (tok_eq(args_json, &tokens[i], "source") &&
                   tokens[i + 1].type == JSMN_STRING) {
            source = tok_arena_str(arena, args_json, &tokens[i + 1]);
            i++;
        }
    }

    if (!text) {
        res.for_llm = arena_strdup(arena, "Error: 'text' is required");
        return res;
    }

    adam_status_t rc = adam_memory_add(mem, text, context, source);
    if (rc != ADAM_OK) {
        res.for_llm = arena_strdup(arena, "Error: failed to add memory");
        return res;
    }

    res.for_llm = arena_strdup(arena, "Memory added successfully.");
    res.success = 1;
    return res;
}

#endif // ADAM_NO_SQLITE
