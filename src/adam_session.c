//
//  adam_session.c
//  Adam — Session persistence (stored in the memory database)
//
//  Stores conversation history (messages, tool calls, attachments)
//  in the same SQLite database as long-term memory.
//
//  Created by Marco Bambini on 16/03/26.
//

#ifndef ADAM_NO_SQLITE

#include "adam.h"
#include "sqlite3.h"
#undef UNUSED_PARAM
#include "dbmem-utils.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Portable strndup. POSIX 2008 (glibc / musl / Apple / Android bionic /
// MinGW-UCRT) all expose strndup, so we alias straight to the system call.
// Only MSVCRT-based MinGW (msys2 `mingw64` toolchain, what GitHub Actions
// uses by default) doesn't ship strndup at all — fall back there.
#if defined(__MINGW32__) && !defined(_UCRT)
static inline char *adam_strndup(const char *s, size_t n) {
    char *p = (char *)malloc(n + 1);
    if (!p) return NULL;
    size_t i;
    for (i = 0; i < n && s[i]; i++) p[i] = s[i];
    p[i] = '\0';
    return p;
}
#else
#define adam_strndup strndup
#endif

// Internal: get the sqlite3* handle from adam_memory_t (defined in adam_memory.c)
extern sqlite3 *adam_memory_db(adam_memory_t *mem);

// ============================================================================
// MARK: - Internal: tool calls JSON serialization
// ============================================================================

// Serialize tool_call_entry_t array to JSON string (malloc'd).
// Format: [{"id":"...","name":"...","arguments":"..."},...]
static char *serialize_tool_calls(const adam_tool_call_entry_t *calls,
                                   size_t count) {
    if (!calls || count == 0) return NULL;

    size_t cap = 256 * count;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    size_t pos = 0;

    buf[pos++] = '[';
    for (size_t i = 0; i < count; i++) {
        if (i > 0) buf[pos++] = ',';

        // Estimate needed space
        size_t need = 128;
        if (calls[i].id) need += strlen(calls[i].id);
        if (calls[i].name) need += strlen(calls[i].name);
        if (calls[i].arguments_json) need += strlen(calls[i].arguments_json) * 2;

        if (pos + need >= cap) {
            cap = (pos + need) * 2;
            char *new_buf = realloc(buf, cap);
            if (!new_buf) { free(buf); return NULL; }
            buf = new_buf;
        }

        pos += (size_t)snprintf(buf + pos, cap - pos,
            "{\"id\":\"%s\",\"name\":\"%s\",\"arguments\":",
            calls[i].id ? calls[i].id : "",
            calls[i].name ? calls[i].name : "");

        // arguments_json is already valid JSON — but needs to be escaped
        // since it's stored as a string value
        const char *args = calls[i].arguments_json;
        if (args) {
            buf[pos++] = '"';
            size_t alen = strlen(args);
            for (size_t j = 0; j < alen && pos < cap - 4; j++) {
                if (args[j] == '"') { buf[pos++] = '\\'; buf[pos++] = '"'; }
                else if (args[j] == '\\') { buf[pos++] = '\\'; buf[pos++] = '\\'; }
                else if (args[j] == '\n') { buf[pos++] = '\\'; buf[pos++] = 'n'; }
                else buf[pos++] = args[j];
            }
            buf[pos++] = '"';
        } else {
            memcpy(buf + pos, "null", 4); pos += 4;
        }

        buf[pos++] = '}';
    }
    buf[pos++] = ']';
    buf[pos] = '\0';
    return buf;
}

// Deserialize tool calls JSON back into adam_tool_call_entry_t array.
// Simple parser — expects the format produced by serialize_tool_calls.
static void deserialize_tool_calls(const char *json,
                                    adam_tool_call_entry_t **out_calls,
                                    size_t *out_count) {
    *out_calls = NULL;
    *out_count = 0;
    if (!json || json[0] != '[') return;

    // Count entries (count '{' at top level)
    size_t cap = 0;
    int depth = 0;
    for (const char *p = json; *p; p++) {
        if (*p == '[' || *p == '{') depth++;
        else if (*p == ']' || *p == '}') depth--;
        if (*p == '{' && depth == 2) cap++; // depth 2 = inside array, opening object
    }
    if (cap == 0) return;

    adam_tool_call_entry_t *calls = calloc(cap, sizeof(adam_tool_call_entry_t));
    if (!calls) return;

    // Simple extraction: find "id":"...", "name":"...", "arguments":"..."
    size_t idx = 0;
    const char *p = json;
    while (*p && idx < cap) {
        // Find next object
        p = strchr(p, '{');
        if (!p) break;
        p++;

        // Extract fields within this object
        const char *obj_end = NULL;
        int d = 1;
        for (const char *q = p; *q; q++) {
            if (*q == '{') d++;
            else if (*q == '}') { d--; if (d == 0) { obj_end = q; break; } }
        }
        if (!obj_end) break;

        // Extract "id":"..."
        const char *id_key = strstr(p, "\"id\":");
        if (id_key && id_key < obj_end) {
            const char *v = strchr(id_key + 5, '"');
            if (v) { v++; const char *e = strchr(v, '"');
                if (e) { calls[idx].id = adam_strndup(v, (size_t)(e - v)); }
            }
        }

        // Extract "name":"..."
        const char *name_key = strstr(p, "\"name\":");
        if (name_key && name_key < obj_end) {
            const char *v = strchr(name_key + 7, '"');
            if (v) { v++; const char *e = strchr(v, '"');
                if (e) { calls[idx].name = adam_strndup(v, (size_t)(e - v)); }
            }
        }

        // Extract "arguments":"..." (may contain escaped quotes)
        const char *args_key = strstr(p, "\"arguments\":");
        if (args_key && args_key < obj_end) {
            const char *v = args_key + 12;
            while (*v == ' ') v++;
            if (*v == '"') {
                v++;
                // Unescape: find closing quote (not preceded by \)
                size_t alen = 0;
                size_t acap = 256;
                char *abuf = malloc(acap);
                const char *s = v;
                while (*s && !(*s == '"' && (s == v || *(s-1) != '\\'))) {
                    if (*s == '\\' && *(s+1)) {
                        s++;
                        if (*s == 'n') { if (abuf) abuf[alen++] = '\n'; }
                        else { if (abuf) abuf[alen++] = *s; }
                    } else {
                        if (abuf) abuf[alen++] = *s;
                    }
                    s++;
                    if (alen >= acap - 1) {
                        acap *= 2;
                        char *nb = realloc(abuf, acap);
                        if (!nb) { free(abuf); abuf = NULL; break; }
                        abuf = nb;
                    }
                }
                if (abuf) { abuf[alen] = '\0'; calls[idx].arguments_json = abuf; }
            } else if (strncmp(v, "null", 4) == 0) {
                calls[idx].arguments_json = NULL;
            }
        }

        idx++;
        p = obj_end + 1;
    }

    *out_calls = calls;
    *out_count = idx;
}

// ============================================================================
// MARK: - Create
// ============================================================================

adam_status_t adam_session_create(adam_memory_t *mem, char *out_id,
                                  size_t out_id_size) {
    if (!mem || !out_id || out_id_size < DBMEM_UUID_STR_MAXLEN)
        return ADAM_ERR_INVALID_PARAM;

    sqlite3 *db = adam_memory_db(mem);
    if (!db) return ADAM_ERR_SQLITE;

    // Generate UUIDv7
    char uuid[DBMEM_UUID_STR_MAXLEN];
    dbmem_uuid_v7(uuid);

    // Insert session row
    sqlite3_stmt *ins = NULL;
    int rc = sqlite3_prepare_v2(db,
        "INSERT INTO sessions(id) VALUES(?1)", -1, &ins, NULL);
    if (rc != SQLITE_OK) return ADAM_ERR_SQLITE;

    sqlite3_bind_text(ins, 1, uuid, -1, SQLITE_TRANSIENT);
    rc = sqlite3_step(ins);
    sqlite3_finalize(ins);

    if (rc != SQLITE_DONE) return ADAM_ERR_SQLITE;

    memcpy(out_id, uuid, DBMEM_UUID_STR_MAXLEN);
    return ADAM_OK;
}

// ============================================================================
// MARK: - Save
// ============================================================================

adam_status_t adam_session_save(adam_memory_t *mem, const char *session_id,
                                const adam_history_t *h) {
    if (!mem || !session_id || !h) return ADAM_ERR_INVALID_PARAM;

    sqlite3 *db = adam_memory_db(mem);
    if (!db) return ADAM_ERR_SQLITE;

    // Begin transaction
    sqlite3_exec(db, "BEGIN IMMEDIATE", NULL, NULL, NULL);

    // Upsert session row
    sqlite3_stmt *upsert;
    int rc = sqlite3_prepare_v2(db,
        "INSERT INTO sessions(id, updated_at) VALUES(?1, unixepoch()) "
        "ON CONFLICT(id) DO UPDATE SET updated_at=unixepoch()",
        -1, &upsert, NULL);
    if (rc != SQLITE_OK) goto fail;
    sqlite3_bind_text(upsert, 1, session_id, -1, SQLITE_STATIC);
    sqlite3_step(upsert);
    sqlite3_finalize(upsert);

    // Delete existing messages for this session
    sqlite3_stmt *del;
    rc = sqlite3_prepare_v2(db,
        "DELETE FROM messages WHERE session_id=?1", -1, &del, NULL);
    if (rc != SQLITE_OK) goto fail;
    sqlite3_bind_text(del, 1, session_id, -1, SQLITE_STATIC);
    sqlite3_step(del);
    sqlite3_finalize(del);

    // Insert all current messages
    sqlite3_stmt *ins;
    rc = sqlite3_prepare_v2(db,
        "INSERT INTO messages(session_id, idx, role, content, content_len, "
        "tool_call_id, tool_calls_json) VALUES(?1,?2,?3,?4,?5,?6,?7)",
        -1, &ins, NULL);
    if (rc != SQLITE_OK) goto fail;

    for (size_t i = 0; i < h->count; i++) {
        const adam_message_t *m = &h->items[i];
        sqlite3_bind_text(ins, 1, session_id, -1, SQLITE_STATIC);
        sqlite3_bind_int(ins, 2, (int)i);
        sqlite3_bind_int(ins, 3, (int)m->role);
        sqlite3_bind_text(ins, 4, m->content, -1, SQLITE_STATIC);
        sqlite3_bind_int(ins, 5, (int)m->content_len);
        sqlite3_bind_text(ins, 6, m->tool_call_id, -1, SQLITE_STATIC);

        if (m->tool_call_count > 0 && m->tool_calls) {
            char *tc_json = serialize_tool_calls(m->tool_calls,
                                                  m->tool_call_count);
            sqlite3_bind_text(ins, 7, tc_json, -1, SQLITE_TRANSIENT);
            free(tc_json);
        } else {
            sqlite3_bind_null(ins, 7);
        }

        sqlite3_step(ins);
        sqlite3_reset(ins);
    }
    sqlite3_finalize(ins);

    sqlite3_exec(db, "COMMIT", NULL, NULL, NULL);
    return ADAM_OK;

fail:
    sqlite3_exec(db, "ROLLBACK", NULL, NULL, NULL);
    return ADAM_ERR_SQLITE;
}

// ============================================================================
// MARK: - Load
// ============================================================================

adam_status_t adam_session_load(adam_memory_t *mem, const char *session_id,
                                adam_history_t *h) {
    if (!mem || !session_id || !h) return ADAM_ERR_INVALID_PARAM;

    sqlite3 *db = adam_memory_db(mem);
    if (!db) return ADAM_ERR_SQLITE;

    // Check session exists
    sqlite3_stmt *chk;
    int rc = sqlite3_prepare_v2(db,
        "SELECT 1 FROM sessions WHERE id=?1", -1, &chk, NULL);
    if (rc != SQLITE_OK) return ADAM_ERR_SQLITE;
    sqlite3_bind_text(chk, 1, session_id, -1, SQLITE_STATIC);
    rc = sqlite3_step(chk);
    sqlite3_finalize(chk);
    if (rc != SQLITE_ROW) return ADAM_ERR_INVALID_PARAM; // session not found

    // Clear existing history
    adam_history_clear(h);

    // Load messages in order
    sqlite3_stmt *sel;
    rc = sqlite3_prepare_v2(db,
        "SELECT role, content, tool_call_id, tool_calls_json "
        "FROM messages WHERE session_id=?1 ORDER BY idx",
        -1, &sel, NULL);
    if (rc != SQLITE_OK) return ADAM_ERR_SQLITE;
    sqlite3_bind_text(sel, 1, session_id, -1, SQLITE_STATIC);

    while (sqlite3_step(sel) == SQLITE_ROW) {
        adam_role_t role = (adam_role_t)sqlite3_column_int(sel, 0);
        const char *content = (const char *)sqlite3_column_text(sel, 1);
        const char *tc_id = (const char *)sqlite3_column_text(sel, 2);
        const char *tc_json = (const char *)sqlite3_column_text(sel, 3);

        if (role == ADAM_ROLE_ASSISTANT) {
            // Assistant message (with or without tool calls)
            adam_tool_call_entry_t *calls = NULL;
            size_t call_count = 0;
            if (tc_json)
                deserialize_tool_calls(tc_json, &calls, &call_count);

            // Convert to adam_tool_call_t for the append function
            adam_tool_call_t *tc_arr = NULL;
            if (call_count > 0) {
                tc_arr = calloc(call_count, sizeof(adam_tool_call_t));
                for (size_t i = 0; i < call_count; i++) {
                    tc_arr[i].id = calls[i].id;
                    tc_arr[i].name = calls[i].name;
                    tc_arr[i].arguments_json = calls[i].arguments_json;
                }
            }

            adam_history_append_assistant(h, content ? content : "",
                                          tc_arr, call_count);

            // Free temporary tc_arr (append copies the data)
            free(tc_arr);
            // Free deserialized calls
            for (size_t i = 0; i < call_count; i++) {
                free(calls[i].id);
                free(calls[i].name);
                free(calls[i].arguments_json);
            }
            free(calls);
        } else if (role == ADAM_ROLE_TOOL) {
            adam_history_append_tool(h, content ? content : "",
                                     tc_id);
        } else if (role == ADAM_ROLE_USER) {
            adam_history_append_user(h, content ? content : "");
        } else {
            // System or unknown — append as user (system gets rebuilt)
            adam_history_append_user(h, content ? content : "");
        }
    }
    sqlite3_finalize(sel);

    return ADAM_OK;
}

// ============================================================================
// MARK: - List Sessions
// ============================================================================

adam_status_t adam_session_list(adam_memory_t *mem,
                                char ***ids, size_t *count) {
    if (!mem || !ids || !count) return ADAM_ERR_INVALID_PARAM;
    *ids = NULL;
    *count = 0;

    sqlite3 *db = adam_memory_db(mem);
    if (!db) return ADAM_ERR_SQLITE;

    // Count sessions
    sqlite3_stmt *cnt;
    int rc = sqlite3_prepare_v2(db,
        "SELECT COUNT(*) FROM sessions", -1, &cnt, NULL);
    if (rc != SQLITE_OK) return ADAM_ERR_SQLITE;
    sqlite3_step(cnt);
    size_t n = (size_t)sqlite3_column_int(cnt, 0);
    sqlite3_finalize(cnt);

    if (n == 0) return ADAM_OK;

    char **arr = calloc(n, sizeof(char *));
    if (!arr) return ADAM_ERR_ALLOC;

    sqlite3_stmt *sel;
    rc = sqlite3_prepare_v2(db,
        "SELECT id FROM sessions ORDER BY updated_at DESC", -1, &sel, NULL);
    if (rc != SQLITE_OK) { free(arr); return ADAM_ERR_SQLITE; }

    size_t i = 0;
    while (sqlite3_step(sel) == SQLITE_ROW && i < n) {
        const char *id = (const char *)sqlite3_column_text(sel, 0);
        arr[i] = strdup(id ? id : "");
        if (!arr[i]) break;
        i++;
    }
    sqlite3_finalize(sel);

    *ids = arr;
    *count = i;
    return ADAM_OK;
}

// ============================================================================
// MARK: - Delete Session
// ============================================================================

adam_status_t adam_session_delete(adam_memory_t *mem, const char *session_id) {
    if (!mem || !session_id) return ADAM_ERR_INVALID_PARAM;

    sqlite3 *db = adam_memory_db(mem);
    if (!db) return ADAM_ERR_SQLITE;

    // Foreign key cascade deletes messages automatically
    sqlite3_stmt *del;
    int rc = sqlite3_prepare_v2(db,
        "DELETE FROM sessions WHERE id=?1", -1, &del, NULL);
    if (rc != SQLITE_OK) return ADAM_ERR_SQLITE;
    sqlite3_bind_text(del, 1, session_id, -1, SQLITE_STATIC);
    rc = sqlite3_step(del);
    sqlite3_finalize(del);

    return (rc == SQLITE_DONE) ? ADAM_OK : ADAM_ERR_SQLITE;
}

// ============================================================================
// MARK: - Sync Flag
// ============================================================================

adam_status_t adam_session_set_sync(adam_memory_t *mem, const char *session_id,
                                    int sync) {
    if (!mem || !session_id) return ADAM_ERR_INVALID_PARAM;

    sqlite3 *db = adam_memory_db(mem);
    if (!db) return ADAM_ERR_SQLITE;

    sqlite3_stmt *upd;
    int rc = sqlite3_prepare_v2(db,
        "UPDATE sessions SET sync=?2 WHERE id=?1", -1, &upd, NULL);
    if (rc != SQLITE_OK) return ADAM_ERR_SQLITE;
    sqlite3_bind_text(upd, 1, session_id, -1, SQLITE_STATIC);
    sqlite3_bind_int(upd, 2, sync ? 1 : 0);
    rc = sqlite3_step(upd);
    sqlite3_finalize(upd);

    return (rc == SQLITE_DONE) ? ADAM_OK : ADAM_ERR_SQLITE;
}

#endif // ADAM_NO_SQLITE
