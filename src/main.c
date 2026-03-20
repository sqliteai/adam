//
//  main.c
//  Adam CLI — Interactive agent with all features accessible via /commands
//
//  Usage:
//    adam --anthropic KEY [--model MODEL]
//    adam --openai KEY [--model MODEL] [--base-url URL]
//    adam --gemini KEY [--model MODEL]
//    adam --local model.gguf [--mmproj mmproj.gguf] [--ctx N]
//

#include "adam.h"
#ifndef ADAM_NO_SQLITE
#include "sqlite3.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>

// ============================================================================
// MARK: - Globals
// ============================================================================

static adam_settings_t  *g_settings = NULL;
static adam_history_t   *g_history  = NULL;
static adam_memory_t    *g_memory   = NULL;
static adam_memory_t    *g_userdb   = NULL;   // user's SQLite database (/db)
static adam_cache_t     *g_cache    = NULL;
static int               g_streaming = 1;
static int               g_turn = 0;
static char             *g_image_path = NULL;  // pending image attachment

// ============================================================================
// MARK: - Helpers
// ============================================================================

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

static int g_stream_started = 0;

static void on_stream(void *ctx, const char *chunk, size_t len, int is_done) {
    UNUSED_PARAM(ctx);
    if (len > 0) {
        if (!g_stream_started) {
            g_stream_started = 1;
        }
        fwrite(chunk, 1, len, stdout);
    }
    if (is_done) {
        printf("\n");
        g_stream_started = 0;
    }
    fflush(stdout);
}

static void signal_handler(int sig) {
    UNUSED_PARAM(sig);
    if (g_settings) adam_abort(g_settings);
}

static uint8_t *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len <= 0) { fclose(f); return NULL; }
    uint8_t *buf = malloc((size_t)len);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)len, f);
    fclose(f);
    if ((long)rd != len) { free(buf); return NULL; }
    *out_len = (size_t)len;
    return buf;
}

static adam_media_type_t guess_media_type(const char *path) {
    const char *ext = strrchr(path, '.');
    if (!ext) return ADAM_MEDIA_IMAGE_JPEG;
    if (strcasecmp(ext, ".png") == 0) return ADAM_MEDIA_IMAGE_PNG;
    if (strcasecmp(ext, ".gif") == 0) return ADAM_MEDIA_IMAGE_GIF;
    if (strcasecmp(ext, ".webp") == 0) return ADAM_MEDIA_IMAGE_WEBP;
    return ADAM_MEDIA_IMAGE_JPEG;
}

static void print_result_stats(const adam_run_result_t *r) {
    printf("  [%d in / %d out", r->input_tokens, r->output_tokens);
    if (r->cost_usd > 0) printf(" | $%.4f", r->cost_usd);
    printf(" | %.0fms", r->elapsed_ms);
    if (r->total_iterations > 0) printf(" | %d iters", r->total_iterations);
    printf("]\n");
}

// ============================================================================
// MARK: - Tool registration
// ============================================================================

static void register_tools(void) {
    adam_settings_t *s = g_settings;

    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "file_read", .description = "Read a file",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}",
        .execute = adam_tool_file_read, .ctx = s });
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "file_write", .description = "Write content to a file",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"},\"append\":{\"type\":\"boolean\"}},\"required\":[\"path\",\"content\"]}",
        .execute = adam_tool_file_write, .ctx = s });
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "list_directory", .description = "List directory contents",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}",
        .execute = adam_tool_list_directory, .ctx = s });
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "shell_exec", .description = "Execute a shell command",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\"},\"timeout\":{\"type\":\"integer\"}},\"required\":[\"command\"]}",
        .execute = adam_tool_shell_exec, .ctx = s });
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "calculator", .description = "Evaluate a math expression",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"expression\":{\"type\":\"string\"}},\"required\":[\"expression\"]}",
        .execute = adam_tool_calculator });
    adam_settings_add_tool(s, (adam_tool_def_t){
        .name = "web_search", .description = "Search the web",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"},\"count\":{\"type\":\"integer\"}},\"required\":[\"query\"]}",
        .execute = adam_tool_web_search, .ctx = s });
#ifndef ADAM_NO_SQLITE
    if (g_memory) {
        adam_settings_add_tool(s, (adam_tool_def_t){
            .name = "memory_search", .description = "Search long-term memory",
            .parameters_json = "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"},\"limit\":{\"type\":\"integer\"}},\"required\":[\"query\"]}",
            .execute = adam_tool_memory_search, .ctx = g_memory });
        adam_settings_add_tool(s, (adam_tool_def_t){
            .name = "memory_add", .description = "Add a fact to long-term memory",
            .parameters_json = "{\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"},\"context\":{\"type\":\"string\"},\"source\":{\"type\":\"string\"}},\"required\":[\"text\"]}",
            .execute = adam_tool_memory_add, .ctx = g_memory });
        adam_settings_add_tool(s, (adam_tool_def_t){
            .name = "sql_query", .description = "Execute SQL against the database",
            .parameters_json = "{\"type\":\"object\",\"properties\":{\"sql\":{\"type\":\"string\"}},\"required\":[\"sql\"]}",
            .execute = adam_tool_sql_query, .ctx = g_memory });
    }
#endif

    printf("  Tools enabled: file_read, file_write, list_directory, shell_exec, calculator, web_search");
#ifndef ADAM_NO_SQLITE
    if (g_memory) printf(", memory_search, memory_add, sql_query");
#endif
    printf("\n");
}

// ============================================================================
// MARK: - Slash command handlers
// ============================================================================

static void cmd_help(void) {
    printf("\n");
    printf("  Chat commands:\n");
    printf("    /image <path>       Attach image to next message\n");
    printf("    /clear              Clear conversation history\n");
    printf("    /history            Show message count and token estimate\n");
    printf("    /stream             Toggle streaming (currently %s)\n", g_streaming ? "on" : "off");
    printf("\n");
    printf("  Agent configuration:\n");
    printf("    /identity <text>    Set system identity\n");
    printf("    /instructions <text> Set additional instructions\n");
    printf("    /tools              Enable built-in tools (file, shell, calc, web)\n");
    printf("    /sandbox <dir>      Allow directory for file/shell tools\n");
    printf("    /cache              Enable response cache\n");
    printf("\n");
    printf("  Database & memory:\n");
    printf("    /db <path>          Open a SQLite database (enables sql_query tool)\n");
    printf("    /memory [db_path]   Enable memory system (default: adam_memory.db)\n");
    printf("    /session [id]       Create or load a session\n");
    printf("    /sessions           List all saved sessions\n");
    printf("\n");
    printf("  Advanced modes:\n");
    printf("    /json [hint]        Run next message in JSON mode\n");
    printf("    /research <question> Run autonomous research\n");
    printf("    /evolve <task>      Run evolution loop\n");
    printf("\n");
#if !defined(ADAM_NO_VOICE) && !defined(ADAM_NO_PTHREADS)
    printf("  Voice:\n");
    printf("    /tts                Enable text-to-speech (system)\n");
    printf("    /speak <text>       Speak text aloud\n");
    printf("    /talk               Start voice conversation mode\n");
    printf("\n");
#endif
#ifndef ADAM_NO_LOCAL
    printf("  Local model:\n");
    printf("    /verbose            Toggle llama.cpp verbose logging\n");
    printf("\n");
#endif
    printf("  Integrations:\n");
    printf("    /telegram           Continue this session on Telegram\n");
    printf("\n");
    printf("  General:\n");
    printf("    /status             Show current configuration\n");
    printf("    /help               Show this help\n");
    printf("    /quit               Exit\n");
    printf("\n");
}

static void cmd_status(void) {
    adam_settings_t *s = g_settings;
    printf("\n  Configuration:\n");

#ifndef ADAM_NO_LOCAL
    if (s->gguf_path) {
        printf("    Provider:     local (llama.cpp)\n");
        printf("    Model:        %s\n", s->gguf_path);
        if (s->mmproj_path)
            printf("    MMProj:       %s\n", s->mmproj_path);
        printf("    GPU layers:   %d\n", s->local_gpu_layers);
        printf("    Context:      %d\n", s->local_ctx_size);
        printf("    Verbose:      %s\n", s->local_verbose ? "on" : "off");
    } else
#endif
    {
        const char *fmt = "none";
        if (s->api_format == ADAM_API_ANTHROPIC) fmt = "anthropic";
        else if (s->api_format == ADAM_API_OPENAI) fmt = "openai";
        else if (s->api_format == ADAM_API_GEMINI) fmt = "gemini";
        printf("    Provider:     %s\n", fmt);
        printf("    Model:        %s\n", s->model ? s->model : "(default)");
        if (s->base_url)
            printf("    Base URL:     %s\n", s->base_url);
    }

    printf("    Temperature:  %.1f\n", s->temperature);
    printf("    Max tokens:   %d\n", s->max_tokens);
    printf("    Streaming:    %s\n", g_streaming ? "on" : "off");
    printf("    Tools:        %zu registered\n", s->tool_count);
    printf("    History:      %zu messages (~%zu tokens)\n",
           adam_history_count(g_history),
           adam_history_estimate_tokens(g_history));
    if (s->identity)
        printf("    Identity:     %.60s%s\n", s->identity,
               strlen(s->identity) > 60 ? "..." : "");
    if (g_memory)
        printf("    Memory:       enabled\n");
    if (s->session_id)
        printf("    Session:      %s\n", s->session_id);
    if (g_cache)
        printf("    Cache:        %zu entries, %zu hits, %zu misses\n",
               adam_cache_count(g_cache), adam_cache_hits(g_cache),
               adam_cache_misses(g_cache));
    printf("\n");
}

static void cmd_image(const char *arg) {
    if (!arg || !*arg) {
        printf("  Usage: /image <path>\n");
        return;
    }
    free(g_image_path);
    g_image_path = strdup(arg);
    printf("  Image queued: %s (will be attached to next message)\n", g_image_path);
}

static void cmd_clear(void) {
    adam_history_clear(g_history);
    printf("  History cleared.\n");
}

static void cmd_history(void) {
    printf("  %zu messages, ~%zu tokens\n",
           adam_history_count(g_history),
           adam_history_estimate_tokens(g_history));
}

static void cmd_stream(void) {
    g_streaming = !g_streaming;
    if (g_streaming)
        adam_settings_set_stream(g_settings, on_stream, NULL);
    else
        adam_settings_set_stream(g_settings, NULL, NULL);
    printf("  Streaming %s.\n", g_streaming ? "on" : "off");
}

static void cmd_identity(const char *arg) {
    if (!arg || !*arg) {
        printf("  Usage: /identity <text>\n");
        return;
    }
    adam_settings_set_identity(g_settings, arg);
    printf("  Identity set.\n");
}

static void cmd_instructions(const char *arg) {
    if (!arg || !*arg) {
        printf("  Usage: /instructions <text>\n");
        return;
    }
    adam_settings_set_instructions(g_settings, arg);
    printf("  Instructions set.\n");
}

static void cmd_sandbox(const char *arg) {
    if (!arg || !*arg) {
        printf("  Usage: /sandbox <directory>\n");
        return;
    }
    adam_status_t rc = adam_settings_allow_dir(g_settings, arg);
    if (rc == ADAM_OK)
        printf("  Sandbox: allowed %s\n", arg);
    else
        printf("  Failed to allow directory: %s\n", adam_status_string(rc));
}

static void cmd_cache(void) {
    if (g_cache) {
        printf("  Cache already enabled (%zu entries).\n", adam_cache_count(g_cache));
        return;
    }
    g_cache = adam_cache_create(256);
    g_settings->cache = g_cache;
    printf("  Response cache enabled (256 entries).\n");
}

#ifndef ADAM_NO_SQLITE

extern sqlite3 *adam_memory_db(adam_memory_t *mem);

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

    // Add row counts for each table
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

static void cmd_db(const char *arg) {
    if (!arg || !*arg) {
        printf("  Usage: /db <path_to_database.db>\n");
        return;
    }
    if (g_userdb) {
        adam_memory_close(g_userdb);
        adam_settings_remove_tool(g_settings, "sql_query");
        g_userdb = NULL;
    }

    g_userdb = adam_memory_open(arg);
    if (!g_userdb) {
        printf("  Failed to open database: %s\n", arg);
        return;
    }

    // Read schema and inject as instructions (kept alive — settings stores pointer)
    static char *db_instructions = NULL;
    free(db_instructions);
    char *schema = read_db_schema(g_userdb);
    if (schema) {
        size_t instr_len = strlen(schema) + 256;
        db_instructions = malloc(instr_len);
        if (db_instructions) {
            snprintf(db_instructions, instr_len,
                "You have access to a SQLite database. Use the sql_query tool "
                "to query it. Here is the database schema:\n%s", schema);
            adam_settings_set_instructions(g_settings, db_instructions);
        }
        printf("  Database schema:\n%s\n", schema);
        free(schema);
    }

    // Register sql_query tool pointing to user's database
    adam_settings_add_tool(g_settings, (adam_tool_def_t){
        .name = "sql_query",
        .description = "Execute a SQL query against the connected SQLite database",
        .parameters_json = "{\"type\":\"object\",\"properties\":"
                           "{\"sql\":{\"type\":\"string\",\"description\":\"SQL query to execute\"}},"
                           "\"required\":[\"sql\"]}",
        .execute = adam_tool_sql_query,
        .ctx = g_userdb
    });

    printf("  Database opened: %s\n", arg);
    printf("  Tool registered: sql_query\n");
}

static void cmd_memory(const char *arg) {
    if (g_memory) {
        printf("  Memory already enabled.\n");
        return;
    }
    const char *db_path = (arg && *arg) ? arg : "adam_memory.db";
    g_memory = adam_memory_open(db_path);
    if (!g_memory) {
        printf("  Failed to open memory database: %s\n", db_path);
        return;
    }
    adam_memory_configure(g_memory, g_settings);
    g_settings->memory = g_memory;
    g_settings->inject_memory = 1;
    g_settings->memory_extract = 1;
    printf("  Memory enabled: %s\n", db_path);
}

static void cmd_session(const char *arg) {
    if (!g_memory) {
        printf("  Memory must be enabled first. Use /memory\n");
        return;
    }
    static char session_id[64];

    if (arg && *arg) {
        // Load existing session
        strncpy(session_id, arg, sizeof(session_id) - 1);
        session_id[sizeof(session_id) - 1] = '\0';
        adam_history_clear(g_history);
        adam_status_t rc = adam_session_load(g_memory, session_id, g_history);
        if (rc == ADAM_OK) {
            g_settings->session_id = session_id;
            g_settings->auto_save = 1;
            printf("  Session loaded: %s (%zu messages)\n",
                   session_id, adam_history_count(g_history));
        } else {
            printf("  Failed to load session: %s\n", adam_status_string(rc));
        }
    } else {
        // Create new session
        adam_status_t rc = adam_session_create(g_memory, session_id,
                                               sizeof(session_id));
        if (rc == ADAM_OK) {
            g_settings->session_id = session_id;
            g_settings->auto_save = 1;
            printf("  New session: %s\n", session_id);
        } else {
            printf("  Failed to create session: %s\n", adam_status_string(rc));
        }
    }
}

static void cmd_sessions(void) {
    if (!g_memory) {
        printf("  Memory must be enabled first. Use /memory\n");
        return;
    }
    char **ids = NULL;
    size_t count = 0;
    adam_status_t rc = adam_session_list(g_memory, &ids, &count);
    if (rc != ADAM_OK) {
        printf("  Failed to list sessions: %s\n", adam_status_string(rc));
        return;
    }
    if (count == 0) {
        printf("  No saved sessions.\n");
    } else {
        printf("  Sessions (%zu):\n", count);
        for (size_t i = 0; i < count; i++) {
            printf("    %s%s\n", ids[i],
                   (g_settings->session_id &&
                    strcmp(ids[i], g_settings->session_id) == 0) ? " (active)" : "");
            free(ids[i]);
        }
        free(ids);
    }
}
#endif // ADAM_NO_SQLITE

static void cmd_json(const char *arg) {
    g_settings->response_format = "json";
    printf("  JSON mode enabled for next message.\n");
    if (arg && *arg) printf("  Hint: %s\n", arg);
}

static void cmd_research(const char *arg) {
    if (!arg || !*arg) {
        printf("  Usage: /research <question>\n");
        return;
    }

    adam_research_config_t cfg = adam_research_config_defaults();
    cfg.question = arg;
    cfg.max_iterations = 5;
    cfg.on_progress = NULL;

    printf("  Researching: %s\n\n", arg);
    adam_research_result_t r = adam_research(g_settings, &cfg);

    if (r.status == ADAM_OK && r.report) {
        printf("%s\n\n", r.report);
        printf("  [%d iterations, %zu findings, %d in / %d out, $%.4f, %.0fms]\n",
               r.total_iterations, r.finding_count,
               r.total_input_tokens, r.total_output_tokens,
               r.total_cost_usd, r.elapsed_ms);
    } else {
        printf("  Research failed: %s\n", adam_status_string(r.status));
    }
    adam_research_result_free(&r);
}

static int evolve_eval(void *ctx, const char *output, int iteration) {
    UNUSED_PARAM(ctx);
    UNUSED_PARAM(iteration);
    if (!output) return 0;
    int score = 0;
    size_t len = strlen(output);
    if (len >= 200) score += 40;
    else if (len >= 100) score += 25;
    else if (len >= 50) score += 10;
    if (strstr(output, "1.") && strstr(output, "2.")) score += 30;
    if (len >= 300) score += 30;
    return score > 100 ? 100 : score;
}

static void cmd_evolve(const char *arg) {
    if (!arg || !*arg) {
        printf("  Usage: /evolve <task>\n");
        return;
    }

    adam_evolve_config_t cfg = adam_evolve_config_defaults();
    cfg.task = arg;
    cfg.max_iterations = 5;
    cfg.target_score = 90;
    cfg.eval_fn = evolve_eval;

    printf("  Evolving: %s\n\n", arg);
    adam_evolve_result_t r = adam_evolve(g_settings, &cfg);

    if (r.status == ADAM_OK && r.best_output) {
        printf("Best (score %d, iteration %d):\n%s\n\n", r.best_score,
               r.best_iteration, r.best_output);
        if (r.strategy) printf("Strategy: %s\n\n", r.strategy);
        printf("  [%d iterations, %d in / %d out, $%.4f, %.0fms]\n",
               r.attempt_total, r.total_input_tokens,
               r.total_output_tokens, r.total_cost_usd, r.elapsed_ms);
    } else {
        printf("  Evolution failed: %s\n", adam_status_string(r.status));
    }
    adam_evolve_result_free(&r);
}

#if !defined(ADAM_NO_VOICE) && !defined(ADAM_NO_PTHREADS)
static void cmd_tts(void) {
    adam_settings_set_tts(g_settings, ADAM_TTS_SYSTEM, NULL, NULL, NULL, NULL);
    printf("  TTS enabled (system voice).\n");
}

static void cmd_speak(const char *arg) {
    if (!arg || !*arg) {
        printf("  Usage: /speak <text>\n");
        return;
    }
    adam_status_t rc = adam_tts_speak(g_settings, arg);
    if (rc != ADAM_OK)
        printf("  TTS error: %s\n", adam_status_string(rc));
}

static void cmd_talk(void) {
    const char *openai_key = env_load("OPENAI_API_KEY");
    if (!openai_key) openai_key = g_settings->api_key;
    if (!openai_key) {
        printf("  Need OPENAI_API_KEY in .env for Whisper STT.\n");
        return;
    }
    adam_settings_set_stt(g_settings, ADAM_STT_CLOUD, NULL, openai_key, NULL);
    adam_settings_set_tts(g_settings, ADAM_TTS_SYSTEM, NULL, NULL, NULL, NULL);
    g_settings->voice_enabled = 1;

    printf("  Starting voice mode. Speak into the microphone. Ctrl+C to stop.\n");
    adam_status_t rc = adam_voice_start(g_settings, g_history);
    if (rc != ADAM_OK) {
        printf("  Voice start failed: %s\n", adam_status_string(rc));
        return;
    }

    // Block until Ctrl+C
    while (adam_voice_is_running(g_settings)) {
        struct timespec ts = {0, 100000000}; // 100ms
        nanosleep(&ts, NULL);
    }
    adam_voice_stop(g_settings);
    printf("  Voice mode stopped.\n");
}
#endif // !ADAM_NO_VOICE && !ADAM_NO_PTHREADS

#ifndef ADAM_NO_LOCAL
static void cmd_verbose(void) {
    g_settings->local_verbose = !g_settings->local_verbose;
    printf("  llama.cpp verbose: %s\n", g_settings->local_verbose ? "on" : "off");
}
#endif

// ============================================================================
// MARK: - Telegram integration (/telegram command)
// ============================================================================

#include "adam_net.h"

static char *g_tg_token = NULL;
static int64_t g_tg_chat_id = 0;
static int g_tg_offset = 0;

static char *tg_url(const char *method) {
    static char url[512];
    snprintf(url, sizeof(url), "https://api.telegram.org/bot%s/%s",
             g_tg_token, method);
    return url;
}

static char *tg_post(const char *method, const char *body) {
    arena_t *a = arena_create(8192);
    if (!a) return NULL;
    adam_net_response_t r = adam_net_post_json(
        g_settings, a, tg_url(method), "", body, NULL, 0);
    char *out = NULL;
    if (r.error == ADAM_OK && r.data && r.data_len > 0) {
        out = malloc(r.data_len + 1);
        if (out) { memcpy(out, r.data, r.data_len); out[r.data_len] = '\0'; }
    }
    arena_destroy(a);
    return out;
}

static void tg_send(int64_t chat_id, const char *text) {
    if (!text || !*text) return;
    size_t tlen = strlen(text);
    size_t est = tlen * 2 + 128;
    char *body = malloc(est);
    if (!body) return;
    size_t pos = (size_t)snprintf(body, est,
        "{\"chat_id\":%lld,\"text\":\"", (long long)chat_id);
    for (size_t i = 0; i < tlen && pos < est - 4; i++) {
        char c = text[i];
        if (c == '"')       { body[pos++] = '\\'; body[pos++] = '"'; }
        else if (c == '\\') { body[pos++] = '\\'; body[pos++] = '\\'; }
        else if (c == '\n') { body[pos++] = '\\'; body[pos++] = 'n'; }
        else if (c == '\r') { body[pos++] = '\\'; body[pos++] = 'r'; }
        else if (c == '\t') { body[pos++] = '\\'; body[pos++] = 't'; }
        else body[pos++] = c;
    }
    snprintf(body + pos, est - pos, "\"}");
    char *r = tg_post("sendMessage", body);
    free(r); free(body);
}

static void tg_typing(int64_t chat_id) {
    char body[128];
    snprintf(body, sizeof(body),
             "{\"chat_id\":%lld,\"action\":\"typing\"}", (long long)chat_id);
    char *r = tg_post("sendChatAction", body);
    free(r);
}

// Simple JSON helpers (same as telegram example)
static char *tg_json_str(const char *json, const char *key) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return NULL;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    if (*p != '"') return NULL;
    p++;
    const char *end = p;
    while (*end && *end != '"') { if (*end == '\\') end++; end++; }
    size_t len = (size_t)(end - p);
    char *val = malloc(len + 1);
    if (val) { memcpy(val, p, len); val[len] = '\0'; }
    return val;
}

static int64_t tg_json_int(const char *json, const char *key) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return 0;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    return strtoll(p, NULL, 10);
}

static uint8_t *tg_download(const char *file_id, size_t *out_len) {
    *out_len = 0;
    char body[256];
    snprintf(body, sizeof(body), "{\"file_id\":\"%s\"}", file_id);
    char *resp = tg_post("getFile", body);
    if (!resp) return NULL;
    char *file_path = tg_json_str(resp, "file_path");
    free(resp);
    if (!file_path) return NULL;

    char url[512];
    snprintf(url, sizeof(url), "https://api.telegram.org/file/bot%s/%s",
             g_tg_token, file_path);
    free(file_path);

    char tmp[256], cmd[768];
    snprintf(tmp, sizeof(tmp), "/tmp/adam_tg_%d", (int)getpid());
    snprintf(cmd, sizeof(cmd), "curl -s -o '%s' '%s'", tmp, url);
    if (system(cmd) != 0) return NULL;

    FILE *fp = fopen(tmp, "rb");
    if (!fp) { unlink(tmp); return NULL; }
    fseek(fp, 0, SEEK_END);
    long flen = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (flen <= 0) { fclose(fp); unlink(tmp); return NULL; }
    uint8_t *data = malloc((size_t)flen);
    if (data) { fread(data, 1, (size_t)flen, fp); *out_len = (size_t)flen; }
    fclose(fp); unlink(tmp);
    return data;
}

static void tg_process_update(const char *upd) {
    const char *msg = strstr(upd, "\"message\"");
    if (!msg) return;

    const char *chat_obj = strstr(msg, "\"chat\"");
    int64_t chat_id = chat_obj ? tg_json_int(chat_obj, "id") : 0;
    if (chat_id == 0) return;
    g_tg_chat_id = chat_id;

    char *text = tg_json_str(msg, "text");

    // Photo
    const char *photo = strstr(msg, "\"photo\"");
    if (photo) {
        const char *last_fid = NULL, *p = photo;
        while ((p = strstr(p, "\"file_id\"")) != NULL) { last_fid = p; p += 9; }
        if (last_fid) {
            char *fid = tg_json_str(last_fid - 1, "file_id");
            if (fid) {
                printf("  [TG] Photo received\n");
                size_t img_len = 0;
                uint8_t *img = tg_download(fid, &img_len);
                free(fid);
                if (img && img_len > 0) {
                    if (!text) text = tg_json_str(msg, "caption");
                    if (!text) text = strdup("Describe this image.");
                    adam_history_append_user(g_history, text);
                    adam_history_attach(g_history, ADAM_MEDIA_IMAGE_JPEG,
                                        img, img_len, "photo.jpg");
                    free(img);
                    tg_typing(chat_id);
                    adam_run_result_t r = adam_run(g_settings, g_history, NULL);
                    if (r.status == ADAM_OK && r.final_response) {
                        tg_send(chat_id, r.final_response);
                        printf("  [TG] → %s\n", r.final_response);
                    }
                    adam_run_result_free(&r);
                    free(text);
                    return;
                }
            }
        }
    }

    // Voice
#if !defined(ADAM_NO_VOICE) && !defined(ADAM_NO_PTHREADS)
    const char *voice = strstr(msg, "\"voice\"");
    if (voice && g_settings->stt_backend != ADAM_STT_NONE) {
        char *vfid = tg_json_str(voice, "file_id");
        if (vfid) {
            printf("  [TG] Voice received\n");
            size_t alen = 0;
            uint8_t *aud = tg_download(vfid, &alen);
            free(vfid);
            if (aud && alen > 0) {
                arena_t *sa = arena_create(4096);
                const char *transcript = NULL;
                adam_status_t src = adam_stt_transcribe(g_settings, sa,
                    aud, alen, ADAM_AUDIO_OGG_OPUS, &transcript);
                free(aud);
                if (src == ADAM_OK && transcript && *transcript) {
                    printf("  [TG] Transcribed: %s\n", transcript);
                    tg_typing(chat_id);
                    adam_run_result_t r = adam_run(g_settings, g_history,
                                                   transcript);
                    arena_destroy(sa);
                    if (r.status == ADAM_OK && r.final_response) {
                        tg_send(chat_id, r.final_response);
                        printf("  [TG] → %s\n", r.final_response);
                    }
                    adam_run_result_free(&r);
                } else {
                    arena_destroy(sa);
                    tg_send(chat_id, "Could not transcribe voice message.");
                }
                free(text);
                return;
            }
        }
    }
#endif

    if (!text || !*text) { free(text); return; }

    printf("  [TG] %s\n", text);
    tg_typing(chat_id);
    adam_run_result_t r = adam_run(g_settings, g_history, text);
    if (r.status == ADAM_OK && r.final_response) {
        tg_send(chat_id, r.final_response);
        printf("  [TG] → %s\n", r.final_response);
    }
    adam_run_result_free(&r);
    free(text);
}

static void cmd_telegram(void) {
    if (!g_tg_token) g_tg_token = env_load("TELEGRAM_BOT_TOKEN");
    if (!g_tg_token) {
        printf("  Set TELEGRAM_BOT_TOKEN in .env first.\n");
        printf("  Get one from @BotFather on Telegram.\n");
        return;
    }

    // Enable STT for voice messages if OpenAI key available
#if !defined(ADAM_NO_VOICE) && !defined(ADAM_NO_PTHREADS)
    if (g_settings->stt_backend == ADAM_STT_NONE) {
        char *oai = env_load("OPENAI_API_KEY");
        if (oai) adam_settings_set_stt(g_settings, ADAM_STT_CLOUD,
                                        NULL, oai, "whisper-1");
    }
#endif

    printf("  Telegram mode. Session continues on Telegram.\n");
    printf("  Send a message to the bot to start. Ctrl+C to return to CLI.\n");
    printf("  History: %zu messages carried over.\n\n",
           adam_history_count(g_history));

    // Polling loop (shares g_settings + g_history with CLI)
    while (!g_settings->abort_flag) {
        char body[128];
        snprintf(body, sizeof(body),
            "{\"offset\":%d,\"timeout\":30,\"allowed_updates\":[\"message\"]}",
            g_tg_offset);
        char *resp = tg_post("getUpdates", body);
        if (!resp) continue;

        if (!strstr(resp, "\"ok\":true")) {
            fprintf(stderr, "  Telegram error: %.100s\n", resp);
            free(resp);
            continue;
        }

        const char *p = strstr(resp, "\"result\"");
        if (p) {
            while ((p = strstr(p, "\"update_id\"")) != NULL) {
                int uid = (int)tg_json_int(p - 1, "update_id");
                if (uid >= g_tg_offset) g_tg_offset = uid + 1;
                tg_process_update(p);
                p += 11;
            }
        }
        free(resp);
    }

    adam_abort_reset(g_settings);
    printf("  Telegram mode ended. Back to CLI.\n\n");
}

// ============================================================================
// MARK: - Command dispatch
// ============================================================================

// Returns 1 if the input was a command (handled), 0 if it's a chat message.
static int handle_command(const char *input) {
    if (input[0] != '/') return 0;

    const char *cmd = input + 1;
    const char *arg = strchr(cmd, ' ');
    size_t cmd_len = arg ? (size_t)(arg - cmd) : strlen(cmd);
    if (arg) arg++; // skip space
    while (arg && *arg == ' ') arg++; // skip extra spaces

    #define CMD(name) (cmd_len == strlen(name) && strncmp(cmd, name, cmd_len) == 0)

    if (CMD("help") || CMD("?"))        { cmd_help(); return 1; }
    if (CMD("quit") || CMD("exit")
        || CMD("q"))                    { exit(0); }
    if (CMD("status"))                  { cmd_status(); return 1; }
    if (CMD("clear"))                   { cmd_clear(); return 1; }
    if (CMD("history"))                 { cmd_history(); return 1; }
    if (CMD("stream"))                  { cmd_stream(); return 1; }
    if (CMD("image"))                   { cmd_image(arg); return 1; }
    if (CMD("identity"))                { cmd_identity(arg); return 1; }
    if (CMD("instructions"))            { cmd_instructions(arg); return 1; }
    if (CMD("tools"))                   { register_tools(); return 1; }
    if (CMD("sandbox"))                 { cmd_sandbox(arg); return 1; }
    if (CMD("cache"))                   { cmd_cache(); return 1; }
    if (CMD("json"))                    { cmd_json(arg); return 1; }
    if (CMD("research"))                { cmd_research(arg); return 1; }
    if (CMD("evolve"))                  { cmd_evolve(arg); return 1; }
    if (CMD("telegram"))                { cmd_telegram(); return 1; }

#ifndef ADAM_NO_SQLITE
    if (CMD("db"))                      { cmd_db(arg); return 1; }
    if (CMD("memory"))                  { cmd_memory(arg); return 1; }
    if (CMD("session"))                 { cmd_session(arg); return 1; }
    if (CMD("sessions"))                { cmd_sessions(); return 1; }
#endif

#if !defined(ADAM_NO_VOICE) && !defined(ADAM_NO_PTHREADS)
    if (CMD("tts"))                     { cmd_tts(); return 1; }
    if (CMD("speak"))                   { cmd_speak(arg); return 1; }
    if (CMD("talk"))                    { cmd_talk(); return 1; }
#endif

#ifndef ADAM_NO_LOCAL
    if (CMD("verbose"))                 { cmd_verbose(); return 1; }
#endif

    #undef CMD

    printf("  Unknown command: /%.*s (type /help for commands)\n",
           (int)cmd_len, cmd);
    return 1;
}

// ============================================================================
// MARK: - Chat turn
// ============================================================================

static void run_turn(const char *input) {
    g_turn++;
    adam_abort_reset(g_settings);

    // If an image is queued, attach it to the message
    if (g_image_path) {
        size_t img_len = 0;
        uint8_t *img_data = read_file(g_image_path, &img_len);
        if (img_data) {
            adam_history_append_user(g_history, input);
            adam_history_attach(g_history, guess_media_type(g_image_path),
                                img_data, img_len, g_image_path);
            free(img_data);

            printf("[%d] Adam:\n", g_turn);
            fflush(stdout);

            adam_run_result_t r = adam_run(g_settings, g_history, NULL);

            if (r.status != ADAM_OK) {
                printf("(error: %s)\n",
                       r.final_response ? r.final_response
                                        : adam_status_string(r.status));
            } else if (!g_streaming && r.final_response) {
                printf("%s\n", r.final_response);
            }

            print_result_stats(&r);
            adam_run_result_free(&r);
        } else {
            printf("  Failed to read image: %s\n", g_image_path);
        }
        free(g_image_path);
        g_image_path = NULL;
        return;
    }

    // JSON mode: use adam_run_json for this turn
    if (g_settings->response_format &&
        strcmp(g_settings->response_format, "json") == 0) {
        printf("[%d] Adam (JSON):\n", g_turn);
        fflush(stdout);

        adam_json_result_t r = adam_run_json(g_settings, g_history, input,
                                             NULL, 3);

        if (r.base.status != ADAM_OK) {
            printf("(error: %s)\n", adam_status_string(r.base.status));
        } else {
            if (!g_streaming && r.base.final_response)
                printf("%s\n", r.base.final_response);
            if (!r.json_valid)
                printf("  (warning: response was not valid JSON after %d retries)\n",
                       r.retries_used);
        }

        print_result_stats(&r.base);
        adam_json_result_free(&r);
        g_settings->response_format = NULL; // one-shot
        return;
    }

    // Normal chat turn
    printf("[%d] Adam:\n", g_turn);
    fflush(stdout);

    adam_run_result_t r = adam_run(g_settings, g_history, input);

    if (r.status != ADAM_OK) {
        printf("(error: %s)\n",
               r.final_response ? r.final_response
                                : adam_status_string(r.status));
    } else if (!g_streaming && r.final_response) {
        printf("%s\n", r.final_response);
    }

    // TTS: speak the response if TTS is configured
#if !defined(ADAM_NO_VOICE) && !defined(ADAM_NO_PTHREADS)
    if (r.status == ADAM_OK && r.final_response &&
        g_settings->tts_backend != ADAM_TTS_NONE &&
        !g_settings->voice_enabled) {
        adam_tts_speak(g_settings, r.final_response);
    }
#endif

    print_result_stats(&r);
    adam_run_result_free(&r);
}

// ============================================================================
// MARK: - Argument parsing
// ============================================================================

static void print_usage(const char *prog) {
    printf("Usage:\n");
    printf("  %s --anthropic <api_key> [--model <model>]\n", prog);
    printf("  %s --openai <api_key> [--model <model>] [--base-url <url>]\n", prog);
    printf("  %s --gemini <api_key> [--model <model>]\n", prog);
#ifndef ADAM_NO_LOCAL
    printf("  %s --local <model.gguf> [--mmproj <mmproj.gguf>] [--ctx <size>]\n", prog);
#endif
    printf("\nOptions:\n");
    printf("  --model <name>     Model name (provider-specific default if omitted)\n");
    printf("  --base-url <url>   Override API endpoint (for Groq, Together, Ollama, etc.)\n");
#ifndef ADAM_NO_LOCAL
    printf("  --mmproj <path>    Multimodal projector GGUF for vision models\n");
    printf("  --ctx <size>       Context window size (default: 4096)\n");
#endif
    printf("  --temperature <f>  Generation temperature (default: 0.7)\n");
    printf("  --max-tokens <n>   Max output tokens (default: 4096)\n");
    printf("  --identity <text>  System identity prompt\n");
    printf("  --embedding-model <path>  GGUF embedding model for local memory\n");
    printf("\nType /help during chat for interactive commands.\n");
}

static int parse_args(int argc, char **argv) {
    adam_settings_t *s = g_settings;
    const char *model = NULL;
    const char *base_url = NULL;
    adam_api_format_t format = ADAM_API_NONE;
    const char *api_key = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--anthropic") == 0 && i + 1 < argc) {
            format = ADAM_API_ANTHROPIC;
            api_key = argv[++i];
        } else if (strcmp(argv[i], "--openai") == 0 && i + 1 < argc) {
            format = ADAM_API_OPENAI;
            api_key = argv[++i];
        } else if (strcmp(argv[i], "--gemini") == 0 && i + 1 < argc) {
            format = ADAM_API_GEMINI;
            api_key = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model = argv[++i];
#ifndef ADAM_NO_LOCAL
        } else if (strcmp(argv[i], "--local") == 0 && i + 1 < argc) {
            adam_settings_set_local(s, argv[++i], -1, 4096);
        } else if (strcmp(argv[i], "--mmproj") == 0 && i + 1 < argc) {
            adam_settings_set_mmproj(s, argv[++i]);
        } else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            s->local_ctx_size = atoi(argv[++i]);
#endif
        } else if (strcmp(argv[i], "--base-url") == 0 && i + 1 < argc) {
            base_url = argv[++i];
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            s->temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            s->max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--identity") == 0 && i + 1 < argc) {
            adam_settings_set_identity(s, argv[++i]);
        } else if (strcmp(argv[i], "--embedding-model") == 0 && i + 1 < argc) {
            s->memory_embedding_model = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return -1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    // Set up remote provider if specified
    if (format != ADAM_API_NONE && api_key) {
        adam_settings_set_provider(s, format, api_key, model);
        if (base_url)
            adam_settings_set_base_url(s, base_url);
    }

    // Auto-detect from .env if no provider set and no local model
#ifndef ADAM_NO_LOCAL
    if (format == ADAM_API_NONE && !s->gguf_path) {
#else
    if (format == ADAM_API_NONE) {
#endif
        char *key;
        if ((key = env_load("ANTHROPIC_API_KEY"))) {
            adam_settings_set_provider(s, ADAM_API_ANTHROPIC, key, model);
        } else if ((key = env_load("GEMINI_API_KEY"))) {
            adam_settings_set_provider(s, ADAM_API_GEMINI, key,
                                       model ? model : "gemini-2.5-flash");
        } else if ((key = env_load("OPENAI_API_KEY"))) {
            adam_settings_set_provider(s, ADAM_API_OPENAI, key,
                                       model ? model : "gpt-4o-mini");
        } else {
            fprintf(stderr, "No provider configured. Use --anthropic/--openai/--gemini/--local or set API key in .env\n");
            print_usage(argv[0]);
            return 1;
        }
    }

    return 0;
}

// ============================================================================
// MARK: - Main
// ============================================================================

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    adam_init();

    g_settings = adam_create_settings();
    g_history = adam_history_create();

    int rc = parse_args(argc, argv);
    if (rc != 0) {
        adam_history_destroy(g_history);
        adam_settings_destroy(g_settings);
        adam_cleanup();
        return rc < 0 ? 0 : rc; // -1 = --help (success)
    }

    // Enable streaming by default
    if (g_streaming)
        adam_settings_set_stream(g_settings, on_stream, NULL);

    // Handle Ctrl+C gracefully
    signal(SIGINT, signal_handler);

    // Print banner
    printf("Adam %s\n", ADAM_VERSION_STRING);
#ifndef ADAM_NO_LOCAL
    if (g_settings->gguf_path) {
        printf("  Model:  %s (local)\n", g_settings->gguf_path);
        if (g_settings->mmproj_path)
            printf("  MMProj: %s\n", g_settings->mmproj_path);
    } else
#endif
    {
        printf("  Model:  %s\n", g_settings->model ? g_settings->model : "(default)");
    }
    printf("  Type /help for commands. Ctrl+C to abort. Ctrl+D to quit.\n\n");

    // Main loop
    char input[8192];
    while (1) {
        printf("[%d] You: ", g_turn + 1);
        fflush(stdout);

        if (!fgets(input, sizeof(input), stdin)) break; // EOF

        // Trim trailing newline
        size_t len = strlen(input);
        while (len > 0 && (input[len-1] == '\n' || input[len-1] == '\r'))
            input[--len] = '\0';

        if (len == 0) continue;

        if (handle_command(input)) continue;

        run_turn(input);
        printf("\n");
    }

    printf("\nGoodbye!\n");

    // Cleanup
    free(g_image_path);
    if (g_cache) adam_cache_destroy(g_cache);
#ifndef ADAM_NO_SQLITE
    if (g_userdb) adam_memory_close(g_userdb);
    if (g_memory) adam_memory_close(g_memory);
#endif
    adam_history_destroy(g_history);
    adam_settings_destroy(g_settings);
    adam_cleanup();
    return 0;
}
