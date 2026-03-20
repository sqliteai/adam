//
//  telegram/main.c
//  Adam Telegram Bot — Connect Adam to a Telegram chat
//
//  Each Telegram chat gets its own Adam session with full feature support:
//  text, images, tools, memory, sessions, /commands.
//
//  Usage: ./telegram [--local model.gguf] [--mmproj mmproj.gguf]
//  Requires TELEGRAM_BOT_TOKEN in .env (and an LLM provider key).
//

#include "adam.h"
#include "adam_net.h"
#ifndef ADAM_NO_SQLITE
#include "sqlite3.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>

// ============================================================================
// MARK: - Globals
// ============================================================================

static adam_settings_t *g_settings  = NULL;
static adam_memory_t   *g_memory    = NULL;
static volatile int     g_running   = 1;
static char            *g_bot_token = NULL;
static int              g_offset    = 0;   // Telegram update offset

// Per-chat state (simple: support one active chat)
static adam_history_t  *g_history   = NULL;
static int64_t          g_chat_id   = 0;

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

static void signal_handler(int sig) {
    UNUSED_PARAM(sig);
    g_running = 0;
    if (g_settings) adam_abort(g_settings);
}

// Simple JSON string extractor (no jsmn dependency — just find "key":"value")
static char *json_get_string(const char *json, const char *key) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return NULL;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    if (*p != '"') return NULL;
    p++; // skip opening quote
    const char *end = p;
    while (*end && *end != '"') {
        if (*end == '\\') end++; // skip escaped char
        end++;
    }
    size_t len = (size_t)(end - p);
    char *val = malloc(len + 1);
    if (!val) return NULL;
    memcpy(val, p, len);
    val[len] = '\0';
    return val;
}

static int64_t json_get_int(const char *json, const char *key) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return 0;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    return strtoll(p, NULL, 10);
}

// ============================================================================
// MARK: - Telegram API helpers
// ============================================================================

static char *tg_api_url(const char *method) {
    static char url[512];
    snprintf(url, sizeof(url), "https://api.telegram.org/bot%s/%s",
             g_bot_token, method);
    return url;
}

// POST JSON to Telegram API, return malloc'd response body (caller frees)
static char *tg_post(const char *method, const char *body) {
    arena_t *arena = arena_create(8192);
    if (!arena) return NULL;

    adam_net_response_t resp = adam_net_post_json(
        g_settings, arena, tg_api_url(method), "", body, NULL, 0);

    char *result = NULL;
    if (resp.error == ADAM_OK && resp.data && resp.data_len > 0) {
        result = malloc(resp.data_len + 1);
        if (result) {
            memcpy(result, resp.data, resp.data_len);
            result[resp.data_len] = '\0';
        }
    }

    arena_destroy(arena);
    return result;
}

// Send a text message to a chat
static void tg_send_message(int64_t chat_id, const char *text) {
    if (!text || !*text) return;

    // Escape for JSON
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
    pos += (size_t)snprintf(body + pos, est - pos, "\"}");

    char *resp = tg_post("sendMessage", body);
    free(resp);
    free(body);
}

// Send a "typing..." indicator
static void tg_send_typing(int64_t chat_id) {
    char body[128];
    snprintf(body, sizeof(body),
             "{\"chat_id\":%lld,\"action\":\"typing\"}", (long long)chat_id);
    char *resp = tg_post("sendChatAction", body);
    free(resp);
}

// Download a file by file_id, return malloc'd data (caller frees)
static uint8_t *tg_download_file(const char *file_id, size_t *out_len) {
    *out_len = 0;

    // Step 1: getFile to get file_path
    char body[256];
    snprintf(body, sizeof(body), "{\"file_id\":\"%s\"}", file_id);
    char *resp = tg_post("getFile", body);
    if (!resp) return NULL;

    char *file_path = json_get_string(resp, "file_path");
    free(resp);
    if (!file_path) return NULL;

    // Step 2: Download via URL
    char url[512];
    snprintf(url, sizeof(url), "https://api.telegram.org/file/bot%s/%s",
             g_bot_token, file_path);
    free(file_path);

    arena_t *arena = arena_create(1024 * 1024); // 1MB for photos
    if (!arena) return NULL;

    adam_net_response_t net = adam_net_post_json(
        g_settings, arena, url, "", "", NULL, 0);

    uint8_t *data = NULL;
    if (net.error == ADAM_OK && net.data && net.data_len > 0) {
        data = malloc(net.data_len);
        if (data) {
            memcpy(data, net.data, net.data_len);
            *out_len = net.data_len;
        }
    }

    arena_destroy(arena);
    return data;
}

// ============================================================================
// MARK: - Process a single Telegram message
// ============================================================================

static void process_message(const char *update_json) {
    // Extract chat_id and text
    // Find the "message" object first
    const char *msg = strstr(update_json, "\"message\"");
    if (!msg) return;

    int64_t chat_id = json_get_int(msg, "chat_id");
    if (chat_id == 0) return;

    // Reset history if chat changed
    if (chat_id != g_chat_id) {
        adam_history_clear(g_history);
        g_chat_id = chat_id;
        printf("  New chat: %lld\n", (long long)chat_id);
    }

    char *text = json_get_string(msg, "text");

    // Check for photo attachments
    const char *photo = strstr(msg, "\"photo\"");
    if (photo) {
        // Find the largest photo (last in array — has highest resolution)
        const char *last_file_id = NULL;
        const char *p = photo;
        while ((p = strstr(p, "\"file_id\"")) != NULL) {
            last_file_id = p;
            p += 9;
        }

        if (last_file_id) {
            char *file_id = json_get_string(last_file_id - 1, "file_id");
            if (file_id) {
                printf("  Downloading photo: %s\n", file_id);
                size_t img_len = 0;
                uint8_t *img_data = tg_download_file(file_id, &img_len);
                free(file_id);

                if (img_data && img_len > 0) {
                    // Get caption as text if no text field
                    if (!text) text = json_get_string(msg, "caption");
                    if (!text) text = strdup("Describe this image.");

                    adam_history_append_user(g_history, text);
                    adam_history_attach(g_history, ADAM_MEDIA_IMAGE_JPEG,
                                        img_data, img_len, "photo.jpg");
                    free(img_data);

                    tg_send_typing(chat_id);
                    adam_run_result_t r = adam_run(g_settings, g_history, NULL);

                    if (r.status == ADAM_OK && r.final_response)
                        tg_send_message(chat_id, r.final_response);
                    else
                        tg_send_message(chat_id, "(error processing image)");

                    printf("  [%d in / %d out | %.0fms]\n",
                           r.input_tokens, r.output_tokens, r.elapsed_ms);
                    adam_run_result_free(&r);
                    free(text);
                    return;
                }
            }
        }
    }

    if (!text || !*text) { free(text); return; }

    printf("  [%lld] %s\n", (long long)chat_id, text);

    // Handle /commands
    if (text[0] == '/') {
        if (strcmp(text, "/start") == 0) {
            tg_send_message(chat_id,
                "Hello! I'm Adam, an AI assistant.\n\n"
                "Send me text or images and I'll respond.\n\n"
                "Commands:\n"
                "/tools - Enable tools (file, calc, web)\n"
                "/memory - Enable persistent memory\n"
                "/clear - Clear conversation\n"
                "/status - Show configuration");
            free(text);
            return;
        }
        if (strcmp(text, "/clear") == 0) {
            adam_history_clear(g_history);
            tg_send_message(chat_id, "History cleared.");
            free(text);
            return;
        }
        if (strcmp(text, "/tools") == 0) {
            // Register built-in tools
            adam_settings_add_tool(g_settings, (adam_tool_def_t){
                .name = "calculator", .description = "Evaluate a math expression",
                .parameters_json = "{\"type\":\"object\",\"properties\":{\"expression\":{\"type\":\"string\"}},\"required\":[\"expression\"]}",
                .execute = adam_tool_calculator });
            adam_settings_add_tool(g_settings, (adam_tool_def_t){
                .name = "web_search", .description = "Search the web",
                .parameters_json = "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}},\"required\":[\"query\"]}",
                .execute = adam_tool_web_search, .ctx = g_settings });
            tg_send_message(chat_id, "Tools enabled: calculator, web_search");
            free(text);
            return;
        }
#ifndef ADAM_NO_SQLITE
        if (strcmp(text, "/memory") == 0) {
            if (!g_memory) {
                g_memory = adam_memory_open("adam_telegram.db");
                if (g_memory) {
                    adam_memory_configure(g_memory, g_settings);
                    g_settings->memory = g_memory;
                    g_settings->inject_memory = 1;
                    g_settings->memory_extract = 1;
                }
            }
            tg_send_message(chat_id, g_memory
                ? "Memory enabled." : "Failed to open memory database.");
            free(text);
            return;
        }
#endif
        if (strcmp(text, "/status") == 0) {
            char buf[512];
            snprintf(buf, sizeof(buf),
                "Model: %s\n"
                "Tools: %zu\n"
                "History: %zu messages\n"
                "Memory: %s",
                g_settings->model ? g_settings->model : "(local)",
                g_settings->tool_count,
                adam_history_count(g_history),
                g_memory ? "on" : "off");
            tg_send_message(chat_id, buf);
            free(text);
            return;
        }
    }

    // Regular chat message
    tg_send_typing(chat_id);
    adam_run_result_t r = adam_run(g_settings, g_history, text);

    if (r.status == ADAM_OK && r.final_response)
        tg_send_message(chat_id, r.final_response);
    else
        tg_send_message(chat_id, r.final_response
            ? r.final_response : "(error)");

    printf("  [%d in / %d out | $%.4f | %.0fms]\n",
           r.input_tokens, r.output_tokens, r.cost_usd, r.elapsed_ms);

    adam_run_result_free(&r);
    free(text);
}

// ============================================================================
// MARK: - Main polling loop
// ============================================================================

static void poll_updates(void) {
    char body[128];
    snprintf(body, sizeof(body),
             "{\"offset\":%d,\"timeout\":30,\"allowed_updates\":[\"message\"]}",
             g_offset);

    char *resp = tg_post("getUpdates", body);
    if (!resp) return;

    // Check "ok":true
    if (!strstr(resp, "\"ok\":true")) {
        fprintf(stderr, "Telegram API error: %.200s\n", resp);
        free(resp);
        return;
    }

    // Walk through "result" array — find each "update_id"
    const char *p = strstr(resp, "\"result\"");
    if (!p) { free(resp); return; }

    while ((p = strstr(p, "\"update_id\"")) != NULL) {
        int update_id = (int)json_get_int(p - 1, "update_id");
        if (update_id >= g_offset) g_offset = update_id + 1;

        process_message(p);

        p += 11; // advance past "update_id"
    }

    free(resp);
}

// ============================================================================
// MARK: - Argument parsing and main
// ============================================================================

int main(int argc, char **argv) {
    adam_init();

    g_settings = adam_create_settings();
    g_history = adam_history_create();

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--local") == 0 && i + 1 < argc) {
            adam_settings_set_local(g_settings, argv[++i], -1, 4096);
        } else if (strcmp(argv[i], "--mmproj") == 0 && i + 1 < argc) {
            adam_settings_set_mmproj(g_settings, argv[++i]);
        } else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            g_settings->local_ctx_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            g_settings->model = argv[++i];
        }
    }

    // Load tokens from .env
    g_bot_token = env_load("TELEGRAM_BOT_TOKEN");
    if (!g_bot_token) {
        fprintf(stderr, "Missing TELEGRAM_BOT_TOKEN in .env\n");
        fprintf(stderr, "Get one from @BotFather on Telegram.\n");
        adam_history_destroy(g_history);
        adam_settings_destroy(g_settings);
        adam_cleanup();
        return 1;
    }

    // Auto-detect LLM provider from .env
#ifndef ADAM_NO_LOCAL
    if (!g_settings->gguf_path) {
#endif
        char *key;
        if ((key = env_load("ANTHROPIC_API_KEY")))
            adam_settings_set_provider(g_settings, ADAM_API_ANTHROPIC, key,
                                       "claude-sonnet-4-20250514");
        else if ((key = env_load("GEMINI_API_KEY")))
            adam_settings_set_provider(g_settings, ADAM_API_GEMINI, key,
                                       "gemini-2.5-flash");
        else if ((key = env_load("OPENAI_API_KEY")))
            adam_settings_set_provider(g_settings, ADAM_API_OPENAI, key,
                                       "gpt-4o-mini");
        else {
            fprintf(stderr, "No LLM provider in .env and no --local model.\n");
            free(g_bot_token);
            adam_history_destroy(g_history);
            adam_settings_destroy(g_settings);
            adam_cleanup();
            return 1;
        }
#ifndef ADAM_NO_LOCAL
    }
#endif

    adam_settings_set_identity(g_settings,
        "You are Adam, a helpful AI assistant on Telegram. "
        "Be concise and friendly. Use markdown formatting when helpful.");

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    printf("Adam Telegram Bot started.\n");
    printf("  Model: %s\n",
#ifndef ADAM_NO_LOCAL
           g_settings->gguf_path ? g_settings->gguf_path :
#endif
           (g_settings->model ? g_settings->model : "(none)"));
    printf("  Token: %s...%s\n", g_bot_token,
           strlen(g_bot_token) > 8 ? g_bot_token + strlen(g_bot_token) - 4 : "");
    printf("  Polling for messages... (Ctrl+C to stop)\n\n");

    // Main loop
    while (g_running) {
        poll_updates();
    }

    printf("\nShutting down...\n");

    // Cleanup
    free(g_bot_token);
#ifndef ADAM_NO_SQLITE
    if (g_memory) adam_memory_close(g_memory);
#endif
    adam_history_destroy(g_history);
    adam_settings_destroy(g_settings);
    adam_cleanup();
    return 0;
}
