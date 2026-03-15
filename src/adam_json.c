//
//  adam_json.c
//  Adam — JSON building and parsing via jsmn
//
//  Created by Marco Bambini on 15/03/26.
//

#include "adam_json.h"
#include "jsmn.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#define UNUSED_PARAM(p) ((void)(p))

// ============================================================================
// MARK: - Growable arena buffer
// ============================================================================

// A simple string builder that uses the arena for storage.
// On realloc, the old buffer stays in the arena (freed on arena_reset).
typedef struct {
    arena_t *arena;
    char    *buf;
    size_t   len;
    size_t   cap;
} abuf_t;

static abuf_t abuf_new(arena_t *arena, size_t initial_cap) {
    char *buf = arena_alloc(arena, initial_cap);
    return (abuf_t){ .arena = arena, .buf = buf, .len = 0, .cap = buf ? initial_cap : 0 };
}

static int abuf_grow(abuf_t *b, size_t need) {
    if (b->len + need < b->cap) return 1;
    size_t new_cap = b->cap * 2;
    while (new_cap < b->len + need + 1) new_cap *= 2;
    char *new_buf = arena_alloc(b->arena, new_cap);
    if (!new_buf) return 0;
    memcpy(new_buf, b->buf, b->len);
    b->buf = new_buf;
    b->cap = new_cap;
    return 1;
}

static void abuf_append(abuf_t *b, const char *s, size_t n) {
    if (!abuf_grow(b, n)) return;
    memcpy(b->buf + b->len, s, n);
    b->len += n;
    b->buf[b->len] = '\0';
}

static void abuf_str(abuf_t *b, const char *s) {
    abuf_append(b, s, strlen(s));
}

static void abuf_char(abuf_t *b, char c) {
    abuf_append(b, &c, 1);
}

static void abuf_fmt(abuf_t *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char tmp[256];
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n > 0) abuf_append(b, tmp, (size_t)n);
}

// ============================================================================
// MARK: - JSON string escaping
// ============================================================================

static void abuf_json_string(abuf_t *b, const char *s, size_t len) {
    abuf_char(b, '"');
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"':  abuf_str(b, "\\\""); break;
        case '\\': abuf_str(b, "\\\\"); break;
        case '\b': abuf_str(b, "\\b");  break;
        case '\f': abuf_str(b, "\\f");  break;
        case '\n': abuf_str(b, "\\n");  break;
        case '\r': abuf_str(b, "\\r");  break;
        case '\t': abuf_str(b, "\\t");  break;
        default:
            if (c < 0x20) {
                char esc[8];
                snprintf(esc, sizeof(esc), "\\u%04x", c);
                abuf_str(b, esc);
            } else {
                abuf_char(b, (char)c);
            }
        }
    }
    abuf_char(b, '"');
}

static void abuf_json_cstr(abuf_t *b, const char *s) {
    if (!s) { abuf_str(b, "null"); return; }
    abuf_json_string(b, s, strlen(s));
}

// ============================================================================
// MARK: - jsmn helpers
// ============================================================================

static int tok_eq(const char *json, const jsmntok_t *tok, const char *s) {
    size_t slen = strlen(s);
    return (tok->type == JSMN_STRING
         && (size_t)(tok->end - tok->start) == slen
         && memcmp(json + tok->start, s, slen) == 0);
}

// Extract a raw token value (no JSON unescaping — for keys, IDs, names)
static const char *tok_str(arena_t *arena, const char *json, const jsmntok_t *tok) {
    size_t len = (size_t)(tok->end - tok->start);
    char *s = arena_alloc(arena, len + 1);
    if (!s) return NULL;
    memcpy(s, json + tok->start, len);
    s[len] = '\0';
    return s;
}

// Extract a string token with JSON unescaping (for content text)
static const char *tok_str_unesc(arena_t *arena, const char *json, const jsmntok_t *tok) {
    size_t src_len = (size_t)(tok->end - tok->start);
    const char *src = json + tok->start;
    // Output can only be shorter or equal
    char *dst = arena_alloc(arena, src_len + 1);
    if (!dst) return NULL;
    size_t j = 0;
    for (size_t i = 0; i < src_len; i++) {
        if (src[i] == '\\' && i + 1 < src_len) {
            i++;
            switch (src[i]) {
            case '"':  dst[j++] = '"';  break;
            case '\\': dst[j++] = '\\'; break;
            case '/':  dst[j++] = '/';  break;
            case 'b':  dst[j++] = '\b'; break;
            case 'f':  dst[j++] = '\f'; break;
            case 'n':  dst[j++] = '\n'; break;
            case 'r':  dst[j++] = '\r'; break;
            case 't':  dst[j++] = '\t'; break;
            case 'u':
                // \uXXXX — for now, pass through as-is for non-ASCII
                dst[j++] = '\\'; dst[j++] = 'u';
                break;
            default:   dst[j++] = src[i]; break;
            }
        } else {
            dst[j++] = src[i];
        }
    }
    dst[j] = '\0';
    return dst;
}

static int tok_int(const char *json, const jsmntok_t *tok) {
    char buf[32];
    size_t len = (size_t)(tok->end - tok->start);
    if (len >= sizeof(buf)) len = sizeof(buf) - 1;
    memcpy(buf, json + tok->start, len);
    buf[len] = '\0';
    return atoi(buf);
}

static int tok_is(const char *json, const jsmntok_t *tok, const char *val) {
    size_t len = strlen(val);
    return ((size_t)(tok->end - tok->start) == len
         && memcmp(json + tok->start, val, len) == 0);
}

// Skip a jsmn token and all its children. Returns the index of the next sibling.
static int tok_skip(const jsmntok_t *tokens, int pos, int ntok) {
    if (pos >= ntok) return ntok;
    const jsmntok_t *t = &tokens[pos];
    if (t->type == JSMN_PRIMITIVE || t->type == JSMN_STRING) {
        return pos + 1;
    }
    int count = t->size;
    int i = pos + 1;
    if (t->type == JSMN_OBJECT) count *= 2; // key+value pairs
    for (int c = 0; c < count && i < ntok; c++) {
        i = tok_skip(tokens, i, ntok);
    }
    return i;
}

// ============================================================================
// MARK: - Build request: Anthropic Messages API
// ============================================================================

static void build_anthropic(
    abuf_t *b,
    const char *model,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count,
    float temperature, int max_tokens, const char *response_format
) {
    abuf_str(b, "{\"model\":");
    abuf_json_cstr(b, model);
    abuf_fmt(b, ",\"max_tokens\":%d", max_tokens);
    abuf_fmt(b, ",\"temperature\":%.2f", (double)temperature);

    // System message (Anthropic puts it outside the messages array)
    if (msg_count > 0 && msgs[0].role == ADAM_ROLE_SYSTEM) {
        abuf_str(b, ",\"system\":[{\"type\":\"text\",\"text\":");
        abuf_json_string(b, msgs[0].content, msgs[0].content_len);
        abuf_str(b, ",\"cache_control\":{\"type\":\"ephemeral\"}}]");
    }

    // Messages array
    abuf_str(b, ",\"messages\":[");
    int first = 1;
    for (size_t i = 0; i < msg_count; i++) {
        if (msgs[i].role == ADAM_ROLE_SYSTEM) continue; // handled above

        if (!first) abuf_char(b, ',');
        first = 0;

        const char *role = (msgs[i].role == ADAM_ROLE_USER) ? "user" : "assistant";

        if (msgs[i].role == ADAM_ROLE_TOOL) {
            // Tool results in Anthropic format
            abuf_str(b, "{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":");
            abuf_json_cstr(b, msgs[i].tool_call_id);
            abuf_str(b, ",\"content\":");
            abuf_json_string(b, msgs[i].content, msgs[i].content_len);
            abuf_str(b, "}]}");
        } else if (msgs[i].role == ADAM_ROLE_ASSISTANT && msgs[i].tool_call_count > 0) {
            // Assistant message with tool calls
            abuf_fmt(b, "{\"role\":\"%s\",\"content\":[", role);
            // Text content first (if any)
            int has_text = (msgs[i].content_len > 0);
            if (has_text) {
                abuf_str(b, "{\"type\":\"text\",\"text\":");
                abuf_json_string(b, msgs[i].content, msgs[i].content_len);
                abuf_char(b, '}');
            }
            // Tool use blocks
            for (size_t t = 0; t < msgs[i].tool_call_count; t++) {
                if (t > 0 || has_text) abuf_char(b, ',');
                abuf_str(b, "{\"type\":\"tool_use\",\"id\":");
                abuf_json_cstr(b, msgs[i].tool_calls[t].id);
                abuf_str(b, ",\"name\":");
                abuf_json_cstr(b, msgs[i].tool_calls[t].name);
                abuf_str(b, ",\"input\":");
                // arguments_json is already valid JSON
                abuf_str(b, msgs[i].tool_calls[t].arguments_json
                            ? msgs[i].tool_calls[t].arguments_json : "{}");
                abuf_char(b, '}');
            }
            abuf_str(b, "]}");
        } else {
            // Simple text message
            abuf_fmt(b, "{\"role\":\"%s\",\"content\":", role);
            abuf_json_string(b, msgs[i].content, msgs[i].content_len);
            abuf_char(b, '}');
        }
    }
    abuf_char(b, ']');

    // Tools
    if (tool_count > 0) {
        abuf_str(b, ",\"tools\":[");
        for (size_t i = 0; i < tool_count; i++) {
            if (i > 0) abuf_char(b, ',');
            abuf_str(b, "{\"name\":");
            abuf_json_cstr(b, tools[i].name);
            abuf_str(b, ",\"description\":");
            abuf_json_cstr(b, tools[i].description ? tools[i].description : "");
            abuf_str(b, ",\"input_schema\":");
            abuf_str(b, tools[i].parameters_json
                        ? tools[i].parameters_json
                        : "{\"type\":\"object\",\"properties\":{}}");
            abuf_char(b, '}');
        }
        abuf_char(b, ']');
    }

    // Response format (Anthropic doesn't have a direct json_mode,
    // but we can hint via tool_choice or prefill)
    UNUSED_PARAM(response_format);

    abuf_char(b, '}');
}

// ============================================================================
// MARK: - Build request: OpenAI Chat Completions
// ============================================================================

static void build_openai(
    abuf_t *b,
    const char *model,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count,
    float temperature, int max_tokens, float top_p,
    const char *response_format
) {
    abuf_str(b, "{\"model\":");
    abuf_json_cstr(b, model);
    abuf_fmt(b, ",\"max_completion_tokens\":%d", max_tokens);
    abuf_fmt(b, ",\"temperature\":%.2f", (double)temperature);
    if (top_p < 0.99f) abuf_fmt(b, ",\"top_p\":%.2f", (double)top_p);

    // Messages array
    abuf_str(b, ",\"messages\":[");
    int first = 1;
    for (size_t i = 0; i < msg_count; i++) {
        if (!first) abuf_char(b, ',');
        first = 0;

        const char *role;
        switch (msgs[i].role) {
        case ADAM_ROLE_SYSTEM:    role = "system";    break;
        case ADAM_ROLE_ASSISTANT: role = "assistant";  break;
        case ADAM_ROLE_TOOL:      role = "tool";       break;
        default:                 role = "user";        break;
        }

        if (msgs[i].role == ADAM_ROLE_TOOL) {
            abuf_fmt(b, "{\"role\":\"tool\",\"tool_call_id\":");
            abuf_json_cstr(b, msgs[i].tool_call_id);
            abuf_str(b, ",\"content\":");
            abuf_json_string(b, msgs[i].content, msgs[i].content_len);
            abuf_char(b, '}');
        } else if (msgs[i].role == ADAM_ROLE_ASSISTANT && msgs[i].tool_call_count > 0) {
            abuf_fmt(b, "{\"role\":\"%s\"", role);
            if (msgs[i].content_len > 0) {
                abuf_str(b, ",\"content\":");
                abuf_json_string(b, msgs[i].content, msgs[i].content_len);
            } else {
                abuf_str(b, ",\"content\":null");
            }
            abuf_str(b, ",\"tool_calls\":[");
            for (size_t t = 0; t < msgs[i].tool_call_count; t++) {
                if (t > 0) abuf_char(b, ',');
                abuf_str(b, "{\"id\":");
                abuf_json_cstr(b, msgs[i].tool_calls[t].id);
                abuf_str(b, ",\"type\":\"function\",\"function\":{\"name\":");
                abuf_json_cstr(b, msgs[i].tool_calls[t].name);
                abuf_str(b, ",\"arguments\":");
                abuf_json_cstr(b, msgs[i].tool_calls[t].arguments_json
                                   ? msgs[i].tool_calls[t].arguments_json : "{}");
                abuf_str(b, "}}");
            }
            abuf_str(b, "]}");
        } else {
            abuf_fmt(b, "{\"role\":\"%s\",\"content\":", role);
            abuf_json_string(b, msgs[i].content, msgs[i].content_len);
            abuf_char(b, '}');
        }
    }
    abuf_char(b, ']');

    // Tools
    if (tool_count > 0) {
        abuf_str(b, ",\"tools\":[");
        for (size_t i = 0; i < tool_count; i++) {
            if (i > 0) abuf_char(b, ',');
            abuf_str(b, "{\"type\":\"function\",\"function\":{\"name\":");
            abuf_json_cstr(b, tools[i].name);
            abuf_str(b, ",\"description\":");
            abuf_json_cstr(b, tools[i].description ? tools[i].description : "");
            abuf_str(b, ",\"parameters\":");
            abuf_str(b, tools[i].parameters_json
                        ? tools[i].parameters_json
                        : "{\"type\":\"object\",\"properties\":{}}");
            abuf_str(b, "}}");
        }
        abuf_char(b, ']');
    }

    // Response format
    if (response_format && strcmp(response_format, "json") == 0) {
        abuf_str(b, ",\"response_format\":{\"type\":\"json_object\"}");
    }

    abuf_char(b, '}');
}

// ============================================================================
// MARK: - Build request (public)
// ============================================================================

const char *adam_json_build_request(
    arena_t *arena, adam_api_format_t format,
    const char *model,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count,
    float temperature, int max_tokens, float top_p,
    const char *response_format
) {
    // Estimate initial buffer size
    size_t est = 4096;
    for (size_t i = 0; i < msg_count; i++)
        est += msgs[i].content_len * 2 + 256;
    for (size_t i = 0; i < tool_count; i++)
        est += (tools[i].parameters_json ? strlen(tools[i].parameters_json) : 64) + 256;

    abuf_t b = abuf_new(arena, est);
    if (!b.buf) return NULL;

    if (format == ADAM_API_ANTHROPIC) {
        build_anthropic(&b, model, msgs, msg_count, tools, tool_count,
                        temperature, max_tokens, response_format);
    } else {
        build_openai(&b, model, msgs, msg_count, tools, tool_count,
                     temperature, max_tokens, top_p, response_format);
    }

    return b.buf;
}

// ============================================================================
// MARK: - Parse response: Anthropic Messages API
// ============================================================================

// Anthropic response format:
// {
//   "content": [
//     {"type":"text","text":"..."},
//     {"type":"tool_use","id":"...","name":"...","input":{...}}
//   ],
//   "stop_reason": "end_turn" | "tool_use",
//   "usage": {"input_tokens": N, "output_tokens": N}
// }

static adam_llm_response_t parse_anthropic(
    arena_t *arena, const char *json, const jsmntok_t *tokens, int ntok
) {
    adam_llm_response_t resp = {0};

    // Walk top-level object
    if (ntok < 1 || tokens[0].type != JSMN_OBJECT) {
        resp.error = ADAM_ERR_JSON;
        resp.error_msg = arena_strdup(arena, "expected JSON object");
        return resp;
    }

    // Check for error response
    for (int i = 1; i < ntok - 1; i++) {
        if (tok_eq(json, &tokens[i], "type") && tok_is(json, &tokens[i+1], "error")) {
            resp.error = ADAM_ERR_PROVIDER;
            // Find error.message
            for (int j = i; j < ntok - 1; j++) {
                if (tok_eq(json, &tokens[j], "message") && tokens[j+1].type == JSMN_STRING) {
                    resp.error_msg = tok_str_unesc(arena, json, &tokens[j+1]);
                    return resp;
                }
            }
            resp.error_msg = arena_strdup(arena, "unknown API error");
            return resp;
        }
    }

    // Collect text and tool_use blocks from content array
    abuf_t text_buf = abuf_new(arena, 1024);
    size_t tc_cap = 8;
    adam_tool_call_t *tc_arr = arena_alloc(arena, tc_cap * sizeof(adam_tool_call_t));
    size_t tc_count = 0;

    int i = 1;
    while (i < ntok - 1) {
        if (tok_eq(json, &tokens[i], "content") && tokens[i+1].type == JSMN_ARRAY) {
            int arr_size = tokens[i+1].size;
            int j = i + 2;
            for (int elem = 0; elem < arr_size && j < ntok; elem++) {
                // Each element is an object
                if (tokens[j].type != JSMN_OBJECT) { j = tok_skip(tokens, j, ntok); continue; }
                int obj_size = tokens[j].size;
                int k = j + 1;
                const char *block_type = NULL;
                const char *block_text = NULL;
                const char *tc_id = NULL;
                const char *tc_name = NULL;
                int input_start = -1, input_end = -1;

                for (int f = 0; f < obj_size && k < ntok - 1; f++) {
                    if (tok_eq(json, &tokens[k], "type")) {
                        block_type = tok_str(arena, json, &tokens[k+1]);
                        k += 2;
                    } else if (tok_eq(json, &tokens[k], "text")) {
                        block_text = tok_str_unesc(arena, json, &tokens[k+1]);
                        k += 2;
                    } else if (tok_eq(json, &tokens[k], "id")) {
                        tc_id = tok_str(arena, json, &tokens[k+1]);
                        k += 2;
                    } else if (tok_eq(json, &tokens[k], "name")) {
                        tc_name = tok_str(arena, json, &tokens[k+1]);
                        k += 2;
                    } else if (tok_eq(json, &tokens[k], "input")) {
                        input_start = tokens[k+1].start;
                        input_end = tokens[k+1].end;
                        k++;
                        k = tok_skip(tokens, k, ntok);
                    } else {
                        k++;
                        k = tok_skip(tokens, k, ntok);
                    }
                }

                if (block_type && strcmp(block_type, "text") == 0 && block_text) {
                    abuf_str(&text_buf, block_text);
                } else if (block_type && strcmp(block_type, "tool_use") == 0 && tc_name) {
                    if (tc_count >= tc_cap) {
                        tc_cap *= 2;
                        adam_tool_call_t *new_arr = arena_alloc(arena, tc_cap * sizeof(adam_tool_call_t));
                        memcpy(new_arr, tc_arr, tc_count * sizeof(adam_tool_call_t));
                        tc_arr = new_arr;
                    }
                    tc_arr[tc_count].id = tc_id;
                    tc_arr[tc_count].name = tc_name;
                    if (input_start >= 0 && input_end > input_start) {
                        size_t ilen = (size_t)(input_end - input_start);
                        char *inp = arena_alloc(arena, ilen + 1);
                        memcpy(inp, json + input_start, ilen);
                        inp[ilen] = '\0';
                        tc_arr[tc_count].arguments_json = inp;
                    } else {
                        tc_arr[tc_count].arguments_json = arena_strdup(arena, "{}");
                    }
                    tc_count++;
                }
                j = tok_skip(tokens, j, ntok);
            }
            i = tok_skip(tokens, i + 1, ntok);
        } else if (tok_eq(json, &tokens[i], "usage") && tokens[i+1].type == JSMN_OBJECT) {
            int obj_size = tokens[i+1].size;
            int k = i + 2;
            for (int f = 0; f < obj_size && k < ntok - 1; f++) {
                if (tok_eq(json, &tokens[k], "input_tokens")) {
                    resp.input_tokens = tok_int(json, &tokens[k+1]);
                    k += 2;
                } else if (tok_eq(json, &tokens[k], "output_tokens")) {
                    resp.output_tokens = tok_int(json, &tokens[k+1]);
                    k += 2;
                } else {
                    k++;
                    k = tok_skip(tokens, k, ntok);
                }
            }
            i = tok_skip(tokens, i + 1, ntok);
        } else {
            i++;
            i = tok_skip(tokens, i, ntok);
        }
    }

    resp.content = (text_buf.len > 0) ? text_buf.buf : NULL;
    if (tc_count > 0) {
        resp.tool_calls = tc_arr;
        resp.tool_call_count = tc_count;
    }
    return resp;
}

// ============================================================================
// MARK: - Parse response: OpenAI Chat Completions
// ============================================================================

// OpenAI response format:
// {
//   "choices": [{
//     "message": {
//       "role": "assistant",
//       "content": "...",
//       "tool_calls": [{"id":"...","type":"function","function":{"name":"...","arguments":"..."}}]
//     },
//     "finish_reason": "stop" | "tool_calls"
//   }],
//   "usage": {"prompt_tokens": N, "completion_tokens": N}
// }

static adam_llm_response_t parse_openai(
    arena_t *arena, const char *json, const jsmntok_t *tokens, int ntok
) {
    adam_llm_response_t resp = {0};

    if (ntok < 1 || tokens[0].type != JSMN_OBJECT) {
        resp.error = ADAM_ERR_JSON;
        resp.error_msg = arena_strdup(arena, "expected JSON object");
        return resp;
    }

    // Check for error response: {"error":{"message":"..."}}
    for (int i = 1; i < ntok - 1; i++) {
        if (tok_eq(json, &tokens[i], "error") && tokens[i+1].type == JSMN_OBJECT) {
            resp.error = ADAM_ERR_PROVIDER;
            int k = i + 2;
            int obj_size = tokens[i+1].size;
            for (int f = 0; f < obj_size && k < ntok - 1; f++) {
                if (tok_eq(json, &tokens[k], "message") && tokens[k+1].type == JSMN_STRING) {
                    resp.error_msg = tok_str_unesc(arena, json, &tokens[k+1]);
                    return resp;
                }
                k++;
                k = tok_skip(tokens, k, ntok);
            }
            resp.error_msg = arena_strdup(arena, "unknown API error");
            return resp;
        }
    }

    size_t tc_cap = 8;
    adam_tool_call_t *tc_arr = arena_alloc(arena, tc_cap * sizeof(adam_tool_call_t));
    size_t tc_count = 0;

    // Walk top-level keys
    int i = 1;
    while (i < ntok - 1) {
        if (tok_eq(json, &tokens[i], "choices") && tokens[i+1].type == JSMN_ARRAY) {
            // Parse first choice
            int arr_size = tokens[i+1].size;
            if (arr_size < 1) { i = tok_skip(tokens, i + 1, ntok); continue; }
            // Find message object in first choice
            int choice = i + 2; // first choice object
            if (tokens[choice].type != JSMN_OBJECT) { i = tok_skip(tokens, i + 1, ntok); continue; }
            int choice_size = tokens[choice].size;
            int k = choice + 1;
            for (int f = 0; f < choice_size && k < ntok - 1; f++) {
                if (tok_eq(json, &tokens[k], "message") && tokens[k+1].type == JSMN_OBJECT) {
                    int msg_size = tokens[k+1].size;
                    int m = k + 2;
                    for (int mf = 0; mf < msg_size && m < ntok - 1; mf++) {
                        if (tok_eq(json, &tokens[m], "content")) {
                            if (tokens[m+1].type == JSMN_STRING) {
                                resp.content = tok_str_unesc(arena, json, &tokens[m+1]);
                            }
                            m += 2;
                        } else if (tok_eq(json, &tokens[m], "tool_calls") && tokens[m+1].type == JSMN_ARRAY) {
                            int tc_arr_size = tokens[m+1].size;
                            int t = m + 2;
                            for (int te = 0; te < tc_arr_size && t < ntok; te++) {
                                if (tokens[t].type != JSMN_OBJECT) { t = tok_skip(tokens, t, ntok); continue; }
                                const char *tc_id = NULL;
                                const char *tc_name = NULL;
                                const char *tc_args = NULL;
                                int tc_obj_size = tokens[t].size;
                                int u = t + 1;
                                for (int uf = 0; uf < tc_obj_size && u < ntok - 1; uf++) {
                                    if (tok_eq(json, &tokens[u], "id")) {
                                        tc_id = tok_str(arena, json, &tokens[u+1]);
                                        u += 2;
                                    } else if (tok_eq(json, &tokens[u], "function") && tokens[u+1].type == JSMN_OBJECT) {
                                        int fn_size = tokens[u+1].size;
                                        int v = u + 2;
                                        for (int vf = 0; vf < fn_size && v < ntok - 1; vf++) {
                                            if (tok_eq(json, &tokens[v], "name")) {
                                                tc_name = tok_str(arena, json, &tokens[v+1]);
                                                v += 2;
                                            } else if (tok_eq(json, &tokens[v], "arguments")) {
                                                tc_args = tok_str(arena, json, &tokens[v+1]);
                                                v += 2;
                                            } else { v++; v = tok_skip(tokens, v, ntok); }
                                        }
                                        u++;
                                        u = tok_skip(tokens, u, ntok);
                                    } else { u++; u = tok_skip(tokens, u, ntok); }
                                }
                                if (tc_name) {
                                    if (tc_count >= tc_cap) {
                                        tc_cap *= 2;
                                        adam_tool_call_t *new_a = arena_alloc(arena, tc_cap * sizeof(adam_tool_call_t));
                                        memcpy(new_a, tc_arr, tc_count * sizeof(adam_tool_call_t));
                                        tc_arr = new_a;
                                    }
                                    tc_arr[tc_count].id = tc_id;
                                    tc_arr[tc_count].name = tc_name;
                                    tc_arr[tc_count].arguments_json = tc_args ? tc_args : arena_strdup(arena, "{}");
                                    tc_count++;
                                }
                                t = tok_skip(tokens, t, ntok);
                            }
                            m++;
                            m = tok_skip(tokens, m, ntok);
                        } else { m++; m = tok_skip(tokens, m, ntok); }
                    }
                    k++;
                    k = tok_skip(tokens, k, ntok);
                } else { k++; k = tok_skip(tokens, k, ntok); }
            }
            i = tok_skip(tokens, i + 1, ntok);
        } else if (tok_eq(json, &tokens[i], "usage") && tokens[i+1].type == JSMN_OBJECT) {
            int obj_size = tokens[i+1].size;
            int k = i + 2;
            for (int f = 0; f < obj_size && k < ntok - 1; f++) {
                if (tok_eq(json, &tokens[k], "prompt_tokens")) {
                    resp.input_tokens = tok_int(json, &tokens[k+1]);
                    k += 2;
                } else if (tok_eq(json, &tokens[k], "completion_tokens")) {
                    resp.output_tokens = tok_int(json, &tokens[k+1]);
                    k += 2;
                } else { k++; k = tok_skip(tokens, k, ntok); }
            }
            i = tok_skip(tokens, i + 1, ntok);
        } else {
            i++;
            i = tok_skip(tokens, i, ntok);
        }
    }

    if (tc_count > 0) {
        resp.tool_calls = tc_arr;
        resp.tool_call_count = tc_count;
    }
    return resp;
}

// ============================================================================
// MARK: - Parse response (public)
// ============================================================================

adam_llm_response_t adam_json_parse_response(
    arena_t *arena, adam_api_format_t format,
    const char *json, size_t json_len
) {
    adam_llm_response_t resp = {0};

    if (!json || json_len == 0) {
        resp.error = ADAM_ERR_JSON;
        resp.error_msg = arena_strdup(arena, "empty response");
        return resp;
    }

    // Allocate jsmn tokens from arena
    size_t max_tokens = json_len / 4 + 128;
    jsmntok_t *tokens = arena_alloc(arena, max_tokens * sizeof(jsmntok_t));
    if (!tokens) {
        resp.error = ADAM_ERR_ALLOC;
        resp.error_msg = arena_strdup(arena, "token alloc failed");
        return resp;
    }

    jsmn_parser parser;
    jsmn_init(&parser);
    int ntok = jsmn_parse(&parser, json, json_len, tokens, (unsigned)max_tokens);

    if (ntok < 0) {
        resp.error = ADAM_ERR_JSON;
        if (ntok == JSMN_ERROR_NOMEM)
            resp.error_msg = arena_strdup(arena, "JSON too large for token buffer");
        else if (ntok == JSMN_ERROR_INVAL)
            resp.error_msg = arena_strdup(arena, "invalid JSON");
        else
            resp.error_msg = arena_strdup(arena, "incomplete JSON");
        return resp;
    }

    if (format == ADAM_API_ANTHROPIC) {
        return parse_anthropic(arena, json, tokens, ntok);
    } else {
        return parse_openai(arena, json, tokens, ntok);
    }
}
