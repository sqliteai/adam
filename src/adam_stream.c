//
//  adam_stream.c
//  Adam — SSE streaming for cloud LLM providers
//
//  Parses Server-Sent Events (SSE) from Anthropic and OpenAI streaming
//  endpoints. Text tokens are sent to on_stream in real-time. Tool calls
//  are buffered until complete. Returns a full adam_llm_response_t.
//
//  SSE format (both providers):
//    data: {"json":"payload"}\n\n
//    data: [DONE]\n\n
//
//  Created by Marco Bambini on 16/03/26.
//

#include "adam_stream.h"
#include "adam_json.h"
#include "adam_net.h"
#define JSMN_STATIC
#include "jsmn.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define AUTH_HDR_EXTRA 32
#define MAX_TOOL_CALLS 16

// ============================================================================
// MARK: - SSE accumulator (fed by adam_net_post_streaming callback)
// ============================================================================

typedef struct {
    // Settings
    adam_settings_t     *settings;
    adam_api_format_t    format;

    // Line buffer for SSE parsing (lines end with \n)
    char                *line_buf;
    size_t               line_len;
    size_t               line_cap;

    // Accumulated text content
    char                *text_buf;
    size_t               text_len;
    size_t               text_cap;

    // Accumulated tool calls
    struct {
        char *id;
        char *name;
        char *args;       // accumulated arguments string
        size_t args_len;
        size_t args_cap;
    } tool_calls[MAX_TOOL_CALLS];
    size_t               tc_count;
    int                  tc_current; // index of tool call being streamed (-1 = none)

    // Token counts
    int                  input_tokens;
    int                  output_tokens;

    // Error
    int                  had_error;
    char                *error_msg;

    // Debug
    int                  chunk_count;
} sse_ctx_t;

// ============================================================================
// MARK: - jsmn helpers (minimal, inline)
// ============================================================================

static int sse_tok_eq(const char *json, const jsmntok_t *tok, const char *s) {
    size_t sl = strlen(s);
    return (tok->type == JSMN_STRING
         && (size_t)(tok->end - tok->start) == sl
         && memcmp(json + tok->start, s, sl) == 0);
}

static const char *sse_tok_str(const char *json, const jsmntok_t *tok,
                                char *buf, size_t bufsize) {
    size_t len = (size_t)(tok->end - tok->start);
    if (len >= bufsize) len = bufsize - 1;
    memcpy(buf, json + tok->start, len);
    buf[len] = '\0';
    return buf;
}

static int sse_tok_int(const char *json, const jsmntok_t *tok) {
    char buf[32];
    sse_tok_str(json, tok, buf, sizeof(buf));
    return atoi(buf);
}

// ============================================================================
// MARK: - JSON string unescaping
// ============================================================================

// Unescape a JSON string value in-place. Returns the new length.
// Handles: \n \r \t \\ \" \/ \b \f \uXXXX (including surrogate pairs for emoji)
static uint32_t hex4(const char *s) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        v <<= 4;
        char c = s[i];
        if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
    }
    return v;
}

static size_t utf8_encode(uint32_t cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

static size_t json_unescape(char *buf, size_t len) {
    size_t r = 0, w = 0;
    while (r < len) {
        if (buf[r] == '\\' && r + 1 < len) {
            r++;
            switch (buf[r]) {
            case 'n':  buf[w++] = '\n'; break;
            case 'r':  buf[w++] = '\r'; break;
            case 't':  buf[w++] = '\t'; break;
            case '\\': buf[w++] = '\\'; break;
            case '"':  buf[w++] = '"';  break;
            case '/':  buf[w++] = '/';  break;
            case 'b':  buf[w++] = '\b'; break;
            case 'f':  buf[w++] = '\f'; break;
            case 'u': {
                r++;
                if (r + 4 > len) { buf[w++] = '?'; break; }
                uint32_t cp = hex4(buf + r);
                r += 4;
                // Handle surrogate pairs (emoji): \uD800-\uDBFF \uDC00-\uDFFF
                if (cp >= 0xD800 && cp <= 0xDBFF && r + 5 < len
                    && buf[r] == '\\' && buf[r+1] == 'u') {
                    uint32_t lo = hex4(buf + r + 2);
                    if (lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        r += 6; // skip \uXXXX
                    }
                }
                w += utf8_encode(cp, buf + w);
                continue; // r already advanced
            }
            default:   buf[w++] = buf[r]; break;
            }
            r++;
        } else {
            buf[w++] = buf[r++];
        }
    }
    buf[w] = '\0';
    return w;
}

// ============================================================================
// MARK: - Process a single SSE data line
// ============================================================================

// #define SSE_DEBUG  // uncomment to dump raw SSE lines

static void sse_process_line(sse_ctx_t *ctx, const char *data, size_t data_len) {
    if (data_len == 0) return;
    if (data_len == 6 && memcmp(data, "[DONE]", 6) == 0) return;

#ifdef SSE_DEBUG
    fprintf(stderr, "[SSE] %.*s\n", (int)data_len, data);
#endif

    // Parse JSON
    jsmntok_t tokens[128];
    jsmn_parser parser;
    jsmn_init(&parser);
    int ntok = jsmn_parse(&parser, data, data_len, tokens, 128);
    if (ntok < 1) return;

    if (ctx->format == ADAM_API_ANTHROPIC) {
        // Anthropic streaming events:
        //   {"type":"content_block_delta","delta":{"type":"text_delta","text":"..."}}
        //   {"type":"content_block_start","content_block":{"type":"tool_use","id":"...","name":"..."}}
        //   {"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"..."}}
        //   {"type":"message_delta","usage":{"output_tokens":N}}
        //   {"type":"message_start","message":{"usage":{"input_tokens":N}}}

        const char *event_type = NULL;
        char tbuf[64];

        for (int i = 1; i < ntok - 1; i++) {
            if (sse_tok_eq(data, &tokens[i], "type") && tokens[i+1].type == JSMN_STRING) {
                event_type = sse_tok_str(data, &tokens[i+1], tbuf, sizeof(tbuf));
                break;
            }
        }
        if (!event_type) return;

        if (strcmp(event_type, "content_block_delta") == 0) {
            // Find delta.text or delta.partial_json
            for (int i = 1; i < ntok - 1; i++) {
                if (sse_tok_eq(data, &tokens[i], "text") && tokens[i+1].type == JSMN_STRING) {
                    size_t tlen = (size_t)(tokens[i+1].end - tokens[i+1].start);

                    // Copy and unescape JSON string escapes (\n, \t, \\, etc.)
                    char *text = malloc(tlen + 1);
                    if (!text) continue;
                    memcpy(text, data + tokens[i+1].start, tlen);
                    text[tlen] = '\0';
                    tlen = json_unescape(text, tlen);

                    // Append to text buffer
                    if (ctx->text_len + tlen >= ctx->text_cap) {
                        size_t new_cap = (ctx->text_cap + tlen) * 2;
                        char *nb = realloc(ctx->text_buf, new_cap);
                        if (!nb) { free(text); continue; }
                        ctx->text_buf = nb;
                        ctx->text_cap = new_cap;
                    }
                    memcpy(ctx->text_buf + ctx->text_len, text, tlen);
                    ctx->text_len += tlen;
                    ctx->text_buf[ctx->text_len] = '\0';

                    // Stream to callback
                    if (ctx->settings->on_stream)
                        ctx->settings->on_stream(ctx->settings->stream_ctx,
                                                  text, tlen, 0);
                    free(text);
                    break;
                }
                if (sse_tok_eq(data, &tokens[i], "partial_json") && tokens[i+1].type == JSMN_STRING) {
                    // Accumulate tool call arguments (unescape JSON string first)
                    if (ctx->tc_current >= 0 && ctx->tc_current < MAX_TOOL_CALLS) {
                        size_t plen = (size_t)(tokens[i+1].end - tokens[i+1].start);
                        char *pj = malloc(plen + 1);
                        if (!pj) continue;
                        memcpy(pj, data + tokens[i+1].start, plen);
                        pj[plen] = '\0';
                        plen = json_unescape(pj, plen);

                        size_t idx = (size_t)ctx->tc_current;
                        if (ctx->tool_calls[idx].args_len + plen >= ctx->tool_calls[idx].args_cap) {
                            size_t new_cap = (ctx->tool_calls[idx].args_cap + plen) * 2;
                            char *nb = realloc(ctx->tool_calls[idx].args, new_cap);
                            if (!nb) { free(pj); continue; }
                            ctx->tool_calls[idx].args = nb;
                            ctx->tool_calls[idx].args_cap = new_cap;
                        }
                        memcpy(ctx->tool_calls[idx].args + ctx->tool_calls[idx].args_len, pj, plen);
                        ctx->tool_calls[idx].args_len += plen;
                        ctx->tool_calls[idx].args[ctx->tool_calls[idx].args_len] = '\0';
                        free(pj);
                    }
                    break;
                }
            }
        } else if (strcmp(event_type, "content_block_start") == 0) {
            // Check if it's a tool_use block
            for (int i = 1; i < ntok - 1; i++) {
                if (sse_tok_eq(data, &tokens[i], "content_block") && tokens[i+1].type == JSMN_OBJECT) {
                    const char *block_type = NULL;
                    const char *tc_id = NULL;
                    const char *tc_name = NULL;
                    char buf1[64], buf2[128], buf3[128];
                    int k = i + 2;
                    int obj_size = tokens[i+1].size;
                    for (int f = 0; f < obj_size && k < ntok - 1; f++) {
                        if (sse_tok_eq(data, &tokens[k], "type"))
                            { block_type = sse_tok_str(data, &tokens[k+1], buf1, sizeof(buf1)); k += 2; }
                        else if (sse_tok_eq(data, &tokens[k], "id"))
                            { tc_id = sse_tok_str(data, &tokens[k+1], buf2, sizeof(buf2)); k += 2; }
                        else if (sse_tok_eq(data, &tokens[k], "name"))
                            { tc_name = sse_tok_str(data, &tokens[k+1], buf3, sizeof(buf3)); k += 2; }
                        else k += 2;
                    }
                    if (block_type && strcmp(block_type, "tool_use") == 0
                        && tc_name && ctx->tc_count < MAX_TOOL_CALLS) {
                        char *dup_name = strdup(tc_name);
                        char *dup_args = calloc(1, 256);
                        if (!dup_name || !dup_args) {
                            free(dup_name); free(dup_args);
                        } else {
                            size_t idx = ctx->tc_count++;
                            ctx->tool_calls[idx].id = tc_id ? strdup(tc_id) : NULL;
                            ctx->tool_calls[idx].name = dup_name;
                            ctx->tool_calls[idx].args = dup_args;
                            ctx->tool_calls[idx].args_cap = 256;
                            ctx->tc_current = (int)idx;
                        }
                    }
                    break;
                }
            }
        } else if (strcmp(event_type, "content_block_stop") == 0) {
            ctx->tc_current = -1;
        } else if (strcmp(event_type, "message_start") == 0) {
            // Extract input_tokens from message.usage
            for (int i = 1; i < ntok - 1; i++) {
                if (sse_tok_eq(data, &tokens[i], "input_tokens") && tokens[i+1].type == JSMN_PRIMITIVE) {
                    ctx->input_tokens = sse_tok_int(data, &tokens[i+1]);
                    break;
                }
            }
        } else if (strcmp(event_type, "message_delta") == 0) {
            // Extract output_tokens
            for (int i = 1; i < ntok - 1; i++) {
                if (sse_tok_eq(data, &tokens[i], "output_tokens") && tokens[i+1].type == JSMN_PRIMITIVE) {
                    ctx->output_tokens = sse_tok_int(data, &tokens[i+1]);
                    break;
                }
            }
        } else if (strcmp(event_type, "error") == 0) {
            ctx->had_error = 1;
            for (int i = 1; i < ntok - 1; i++) {
                if (sse_tok_eq(data, &tokens[i], "message") && tokens[i+1].type == JSMN_STRING) {
                    size_t ml = (size_t)(tokens[i+1].end - tokens[i+1].start);
                    ctx->error_msg = malloc(ml + 1);
                    if (ctx->error_msg) {
                        memcpy(ctx->error_msg, data + tokens[i+1].start, ml);
                        ctx->error_msg[ml] = '\0';
                    }
                    break;
                }
            }
        }

    } else {
        // OpenAI / Gemini streaming events:
        //   OpenAI: {"choices":[{"delta":{"content":"..."},"finish_reason":null}]}
        //   Gemini: {"candidates":[{"content":{"parts":[{"text":"..."}]}}]}
        //   {"usage":{"prompt_tokens":N,"completion_tokens":N}}

        for (int i = 1; i < ntok - 1; i++) {
            // Match "content" (OpenAI) or "text" (Gemini parts)
            if ((sse_tok_eq(data, &tokens[i], "content") || sse_tok_eq(data, &tokens[i], "text"))
                && tokens[i+1].type == JSMN_STRING) {
                size_t tlen = (size_t)(tokens[i+1].end - tokens[i+1].start);

                // Copy and unescape JSON string escapes
                char *text = malloc(tlen + 1);
                if (!text) continue;
                memcpy(text, data + tokens[i+1].start, tlen);
                text[tlen] = '\0';
                tlen = json_unescape(text, tlen);

                if (ctx->text_len + tlen >= ctx->text_cap) {
                    size_t new_cap = (ctx->text_cap + tlen) * 2;
                    char *nb = realloc(ctx->text_buf, new_cap);
                    if (!nb) { free(text); continue; }
                    ctx->text_buf = nb;
                    ctx->text_cap = new_cap;
                }
                memcpy(ctx->text_buf + ctx->text_len, text, tlen);
                ctx->text_len += tlen;
                ctx->text_buf[ctx->text_len] = '\0';

                if (ctx->settings->on_stream)
                    ctx->settings->on_stream(ctx->settings->stream_ctx,
                                              text, tlen, 0);
                free(text);
                break;
            }

            // Tool call deltas
            if (sse_tok_eq(data, &tokens[i], "tool_calls") && tokens[i+1].type == JSMN_ARRAY) {
                // Walk array items — each has index, id, function.name, function.arguments
                int arr_size = tokens[i+1].size;
                int k = i + 2;
                for (int a = 0; a < arr_size && k < ntok; a++) {
                    int idx = -1;
                    const char *tc_id = NULL, *tc_name = NULL, *tc_args = NULL;
                    size_t tc_args_len = 0;
                    char buf1[128], buf2[128];

                    if (tokens[k].type != JSMN_OBJECT) { k++; continue; }
                    int obj_size = tokens[k].size;
                    int m = k + 1;
                    for (int f = 0; f < obj_size && m < ntok - 1; f++) {
                        if (sse_tok_eq(data, &tokens[m], "index") && tokens[m+1].type == JSMN_PRIMITIVE) {
                            idx = sse_tok_int(data, &tokens[m+1]); m += 2;
                        } else if (sse_tok_eq(data, &tokens[m], "id") && tokens[m+1].type == JSMN_STRING) {
                            tc_id = sse_tok_str(data, &tokens[m+1], buf1, sizeof(buf1)); m += 2;
                        } else if (sse_tok_eq(data, &tokens[m], "function") && tokens[m+1].type == JSMN_OBJECT) {
                            int fn_size = tokens[m+1].size;
                            int v = m + 2;
                            for (int vf = 0; vf < fn_size && v < ntok - 1; vf++) {
                                if (sse_tok_eq(data, &tokens[v], "name") && tokens[v+1].type == JSMN_STRING) {
                                    tc_name = sse_tok_str(data, &tokens[v+1], buf2, sizeof(buf2)); v += 2;
                                } else if (sse_tok_eq(data, &tokens[v], "arguments") && tokens[v+1].type == JSMN_STRING) {
                                    tc_args = data + tokens[v+1].start;
                                    tc_args_len = (size_t)(tokens[v+1].end - tokens[v+1].start);
                                    v += 2;
                                } else { v += 2; }
                            }
                            m += 2; // skip function object
                            // skip remaining tokens of function object
                        } else { m += 2; }
                    }

                    if (idx < 0) idx = (int)ctx->tc_count;

                    // Initialize new tool call or append to existing
                    if ((size_t)idx >= ctx->tc_count && (size_t)idx < MAX_TOOL_CALLS) {
                        ctx->tc_count = (size_t)(idx + 1);
                        if (tc_id) ctx->tool_calls[idx].id = strdup(tc_id);
                        if (tc_name) ctx->tool_calls[idx].name = strdup(tc_name);
                        if (!ctx->tool_calls[idx].args) {
                            ctx->tool_calls[idx].args = calloc(1, 256);
                            ctx->tool_calls[idx].args_cap = 256;
                        }
                    }

                    // Append arguments (unescape JSON string escapes)
                    if (tc_args && tc_args_len > 0 && (size_t)idx < MAX_TOOL_CALLS) {
                        char *unesc = malloc(tc_args_len + 1);
                        if (!unesc) continue;
                        memcpy(unesc, tc_args, tc_args_len);
                        unesc[tc_args_len] = '\0';
                        tc_args_len = json_unescape(unesc, tc_args_len);

                        size_t ci = (size_t)idx;
                        if (ctx->tool_calls[ci].args_len + tc_args_len >= ctx->tool_calls[ci].args_cap) {
                            size_t new_cap = (ctx->tool_calls[ci].args_cap + tc_args_len) * 2;
                            char *nb = realloc(ctx->tool_calls[ci].args, new_cap);
                            if (!nb) { free(unesc); continue; }
                            ctx->tool_calls[ci].args = nb;
                            ctx->tool_calls[ci].args_cap = new_cap;
                        }
                        memcpy(ctx->tool_calls[ci].args + ctx->tool_calls[ci].args_len, unesc, tc_args_len);
                        ctx->tool_calls[ci].args_len += tc_args_len;
                        ctx->tool_calls[ci].args[ctx->tool_calls[ci].args_len] = '\0';
                        free(unesc);
                    }

                    // Skip to next array element
                    k = m;
                }
                break;
            }

            // Gemini tool calls: "functionCall":{"name":"...","args":{...}}
            if (sse_tok_eq(data, &tokens[i], "functionCall") && tokens[i+1].type == JSMN_OBJECT) {
                const char *fc_name = NULL;
                char nbuf[128];
                int fc_obj_size = tokens[i+1].size;
                int fc_args_start = -1, fc_args_end = -1;
                int m = i + 2;
                for (int f = 0; f < fc_obj_size && m < ntok - 1; f++) {
                    if (sse_tok_eq(data, &tokens[m], "name") && tokens[m+1].type == JSMN_STRING) {
                        fc_name = sse_tok_str(data, &tokens[m+1], nbuf, sizeof(nbuf));
                        m += 2;
                    } else if (sse_tok_eq(data, &tokens[m], "args")) {
                        fc_args_start = tokens[m+1].start;
                        fc_args_end = tokens[m+1].end;
                        // Skip past the args object/value
                        m++;
                        int depth = 1;
                        while (m < ntok && depth > 0) {
                            if (tokens[m].type == JSMN_OBJECT || tokens[m].type == JSMN_ARRAY)
                                depth += tokens[m].size;
                            depth--;
                            m++;
                        }
                    } else {
                        m += 2;
                    }
                }

                if (fc_name && ctx->tc_count < MAX_TOOL_CALLS) {
                    size_t idx = ctx->tc_count++;
                    // Gemini has no tool call IDs — generate one
                    char id_buf[32];
                    snprintf(id_buf, sizeof(id_buf), "gemini_%zu", idx);
                    ctx->tool_calls[idx].id = strdup(id_buf);
                    ctx->tool_calls[idx].name = strdup(fc_name);
                    if (fc_args_start >= 0 && fc_args_end > fc_args_start) {
                        size_t alen = (size_t)(fc_args_end - fc_args_start);
                        ctx->tool_calls[idx].args = malloc(alen + 1);
                        if (ctx->tool_calls[idx].args) {
                            memcpy(ctx->tool_calls[idx].args, data + fc_args_start, alen);
                            ctx->tool_calls[idx].args[alen] = '\0';
                            ctx->tool_calls[idx].args_len = alen;
                            ctx->tool_calls[idx].args_cap = alen + 1;
                        }
                    } else {
                        ctx->tool_calls[idx].args = strdup("{}");
                        ctx->tool_calls[idx].args_len = 2;
                        ctx->tool_calls[idx].args_cap = 3;
                    }
                }
                break;
            }

            // Usage tokens (OpenAI: prompt_tokens/completion_tokens, Gemini: promptTokenCount/candidatesTokenCount)
            if ((sse_tok_eq(data, &tokens[i], "prompt_tokens") || sse_tok_eq(data, &tokens[i], "promptTokenCount"))
                && tokens[i+1].type == JSMN_PRIMITIVE) {
                ctx->input_tokens = sse_tok_int(data, &tokens[i+1]);
            }
            if ((sse_tok_eq(data, &tokens[i], "completion_tokens") || sse_tok_eq(data, &tokens[i], "candidatesTokenCount"))
                && tokens[i+1].type == JSMN_PRIMITIVE) {
                ctx->output_tokens = sse_tok_int(data, &tokens[i+1]);
            }
        }
    }
}

// ============================================================================
// MARK: - SSE chunk callback (called from adam_net_post_streaming)
// ============================================================================

static int sse_chunk_cb(void *ctx_ptr, const uint8_t *chunk, size_t len) {
    sse_ctx_t *ctx = (sse_ctx_t *)ctx_ptr;
    ctx->chunk_count++;

    for (size_t i = 0; i < len; i++) {
        char c = (char)chunk[i];

        if (c == '\n') {
            if (ctx->line_len == 0) continue; // empty line (event separator)
            ctx->line_buf[ctx->line_len] = '\0';

            // Parse "data: " prefix
            if (ctx->line_len > 6 && strncmp(ctx->line_buf, "data: ", 6) == 0) {
                sse_process_line(ctx, ctx->line_buf + 6, ctx->line_len - 6);
            }
            // Also handle "data:" without space (some servers)
            else if (ctx->line_len > 5 && strncmp(ctx->line_buf, "data:", 5) == 0) {
                sse_process_line(ctx, ctx->line_buf + 5, ctx->line_len - 5);
            }
            // Ignore "event:", "id:", "retry:" lines

            ctx->line_len = 0;
        } else {
            // Grow line buffer if needed
            if (ctx->line_len >= ctx->line_cap - 1) {
                ctx->line_cap = ctx->line_cap ? ctx->line_cap * 2 : 4096;
                char *nb = realloc(ctx->line_buf, ctx->line_cap);
                if (!nb) return -1;
                ctx->line_buf = nb;
            }
            ctx->line_buf[ctx->line_len++] = c;
        }
    }

    return 0; // continue
}

// ============================================================================
// MARK: - Public: streaming LLM call
// ============================================================================

adam_llm_response_t adam_llm_call_http_stream(
    arena_t *arena, adam_settings_t *s,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
) {
    adam_llm_response_t resp = {0};

    // Build request JSON with stream:true
    const char *base_body = adam_json_build_request(
        arena, s->api_format, s->model,
        msgs, msg_count, tools, tool_count,
        s->temperature, s->max_tokens, s->top_p,
        s->response_format
    );
    if (!base_body) {
        resp.error = ADAM_ERR_JSON;
        resp.error_msg = arena_strdup(arena, "failed to build request JSON");
        return resp;
    }

    // Inject "stream":true for Anthropic/OpenAI (Gemini uses alt=sse in URL)
    const char *body;
    if (s->api_format == ADAM_API_GEMINI) {
        body = base_body;
    } else {
        size_t base_len = strlen(base_body);
        const char *last_brace = strrchr(base_body, '}');
        if (!last_brace) {
            resp.error = ADAM_ERR_JSON;
            resp.error_msg = arena_strdup(arena, "malformed JSON body");
            return resp;
        }
        size_t prefix_len = (size_t)(last_brace - base_body);
        char *mbody = arena_alloc(arena, base_len + 32);
        if (!mbody) { resp.error = ADAM_ERR_ALLOC; return resp; }
        memcpy(mbody, base_body, prefix_len);
        size_t pos = prefix_len;
        pos += (size_t)snprintf(mbody + pos, base_len + 32 - pos, ",\"stream\":true}");
        mbody[pos] = '\0';
        body = mbody;
    }

    // URL
    const char *url = s->base_url;
    if (!url) {
        if (s->api_format == ADAM_API_GEMINI) {
            const char *model = s->model ? s->model : "gemini-2.0-flash";
            size_t gurl_len = strlen(model) + (s->api_key ? strlen(s->api_key) : 0) + 128;
            char *gurl = arena_alloc(arena, gurl_len);
            if (gurl) {
                snprintf(gurl, gurl_len,
                    "https://generativelanguage.googleapis.com/v1beta/models/%s:streamGenerateContent?alt=sse&key=%s",
                    model, s->api_key ? s->api_key : "");
                url = gurl;
            }
        } else if (s->api_format == ADAM_API_ANTHROPIC) {
            url = "https://api.anthropic.com/v1/messages";
        } else {
            url = "https://api.openai.com/v1/chat/completions";
        }
    }

    // Auth
    if (!s->api_key) {
        resp.error = ADAM_ERR_AUTH;
        resp.error_msg = arena_strdup(arena, "api_key is NULL");
        return resp;
    }
    size_t auth_len = strlen(s->api_key) + AUTH_HDR_EXTRA;
    char *auth = arena_alloc(arena, auth_len);
    if (!auth) { resp.error = ADAM_ERR_ALLOC; return resp; }

    if (s->api_format == ADAM_API_ANTHROPIC) {
        snprintf(auth, auth_len, "x-api-key: %s", s->api_key);
    } else if (s->api_format == ADAM_API_GEMINI) {
        auth[0] = '\0'; // key in URL query string
    } else {
        snprintf(auth, auth_len, "Authorization: Bearer %s", s->api_key);
    }

    // Initialize SSE context
    sse_ctx_t ctx = {0};
    ctx.settings = s;
    ctx.format = s->api_format;
    ctx.text_buf = calloc(1, 4096);
    ctx.text_cap = 4096;
    ctx.line_buf = calloc(1, 4096);
    ctx.line_cap = 4096;
    ctx.tc_current = -1;

    // Extra headers for Anthropic
    const char *extra_hdrs[3] = {NULL, NULL, NULL};
    int eh = 0;
    if (s->api_format == ADAM_API_ANTHROPIC) {
        extra_hdrs[eh++] = "anthropic-version: 2023-06-01";
    }

    adam_net_response_t net = adam_net_post_streaming(
        s, url, auth, body, extra_hdrs[0] ? extra_hdrs : NULL,
        sse_chunk_cb, &ctx, /*handle_id=*/0
    );

    // Signal done to stream callback
    if (s->on_stream)
        s->on_stream(s->stream_ctx, "", 0, 1);

    // Build response from accumulated data
    if (net.error != ADAM_OK) {
        resp.error = net.error;
        resp.error_msg = arena_strdup(arena, "streaming HTTP request failed");
    } else if (net.http_code == 429) {
        resp.error = ADAM_ERR_RATE_LIMIT;
        resp.error_msg = arena_strdup(arena, "rate limited (429)");
    } else if (net.http_code == 401 || net.http_code == 403) {
        resp.error = ADAM_ERR_AUTH;
        resp.error_msg = arena_strdup(arena, "authentication error");
    } else if (net.http_code >= 400 || ctx.had_error) {
        resp.error = ADAM_ERR_PROVIDER;
        resp.error_msg = ctx.error_msg
            ? arena_strdup(arena, ctx.error_msg)
            : arena_strdup(arena, "stream error");
    } else {
        // Copy text content to arena
        if (ctx.text_len > 0) {
            resp.content = arena_strdup(arena, ctx.text_buf);
        }

        // Copy tool calls to arena
        if (ctx.tc_count > 0) {
            resp.tool_calls = arena_alloc(arena,
                ctx.tc_count * sizeof(adam_tool_call_t));
            resp.tool_call_count = ctx.tc_count;
            for (size_t i = 0; i < ctx.tc_count; i++) {
                resp.tool_calls[i].id = ctx.tool_calls[i].id
                    ? arena_strdup(arena, ctx.tool_calls[i].id) : NULL;
                resp.tool_calls[i].name = ctx.tool_calls[i].name
                    ? arena_strdup(arena, ctx.tool_calls[i].name) : NULL;
                resp.tool_calls[i].arguments_json = ctx.tool_calls[i].args
                    ? arena_strdup(arena, ctx.tool_calls[i].args) : arena_strdup(arena, "{}");
            }
        }

        resp.input_tokens = ctx.input_tokens;
        resp.output_tokens = ctx.output_tokens;
    }

    // Cleanup SSE context (all malloc'd buffers)
    free(ctx.text_buf);
    free(ctx.line_buf);
    free(ctx.error_msg);
    for (size_t i = 0; i < ctx.tc_count; i++) {
        free(ctx.tool_calls[i].id);
        free(ctx.tool_calls[i].name);
        free(ctx.tool_calls[i].args);
    }

    return resp;
}
