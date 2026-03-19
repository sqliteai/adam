//
//  adam_local.c
//  Adam — Local GGUF inference via llama.cpp (with multimodal vision support)
//
//  Created by Marco Bambini on 16/03/26.
//

#ifndef ADAM_NO_LOCAL

#include "adam.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// ============================================================================
// MARK: - Local context (persistent across turns)
// ============================================================================

typedef struct {
    struct llama_model   *model;
    struct llama_context *ctx;
    struct llama_sampler *sampler;
    const struct llama_vocab *vocab;
    int                   n_ctx;
} adam_local_ctx_t;

// ============================================================================
// MARK: - Init / Free
// ============================================================================

static adam_local_ctx_t *local_init(const adam_settings_t *s) {
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = s->local_gpu_layers;

    struct llama_model *model = llama_model_load_from_file(
        s->gguf_path, mparams);
    if (!model) {
        fprintf(stderr, "[adam_local] failed to load model: %s\n", s->gguf_path);
        return NULL;
    }

    int n_ctx = s->local_ctx_size > 0 ? s->local_ctx_size : 4096;
    int n_batch = s->local_batch_size > 0 ? s->local_batch_size : 512;

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t)n_ctx;
    cparams.n_batch = (uint32_t)n_batch;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "[adam_local] failed to create context\n");
        llama_model_free(model);
        return NULL;
    }

    // Sampler chain: temperature → distribution sampling
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler *sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler,
        llama_sampler_init_temp(s->temperature));
    llama_sampler_chain_add(sampler,
        llama_sampler_init_dist(0));

    adam_local_ctx_t *lctx = calloc(1, sizeof(adam_local_ctx_t));
    if (!lctx) {
        llama_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }

    lctx->model = model;
    lctx->ctx = ctx;
    lctx->sampler = sampler;
    lctx->vocab = llama_model_get_vocab(model);
    lctx->n_ctx = n_ctx;

    return lctx;
}

static void local_free(adam_local_ctx_t *lctx) {
    if (!lctx) return;
    llama_sampler_free(lctx->sampler);
    llama_free(lctx->ctx);
    llama_model_free(lctx->model);
    free(lctx);
}

// ============================================================================
// MARK: - Multimodal (mtmd) init / free
// ============================================================================

static mtmd_context *vision_init(const adam_settings_t *s,
                                  const struct llama_model *model) {
    struct mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = true;
    mparams.n_threads = 4;
    mparams.print_timings = false;

    mtmd_context *ctx = mtmd_init_from_file(s->mmproj_path, model, mparams);
    if (!ctx) {
        fprintf(stderr, "[adam_local] failed to load mmproj: %s\n",
                s->mmproj_path);
        return NULL;
    }

    if (!mtmd_support_vision(ctx)) {
        fprintf(stderr, "[adam_local] mmproj does not support vision\n");
        mtmd_free(ctx);
        return NULL;
    }

    return ctx;
}

// ============================================================================
// MARK: - Build prompt using llama_chat_apply_template
// ============================================================================

static char *build_chat_prompt(arena_t *arena,
                                const struct llama_model *model,
                                const adam_message_t *msgs, size_t msg_count,
                                const adam_tool_def_t *tools, size_t tool_count) {
    UNUSED_PARAM(model);
    UNUSED_PARAM(tools);
    UNUSED_PARAM(tool_count);

    // Convert adam messages to llama_chat_message format
    struct llama_chat_message *chat = arena_alloc(arena,
        msg_count * sizeof(struct llama_chat_message));
    if (!chat) return NULL;

    for (size_t i = 0; i < msg_count; i++) {
        switch (msgs[i].role) {
        case ADAM_ROLE_SYSTEM:    chat[i].role = "system";    break;
        case ADAM_ROLE_ASSISTANT: chat[i].role = "assistant";  break;
        case ADAM_ROLE_TOOL:      chat[i].role = "tool";       break;
        default:                 chat[i].role = "user";        break;
        }
        chat[i].content = msgs[i].content ? msgs[i].content : "";
    }

    // First call: determine buffer size needed
    int32_t needed = llama_chat_apply_template(
        NULL, /* tmpl = NULL → use model's built-in template */
        chat, msg_count,
        true, /* add_ass = true → add assistant prompt start */
        NULL, 0);

    if (needed <= 0) {
        // Fallback: concatenate messages manually (ChatML-like)
        size_t est = 256;
        for (size_t i = 0; i < msg_count; i++) est += msgs[i].content_len + 64;
        char *buf = arena_alloc(arena, est);
        if (!buf) return NULL;
        size_t pos = 0;
        for (size_t i = 0; i < msg_count; i++) {
            pos += (size_t)snprintf(buf + pos, est - pos,
                "<|im_start|>%s\n%s<|im_end|>\n", chat[i].role, chat[i].content);
        }
        pos += (size_t)snprintf(buf + pos, est - pos, "<|im_start|>assistant\n");
        return buf;
    }

    // Second call: fill buffer
    char *buf = arena_alloc(arena, (size_t)needed + 1);
    if (!buf) return NULL;
    llama_chat_apply_template(NULL, chat, msg_count, true, buf, needed + 1);
    buf[needed] = '\0';
    return buf;
}

// ============================================================================
// MARK: - Build prompt with media markers for vision
// ============================================================================

// Build a chat prompt where user messages with attachments get media markers
// injected before the text content so mtmd can replace them with image tokens.
static char *build_chat_prompt_vision(arena_t *arena,
                                       const struct llama_model *model,
                                       const adam_message_t *msgs,
                                       size_t msg_count,
                                       const adam_tool_def_t *tools,
                                       size_t tool_count) {
    UNUSED_PARAM(model);
    UNUSED_PARAM(tools);
    UNUSED_PARAM(tool_count);

    const char *marker = mtmd_default_marker();

    struct llama_chat_message *chat = arena_alloc(arena,
        msg_count * sizeof(struct llama_chat_message));
    if (!chat) return NULL;

    // We need temp buffers for messages that get markers prepended
    for (size_t i = 0; i < msg_count; i++) {
        switch (msgs[i].role) {
        case ADAM_ROLE_SYSTEM:    chat[i].role = "system";    break;
        case ADAM_ROLE_ASSISTANT: chat[i].role = "assistant";  break;
        case ADAM_ROLE_TOOL:      chat[i].role = "tool";       break;
        default:                 chat[i].role = "user";        break;
        }

        // For user messages with image attachments, prepend media markers
        if (msgs[i].role == ADAM_ROLE_USER && msgs[i].attachment_count > 0) {
            size_t n_images = 0;
            for (size_t j = 0; j < msgs[i].attachment_count; j++) {
                if (msgs[i].attachments[j].type <= ADAM_MEDIA_IMAGE_WEBP)
                    n_images++;
            }

            if (n_images > 0) {
                const char *text = msgs[i].content ? msgs[i].content : "";
                size_t marker_len = strlen(marker);
                size_t buf_len = n_images * (marker_len + 1) + strlen(text) + 1;
                char *buf = arena_alloc(arena, buf_len);
                if (!buf) return NULL;
                size_t pos = 0;
                for (size_t j = 0; j < n_images; j++) {
                    memcpy(buf + pos, marker, marker_len);
                    pos += marker_len;
                    buf[pos++] = '\n';
                }
                size_t tlen = strlen(text);
                memcpy(buf + pos, text, tlen);
                pos += tlen;
                buf[pos] = '\0';
                chat[i].content = buf;
                continue;
            }
        }

        chat[i].content = msgs[i].content ? msgs[i].content : "";
    }

    // Apply chat template
    int32_t needed = llama_chat_apply_template(NULL, chat, msg_count,
                                                true, NULL, 0);
    if (needed <= 0) {
        size_t est = 256;
        for (size_t i = 0; i < msg_count; i++)
            est += strlen(chat[i].content) + 64;
        char *buf = arena_alloc(arena, est);
        if (!buf) return NULL;
        size_t pos = 0;
        for (size_t i = 0; i < msg_count; i++) {
            pos += (size_t)snprintf(buf + pos, est - pos,
                "<|im_start|>%s\n%s<|im_end|>\n",
                chat[i].role, chat[i].content);
        }
        pos += (size_t)snprintf(buf + pos, est - pos, "<|im_start|>assistant\n");
        return buf;
    }

    char *buf = arena_alloc(arena, (size_t)needed + 1);
    if (!buf) return NULL;
    llama_chat_apply_template(NULL, chat, msg_count, true, buf, needed + 1);
    buf[needed] = '\0';
    return buf;
}

// ============================================================================
// MARK: - Check if messages contain image attachments
// ============================================================================

static int has_image_attachments(const adam_message_t *msgs, size_t msg_count) {
    for (size_t i = 0; i < msg_count; i++) {
        for (size_t j = 0; j < msgs[i].attachment_count; j++) {
            if (msgs[i].attachments[j].type <= ADAM_MEDIA_IMAGE_WEBP)
                return 1;
        }
    }
    return 0;
}

// ============================================================================
// MARK: - Collect image bitmaps from messages
// ============================================================================

static mtmd_bitmap **collect_bitmaps(mtmd_context *vctx,
                                      const adam_message_t *msgs,
                                      size_t msg_count,
                                      size_t *out_count) {
    // Count total images
    size_t n_images = 0;
    for (size_t i = 0; i < msg_count; i++) {
        for (size_t j = 0; j < msgs[i].attachment_count; j++) {
            if (msgs[i].attachments[j].type <= ADAM_MEDIA_IMAGE_WEBP)
                n_images++;
        }
    }

    if (n_images == 0) { *out_count = 0; return NULL; }

    mtmd_bitmap **bitmaps = calloc(n_images, sizeof(mtmd_bitmap *));
    if (!bitmaps) { *out_count = 0; return NULL; }

    size_t idx = 0;
    for (size_t i = 0; i < msg_count; i++) {
        for (size_t j = 0; j < msgs[i].attachment_count; j++) {
            adam_attachment_t *att = &msgs[i].attachments[j];
            if (att->type > ADAM_MEDIA_IMAGE_WEBP) continue;

            mtmd_bitmap *bmp = mtmd_helper_bitmap_init_from_buf(
                vctx, att->data, att->data_len);
            if (!bmp) {
                fprintf(stderr, "[adam_local] failed to decode image %zu\n", idx);
                // Free already allocated bitmaps
                for (size_t k = 0; k < idx; k++)
                    mtmd_bitmap_free(bitmaps[k]);
                free(bitmaps);
                *out_count = 0;
                return NULL;
            }
            bitmaps[idx++] = bmp;
        }
    }

    *out_count = idx;
    return bitmaps;
}

// ============================================================================
// MARK: - Token generation (shared between text-only and vision paths)
// ============================================================================

static adam_llm_response_t generate_tokens(arena_t *arena,
                                            adam_settings_t *s,
                                            adam_local_ctx_t *lctx,
                                            int n_prompt_tokens) {
    adam_llm_response_t resp = {0};
    resp.input_tokens = n_prompt_tokens;

    int max_tokens = s->max_tokens > 0 ? s->max_tokens : 4096;
    int max_gen = lctx->n_ctx - n_prompt_tokens;
    if (max_gen < 1) {
        resp.error = ADAM_ERR_CONTEXT_OVERFLOW;
        resp.error_msg = arena_strdup(arena, "prompt too long for context window");
        return resp;
    }
    if (max_tokens > max_gen) max_tokens = max_gen;

    size_t out_cap = 4096;
    char *output = arena_alloc(arena, out_cap);
    if (!output) { resp.error = ADAM_ERR_ALLOC; return resp; }
    size_t out_len = 0;
    int n_generated = 0;

    for (int i = 0; i < max_tokens; i++) {
        llama_token new_token = llama_sampler_sample(
            lctx->sampler, lctx->ctx, -1);

        if (llama_vocab_is_eog(lctx->vocab, new_token))
            break;

        char piece[256];
        int n = llama_token_to_piece(lctx->vocab, new_token,
                                      piece, sizeof(piece), 0, true);
        if (n < 0) break;

        if (out_len + (size_t)n >= out_cap) {
            out_cap *= 2;
            char *new_buf = arena_alloc(arena, out_cap);
            if (!new_buf) break;
            memcpy(new_buf, output, out_len);
            output = new_buf;
        }
        memcpy(output + out_len, piece, (size_t)n);
        out_len += (size_t)n;
        n_generated++;

        if (s->on_stream)
            s->on_stream(s->stream_ctx, piece, (size_t)n, 0);

        if (s->abort_flag) break;

        struct llama_batch next = llama_batch_get_one(&new_token, 1);
        if (llama_decode(lctx->ctx, next) != 0) break;
    }

    output[out_len] = '\0';
    if (s->on_stream)
        s->on_stream(s->stream_ctx, "", 0, 1);

    resp.content = output;
    resp.output_tokens = n_generated;
    return resp;
}

// ============================================================================
// MARK: - Text-only local inference
// ============================================================================

static adam_llm_response_t call_local_text(
    arena_t *arena, adam_settings_t *s, adam_local_ctx_t *lctx,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
) {
    adam_llm_response_t resp = {0};

    char *prompt = build_chat_prompt(arena, lctx->model,
                                      msgs, msg_count, tools, tool_count);
    if (!prompt) {
        resp.error = ADAM_ERR_LOCAL;
        resp.error_msg = arena_strdup(arena, "failed to build prompt");
        return resp;
    }

    // Tokenize
    int n_prompt_max = lctx->n_ctx;
    llama_token *tokens = arena_alloc(arena,
        (size_t)n_prompt_max * sizeof(llama_token));
    if (!tokens) { resp.error = ADAM_ERR_ALLOC; return resp; }

    int n_prompt = llama_tokenize(lctx->vocab,
        prompt, (int32_t)strlen(prompt),
        tokens, n_prompt_max, true, true);

    if (n_prompt < 0) {
        resp.error = ADAM_ERR_LOCAL;
        resp.error_msg = arena_strdup(arena, "tokenization failed");
        return resp;
    }

    // Clear KV cache
    llama_memory_clear(llama_get_memory(lctx->ctx), true);

    // Decode prompt in batches
    int n_batch = s->local_batch_size > 0 ? s->local_batch_size : 512;
    for (int i = 0; i < n_prompt; i += n_batch) {
        int n_chunk = n_prompt - i;
        if (n_chunk > n_batch) n_chunk = n_batch;
        struct llama_batch batch = llama_batch_get_one(tokens + i, n_chunk);
        if (llama_decode(lctx->ctx, batch) != 0) {
            resp.error = ADAM_ERR_LOCAL;
            resp.error_msg = arena_strdup(arena, "prompt decode failed");
            return resp;
        }
    }

    return generate_tokens(arena, s, lctx, n_prompt);
}

// ============================================================================
// MARK: - Vision (multimodal) local inference
// ============================================================================

static adam_llm_response_t call_local_vision(
    arena_t *arena, adam_settings_t *s, adam_local_ctx_t *lctx,
    mtmd_context *vctx,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
) {
    adam_llm_response_t resp = {0};

    // Build prompt with media markers
    char *prompt = build_chat_prompt_vision(arena, lctx->model,
                                             msgs, msg_count, tools, tool_count);
    if (!prompt) {
        resp.error = ADAM_ERR_LOCAL;
        resp.error_msg = arena_strdup(arena, "failed to build vision prompt");
        return resp;
    }

    // Collect image bitmaps from message attachments
    size_t n_bitmaps = 0;
    mtmd_bitmap **bitmaps = collect_bitmaps(vctx, msgs, msg_count, &n_bitmaps);
    if (n_bitmaps == 0) {
        resp.error = ADAM_ERR_LOCAL;
        resp.error_msg = arena_strdup(arena, "no valid images found");
        return resp;
    }

    // Tokenize text + images via mtmd
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    mtmd_input_text input_text;
    input_text.text = prompt;
    input_text.add_special = true;
    input_text.parse_special = true;

    int32_t tok_res = mtmd_tokenize(vctx, chunks,
        &input_text, (const mtmd_bitmap **)bitmaps, n_bitmaps);

    if (tok_res != 0) {
        fprintf(stderr, "[adam_local] mtmd_tokenize failed: %d\n", tok_res);
        resp.error = ADAM_ERR_LOCAL;
        resp.error_msg = arena_strdup(arena,
            tok_res == 1 ? "image count mismatch with markers"
                         : "image preprocessing failed");
        mtmd_input_chunks_free(chunks);
        for (size_t i = 0; i < n_bitmaps; i++) mtmd_bitmap_free(bitmaps[i]);
        free(bitmaps);
        return resp;
    }

    // Get total token count for reporting
    size_t n_tokens = mtmd_helper_get_n_tokens(chunks);

    // Clear KV cache
    llama_memory_clear(llama_get_memory(lctx->ctx), true);

    // Evaluate all chunks (text + image embeddings)
    int n_batch = s->local_batch_size > 0 ? s->local_batch_size : 512;
    llama_pos n_past = 0;
    llama_pos new_n_past = 0;

    int32_t eval_res = mtmd_helper_eval_chunks(
        vctx, lctx->ctx, chunks,
        n_past, 0, n_batch, true, &new_n_past);

    // Cleanup bitmaps and chunks
    mtmd_input_chunks_free(chunks);
    for (size_t i = 0; i < n_bitmaps; i++) mtmd_bitmap_free(bitmaps[i]);
    free(bitmaps);

    if (eval_res != 0) {
        fprintf(stderr, "[adam_local] mtmd_helper_eval_chunks failed: %d\n",
                eval_res);
        resp.error = ADAM_ERR_LOCAL;
        resp.error_msg = arena_strdup(arena, "vision prompt evaluation failed");
        return resp;
    }

    return generate_tokens(arena, s, lctx, (int)n_tokens);
}

// ============================================================================
// MARK: - Public entry point
// ============================================================================

adam_llm_response_t adam_llm_call_local(
    arena_t *arena, adam_settings_t *s,
    const adam_message_t *msgs, size_t msg_count,
    const adam_tool_def_t *tools, size_t tool_count
) {
    adam_llm_response_t resp = {0};

    // Lazy-init the llama.cpp context on first call
    if (!s->_local_ctx) {
        s->_local_ctx = local_init(s);
        if (!s->_local_ctx) {
            resp.error = ADAM_ERR_LOCAL;
            resp.error_msg = arena_strdup(arena, "failed to init local model");
            return resp;
        }
    }

    adam_local_ctx_t *lctx = (adam_local_ctx_t *)s->_local_ctx;

    // Vision path: if mmproj is set and messages contain images
    if (s->mmproj_path && has_image_attachments(msgs, msg_count)) {
        // Lazy-init the mtmd context
        if (!s->_mtmd_ctx) {
            s->_mtmd_ctx = vision_init(s, lctx->model);
            if (!s->_mtmd_ctx) {
                resp.error = ADAM_ERR_LOCAL;
                resp.error_msg = arena_strdup(arena,
                    "failed to init vision model");
                return resp;
            }
        }
        return call_local_vision(arena, s, lctx,
            (mtmd_context *)s->_mtmd_ctx,
            msgs, msg_count, tools, tool_count);
    }

    // Text-only path
    return call_local_text(arena, s, lctx, msgs, msg_count, tools, tool_count);
}

// ============================================================================
// MARK: - Cleanup (called from adam_settings_destroy)
// ============================================================================

void adam_local_cleanup(adam_settings_t *s) {
    if (s && s->_local_ctx) {
        local_free((adam_local_ctx_t *)s->_local_ctx);
        s->_local_ctx = NULL;
    }
}

void adam_local_vision_cleanup(adam_settings_t *s) {
    if (s && s->_mtmd_ctx) {
        mtmd_free((mtmd_context *)s->_mtmd_ctx);
        s->_mtmd_ctx = NULL;
    }
}

#endif // ADAM_NO_LOCAL
