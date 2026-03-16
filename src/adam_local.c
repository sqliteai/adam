//
//  adam_local.c
//  Adam — Local GGUF inference via llama.cpp
//
//  Created by Marco Bambini on 16/03/26.
//

#ifndef ADAM_NO_LOCAL

#include "adam.h"
#include "llama.h"
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
// MARK: - Build prompt using llama_chat_apply_template
// ============================================================================

static char *build_chat_prompt(arena_t *arena,
                                const struct llama_model *model,
                                const adam_message_t *msgs, size_t msg_count,
                                const adam_tool_def_t *tools, size_t tool_count) {
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
// MARK: - Generate (tokenize → decode prompt → sample tokens)
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

    // Build prompt from messages
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
    if (!tokens) {
        resp.error = ADAM_ERR_ALLOC;
        return resp;
    }

    int n_prompt = llama_tokenize(lctx->vocab,
        prompt, (int32_t)strlen(prompt),
        tokens, n_prompt_max,
        true,  /* add_special (BOS) */
        true); /* parse_special */

    if (n_prompt < 0) {
        resp.error = ADAM_ERR_LOCAL;
        resp.error_msg = arena_strdup(arena, "tokenization failed");
        return resp;
    }

    resp.input_tokens = n_prompt;

    // Clear KV cache for fresh generation
    llama_memory_clear(llama_get_memory(lctx->ctx), true);

    // Decode prompt in batches (prompt may exceed n_batch)
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

    // Generate tokens one by one
    int max_tokens = s->max_tokens > 0 ? s->max_tokens : 4096;
    // Don't exceed remaining context
    int max_gen = lctx->n_ctx - n_prompt;
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

        // Check for end of generation
        if (llama_vocab_is_eog(lctx->vocab, new_token))
            break;

        // Detokenize
        char piece[256];
        int n = llama_token_to_piece(lctx->vocab, new_token,
                                      piece, sizeof(piece), 0, true);
        if (n < 0) break;

        // Grow output buffer if needed
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

        // Stream callback
        if (s->on_stream)
            s->on_stream(s->stream_ctx, piece, (size_t)n, 0);

        // Abort check
        if (s->abort_flag) break;

        // Prepare next decode (single token)
        struct llama_batch next = llama_batch_get_one(&new_token, 1);
        if (llama_decode(lctx->ctx, next) != 0) break;
    }

    output[out_len] = '\0';
    if (s->on_stream)
        s->on_stream(s->stream_ctx, "", 0, 1); // signal done

    resp.content = output;
    resp.output_tokens = n_generated;

    // Parse tool calls from text output (models may use <tool_call> tags)
    // TODO: implement tool call extraction for local models

    return resp;
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

#endif // ADAM_NO_LOCAL
