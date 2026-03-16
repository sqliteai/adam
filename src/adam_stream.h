//
//  adam_stream.h
//  Adam — SSE streaming for cloud LLM providers (internal header)
//
//  Created by Marco Bambini on 16/03/26.
//

#pragma once

#include "adam.h"
#include "arena.h"

#ifdef __cplusplus
extern "C" {
#endif

// Streaming LLM call: sends text tokens to on_stream in real-time,
// buffers tool calls until complete, returns full response at the end.
// Falls back to non-streaming adam_llm_call_http if on_stream is NULL.
adam_llm_response_t adam_llm_call_http_stream(
    arena_t             *arena,
    adam_settings_t     *s,
    const adam_message_t *msgs,
    size_t               msg_count,
    const adam_tool_def_t *tools,
    size_t               tool_count
);

#ifdef __cplusplus
}
#endif
