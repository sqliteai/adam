//
//  adam_json.h
//  Adam — JSON building and parsing (internal header)
//
//  Created by Marco Bambini on 15/03/26.
//

#pragma once

#include "adam.h"
#include "arena.h"

#ifdef __cplusplus
extern "C" {
#endif

// Build the request body JSON for an LLM API call.
// Returns arena-allocated JSON string, or NULL on error.
const char *adam_json_build_request(
    arena_t             *arena,
    adam_api_format_t    format,
    const char          *model,
    const adam_message_t *msgs,
    size_t               msg_count,
    const adam_tool_def_t *tools,
    size_t               tool_count,
    float                temperature,
    int                  max_tokens,
    float                top_p,
    const char          *response_format
);

// Parse the response JSON from an LLM API call.
// Populates an adam_llm_response_t with arena-allocated strings.
adam_llm_response_t adam_json_parse_response(
    arena_t             *arena,
    adam_api_format_t    format,
    const char          *json,
    size_t               json_len
);

#ifdef __cplusplus
}
#endif
