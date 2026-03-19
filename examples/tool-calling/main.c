//
//  tool-calling/main.c
//  Custom tool registration and agent tool use
//

#include "adam.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// Mock weather tool implementation
static adam_tool_result_t get_weather(arena_t *arena, void *ctx,
                                      const char *args_json, size_t args_len) {
    UNUSED_PARAM(ctx);
    UNUSED_PARAM(args_len);

    // Extract the "location" field from the JSON arguments.
    // In a real application you would use a JSON parser; here we do a simple check.
    const char *response;
    if (strstr(args_json, "San Francisco") || strstr(args_json, "SF")) {
        response = "{\"location\":\"San Francisco, CA\","
                   "\"temperature_f\":62,"
                   "\"condition\":\"Foggy\","
                   "\"humidity\":78,"
                   "\"wind_mph\":12}";
    } else if (strstr(args_json, "New York") || strstr(args_json, "NYC")) {
        response = "{\"location\":\"New York, NY\","
                   "\"temperature_f\":45,"
                   "\"condition\":\"Cloudy\","
                   "\"humidity\":65,"
                   "\"wind_mph\":8}";
    } else {
        response = "{\"location\":\"Unknown\","
                   "\"temperature_f\":70,"
                   "\"condition\":\"Sunny\","
                   "\"humidity\":50,"
                   "\"wind_mph\":5}";
    }

    // Allocate the response string in the arena (valid until arena reset)
    size_t len = strlen(response);
    char *buf = arena_alloc(arena, len + 1);
    memcpy(buf, response, len + 1);

    return (adam_tool_result_t){
        .for_llm  = buf,
        .for_user = NULL,
        .success  = 1,
    };
}

int main(void) {
    adam_status_t status = adam_init();
    if (status != ADAM_OK) {
        fprintf(stderr, "adam_init failed: %s\n", adam_status_string(status));
        return 1;
    }

    char *api_key = env_load("ANTHROPIC_API_KEY");
    if (!api_key) {
        fprintf(stderr, "Missing ANTHROPIC_API_KEY in .env file\n");
        adam_cleanup();
        return 1;
    }

    adam_settings_t *settings = adam_create_settings();
    adam_settings_set_provider(settings, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");

    // Register the get_weather tool with its JSON schema
    adam_tool_def_t weather_tool = {
        .name            = "get_weather",
        .description     = "Get the current weather for a location.",
        .parameters_json = "{"
            "\"type\":\"object\","
            "\"properties\":{"
                "\"location\":{\"type\":\"string\",\"description\":\"City name, e.g. San Francisco, CA\"}"
            "},"
            "\"required\":[\"location\"]"
        "}",
        .execute = get_weather,
        .ctx     = NULL,
    };
    adam_settings_add_tool(settings, weather_tool);

    adam_history_t *history = adam_history_create();

    printf("User: What's the weather like in San Francisco right now?\n\n");

    adam_run_result_t result = adam_run(settings, history,
        "What's the weather like in San Francisco right now?");

    if (result.status != ADAM_OK) {
        fprintf(stderr, "adam_run failed: %s\n", adam_status_string(result.status));
    } else {
        printf("Assistant: %s\n\n", result.final_response);
        printf("  [iterations: %d, tokens: %d in / %d out, %.1f ms]\n",
               result.total_iterations,
               result.input_tokens, result.output_tokens,
               result.elapsed_ms);
    }

    adam_run_result_free(&result);
    adam_history_destroy(history);
    adam_settings_destroy(settings);
    free(api_key);
    adam_cleanup();
    return 0;
}
