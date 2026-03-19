//
//  guardrails/main.c
//  Input and output validation guardrails
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

// Input guardrail: block messages containing "password"
static int check_input(void *ctx, const adam_message_t *msgs, size_t msg_count) {
    (void)ctx;
    for (size_t i = 0; i < msg_count; i++) {
        if (msgs[i].role == ADAM_ROLE_USER && msgs[i].content) {
            if (strstr(msgs[i].content, "password") != NULL) {
                printf("  [GUARDRAIL] Input blocked: message contains 'password'\n");
                return 1;  // deny
            }
        }
    }
    return 0;  // allow
}

// Output guardrail: block responses containing "SSN"
static int check_output(void *ctx, const char *content,
                         const adam_tool_call_t *tool_calls, size_t tool_call_count) {
    (void)ctx;
    (void)tool_calls;
    (void)tool_call_count;
    if (content && strstr(content, "SSN") != NULL) {
        printf("  [GUARDRAIL] Output blocked: response contains 'SSN'\n");
        return 1;  // deny
    }
    return 0;  // allow
}

static void run_query(adam_settings_t *settings, const char *query) {
    adam_history_t *history = adam_history_create();

    printf("User: %s\n", query);
    adam_run_result_t result = adam_run(settings, history, query);

    if (result.status == ADAM_ERR_GUARDRAIL) {
        printf("Result: BLOCKED by guardrail (ADAM_ERR_GUARDRAIL)\n\n");
    } else if (result.status != ADAM_OK) {
        printf("Result: error - %s\n\n", adam_status_string(result.status));
    } else {
        printf("Assistant: %s\n\n", result.final_response);
    }

    adam_run_result_free(&result);
    adam_history_destroy(history);
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

    // Set guardrails
    settings->on_before_send    = check_input;
    settings->before_send_ctx   = NULL;
    settings->on_after_receive  = check_output;
    settings->after_receive_ctx = NULL;

    // Test 1: Safe query (should succeed)
    printf("--- Test 1: Safe query ---\n");
    run_query(settings, "What is the capital of Japan?");

    // Test 2: Unsafe input (contains "password", should be blocked)
    printf("--- Test 2: Blocked input ---\n");
    run_query(settings, "What is my password for the admin account?");

    adam_settings_destroy(settings);
    free(api_key);
    adam_cleanup();
    return 0;
}
