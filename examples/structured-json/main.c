//
//  structured-json/main.c
//  Validated JSON responses with automatic retries
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

    adam_history_t *history = adam_history_create();

    // Define the expected JSON structure as a hint
    const char *json_hint =
        "Respond with a JSON array of exactly 3 objects, each with "
        "\"name\" (string), \"year\" (integer, year created), and "
        "\"paradigm\" (string). Example: "
        "[{\"name\":\"C\",\"year\":1972,\"paradigm\":\"procedural\"}]";

    printf("Asking for 3 programming languages as structured JSON...\n\n");

    // adam_run_json validates the response is valid JSON and retries up to 3 times if not
    adam_json_result_t result = adam_run_json(settings, history,
        "List 3 influential programming languages with their creation year and primary paradigm.",
        json_hint,
        3);  // max_retries

    if (result.base.status != ADAM_OK) {
        fprintf(stderr, "adam_run_json failed: %s\n", adam_status_string(result.base.status));
    } else {
        printf("JSON valid: %s\n", result.json_valid ? "yes" : "no");
        printf("Retries used: %d\n\n", result.retries_used);
        printf("Response:\n%s\n\n", result.base.final_response);
        printf("  [tokens: %d in / %d out, %.1f ms]\n",
               result.base.input_tokens, result.base.output_tokens,
               result.base.elapsed_ms);
    }

    adam_json_result_free(&result);
    adam_history_destroy(history);
    adam_settings_destroy(settings);
    free(api_key);
    adam_cleanup();
    return 0;
}
