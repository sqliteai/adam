//
//  filesystem-sandbox/main.c
//  Tools restricted to allowed directories
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
    adam_settings_set_identity(settings, "You are a helpful file management assistant.");

    // Restrict filesystem access to /tmp only
    adam_settings_allow_dir(settings, "/tmp");

    // Register filesystem and utility tools (ctx = settings for sandbox checks)
    adam_settings_add_tool(settings, (adam_tool_def_t){
        .name            = "file_read",
        .description     = "Read contents of a file",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"File path to read\"}},\"required\":[\"path\"]}",
        .execute         = adam_tool_file_read,
        .ctx             = settings
    });

    adam_settings_add_tool(settings, (adam_tool_def_t){
        .name            = "file_write",
        .description     = "Write content to a file",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"File path to write\"},\"content\":{\"type\":\"string\",\"description\":\"Content to write\"}},\"required\":[\"path\",\"content\"]}",
        .execute         = adam_tool_file_write,
        .ctx             = settings
    });

    adam_settings_add_tool(settings, (adam_tool_def_t){
        .name            = "list_directory",
        .description     = "List contents of a directory",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"Directory path to list\"}},\"required\":[\"path\"]}",
        .execute         = adam_tool_list_directory,
        .ctx             = settings
    });

    adam_settings_add_tool(settings, (adam_tool_def_t){
        .name            = "shell_exec",
        .description     = "Execute a shell command",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\",\"description\":\"Shell command to run\"}},\"required\":[\"command\"]}",
        .execute         = adam_tool_shell_exec,
        .ctx             = settings
    });

    adam_settings_add_tool(settings, (adam_tool_def_t){
        .name            = "calculator",
        .description     = "Evaluate a math expression",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"expression\":{\"type\":\"string\",\"description\":\"Math expression to evaluate\"}},\"required\":[\"expression\"]}",
        .execute         = adam_tool_calculator,
        .ctx             = NULL
    });

    adam_history_t *history = adam_history_create();

    printf("User: Create a file called /tmp/adam_test.txt with the text 'Hello from Adam!' "
           "then list the /tmp directory to confirm it exists.\n\n");

    adam_run_result_t result = adam_run(settings, history,
        "Create a file called /tmp/adam_test.txt with the text 'Hello from Adam!' "
        "then list the /tmp directory to confirm it exists.");

    if (result.status != ADAM_OK) {
        fprintf(stderr, "Run failed: %s\n", adam_status_string(result.status));
    } else {
        printf("Assistant: %s\n\n", result.final_response);
        printf("  [%d iterations, %d in / %d out tokens, %.1f ms]\n",
               result.total_iterations, result.input_tokens,
               result.output_tokens, result.elapsed_ms);
    }

    adam_run_result_free(&result);
    adam_history_destroy(history);
    adam_settings_destroy(settings);
    free(api_key);
    adam_cleanup();
    return 0;
}
