//
//  multi-agent/main.c
//  Multi-agent orchestration — sub-agent as a tool
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

    // --- Create the researcher sub-agent ---
    adam_settings_t *researcher_settings = adam_create_settings();
    adam_settings_set_provider(researcher_settings, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");
    adam_settings_set_identity(researcher_settings,
        "You are a research specialist. When asked a question, provide a detailed, "
        "factual answer with key data points. Be thorough but concise.");

    // The sub-agent context: settings + optional persistent history
    adam_agent_tool_ctx_t researcher_ctx = {
        .settings = researcher_settings,
        .history  = NULL    // fresh history each invocation
    };

    // --- Create the main orchestrator agent ---
    adam_settings_t *main_settings = adam_create_settings();
    adam_settings_set_provider(main_settings, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");
    adam_settings_set_identity(main_settings,
        "You are a project manager. You have a researcher tool that you can delegate "
        "research questions to. Use it to gather information, then synthesize the "
        "findings into a clear summary for the user.");

    // Register the researcher sub-agent as a tool on the main agent
    adam_tool_def_t researcher_tool = {
        .name            = "researcher",
        .description     = "Delegate a research question to a specialist agent. "
                           "Returns a detailed research report.",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"message\":{\"type\":\"string\","
                           "\"description\":\"The research question to investigate\"}},"
                           "\"required\":[\"message\"]}",
        .execute         = adam_tool_agent,
        .ctx             = &researcher_ctx
    };
    adam_settings_add_tool(main_settings, researcher_tool);

    // --- Run the orchestrator ---
    adam_history_t *history = adam_history_create();

    printf("User: Compare the energy density of lithium-ion vs solid-state batteries\n");
    printf("      and explain which is more promising for electric vehicles.\n\n");

    adam_run_result_t result = adam_run(main_settings, history,
        "Compare the energy density of lithium-ion vs solid-state batteries "
        "and explain which is more promising for electric vehicles.");

    if (result.status == ADAM_OK) {
        printf("Orchestrator: %s\n\n", result.final_response);
        printf("  [iterations: %d, tokens: %d in / %d out, %.1f ms]\n",
               result.total_iterations, result.input_tokens,
               result.output_tokens, result.elapsed_ms);
    } else {
        fprintf(stderr, "Run failed: %s\n", adam_status_string(result.status));
    }

    adam_run_result_free(&result);
    adam_history_destroy(history);
    adam_settings_destroy(main_settings);
    adam_settings_destroy(researcher_settings);
    free(api_key);
    adam_cleanup();
    return 0;
}
