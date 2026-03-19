//
//  research/main.c
//  Autonomous research mode with web_fetch tool
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

static void on_progress(void *ctx, int iteration, const char *status) {
    (void)ctx;
    printf("  [iteration %d] %s\n", iteration, status ? status : "(no status)");
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

    // Configure settings with provider and web_fetch tool
    adam_settings_t *settings = adam_create_settings();
    adam_settings_set_provider(settings, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");

    adam_settings_add_tool(settings, (adam_tool_def_t){
        .name            = "web_search",
        .description     = "Search the web for information",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\",\"description\":\"Search query\"},\"count\":{\"type\":\"integer\",\"description\":\"Number of results\"}},\"required\":[\"query\"]}",
        .execute         = adam_tool_web_search,
        .ctx             = settings
    });

    // Configure the research loop
    adam_research_config_t config = adam_research_config_defaults();
    config.question       = "What are the main differences between REST and GraphQL APIs? "
                            "Compare their strengths, weaknesses, and typical use cases.";
    config.max_iterations = 3;
    config.on_progress    = on_progress;
    config.progress_ctx   = NULL;

    printf("Starting research: %s\n\n", config.question);

    // Run the research loop
    adam_research_result_t result = adam_research(settings, &config);

    if (result.status != ADAM_OK) {
        fprintf(stderr, "Research failed: %s\n", adam_status_string(result.status));
    } else {
        printf("\n--- Research Report ---\n%s\n", result.report ? result.report : "(no report)");
        printf("\nFindings: %zu\n", result.finding_count);
        for (size_t i = 0; i < result.finding_count; i++) {
            printf("  [%zu] %s\n", i + 1, result.findings[i].content);
            if (result.findings[i].source)
                printf("       source: %s\n", result.findings[i].source);
        }
        printf("\nStats: %d iterations, %d in / %d out tokens, %.1f ms\n",
               result.total_iterations, result.total_input_tokens,
               result.total_output_tokens, result.elapsed_ms);
    }

    adam_research_result_free(&result);
    adam_settings_destroy(settings);
    free(api_key);
    adam_cleanup();
    return 0;
}
