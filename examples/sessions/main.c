//
//  sessions/main.c
//  Session persistence — save and restore conversations
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

    // Open memory database (sessions are stored here)
    adam_memory_t *mem = adam_memory_open("sessions_demo.db");
    if (!mem) {
        fprintf(stderr, "Failed to open memory database\n");
        free(api_key);
        adam_cleanup();
        return 1;
    }

    // Create a new session
    char session_id[64];
    status = adam_session_create(mem, session_id, sizeof(session_id));
    if (status != ADAM_OK) {
        fprintf(stderr, "Failed to create session: %s\n", adam_status_string(status));
        adam_memory_close(mem);
        free(api_key);
        adam_cleanup();
        return 1;
    }
    printf("Created session: %s\n\n", session_id);

    // Configure settings with auto_save enabled
    adam_settings_t *settings = adam_create_settings();
    adam_settings_set_provider(settings, ADAM_API_ANTHROPIC, api_key, "claude-sonnet-4-20250514");
    settings->memory = mem;
    settings->session_id = session_id;
    settings->auto_save = 1;

    // --- Phase 1: Have a 2-turn conversation ---
    adam_history_t *history = adam_history_create();

    printf("--- Phase 1: Chatting (auto-save on) ---\n\n");

    printf("User: My favorite color is cerulean blue.\n");
    adam_run_result_t r1 = adam_run(settings, history, "My favorite color is cerulean blue.");
    if (r1.status == ADAM_OK) {
        printf("Assistant: %s\n\n", r1.final_response);
    } else {
        fprintf(stderr, "Turn 1 failed: %s\n", adam_status_string(r1.status));
    }
    adam_run_result_free(&r1);

    printf("User: And I have a cat named Pixel.\n");
    adam_run_result_t r2 = adam_run(settings, history, "And I have a cat named Pixel.");
    if (r2.status == ADAM_OK) {
        printf("Assistant: %s\n\n", r2.final_response);
    } else {
        fprintf(stderr, "Turn 2 failed: %s\n", adam_status_string(r2.status));
    }
    adam_run_result_free(&r2);

    printf("Session saved with %zu messages.\n\n", adam_history_count(history));
    adam_history_destroy(history);

    // --- Phase 2: Load the session into a fresh history ---
    printf("--- Phase 2: Restoring session ---\n\n");

    adam_history_t *restored = adam_history_create();
    status = adam_session_load(mem, session_id, restored);
    if (status != ADAM_OK) {
        fprintf(stderr, "Failed to load session: %s\n", adam_status_string(status));
        adam_history_destroy(restored);
        adam_memory_close(mem);
        adam_settings_destroy(settings);
        free(api_key);
        adam_cleanup();
        return 1;
    }

    printf("Restored %zu messages from session %s:\n\n", adam_history_count(restored), session_id);
    for (size_t i = 0; i < adam_history_count(restored); i++) {
        const char *role_str;
        switch (restored->items[i].role) {
            case ADAM_ROLE_USER:      role_str = "User";      break;
            case ADAM_ROLE_ASSISTANT: role_str = "Assistant";  break;
            case ADAM_ROLE_SYSTEM:    role_str = "System";     break;
            case ADAM_ROLE_TOOL:      role_str = "Tool";       break;
            default:                 role_str = "Unknown";    break;
        }
        printf("  [%zu] %s: %.80s%s\n", i, role_str,
               restored->items[i].content,
               strlen(restored->items[i].content) > 80 ? "..." : "");
    }

    printf("\nSession persistence verified.\n");

    // Cleanup
    adam_history_destroy(restored);
    adam_memory_close(mem);
    adam_settings_destroy(settings);
    free(api_key);
    adam_cleanup();
    return 0;
}
