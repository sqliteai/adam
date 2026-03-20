//
//  adam_ext_config.c
//  Config persistence in _adam_config table
//

#include "adam_ext.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int adam_ext_config_init(adam_ext_ctx_t *ctx) {
    return ctx->db.exec_stmt(ctx->db.db_ctx,
        "CREATE TABLE IF NOT EXISTS _adam_config "
        "(key TEXT PRIMARY KEY, value TEXT)");
}

int adam_ext_config_set(adam_ext_ctx_t *ctx,
                         const char *key, const char *value) {
    char sql[1024];
    // Escape single quotes in value
    char escaped_val[512];
    size_t j = 0;
    for (size_t i = 0; value[i] && j < sizeof(escaped_val) - 2; i++) {
        if (value[i] == '\'') escaped_val[j++] = '\'';
        escaped_val[j++] = value[i];
    }
    escaped_val[j] = '\0';

    // Delete + insert (portable across SQLite and PostgreSQL)
    char del_sql[512];
    snprintf(del_sql, sizeof(del_sql),
        "DELETE FROM _adam_config WHERE key='%s'", key);
    ctx->db.exec_stmt(ctx->db.db_ctx, del_sql);

    snprintf(sql, sizeof(sql),
        "INSERT INTO _adam_config(key,value) VALUES('%s','%s')",
        key, escaped_val);
    return ctx->db.exec_stmt(ctx->db.db_ctx, sql);
}

char *adam_ext_config_get(adam_ext_ctx_t *ctx, const char *key) {
    char sql[256];
    snprintf(sql, sizeof(sql),
        "SELECT value FROM _adam_config WHERE key='%s' LIMIT 1", key);

    adam_ext_result_t result = {0};
    if (ctx->db.exec_query(ctx->db.db_ctx, sql, &result) != 0)
        return NULL;

    char *val = NULL;
    if (result.n_rows > 0 && result.values[0])
        val = strdup(result.values[0]);

    adam_ext_result_free(&result);
    return val;
}

// Apply all config from _adam_config to the settings struct.
int adam_ext_config_apply(adam_ext_ctx_t *ctx) {
    if (!ctx->settings) {
        adam_init();
        ctx->settings = adam_create_settings();
        if (!ctx->settings) return -1;
    }

    char *provider = adam_ext_config_get(ctx, "provider");
    char *api_key  = adam_ext_config_get(ctx, "api_key");
    char *model    = adam_ext_config_get(ctx, "model");
    char *identity = adam_ext_config_get(ctx, "identity");
    char *temp_str = adam_ext_config_get(ctx, "temperature");
    char *maxtok   = adam_ext_config_get(ctx, "max_tokens");

    if (provider && api_key) {
        adam_api_format_t fmt = ADAM_API_NONE;
        if (strcmp(provider, "anthropic") == 0) fmt = ADAM_API_ANTHROPIC;
        else if (strcmp(provider, "openai") == 0) fmt = ADAM_API_OPENAI;
        else if (strcmp(provider, "gemini") == 0) fmt = ADAM_API_GEMINI;

        if (fmt != ADAM_API_NONE)
            adam_settings_set_provider(ctx->settings, fmt, api_key, model);
    }

    if (identity)
        adam_settings_set_identity(ctx->settings, identity);

    if (temp_str)
        ctx->settings->temperature = (float)atof(temp_str);

    if (maxtok)
        ctx->settings->max_tokens = atoi(maxtok);

    ctx->initialized = 1;

    // Don't free api_key/model/identity — settings stores the pointer
    free(provider);
    free(temp_str);
    free(maxtok);
    return 0;
}
