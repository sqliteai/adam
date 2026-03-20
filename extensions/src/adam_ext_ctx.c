//
//  adam_ext_ctx.c
//  Extension context lifecycle + result helpers
//

#include "adam_ext.h"
#include <stdlib.h>
#include <string.h>

adam_ext_ctx_t *adam_ext_ctx_create(adam_ext_db_t db) {
    adam_ext_ctx_t *ctx = calloc(1, sizeof(adam_ext_ctx_t));
    if (!ctx) return NULL;
    ctx->db = db;
    return ctx;
}

void adam_ext_ctx_destroy(adam_ext_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->session_id) {
        adam_ext_session_save(ctx);
        free(ctx->session_id);
    }
    if (ctx->history) adam_history_destroy(ctx->history);
    if (ctx->settings) adam_settings_destroy(ctx->settings);
    free(ctx->schema_cache);
    free(ctx);
}

void adam_ext_result_free(adam_ext_result_t *r) {
    if (!r) return;
    if (r->values) {
        for (int i = 0; i < r->n_rows * r->n_cols; i++)
            free(r->values[i]);
        free(r->values);
    }
    if (r->col_names) {
        for (int i = 0; i < r->n_cols; i++)
            free(r->col_names[i]);
        free(r->col_names);
    }
    r->values = NULL;
    r->col_names = NULL;
    r->n_rows = r->n_cols = 0;
}
