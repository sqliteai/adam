//
//  local-vision/main.c
//  Multimodal image understanding with a local vision model
//

#include "adam.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Read entire file into a malloc'd buffer. Caller must free.
static uint8_t *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz <= 0) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) { free(buf); return NULL; }
    *out_len = (size_t)sz;
    return buf;
}

// Guess media type from file extension
static adam_media_type_t guess_media_type(const char *path) {
    const char *dot = strrchr(path, '.');
    if (!dot) return ADAM_MEDIA_IMAGE_JPEG;
    if (strcmp(dot, ".png") == 0 || strcmp(dot, ".PNG") == 0) return ADAM_MEDIA_IMAGE_PNG;
    if (strcmp(dot, ".gif") == 0 || strcmp(dot, ".GIF") == 0) return ADAM_MEDIA_IMAGE_GIF;
    if (strcmp(dot, ".webp") == 0 || strcmp(dot, ".WEBP") == 0) return ADAM_MEDIA_IMAGE_WEBP;
    return ADAM_MEDIA_IMAGE_JPEG;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <mmproj.gguf> [image.jpg]\n", argv[0]);
        return 1;
    }

    const char *gguf_path   = argv[1];
    const char *mmproj_path = argv[2];
    const char *image_path  = (argc >= 4) ? argv[3] : "image.jpg";

    // Load image file
    size_t image_len = 0;
    uint8_t *image_data = read_file(image_path, &image_len);
    if (!image_data) {
        fprintf(stderr, "Failed to read image: %s\n", image_path);
        return 1;
    }
    printf("Loaded image: %s (%zu bytes)\n\n", image_path, image_len);

    adam_status_t status = adam_init();
    if (status != ADAM_OK) {
        fprintf(stderr, "adam_init failed: %s\n", adam_status_string(status));
        free(image_data);
        return 1;
    }

    adam_settings_t *settings = adam_create_settings();

    // Configure local model with vision projector
    status = adam_settings_set_local(settings, gguf_path, -1, 4096);
    if (status != ADAM_OK) {
        fprintf(stderr, "Failed to set local model: %s\n", adam_status_string(status));
        goto cleanup;
    }

    status = adam_settings_set_mmproj(settings, mmproj_path);
    if (status != ADAM_OK) {
        fprintf(stderr, "Failed to set mmproj: %s\n", adam_status_string(status));
        goto cleanup;
    }

    settings->max_tokens = 512;

    adam_history_t *history = adam_history_create();

    // Add user message and attach the image
    adam_history_append_user(history, "Describe this image in detail. What do you see?");
    adam_history_attach(history, guess_media_type(image_path),
                        image_data, image_len, image_path);

    printf("Prompt: Describe this image in detail. What do you see?\n\n");

    // Run with NULL user_message since we already appended it to history
    adam_run_result_t result = adam_run(settings, history, NULL);

    if (result.status != ADAM_OK) {
        fprintf(stderr, "adam_run failed: %s\n", adam_status_string(result.status));
    } else {
        printf("Response:\n%s\n\n", result.final_response);
        printf("  [tokens: %d in / %d out, %.1f ms]\n",
               result.input_tokens, result.output_tokens, result.elapsed_ms);
    }

    adam_run_result_free(&result);
    adam_history_destroy(history);

cleanup:
    adam_settings_destroy(settings);
    free(image_data);
    adam_cleanup();
    return 0;
}
