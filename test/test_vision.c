//
//  test_vision.c
//  Adam — Local multimodal vision test
//
//  Tests local GGUF model + mmproj for image understanding.
//
//  Build & run:
//    make vision GGUF=models/model.gguf MMPROJ=models/mmproj.gguf
//    ./test_vision models/model.gguf models/mmproj.gguf [image_path]
//

#include "adam.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void on_stream(void *ctx, const char *chunk, size_t len, int is_done) {
    UNUSED_PARAM(ctx);
    if (len > 0) fwrite(chunk, 1, len, stdout);
    if (is_done) printf("\n");
    fflush(stdout);
}

static uint8_t *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc((size_t)len);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)len, f);
    fclose(f);
    if ((long)rd != len) { free(buf); return NULL; }
    *out_len = (size_t)len;
    return buf;
}

static adam_media_type_t guess_media_type(const char *path) {
    const char *ext = strrchr(path, '.');
    if (!ext) return ADAM_MEDIA_IMAGE_JPEG;
    if (strcasecmp(ext, ".png") == 0) return ADAM_MEDIA_IMAGE_PNG;
    if (strcasecmp(ext, ".gif") == 0) return ADAM_MEDIA_IMAGE_GIF;
    if (strcasecmp(ext, ".webp") == 0) return ADAM_MEDIA_IMAGE_WEBP;
    return ADAM_MEDIA_IMAGE_JPEG;
}

static int run_vision_test(adam_settings_t *s, const char *image_path,
                            const char *prompt) {
    printf("--- Image: %s ---\n", image_path);
    printf("Prompt: %s\n", prompt);
    printf("Response: ");
    fflush(stdout);

    // Load image
    size_t img_len = 0;
    uint8_t *img_data = read_file(image_path, &img_len);
    if (!img_data) {
        fprintf(stderr, "Failed to read image: %s\n", image_path);
        return 1;
    }

    adam_history_t *h = adam_history_create();

    // Add user message with image attachment
    adam_history_append_user(h, prompt);
    adam_history_attach(h, guess_media_type(image_path),
                        img_data, img_len, image_path);

    adam_run_result_t r = adam_run(s, h, NULL);

    if (r.status != ADAM_OK) {
        printf("(error: %s)\n",
               r.final_response ? r.final_response
                                : adam_status_string(r.status));
    } else if (!s->on_stream && r.final_response) {
        printf("%s\n", r.final_response);
    }

    printf("[%d in / %d out | %.0fms]\n\n",
           r.input_tokens, r.output_tokens, r.elapsed_ms);

    adam_run_result_free(&r);
    adam_history_destroy(h);
    free(img_data);
    return (r.status != ADAM_OK) ? 1 : 0;
}

int main(int argc, char **argv) {
    // Parse args: test_vision <gguf> <mmproj> [image_path]
    const char *gguf_path = NULL;
    const char *mmproj_path = NULL;
    const char *custom_image = NULL;

    for (int i = 1; i < argc; i++) {
        if (strstr(argv[i], "mmproj") && strstr(argv[i], ".gguf")) {
            mmproj_path = argv[i];
        } else if (strstr(argv[i], ".gguf")) {
            gguf_path = argv[i];
        } else {
            custom_image = argv[i];
        }
    }

    if (!gguf_path || !mmproj_path) {
        fprintf(stderr, "Usage: %s <model.gguf> <mmproj.gguf> [image_path]\n",
                argv[0]);
        fprintf(stderr, "  make vision GGUF=models/model.gguf MMPROJ=models/mmproj.gguf\n");
        return 1;
    }

    adam_init();

    adam_settings_t *s = adam_create_settings();
    s->gguf_path = gguf_path;
    s->local_gpu_layers = -1;
    s->local_ctx_size = 8192;
    s->max_tokens = 512;
    s->temperature = 0.3f;
    s->on_stream = on_stream;

    adam_settings_set_mmproj(s, mmproj_path);

    printf("=== Adam Local Vision Test ===\n");
    printf("  Model:  %s\n", gguf_path);
    printf("  MMProj: %s\n", mmproj_path);
    printf("==============================\n\n");

    int failures = 0;

    if (custom_image) {
        // Single image mode
        failures += run_vision_test(s, custom_image,
            "Describe this image in detail.");
    } else {
        // Run tests on all images in test/images/
        const char *test_images[] = {
            "test/images/landscape.jpg",
            "test/images/portrait.jpg",
            "test/images/square.png",
        };
        const char *prompts[] = {
            "Describe this image in detail. What do you see?",
            "What is the main subject of this image?",
            "List the colors you can see in this image.",
        };
        size_t n_tests = sizeof(test_images) / sizeof(test_images[0]);

        for (size_t i = 0; i < n_tests; i++) {
            failures += run_vision_test(s, test_images[i], prompts[i]);
        }

        printf("=== Results: %zu/%zu passed ===\n",
               n_tests - (size_t)failures, n_tests);
    }

    adam_settings_destroy(s);
    adam_cleanup();
    return failures > 0 ? 1 : 0;
}
