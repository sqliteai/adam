//
//  adam_cache.c
//  Adam — LRU response cache
//
//  Created by Marco Bambini on 17/03/26.
//

#include "adam.h"
#include <stdlib.h>
#include <string.h>

// ============================================================================
// MARK: - Internal types
// ============================================================================

typedef struct adam_cache_entry_t {
    uint64_t     hash;
    char        *response;          // malloc'd
    int          input_tokens;
    int          output_tokens;
    struct adam_cache_entry_t *lru_prev;
    struct adam_cache_entry_t *lru_next;
    struct adam_cache_entry_t *bucket_next; // hash collision chain
} adam_cache_entry_t;

struct adam_cache_t {
    adam_cache_entry_t **buckets;
    size_t               bucket_count;
    adam_cache_entry_t  *lru_head;   // most recently used
    adam_cache_entry_t  *lru_tail;   // least recently used
    size_t               count;
    size_t               max_entries;
    size_t               hits;
    size_t               misses;
};

// ============================================================================
// MARK: - FNV-1a hash
// ============================================================================

static uint64_t fnv1a(const void *data, size_t len, uint64_t h) {
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

uint64_t adam_cache_hash(const char *model,
                          const adam_message_t *msgs, size_t msg_count) {
    uint64_t h = 0xcbf29ce484222325ULL; // FNV offset basis

    if (model) h = fnv1a(model, strlen(model), h);

    for (size_t i = 0; i < msg_count; i++) {
        int role = (int)msgs[i].role;
        h = fnv1a(&role, sizeof(role), h);
        if (msgs[i].content)
            h = fnv1a(msgs[i].content, msgs[i].content_len, h);
    }
    return h;
}

// ============================================================================
// MARK: - LRU helpers
// ============================================================================

static void lru_remove(adam_cache_t *c, adam_cache_entry_t *e) {
    if (e->lru_prev) e->lru_prev->lru_next = e->lru_next;
    else c->lru_head = e->lru_next;
    if (e->lru_next) e->lru_next->lru_prev = e->lru_prev;
    else c->lru_tail = e->lru_prev;
    e->lru_prev = e->lru_next = NULL;
}

static void lru_push_front(adam_cache_t *c, adam_cache_entry_t *e) {
    e->lru_prev = NULL;
    e->lru_next = c->lru_head;
    if (c->lru_head) c->lru_head->lru_prev = e;
    c->lru_head = e;
    if (!c->lru_tail) c->lru_tail = e;
}

// ============================================================================
// MARK: - Create / Destroy / Clear
// ============================================================================

adam_cache_t *adam_cache_create(size_t max_entries) {
    if (max_entries == 0) max_entries = 256;

    adam_cache_t *c = calloc(1, sizeof(adam_cache_t));
    if (!c) return NULL;

    c->max_entries = max_entries;
    c->bucket_count = max_entries * 2; // load factor ~0.5
    if (c->bucket_count < 16) c->bucket_count = 16;

    c->buckets = calloc(c->bucket_count, sizeof(adam_cache_entry_t *));
    if (!c->buckets) { free(c); return NULL; }

    return c;
}

static void entry_free(adam_cache_entry_t *e) {
    if (!e) return;
    free(e->response);
    free(e);
}

void adam_cache_clear(adam_cache_t *c) {
    if (!c) return;
    adam_cache_entry_t *e = c->lru_head;
    while (e) {
        adam_cache_entry_t *next = e->lru_next;
        entry_free(e);
        e = next;
    }
    memset(c->buckets, 0, c->bucket_count * sizeof(adam_cache_entry_t *));
    c->lru_head = c->lru_tail = NULL;
    c->count = 0;
}

void adam_cache_destroy(adam_cache_t *c) {
    if (!c) return;
    adam_cache_clear(c);
    free(c->buckets);
    free(c);
}

// ============================================================================
// MARK: - Lookup
// ============================================================================

const char *adam_cache_lookup(adam_cache_t *c, uint64_t hash,
                               int *out_input, int *out_output) {
    if (!c) return NULL;

    size_t idx = (size_t)(hash % c->bucket_count);
    adam_cache_entry_t *e = c->buckets[idx];
    while (e) {
        if (e->hash == hash) {
            // Move to front of LRU
            lru_remove(c, e);
            lru_push_front(c, e);
            c->hits++;
            if (out_input) *out_input = e->input_tokens;
            if (out_output) *out_output = e->output_tokens;
            return e->response;
        }
        e = e->bucket_next;
    }
    c->misses++;
    return NULL;
}

// ============================================================================
// MARK: - Store
// ============================================================================

static void evict_lru(adam_cache_t *c) {
    adam_cache_entry_t *victim = c->lru_tail;
    if (!victim) return;

    // Remove from LRU
    lru_remove(c, victim);

    // Remove from bucket chain
    size_t idx = (size_t)(victim->hash % c->bucket_count);
    adam_cache_entry_t **pp = &c->buckets[idx];
    while (*pp) {
        if (*pp == victim) { *pp = victim->bucket_next; break; }
        pp = &(*pp)->bucket_next;
    }

    entry_free(victim);
    c->count--;
}

void adam_cache_store(adam_cache_t *c, uint64_t hash,
                       const char *response, int input_tok, int output_tok) {
    if (!c || !response) return;

    // Check if already cached (update)
    size_t idx = (size_t)(hash % c->bucket_count);
    adam_cache_entry_t *e = c->buckets[idx];
    while (e) {
        if (e->hash == hash) {
            // Update existing
            free(e->response);
            e->response = strdup(response);
            e->input_tokens = input_tok;
            e->output_tokens = output_tok;
            lru_remove(c, e);
            lru_push_front(c, e);
            return;
        }
        e = e->bucket_next;
    }

    // Evict if full
    while (c->count >= c->max_entries)
        evict_lru(c);

    // Create new entry
    adam_cache_entry_t *ne = calloc(1, sizeof(adam_cache_entry_t));
    if (!ne) return;
    ne->hash = hash;
    ne->response = strdup(response);
    ne->input_tokens = input_tok;
    ne->output_tokens = output_tok;
    if (!ne->response) { free(ne); return; }

    // Insert into bucket
    ne->bucket_next = c->buckets[idx];
    c->buckets[idx] = ne;

    // Insert at front of LRU
    lru_push_front(c, ne);
    c->count++;
}

// ============================================================================
// MARK: - Stats
// ============================================================================

size_t adam_cache_count(const adam_cache_t *c) { return c ? c->count : 0; }
size_t adam_cache_hits(const adam_cache_t *c) { return c ? c->hits : 0; }
size_t adam_cache_misses(const adam_cache_t *c) { return c ? c->misses : 0; }
