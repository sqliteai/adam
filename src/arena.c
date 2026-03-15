//
//  arena.c
//  liteparser
//
//  Created by Marco Bambini on 05/11/25.
//

#include "arena.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#ifndef ARENA_DEFAULT_ALIGN
#define ARENA_DEFAULT_ALIGN (sizeof(max_align_t))
#endif

typedef struct arena_block_t {
    struct arena_block_t    *next;
    size_t                  used;           // bytes consumed in data[]
    size_t                  capacity;       // size of data[]
    unsigned char           data[];
} arena_block_t;

struct arena_t {
    arena_block_t           *head;          // first block
    arena_block_t           *current;       // append here
    size_t                  default_cap;    // default block capacity
};

static int __attribute__((unused)) is_pow2(size_t x) {
    return x && ((x & (x - 1)) == 0);
}

static size_t align_up(size_t x, size_t a) {
    assert(is_pow2(a));
    return (x + (a - 1)) & ~(a - 1);
}

// MARK: -

static arena_block_t *arena_block_new (size_t capacity) {
    // guard against overflow in malloc size computation
    if (capacity > SIZE_MAX - sizeof(arena_block_t)) return NULL;
    
    arena_block_t *b = (arena_block_t*)malloc(sizeof(arena_block_t) + capacity);
    if (!b) return NULL;
    
    b->next = NULL;
    b->used = 0;
    b->capacity = capacity;
    return b;
}

arena_t *arena_create (size_t block_size) {
    if (block_size < 1024) block_size = 1024; // tiny floor
    
    arena_t *a = (arena_t*)calloc(1, sizeof(arena_t));
    if (!a) return NULL;
    a->default_cap = block_size;

    // eagerly allocate first block
    a->head = arena_block_new(block_size);
    if (!a->head) {free(a); return NULL;}
    a->current = a->head;
    return a;
}

void arena_destroy (arena_t *a) {
    if (!a) return;
    
    arena_block_t *b = a->head;
    while (b) {
        arena_block_t *next = b->next;
        free(b);
        b = next;
    }
    free(a);
}

void arena_reset (arena_t *a) {
    if (!a) return;
    
    // keep the first block to avoid churn and free the rest
    arena_block_t *b = a->head ? a->head->next : NULL;
    while (b) {
        arena_block_t *next = b->next;
        free(b);
        b = next;
    }
    
    if (a->head) a->head->next = NULL;
    a->current = a->head;
    if (a->current) a->current->used = 0;
}

void arena_stats(const arena_t *a, size_t *used, size_t *capacity, size_t *blocks) {
    size_t u = 0, c = 0, n = 0;
    if (a) {
        const arena_block_t *b = a->head;
        while (b) { n++; u += b->used; c += b->capacity; b = b->next; }
    }
    if (used) *used = u;
    if (capacity) *capacity = c;
    if (blocks) *blocks = n;
}

void arena_debug (const arena_t *a) {
    if (!a) {printf("Arena is NULL\n"); return;}

    size_t block_count = 0;
    size_t total_capacity = 0;
    size_t total_used = 0;

    const arena_block_t *b = a->head;
    while (b) {
        block_count++;
        total_capacity += b->capacity;
        total_used += b->used;
        b = b->next;
    }

    size_t total_wasted = total_capacity - total_used;

    printf("\n========== ARENA DEBUG ==========\n");
    printf(" Default block size : %zu bytes\n", a->default_cap);
    printf(" Blocks             : %zu\n", block_count);
    printf(" Total capacity     : %zu bytes\n", total_capacity);
    printf(" Total used         : %zu bytes\n", total_used);
    printf(" Total wasted       : %zu bytes\n", total_wasted);
    printf("---------------------------------\n");

    // Print each block with offsets
    size_t idx = 0;
    b = a->head;
    while (b) {
        printf(" Block %zu:\n", idx++);
        printf("   ptr      = %p\n", (void*)b);
        printf("   data     = %p\n", (void*)b->data);
        printf("   capacity = %zu bytes\n", b->capacity);
        printf("   used     = %zu bytes\n", b->used);
        printf("   free     = %zu bytes\n", b->capacity - b->used);
        printf("\n");
        b = b->next;
    }

    printf("=================================\n\n");
}

// MARK: -

static void *arena_alloc_internal (arena_t *a, size_t size, size_t align) {
    assert(a);
    if (align == 0) align = ARENA_DEFAULT_ALIGN;
    assert(is_pow2(align));

    // fast path: try to carve from current block
    arena_block_t *cur = a->current;
    if (!cur) {
        // lazily re-create head if all was freed (shouldn't happen if create succeeded, but be robust)
        a->head = a->current = cur = arena_block_new(a->default_cap);
        if (!cur) return NULL;
    }

    size_t offset = align_up(cur->used, align);
    if (size <= cur->capacity && offset <= cur->capacity && size <= cur->capacity - offset) {
        void* p = cur->data + offset;
        cur->used = offset + size;
        return p;
    }

    // need a new block: pick capacity large enough for this request
    size_t need = size + (align - 1);               // account for potential padding
    size_t cap  = (need > a->default_cap) ? need : a->default_cap;

    arena_block_t *nb = arena_block_new(cap);
    if (!nb) return NULL;
    a->current->next = nb;
    a->current = nb;

    // With a fresh block, aligned offset is 0.
    size_t aligned0 = align_up(0, align);
    nb->used = aligned0 + size;
    return nb->data + aligned0;
}

void *arena_alloc_aligned (arena_t *a, size_t size, size_t align) {
    if (size == 0) return NULL;
    return arena_alloc_internal(a, size, align);
}

void *arena_alloc (arena_t *a, size_t size) {
    return arena_alloc_internal(a, size, ARENA_DEFAULT_ALIGN);
}

void *arena_zeroalloc (arena_t *a, size_t size) {
    if (size == 0) return NULL;
    void *p = arena_alloc(a, size);
    if (p) memset(p, 0, size);
    return p;
}

char* arena_strdup (arena_t *a, const char* s) {
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *p = (char*)arena_alloc(a, n);
    if (p) memcpy(p, s, n);
    return p;
}
