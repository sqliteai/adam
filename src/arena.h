//
//  arena.h
//  liteparser
//
//  Created by Marco Bambini on 05/11/25.
//

#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct arena_t arena_t;

arena_t *arena_create(size_t block_size);       // e.g. 64 * 1024
void    arena_destroy(arena_t *arena);          // frees all blocks & the arena
void    arena_reset(arena_t *arena);            // keep first block, drop the rest
void    arena_debug(const arena_t *arena);      // print arena stats on stdout

void    arena_stats(const arena_t *arena, size_t *used, size_t *capacity, size_t *blocks);

void    *arena_alloc(arena_t *arena, size_t size);      // default align = alignof(max_align_t)
void    *arena_zeroalloc(arena_t *arena, size_t size);  // zeroed
char    *arena_strdup(arena_t *arena, const char* s);   // arena-owned copy

#ifdef __cplusplus
}
#endif
