# Response Cache

Demonstrates Adam's LRU response cache for avoiding redundant LLM calls. A cache is created with `adam_cache_create` and assigned to the settings. The same query is run twice: the first call is a cache miss and hits the LLM, while the second call is a cache hit and returns instantly. After each run, cache statistics (hits, misses, entry count) are printed to show the caching in action.

## Build & Run

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
make
./response-cache
```

## What It Demonstrates

- Creating and configuring `adam_cache_t` with `adam_cache_create`
- Assigning the cache to `settings->cache`
- Observing cache miss on first query and cache hit on second
- Querying cache stats with `adam_cache_hits`, `adam_cache_misses`, `adam_cache_count`
