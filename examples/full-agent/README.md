# Full Agent

Demonstrates all major Adam features combined in a single agent. The agent is configured with a remote provider, custom identity and instructions, a memory database with session persistence, an LRU response cache, input/output guardrails, a filesystem sandbox restricted to `/tmp`, streaming output, logging, and six registered tools: `file_read`, `file_write`, `shell_exec`, `calculator`, `memory_search`, and `sql_query`. A single query exercises file I/O and calculation, then the program prints stats including session ID, cache performance, and token usage.

## Build & Run

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
make
./full-agent
```

## What It Demonstrates

- Provider, identity, and instructions configuration
- Memory database with `adam_memory_open` and session auto-save
- Response caching with `adam_cache_create`
- Guardrails with `on_before_send` and `on_after_receive`
- Filesystem sandbox with `adam_settings_allow_dir`
- Streaming output with `adam_settings_set_stream`
- Logging with `adam_settings_set_logger`
- Registering multiple tools (file, shell, calculator, memory, SQL)
