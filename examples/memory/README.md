# Long-Term Memory

Demonstrates Adam's hybrid BM25 + vector memory system. The program opens an in-memory database, adds several facts about a fictional project, then enables `inject_memory` so the agent's system prompt is automatically enriched with relevant memories. When the agent is asked a question it could not answer from its training data alone, it draws on the injected memory to provide an accurate response.

## Build & Run

```bash
# Create a .env file with your API keys
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
echo "OPENAI_API_KEY=sk-..." >> .env

make
./memory
```

## What It Demonstrates

- Opening and configuring a memory database (`adam_memory_open`, `adam_memory_configure`)
- Adding facts with context labels (`adam_memory_add`)
- Automatic memory injection via `inject_memory = 1`
- Configuring embedding provider with `adam_settings_set_memory`
