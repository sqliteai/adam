# Simple Conversation

Demonstrates multi-turn conversation with the Anthropic Claude API using Adam. The program asks a question, receives a response, then asks two follow-up questions that rely on conversational context. The shared `adam_history_t` object preserves context between turns so the model can resolve references like "its population" back to a previously mentioned city.

## Build & Run

```bash
# Create a .env file with your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

make
./simple-conversation
```

## What It Demonstrates

- Library initialization and cleanup (`adam_init` / `adam_cleanup`)
- Provider configuration with `adam_settings_set_provider`
- Multi-turn conversation using a shared `adam_history_t`
- Reading API keys from a `.env` file
- Inspecting token usage and latency from `adam_run_result_t`
