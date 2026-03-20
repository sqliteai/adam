# Adam SQLite Extension

Embeds the Adam AI agent directly inside SQLite. The agent can query the same database it's loaded in — ask questions in natural language, get answers derived from your actual data.

## Quick Start

```sql
.load adam

-- Configure (persisted in _adam_config table)
SELECT adam_config('provider', 'anthropic');
SELECT adam_config('api_key', 'sk-ant-...');

-- Ask about your data (agent reads schema, writes SQL, executes it)
SELECT adam_ask('How many orders were placed this week?');
-- → "There were 142 orders this week, totaling $28,439."

-- Multi-turn (remembers context)
SELECT adam_ask('Break that down by product category');
-- → "Electronics: 52 orders ($14,200), Clothing: 48 orders ($8,100), ..."

-- Generate SQL without executing
SELECT adam_sql('top 5 customers by total revenue');
-- → "SELECT c.name, SUM(o.total) AS revenue FROM customers c ..."

-- Simple one-shot chat (no tools, no history)
SELECT adam('What is the capital of France?');
-- → "The capital of France is Paris."
```

## Build

```bash
# Requires libadam.a built first (from project root)
cd extensions/sqlite
make
```

Produces `adam.dylib` (macOS) or `adam.so` (Linux).

## SQL Functions

| Function | Description |
|----------|-------------|
| `adam_config(key, value)` | Set configuration (provider, api_key, model, identity, temperature, max_tokens) |
| `adam(message)` | Stateless one-shot chat — no tools, no history |
| `adam_ask(message)` | SQL-aware agent — reads schema, executes queries, multi-turn with session |
| `adam_sql(question)` | Generate SQL from natural language — returns query text without executing |
| `adam_create_session()` | Create a new session, returns UUID |
| `adam_get_session()` | Get current session UUID (NULL if none) |
| `adam_clear_session()` | Clear current session and history |

## How It Works

When you call `adam_ask()`:

1. **Auto-session**: If no session exists, one is created automatically
2. **Schema injection**: Reads `sqlite_master` to get all tables, columns, types. Injected into the system prompt.
3. **Tool registration**: A `sql_query` tool is registered that executes SQL on the **same database connection**
4. **Agent loop**: `adam_run()` calls the LLM → LLM generates SQL → tool executes on live data → result goes back to LLM → LLM synthesizes answer
5. **Session save**: History is saved to `_adam_messages` for multi-turn

## Configuration

Settings persist in the `_adam_config` table:

```sql
SELECT adam_config('provider', 'anthropic');   -- or 'openai', 'gemini'
SELECT adam_config('api_key', 'sk-ant-...');
SELECT adam_config('model', 'claude-sonnet-4-20250514');
SELECT adam_config('identity', 'You are a data analyst.');
SELECT adam_config('temperature', '0.3');
SELECT adam_config('max_tokens', '4096');
```

## Internal Tables

The extension creates these tables (prefixed with `_adam_` to avoid conflicts):

- `_adam_config` — key/value settings
- `_adam_sessions` — session IDs and timestamps
- `_adam_messages` — conversation history per session
