# Adam PostgreSQL Extension

Embeds the Adam AI agent inside PostgreSQL. The agent can query the same database — ask questions in natural language, get answers derived from your data.

## Quick Start

```sql
CREATE EXTENSION adam;

-- Configure
SELECT adam_config('provider', 'anthropic');
SELECT adam_config('api_key', 'sk-ant-...');

-- Ask about your data
SELECT adam_ask('How many orders were placed this week?');
-- → "There were 142 orders this week, totaling $28,439."

-- Generate SQL
SELECT adam_sql('top 5 customers by revenue');
-- → "SELECT c.name, SUM(o.total) AS revenue FROM ..."

-- Simple chat (no tools)
SELECT adam('What is the capital of France?');
```

## Build & Install

```bash
# Requires libadam.a built first (from project root)
cd extensions/postgres
make
sudo make install
```

Then in psql:

```sql
CREATE EXTENSION adam;
```

## SQL Functions

| Function | Description |
|----------|-------------|
| `adam_config(key, value)` | Set configuration (provider, api_key, model, identity, temperature) |
| `adam(message)` | Stateless one-shot chat — no tools, no history |
| `adam_ask(message)` | SQL-aware agent — reads schema, executes queries, multi-turn |
| `adam_sql(question)` | Generate SQL from natural language (returns query text) |
| `adam_create_session()` | Create a new session, returns UUID |
| `adam_get_session()` | Get current session UUID (NULL if none) |
| `adam_clear_session()` | Clear current session and history |

## How It Works

The extension uses PostgreSQL's SPI (Server Programming Interface) to execute queries within the same transaction context. When `adam_ask()` is called:

1. Reads schema from `information_schema.columns` and `pg_stat_user_tables`
2. Injects schema into the LLM system prompt
3. Registers a `sql_query` tool that calls `SPI_execute()` on the same connection
4. The agent loop generates SQL, executes it via SPI, and synthesizes an answer

## Internal Tables

Created in the `public` schema:

- `_adam_config` — key/value settings (persistent)
- `_adam_sessions` — session IDs and timestamps
- `_adam_messages` — conversation history per session

## Requirements

- PostgreSQL 14+ (uses SPI)
- `pg_config` in PATH
- `libadam.a` built from the project root
