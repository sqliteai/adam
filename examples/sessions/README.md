# Session Persistence

Demonstrates saving and restoring conversations using Adam's session system. The program creates a session, chats for two turns with `auto_save` enabled so messages are persisted to a SQLite database after each call to `adam_run`. It then creates a fresh history, loads the saved session back, and prints all restored messages to verify round-trip integrity.

## Build & Run

```bash
# Create a .env file with your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

make
./sessions
```

## What It Demonstrates

- Creating sessions with `adam_session_create`
- Automatic saving via `settings->auto_save = 1` and `settings->session_id`
- Loading a saved session into a new history with `adam_session_load`
- Iterating over restored messages to inspect conversation contents
