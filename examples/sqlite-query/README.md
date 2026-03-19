# sqlite-query

Ask natural language questions about any SQLite database.

This example opens a SQLite database, reads its full schema (tables, views, triggers, and indexes with row counts), injects the schema as context for the LLM, and registers the `sql_query` tool. You can then ask questions about your data in plain English and the agent will write and execute SQL queries to answer them.

## Build

```bash
# From the repository root, build the library first:
make deps && make all

# Then build this example:
cd examples/sqlite-query
make
```

## Usage

```bash
# Place your API key in a .env file (at the repo root or working directory):
# ANTHROPIC_API_KEY=sk-...
# or GEMINI_API_KEY=...
# or OPENAI_API_KEY=sk-...

./sqlite-query path/to/your/database.db
```

The program prints the detected schema and enters an interactive chat loop. Type natural language questions and the agent will query the database using the `sql_query` tool. Type `quit` or `exit` to stop.

## Example session

```
SQLite Query Agent
Database: chinook.db

Schema:
-- tables:
CREATE TABLE albums (AlbumId INTEGER PRIMARY KEY, Title TEXT, ArtistId INTEGER);
CREATE TABLE artists (ArtistId INTEGER PRIMARY KEY, Name TEXT);
...

-- Row counts:
--   albums: 347 rows
--   artists: 275 rows

Ask questions about your data in natural language. Type 'quit' to exit.

You: How many albums does each of the top 5 artists have?