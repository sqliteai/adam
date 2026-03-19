# Tool Calling

Demonstrates how to register a custom tool and have the agent use it automatically. A `get_weather` tool is defined with a JSON Schema for its parameters and a callback that returns mock weather data. When the user asks about weather, the LLM decides to call the tool, receives the result, and formulates a natural-language answer.

## Build & Run

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

make
./tool-calling
```

## What It Demonstrates

- Defining a tool with `adam_tool_def_t` (name, description, JSON Schema, callback)
- Registering tools with `adam_settings_add_tool`
- The agent loop automatically calling the tool and incorporating results
- Arena-based memory for tool results
