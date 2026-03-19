# Research Mode

Demonstrates Adam's autonomous research loop. The agent is given a research question and iterates through multiple rounds of information gathering using the `web_fetch` tool. After each iteration, a progress callback reports status. When the agent determines it has gathered enough information (or hits the iteration limit), it synthesizes a final report from all accumulated findings.

## Build & Run

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
make
./research
```

## What It Demonstrates

- Configuring `adam_research_config_t` with a question and iteration limit
- Adding the built-in `web_fetch` tool for information gathering
- Using a progress callback to monitor research iterations
- Running `adam_research()` and inspecting the report, findings, and stats
