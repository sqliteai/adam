# Multi-Agent

Demonstrates agent-as-tool orchestration where a main agent delegates work to a specialist sub-agent. The program creates a researcher agent with its own identity and settings, then registers it as a tool on the main orchestrator agent using `adam_tool_agent`. When the orchestrator receives a question requiring research, it invokes the researcher tool, receives the findings, and synthesizes them into a final answer.

## Build & Run

```bash
# Create a .env file with your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

make
./multi-agent
```

## What It Demonstrates

- Creating separate settings for main and sub-agents
- Registering a sub-agent as a tool with `adam_tool_agent` and `adam_agent_tool_ctx_t`
- Automatic delegation: the orchestrator decides when to call the researcher
- Multi-step tool loop: the orchestrator calls the tool, gets results, then responds
