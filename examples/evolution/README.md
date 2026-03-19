# Evolution Loop

Demonstrates Adam's self-improving agent via the evolution loop. The program defines an evaluation function that scores LLM output based on length, keyword presence, and structural formatting. It configures `adam_evolve` with a writing task, an initial strategy, and scoring metrics. The loop iterates, refining the agent's approach until the target score is reached or a plateau is detected, then prints the best output and the final evolved strategy.

## Build & Run

```bash
# Create a .env file with your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

make
./evolution
```

## What It Demonstrates

- Defining a custom evaluation function (`adam_evolve_eval_fn`)
- Configuring evolution parameters: task, strategy, metrics, stopping conditions
- Using `adam_evolve_config_defaults` and overriding specific fields
- Progress tracking via `adam_evolve_progress_fn`
- Inspecting results: best output, final strategy, stop reason, token usage
