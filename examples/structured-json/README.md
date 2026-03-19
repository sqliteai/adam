# Structured JSON Output

Demonstrates getting validated JSON responses from the LLM using `adam_run_json`. The function appends a JSON structure hint to the prompt, validates that the response is parseable JSON, and automatically retries if the model produces invalid output.

## Build & Run

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

make
./structured-json
```

## What It Demonstrates

- Using `adam_run_json` for structured output with automatic validation
- Providing a JSON hint to guide the response format
- Configuring retry count for robustness
- Inspecting `json_valid` and `retries_used` from `adam_json_result_t`
