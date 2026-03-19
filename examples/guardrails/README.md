# Guardrails

Demonstrates input and output validation using Adam's guardrail callbacks. An input guardrail checks user messages before they are sent to the LLM and blocks any message containing "password". An output guardrail inspects the LLM response and blocks any reply containing "SSN". When a guardrail fires, `adam_run` returns `ADAM_ERR_GUARDRAIL` instead of a response. The example runs a safe query that succeeds and an unsafe query that gets blocked.

## Build & Run

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
make
./guardrails
```

## What It Demonstrates

- Implementing `adam_guardrail_fn` (input validation)
- Implementing `adam_guardrail_response_fn` (output validation)
- Setting guardrails on `adam_settings_t` via `on_before_send` and `on_after_receive`
- Handling `ADAM_ERR_GUARDRAIL` status from `adam_run`
