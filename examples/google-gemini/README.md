# Google Gemini

Demonstrates using Google's Gemini models through Adam's Gemini API format. The program configures `ADAM_API_GEMINI` with a Gemini API key and sends a simple question to `gemini-2.5-flash`.

## Build & Run

```bash
# Create a .env file with your Gemini API key
echo "GEMINI_API_KEY=AI..." > .env

make
./google-gemini
```

## What It Demonstrates

- Configuring the Gemini provider with `ADAM_API_GEMINI` format
- Using `adam_settings_set_provider` with a Gemini model name
- The same `adam_run` interface works across all providers
