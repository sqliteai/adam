# Voice Agent

Demonstrates configuring Adam's STT (speech-to-text) and TTS (text-to-speech) pipeline. The program sets up cloud-based Whisper for STT and the system's built-in speech synthesizer for TTS, then uses `adam_tts_speak` to speak a greeting aloud. This is a configuration skeleton showing the API setup; a full voice agent would call `adam_voice_start` to run the continuous listen-transcribe-respond loop.

## Build & Run

```bash
# Create a .env file with your API keys
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
echo "OPENAI_API_KEY=sk-..." >> .env

make
./voice
```

## What It Demonstrates

- Configuring STT backend with `adam_settings_set_stt`
- Configuring TTS backend with `adam_settings_set_tts` (system TTS)
- One-shot speech synthesis with `adam_tts_speak`
- The voice thread API (`adam_voice_start` / `adam_voice_stop`) for continuous operation
