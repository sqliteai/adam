# Telegram Bot

Connect Adam to Telegram. Send text or photos to the bot and get AI responses. Supports tools, memory, vision, and all Adam features.

## Telegram Setup

### 1. Create a Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot`
3. Choose a name (e.g. "My Adam Bot")
4. Choose a username (must end in `bot`, e.g. `my_adam_bot`)
5. BotFather will give you a token like `7123456789:AAH...` — copy it

### 2. Configure

Create a `.env` file in this directory (or the project root):

```
TELEGRAM_BOT_TOKEN=7123456789:AAHxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-...
```

Any LLM provider works — set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY`. For local models, use `--local` instead. Add `OPENAI_API_KEY` for voice message support (Whisper STT).

### 3. Build & Run

```bash
make
./telegram
```

Or with a local model:

```bash
./telegram --local ../../models/gemma-3-4b-it-Q4_K_S.gguf
```

With vision:

```bash
./telegram --local ../../models/gemma-3-4b-it-Q4_K_S.gguf --mmproj ../../models/mmproj-gemma-3-4b-it-F16.gguf
```

### 4. Chat

Open your bot in Telegram (search for the username you chose) and press **Start**. Send any message.

## Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message with instructions |
| `/tools` | Enable calculator and web search |
| `/memory` | Enable persistent memory (survives restart) |
| `/clear` | Clear conversation history |
| `/status` | Show current configuration |

## Features

- **Text chat** — multi-turn conversation with full history
- **Voice messages** — send voice notes, transcribed via Whisper STT then processed
- **Image input** — send photos and the bot describes/analyzes them
- **Tool calling** — calculator, web search (enable with `/tools`)
- **Memory** — persistent knowledge across conversations (enable with `/memory`)
- **All providers** — Anthropic, OpenAI, Gemini, or local llama.cpp

## How It Works

The bot uses Telegram's [long polling](https://core.telegram.org/bots/api#getupdates) (`getUpdates` with 30s timeout) — no webhook or public IP needed. Each message is processed sequentially through `adam_run()`.

For photos, the bot downloads the highest-resolution version via `getFile`, attaches it to the conversation via `adam_history_attach()`, and sends the result. Voice messages are downloaded as OGG/Opus files and transcribed via the OpenAI Whisper API (`adam_stt_transcribe`), then the transcribed text is processed through `adam_run()` like a normal message.
