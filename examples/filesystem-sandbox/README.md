# Filesystem Sandbox

Demonstrates restricting filesystem and shell tools to specific directories. The agent is given access to `file_read`, `file_write`, `list_directory`, `shell_exec`, and `calculator` tools, but filesystem operations are sandboxed to `/tmp` via `adam_settings_allow_dir`. Any attempt to access paths outside the allowed directories will be rejected by the library. The tool context is set to the settings struct so the sandbox checks can inspect the allowed directory list.

## Build & Run

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
make
./filesystem-sandbox
```

## What It Demonstrates

- Using `adam_settings_allow_dir()` to restrict filesystem access
- Registering multiple built-in tools (`file_read`, `file_write`, `list_directory`, `shell_exec`, `calculator`)
- Passing `settings` as tool context for sandbox enforcement
- Agent performing multi-step file operations within the sandbox
