# OpenAI-Compatible Anthropic Claude Proxy

This project is a pure-Python Flask proxy that accepts OpenAI Chat Completions requests and forwards them to Anthropic Claude Messages API. It is designed for clients such as Cursor, Cline, Continue.dev, and other tools that support a custom OpenAI API base URL.

The server does not use Bun and does not require AVX CPU instructions.

## Features

- OpenAI-compatible endpoint: `POST /v1/chat/completions`
- Listens on `0.0.0.0:3456` by default
- Supports non-streaming and streaming SSE responses
- Converts OpenAI `messages` into Anthropic `system` plus `messages`
- Converts Anthropic responses and usage into OpenAI-compatible JSON
- Validates proxy user API keys from `VALID_KEYS`
- Optionally maps proxy user keys to different Anthropic API keys
- Dynamically adds/removes user keys through `POST /admin/keys`
- Logs time, client IP, masked key, model, and token counts

## Requirements

- Python 3.9 or newer
- Flask
- requests
- python-dotenv

## Installation

```powershell
cd openai-anthropic-proxy
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux/macOS:

```bash
cd openai-anthropic-proxy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Copy the sample environment file and edit it:

```powershell
copy .env.example .env
```

Example `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-your-main-anthropic-key
VALID_KEYS=user-key-1,user-key-2
ADMIN_KEY=change-this-admin-key
PORT=3456

# Optional per-user Anthropic API key mapping.
USER_KEY_MAP={"user-key-1":"sk-ant-user-1","user-key-2":"sk-ant-user-2"}
```

Environment variables:

- `ANTHROPIC_API_KEY`: Main Anthropic API key used when no per-user mapping exists.
- `VALID_KEYS`: Comma-separated proxy API keys accepted from `Authorization: Bearer <key>`.
- `ADMIN_KEY`: Optional admin key for `POST /admin/keys`.
- `PORT`: Server port. Defaults to `3456`.
- `USER_KEY_MAP`: Optional mapping from proxy user keys to Anthropic API keys. Supports JSON or `user-key:anthropic-key,user-key-2:anthropic-key-2`.

## Start The Server

```powershell
python proxy.py
```

The server listens on:

```text
http://0.0.0.0:3456
```

For OpenAI-compatible tools, use this base URL:

```text
http://YOUR_SERVER_IP:3456/v1
```

Use one of your `VALID_KEYS` values as the OpenAI API key in the client.

## Chat Completion Test

Non-streaming request:

```bash
curl -X POST http://localhost:3456/v1/chat/completions \
  -H "Authorization: Bearer user-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "system", "content": "You are concise."},
      {"role": "user", "content": "Say hello in one short sentence."}
    ],
    "max_tokens": 128,
    "temperature": 0.7,
    "stream": false
  }'
```

Example response:

```json
{
  "id": "chatcmpl-msg_abc123",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "claude-3-5-sonnet-20241022",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello, it is nice to meet you."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 9,
    "total_tokens": 19
  }
}
```

## Streaming Test

```bash
curl -N -X POST http://localhost:3456/v1/chat/completions \
  -H "Authorization: Bearer user-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-haiku-20240307",
    "messages": [
      {"role": "user", "content": "Count from 1 to 3."}
    ],
    "max_tokens": 128,
    "stream": true
  }'
```

Example streamed response:

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1710000000,"model":"claude-3-haiku-20240307","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1710000000,"model":"claude-3-haiku-20240307","choices":[{"index":0,"delta":{"content":"1"}}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1710000000,"model":"claude-3-haiku-20240307","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## Admin Key Management

Add a proxy user key without restarting:

```bash
curl -X POST http://localhost:3456/admin/keys \
  -H "Authorization: Bearer change-this-admin-key" \
  -H "Content-Type: application/json" \
  -d '{
    "action": "add",
    "key": "new-user-key",
    "anthropic_api_key": "sk-ant-optional-dedicated-key"
  }'
```

Remove a proxy user key:

```bash
curl -X POST http://localhost:3456/admin/keys \
  -H "Authorization: Bearer change-this-admin-key" \
  -H "Content-Type: application/json" \
  -d '{
    "action": "remove",
    "key": "new-user-key"
  }'
```

Dynamic admin changes are stored in memory. Update `.env` too if the key should survive a server restart.

## Model Handling

The proxy accepts any incoming `model` value. If it is one of the supported Claude model IDs, it is passed through to Anthropic:

- `claude-3-5-sonnet-20241022`
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

Any other model value is mapped to the default:

```text
claude-3-5-sonnet-20241022
```

## Error Formats

Invalid proxy API key:

```json
{"error":"Invalid API key"}
```

Validation or Anthropic errors:

```json
{
  "error": {
    "message": "error message",
    "type": "invalid_request_error"
  }
}
```

## Logs

Logs are JSON lines, for example:

```json
{"time":"2026-06-17T16:46:00Z","client_ip":"127.0.0.1","api_key":"user...","model":"claude-3-5-sonnet-20241022","input_tokens":10,"output_tokens":20,"status":"ok"}
```
