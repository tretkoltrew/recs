import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, stream_with_context


load_dotenv()

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 1.0
REQUEST_TIMEOUT_SECONDS = 300

SUPPORTED_MODELS = {
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
}

app = Flask(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(message)s")
logger = logging.getLogger("openai_anthropic_proxy")


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_key_mapping() -> Dict[str, str]:
    """Load USER_KEY_MAP as JSON or as user_key:anthropic_key,user_key2:anthropic_key2."""
    raw = os.getenv("USER_KEY_MAP", "").strip()
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items() if k and v}
    except json.JSONDecodeError:
        pass

    mapping: Dict[str, str] = {}
    for pair in _split_csv(raw):
        if ":" in pair:
            user_key, anthropic_key = pair.split(":", 1)
            user_key = user_key.strip()
            anthropic_key = anthropic_key.strip()
            if user_key and anthropic_key:
                mapping[user_key] = anthropic_key
    return mapping


VALID_KEYS = set(_split_csv(os.getenv("VALID_KEYS", "")))
KEY_MAPPING = _load_key_mapping()


def now_unix() -> int:
    return int(time.time())


def mask_key(api_key: Optional[str]) -> str:
    if not api_key:
        return "none"
    return f"{api_key[:4]}..."


def json_error(message: str, error_type: str, status_code: int):
    return jsonify({"error": {"message": message, "type": error_type}}), status_code


def extract_bearer_token() -> Optional[str]:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return None
    return auth_header[7:].strip()


def get_authenticated_user_key() -> Tuple[Optional[str], Optional[Response]]:
    user_key = extract_bearer_token()
    if not user_key or user_key not in VALID_KEYS:
        return None, jsonify({"error": "Invalid API key"})
    return user_key, None


def get_anthropic_key_for_user(user_key: str) -> Optional[str]:
    return KEY_MAPPING.get(user_key) or os.getenv("ANTHROPIC_API_KEY")


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def normalize_anthropic_messages(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, str]]]:
    system_parts: List[str] = []
    anthropic_messages: List[Dict[str, str]] = []

    for item in messages:
        role = item.get("role")
        content = content_to_text(item.get("content"))

        if role == "system":
            if content:
                system_parts.append(content)
            continue

        mapped_role = "assistant" if role == "assistant" else "user"
        if anthropic_messages and anthropic_messages[-1]["role"] == mapped_role:
            anthropic_messages[-1]["content"] = f"{anthropic_messages[-1]['content']}\n\n{content}".strip()
        else:
            anthropic_messages.append({"role": mapped_role, "content": content})

    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, anthropic_messages


def validate_chat_request(payload: Any) -> Optional[Tuple[str, int]]:
    if not isinstance(payload, dict):
        return "Request body must be a JSON object", 400
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return "messages must be a non-empty array", 400
    for message in messages:
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            return "each message must contain role and content", 400
    return None


def build_anthropic_payload(payload: Dict[str, Any], stream: bool) -> Tuple[Dict[str, Any], str]:
    requested_model = payload.get("model")
    anthropic_model = requested_model if requested_model in SUPPORTED_MODELS else DEFAULT_MODEL
    system_prompt, anthropic_messages = normalize_anthropic_messages(payload["messages"])

    anthropic_payload: Dict[str, Any] = {
        "model": anthropic_model,
        "messages": anthropic_messages,
        "max_tokens": int(payload.get("max_tokens") or DEFAULT_MAX_TOKENS),
        "temperature": float(payload.get("temperature") if payload.get("temperature") is not None else DEFAULT_TEMPERATURE),
        "stream": stream,
    }
    if system_prompt:
        anthropic_payload["system"] = system_prompt
    return anthropic_payload, anthropic_model


def anthropic_headers(api_key: str) -> Dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def extract_text_from_anthropic(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type", "text") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def openai_usage(usage: Dict[str, Any]) -> Dict[str, int]:
    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def finish_reason_from_anthropic(stop_reason: Optional[str]) -> str:
    if stop_reason == "max_tokens":
        return "length"
    if stop_reason in {"stop_sequence", "end_turn", "tool_use"}:
        return "stop"
    return "stop"


def transform_anthropic_response(data: Dict[str, Any], model: str) -> Dict[str, Any]:
    usage = openai_usage(data.get("usage") or {})
    return {
        "id": f"chatcmpl-{data.get('id') or uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now_unix(),
        "model": data.get("model") or model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": extract_text_from_anthropic(data.get("content")),
                },
                "finish_reason": finish_reason_from_anthropic(data.get("stop_reason")),
            }
        ],
        "usage": usage,
    }


def transform_anthropic_error(response: requests.Response):
    try:
        body = response.json()
        error = body.get("error") if isinstance(body, dict) else {}
        message = error.get("message") if isinstance(error, dict) else response.text
        error_type = error.get("type") if isinstance(error, dict) else "anthropic_error"
    except ValueError:
        message = response.text or "Anthropic API error"
        error_type = "anthropic_error"
    return jsonify({"error": {"message": message, "type": error_type}}), response.status_code


def log_request(
    user_key: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    status: str = "ok",
) -> None:
    log_record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "client_ip": request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip(),
        "api_key": mask_key(user_key),
        "model": model,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "status": status,
    }
    logger.info(json.dumps(log_record, ensure_ascii=False))


def sse(data: Any) -> str:
    if isinstance(data, str):
        return f"data: {data}\n\n"
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def openai_stream_chunk(
    chunk_id: str,
    model: str,
    delta: Dict[str, Any],
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    choice: Dict[str, Any] = {"index": 0, "delta": delta}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": now_unix(),
        "model": model,
        "choices": [choice],
    }


def iter_anthropic_sse_lines(response: requests.Response) -> Iterable[Dict[str, Any]]:
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            logger.warning("Skipping invalid Anthropic stream event: %s", data)


def stream_openai_response(
    anthropic_response: requests.Response,
    user_key: str,
    model: str,
) -> Generator[str, None, None]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    input_tokens = 0
    output_tokens = 0
    finish_reason = "stop"

    yield sse(openai_stream_chunk(chunk_id, model, {"role": "assistant", "content": ""}))

    try:
        for event in iter_anthropic_sse_lines(anthropic_response):
            event_type = event.get("type")

            if event_type == "message_start":
                usage = (event.get("message") or {}).get("usage") or {}
                input_tokens = int(usage.get("input_tokens") or input_tokens)
                continue

            if event_type == "content_block_delta":
                delta = event.get("delta") or {}
                text = delta.get("text")
                if text:
                    yield sse(openai_stream_chunk(chunk_id, model, {"content": text}))
                continue

            if event_type == "message_delta":
                usage = event.get("usage") or {}
                output_tokens = int(usage.get("output_tokens") or output_tokens)
                finish_reason = finish_reason_from_anthropic((event.get("delta") or {}).get("stop_reason"))
                continue

            if event_type == "error":
                error = event.get("error") or {}
                yield sse(
                    {
                        "error": {
                            "message": error.get("message", "Anthropic stream error"),
                            "type": error.get("type", "anthropic_error"),
                        }
                    }
                )
                break

        yield sse(openai_stream_chunk(chunk_id, model, {}, finish_reason))
        yield sse("[DONE]")
    finally:
        log_request(user_key, model, input_tokens, output_tokens)
        anthropic_response.close()


@app.post("/v1/chat/completions")
def chat_completions():
    user_key, auth_error = get_authenticated_user_key()
    if auth_error is not None:
        return auth_error, 401

    anthropic_api_key = get_anthropic_key_for_user(user_key)
    if not anthropic_api_key:
        return json_error("ANTHROPIC_API_KEY is not configured", "server_error", 500)

    payload = request.get_json(silent=True)
    validation_error = validate_chat_request(payload)
    if validation_error:
        message, status_code = validation_error
        return json_error(message, "invalid_request_error", status_code)

    stream = bool(payload.get("stream"))
    try:
        anthropic_payload, model = build_anthropic_payload(payload, stream)
    except (TypeError, ValueError) as exc:
        return json_error(str(exc), "invalid_request_error", 400)

    try:
        anthropic_response = requests.post(
            ANTHROPIC_MESSAGES_URL,
            headers=anthropic_headers(anthropic_api_key),
            json=anthropic_payload,
            stream=stream,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return json_error(str(exc), "api_connection_error", 502)

    if not anthropic_response.ok:
        return transform_anthropic_error(anthropic_response)

    if stream:
        return Response(
            stream_with_context(stream_openai_response(anthropic_response, user_key, model)),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    data = anthropic_response.json()
    result = transform_anthropic_response(data, model)
    usage = result["usage"]
    log_request(user_key, result["model"], usage["prompt_tokens"], usage["completion_tokens"])
    return jsonify(result)


@app.get("/v1/models")
def list_models():
    user_key, auth_error = get_authenticated_user_key()
    if auth_error is not None:
        return auth_error, 401

    created = now_unix()
    return jsonify(
        {
            "object": "list",
            "data": [
                {"id": model, "object": "model", "created": created, "owned_by": "anthropic"}
                for model in sorted(SUPPORTED_MODELS)
            ],
        }
    )


@app.post("/admin/keys")
def manage_keys():
    admin_key = os.getenv("ADMIN_KEY")
    provided_key = extract_bearer_token()
    if not admin_key or provided_key != admin_key:
        return jsonify({"error": "Invalid admin key"}), 401

    payload = request.get_json(silent=True) or {}
    action = payload.get("action")
    key = payload.get("key")
    anthropic_api_key = payload.get("anthropic_api_key")

    if action not in {"add", "remove"} or not key:
        return json_error("action must be add/remove and key is required", "invalid_request_error", 400)

    if action == "add":
        VALID_KEYS.add(str(key))
        if anthropic_api_key:
            KEY_MAPPING[str(key)] = str(anthropic_api_key)
    else:
        VALID_KEYS.discard(str(key))
        KEY_MAPPING.pop(str(key), None)

    return jsonify({"ok": True, "action": action, "key": mask_key(str(key)), "valid_key_count": len(VALID_KEYS)})


@app.get("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "3456"))
    app.run(host="0.0.0.0", port=port, threaded=True)
