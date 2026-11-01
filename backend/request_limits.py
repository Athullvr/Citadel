"""Small ASGI request-size guard for execution-feedback JSON bodies."""

import json
from typing import Any

from execution_feedback import MAX_EXECUTION_BODY_BYTES


class ExecutionRequestBodyLimitMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive, send) -> None:
        path = scope.get("path", "")
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or not path.startswith("/api/executions")
        ):
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                if int(raw_length) > MAX_EXECUTION_BODY_BYTES:
                    await self._send_too_large(scope, receive, send)
                    return
            except ValueError:
                pass

        messages: list[dict[str, Any]] = []
        body_size = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.request":
                body_size += len(message.get("body", b""))
                if body_size > MAX_EXECUTION_BODY_BYTES:
                    await self._send_too_large(scope, receive, send)
                    return
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                return

        message_index = 0

        async def replay_receive() -> dict[str, Any]:
            nonlocal message_index
            if message_index < len(messages):
                message = messages[message_index]
                message_index += 1
                return message
            return await receive()

        await self.app(scope, replay_receive, send)

    @staticmethod
    async def _send_too_large(scope: dict[str, Any], receive, send) -> None:
        body = json.dumps(
            {
                "error": "payload_too_large",
                "message": "Execution request body exceeds the 256 KiB limit.",
                "detail": "Execution request body exceeds the 256 KiB limit.",
            }
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
