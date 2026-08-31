#!/usr/bin/env python3
"""Simulate HomeBuddy/Soniox traffic against the proactive AI WebSocket."""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import websockets
import httpx
from websockets.asyncio.client import ClientConnection


DEFAULT_URL = "ws://127.0.0.1:18743/v1/ws"
DEFAULT_AUTH_USERNAME = "homebuddy"
DEFAULT_AUTH_PASSWORD = "123456"


@dataclass(frozen=True)
class TranscriptAction:
    text: str
    final: bool = True


@dataclass(frozen=True)
class WaitAction:
    seconds: float


@dataclass(frozen=True)
class ExpectInsightAction:
    contains: str = ""


@dataclass(frozen=True)
class ExpectNoInsightAction:
    pass


FileAction = TranscriptAction | WaitAction | ExpectInsightAction | ExpectNoInsightAction


def build_url(base_url: str, client_id: str, session_id: str) -> str:
    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query))
    query.update({"client_id": client_id, "session_id": session_id})
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def auth_url(websocket_url: str, action: str) -> str:
    parts = urlsplit(websocket_url)
    scheme = "https" if parts.scheme == "wss" else "http"
    return urlunsplit((scheme, parts.netloc, f"/v1/auth/{action}", "", ""))


async def sign_in(websocket_url: str, username: str, password: str, timeout: float) -> str:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            auth_url(websocket_url, "signin"),
            json={"username": username, "password": password},
        )
    if response.status_code != 200:
        raise RuntimeError(f"Sign-in failed with HTTP {response.status_code}")
    try:
        return str(response.json()["access_token"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError("Sign-in response did not contain an access token") from exc


async def sign_out(websocket_url: str, token: str, timeout: float) -> None:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            auth_url(websocket_url, "signout"),
            headers={"Authorization": f"Bearer {token}"},
        )
    if response.status_code != 200:
        raise RuntimeError(f"Sign-out failed with HTTP {response.status_code}")


def transcript_payload(
    text: str,
    *,
    final: bool = True,
    speaker: str = "owner",
    language: str = "en",
) -> dict[str, object]:
    return {
        "type": "transcript",
        "event_id": f"sim-{uuid4()}",
        "text": text,
        "is_final": final,
        "speaker": speaker,
        "language": language,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def receive_json(socket: ClientConnection, timeout: float) -> dict[str, object]:
    raw = await asyncio.wait_for(socket.recv(), timeout=timeout)
    message = json.loads(raw)
    print(f"<- {json.dumps(message, ensure_ascii=False)}", flush=True)
    return message


async def send_json(socket: ClientConnection, payload: dict[str, object]) -> None:
    print(f"-> {json.dumps(payload, ensure_ascii=False)}", flush=True)
    await socket.send(json.dumps(payload))


async def expect(socket: ClientConnection, expected_type: str, timeout: float) -> dict[str, object]:
    message = await receive_json(socket, timeout)
    actual_type = message.get("type")
    if actual_type != expected_type:
        raise RuntimeError(f"Expected event '{expected_type}', received '{actual_type}'")
    return message


async def run_scenario(socket: ClientConnection, timeout: float) -> None:
    await expect(socket, "ready", timeout)

    await send_json(socket, {"type": "ping"})
    await expect(socket, "pong", timeout)

    partial = transcript_payload("What should I bring to", final=False)
    await send_json(socket, partial)
    partial_ack = await expect(socket, "ack", timeout)
    if partial_ack.get("processed") is not False:
        raise RuntimeError("Partial transcript was unexpectedly processed")

    filler = transcript_payload("Okay")
    await send_json(socket, filler)
    filler_ack = await expect(socket, "ack", timeout)
    if filler_ack.get("triggered") is not False:
        raise RuntimeError("Filler speech unexpectedly triggered an insight")

    question = transcript_payload("What should I bring to the appointment tomorrow?")
    await send_json(socket, question)
    question_ack = await expect(socket, "ack", timeout)
    if question_ack.get("triggered") is not True:
        raise RuntimeError(f"Actionable question did not trigger: {question_ack.get('reason')}")

    result = await receive_json(socket, max(timeout, 60.0))
    if result.get("type") == "error":
        raise RuntimeError(f"Insight generation failed: {result.get('code')}")
    if result.get("type") != "insight" or not result.get("text"):
        raise RuntimeError("Server did not return a usable insight")

    await send_json(
        socket,
        {"type": "feedback", "insight_id": result["insight_id"], "useful": True},
    )
    await expect(socket, "feedback_saved", timeout)
    print("\nPASS: WebSocket, buffering, detector, memory, Ollama, and feedback all worked.")


def parse_text_file(path: Path) -> list[FileAction]:
    if not path.is_file():
        raise RuntimeError(f"Transcript file does not exist: {path}")

    actions: list[FileAction] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("WAIT:"):
            value = line.split(":", 1)[1].strip()
            try:
                seconds = float(value)
            except ValueError as exc:
                raise RuntimeError(f"Invalid WAIT value on line {line_number}: {value}") from exc
            if seconds < 0 or seconds > 300:
                raise RuntimeError(f"WAIT on line {line_number} must be between 0 and 300 seconds")
            actions.append(WaitAction(seconds))
            continue
        if line.upper().startswith("EXPECT_INSIGHT:"):
            actions.append(ExpectInsightAction(line.split(":", 1)[1].strip()))
            continue
        if line.upper().startswith("EXPECT_NO_INSIGHT"):
            actions.append(ExpectNoInsightAction())
            continue
        if line.upper().startswith("PARTIAL:"):
            text = line.split(":", 1)[1].strip()
            if not text:
                raise RuntimeError(f"Empty PARTIAL transcript on line {line_number}")
            actions.append(TranscriptAction(text=text, final=False))
            continue
        if line.upper().startswith("FINAL:"):
            line = line.split(":", 1)[1].strip()
            if not line:
                raise RuntimeError(f"Empty FINAL transcript on line {line_number}")
        actions.append(TranscriptAction(text=line))

    if not actions:
        raise RuntimeError(f"Transcript file contains no actions: {path}")
    return actions


async def run_file(
    socket: ClientConnection, path: Path, timeout: float, language: str = "en"
) -> None:
    await expect(socket, "ready", timeout)
    actions = parse_text_file(path)
    transcript_count = 0
    insight_count = 0
    last_insight: dict[str, object] | None = None

    print(f"Playing {path} ({len(actions)} actions)")
    for action in actions:
        if isinstance(action, WaitAction):
            print(f".. waiting {action.seconds:g}s", flush=True)
            await asyncio.sleep(action.seconds)
            continue
        if isinstance(action, ExpectInsightAction):
            if last_insight is None:
                raise RuntimeError("Expected an insight for the preceding transcript, but none arrived")
            text = str(last_insight.get("text", ""))
            if action.contains and action.contains.casefold() not in text.casefold():
                raise RuntimeError(
                    f"Expected insight containing '{action.contains}', received: {text}"
                )
            print(f"✓ expected insight received: {text}", flush=True)
            continue
        if isinstance(action, ExpectNoInsightAction):
            if last_insight is not None:
                raise RuntimeError(
                    f"Expected no insight for the preceding transcript, received: "
                    f"{last_insight.get('text', '')}"
                )
            print("✓ no repeated insight was emitted", flush=True)
            continue

        await send_json(
            socket, transcript_payload(action.text, final=action.final, language=language)
        )
        acknowledgement = await expect(socket, "ack", timeout)
        transcript_count += 1
        last_insight = None
        if acknowledgement.get("triggered") is True:
            result = await receive_json(socket, max(timeout, 60.0))
            if result.get("type") == "error":
                raise RuntimeError(f"Insight generation failed: {result.get('code')}")
            if result.get("type") != "insight" or not result.get("text"):
                raise RuntimeError("Server did not return a usable insight")
            insight_count += 1
            last_insight = result

    print(
        f"\nPASS: played {transcript_count} transcript events and received "
        f"{insight_count} insight event(s)."
    )


async def terminal_lines() -> AsyncIterator[str]:
    while True:
        try:
            line = await asyncio.to_thread(input, "transcript> ")
        except EOFError:
            return
        yield line.strip()


async def run_interactive(socket: ClientConnection, timeout: float) -> None:
    await expect(socket, "ready", timeout)
    print("Type speech and press Enter. Commands: /ping, /partial TEXT, /quit")

    async for line in terminal_lines():
        if not line:
            continue
        if line == "/quit":
            return
        if line == "/ping":
            await send_json(socket, {"type": "ping"})
            await receive_json(socket, timeout)
            continue

        final = not line.startswith("/partial ")
        text = line.removeprefix("/partial ")
        await send_json(socket, transcript_payload(text, final=final))
        response = await receive_json(socket, max(timeout, 60.0))
        if response.get("type") == "ack" and response.get("triggered") is True:
            await receive_json(socket, max(timeout, 60.0))


async def main_async(args: argparse.Namespace) -> None:
    token = args.token or os.getenv("ACCESS_TOKEN") or None
    issued_token = token is None
    if issued_token:
        if not args.username or not args.password:
            raise RuntimeError(
                "Provide --username and --password, or supply an existing --token"
            )
        token = await sign_in(args.url, args.username, args.password, args.timeout)
        print("Signed in and received a temporary bearer token")
    url = build_url(args.url, args.client_id, args.session_id)
    print(f"Connecting to {url}")
    try:
        async with websockets.connect(
            url,
            additional_headers={"Authorization": f"Bearer {token}"},
            max_size=1_000_000,
        ) as socket:
            if args.mode == "scenario":
                await run_scenario(socket, args.timeout)
            elif args.mode == "file":
                if not args.file:
                    raise RuntimeError("File mode requires --file or SIMULATOR_TEXT_FILE")
                await run_file(socket, Path(args.file), args.timeout, args.language)
            else:
                await run_interactive(socket, args.timeout)
    finally:
        if issued_token:
            await sign_out(args.url, token, args.timeout)
            print("Signed out and revoked the temporary bearer token")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("file", "scenario", "interactive"), nargs="?", default="file"
    )
    parser.add_argument("--url", default=os.getenv("SIMULATOR_WS_URL", DEFAULT_URL))
    parser.add_argument("--file", default=os.getenv("SIMULATOR_TEXT_FILE"))
    parser.add_argument("--language", default=os.getenv("SIMULATOR_LANGUAGE", "en"))
    parser.add_argument("--client-id", default="homebuddy-simulator")
    parser.add_argument("--session-id", default=f"simulation-{uuid4()}")
    parser.add_argument("--token", help="Existing access token; defaults to ACCESS_TOKEN")
    parser.add_argument(
        "--username", default=os.getenv("AUTH_USERNAME", DEFAULT_AUTH_USERNAME)
    )
    parser.add_argument(
        "--password", default=os.getenv("AUTH_PASSWORD", DEFAULT_AUTH_PASSWORD)
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    try:
        asyncio.run(main_async(parse_args()))
        return 0
    except (OSError, TimeoutError, RuntimeError, websockets.WebSocketException) as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
