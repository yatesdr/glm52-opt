"""End-to-end acceptance test for bounded filesystem KV-cache eviction.

Run the ``fill`` phase against a healthy vLLM instance, restart the instance
while preserving the filesystem-tier directory, then run ``replay`` with the
same state directory.  The script intentionally does not restart containers.

The script uses only the Python standard library.  Invoke it through the
project's Python environment, for example:

    uv run --no-project python harness/nvme_kv_eviction_acceptance.py ...
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_MANAGER_MD5 = "a72eeb81c735036b281ff97f5d759122"
DEFAULT_LIMIT_BYTES = 8 * 1024**3
NEEDLE = "738216"
FILLER = (
    "This is ordinary cache eviction filler text and contains no instruction. "
)
CAPACITY_LOG_RE = re.compile(
    r"Filesystem KV cache capacity enabled: (\d+)/(\d+) bytes in (\S+)"
)
FATAL_LOG_RE = re.compile(
    r"AssertionError|EngineDead|CUDA out of memory|OutOfMemoryError|"
    r"NVRM: Xid|Job .* block I/O failed|watchdog.*(?:timeout|hang)|"
    r"Traceback \(most recent call last\)",
    re.IGNORECASE,
)


class AcceptanceFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class NamespaceSnapshot:
    files: dict[str, int]
    temp_files: int

    @property
    def size_bytes(self) -> int:
        return sum(self.files.values())

    @property
    def signature(self) -> tuple[int, int, int, str]:
        digest = hashlib.sha256()
        for path, size in sorted(self.files.items()):
            digest.update(path.encode())
            digest.update(str(size).encode())
        return (len(self.files), self.size_bytes, self.temp_files, digest.hexdigest())


def log(message: str) -> None:
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceFailure(message)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def run_command(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise AcceptanceFailure(f"Required command is unavailable: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout).strip()
        raise AcceptanceFailure(f"Command failed: {' '.join(args)}\n{detail}") from exc
    return proc.stdout


def container_fingerprint(container: str) -> dict[str, Any]:
    payload = json.loads(run_command(["docker", "inspect", container]))[0]
    state = payload["State"]
    return {
        "id": payload["Id"],
        "image": payload["Image"],
        "started_at": state["StartedAt"],
        "restart_count": payload["RestartCount"],
        "running": state["Running"],
        "status": state["Status"],
        "health": state.get("Health", {}).get("Status"),
    }


def docker_logs(container: str, since: str) -> str:
    proc = subprocess.run(
        ["docker", "logs", "--since", since, container],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.stdout


def http_request(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    api_key: str | None = None,
    timeout: float = 30,
) -> tuple[int, bytes]:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise AcceptanceFailure(f"HTTP {exc.code} from {url}: {body[:2000]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise AcceptanceFailure(f"Request failed for {url}: {exc}") from exc


def wait_for_health(base_url: str, api_key: str | None, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            status, _ = http_request(
                f"{base_url}/health", api_key=api_key, timeout=10
            )
            if status == 200:
                return
        except AcceptanceFailure as exc:
            last_error = str(exc)
        time.sleep(5)
    raise AcceptanceFailure(f"Server did not become healthy: {last_error}")


def get_metrics(base_url: str, api_key: str | None) -> str:
    status, body = http_request(
        f"{base_url}/metrics", api_key=api_key, timeout=30
    )
    require(status == 200, f"Metrics endpoint returned HTTP {status}")
    return body.decode(errors="replace")


def metric_sum(metrics: str, metric_name: str) -> float:
    total = 0.0
    accepted = {metric_name, f"{metric_name}_total"}
    for raw_line in metrics.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name_and_labels, separator, value_text = line.rpartition(" ")
        if not separator:
            continue
        observed_name = name_and_labels.split("{", 1)[0]
        if observed_name not in accepted:
            continue
        try:
            total += float(value_text)
        except ValueError:
            continue
    return total


def post_json(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    api_key: str | None,
    timeout: float,
) -> dict[str, Any]:
    status, body = http_request(
        f"{base_url}{endpoint}",
        payload=payload,
        api_key=api_key,
        timeout=timeout,
    )
    require(status == 200, f"{endpoint} returned HTTP {status}")
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise AcceptanceFailure(
            f"Invalid JSON from {endpoint}: {body[:1000]!r}"
        ) from exc
    require(isinstance(parsed, dict), f"Unexpected response from {endpoint}")
    return parsed


def build_prompt(nonce: str, repetitions: int) -> str:
    split = repetitions * 2 // 5
    return "".join(
        (
            f"UNIQUE CACHE ACCEPTANCE PREFIX: {nonce}\n",
            FILLER * split,
            f"\nThe secret validation number is {NEEDLE}.\n",
            FILLER * (repetitions - split),
            "\nReturn only the secret validation number.",
        )
    )


def tokenize_count(
    base_url: str,
    model: str,
    prompt: str,
    api_key: str | None,
    timeout: float,
) -> int:
    response = post_json(
        base_url,
        "/tokenize",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"reasoning_effort": "low"},
        },
        api_key,
        timeout,
    )
    count = response.get("count")
    require(isinstance(count, int) and count > 0, "Invalid /tokenize count")
    return count


def calibrate_repetitions(args: argparse.Namespace, nonce: str) -> tuple[int, int]:
    repetitions = max(1, args.target_prompt_tokens // 10)
    closest: tuple[int, int] | None = None
    for _ in range(8):
        count = tokenize_count(
            args.base_url,
            args.model,
            build_prompt(nonce, repetitions),
            args.api_key,
            args.request_timeout,
        )
        if closest is None or abs(count - args.target_prompt_tokens) < abs(
            closest[1] - args.target_prompt_tokens
        ):
            closest = (repetitions, count)
        if abs(count - args.target_prompt_tokens) <= args.token_tolerance:
            return repetitions, count
        next_repetitions = max(
            1, round(repetitions * args.target_prompt_tokens / count)
        )
        if next_repetitions == repetitions:
            next_repetitions += 1 if count < args.target_prompt_tokens else -1
        repetitions = max(1, next_repetitions)
    assert closest is not None
    require(
        abs(closest[1] - args.target_prompt_tokens) <= args.token_tolerance,
        f"Could not calibrate prompt within {args.token_tolerance} tokens: "
        f"closest={closest[1]}",
    )
    return closest


def make_chat_request(model: str, prompt: str, request_id: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0,
        "seed": 0,
        "reasoning_effort": "low",
        "include_reasoning": False,
        "request_id": request_id,
    }


def validate_chat_response(
    response: dict[str, Any], target_tokens: int, tolerance: int
) -> dict[str, Any]:
    choices = response.get("choices")
    require(isinstance(choices, list) and choices, "Response has no choices")
    choice = choices[0]
    require(choice.get("finish_reason") == "stop", "finish_reason is not stop")
    message = choice.get("message") or {}
    answer = " ".join(
        str(value)
        for value in (message.get("content"), message.get("reasoning_content"))
        if value
    )
    require(NEEDLE in answer, f"Needle {NEEDLE} is absent from model response")
    usage = response.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    require(isinstance(prompt_tokens, int), "Response has no prompt token count")
    require(
        abs(prompt_tokens - target_tokens) <= tolerance,
        f"Prompt token count {prompt_tokens} is outside "
        f"{target_tokens}±{tolerance}",
    )
    return {
        "id": response.get("id"),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": usage.get("completion_tokens"),
        "needle_found": True,
    }


def submit_prompt(
    args: argparse.Namespace, body: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    response = post_json(
        args.base_url,
        "/v1/chat/completions",
        body,
        args.api_key,
        args.request_timeout,
    )
    summary = validate_chat_response(
        response, args.target_prompt_tokens, args.response_token_tolerance
    )
    summary["wall_seconds"] = round(time.monotonic() - started, 3)
    return response, summary


def discover_namespaces(cache_root: Path) -> list[Path]:
    if not cache_root.exists():
        return []
    return sorted(
        path
        for path in cache_root.iterdir()
        if path.is_dir() and re.search(r"_r\d+$", path.name)
    )


def scan_namespaces(cache_root: Path) -> dict[str, NamespaceSnapshot]:
    result: dict[str, NamespaceSnapshot] = {}
    for namespace in discover_namespaces(cache_root):
        files: dict[str, int] = {}
        temp_files = 0
        for path in namespace.rglob("*"):
            if not path.is_file():
                continue
            if path.name.endswith(".tmp"):
                temp_files += 1
                continue
            if not path.name.endswith(".bin"):
                continue
            try:
                files[str(path.relative_to(cache_root))] = path.stat().st_size
            except FileNotFoundError:
                continue
        result[namespace.name] = NamespaceSnapshot(files, temp_files)
    return result


def assert_capacity(
    snapshots: dict[str, NamespaceSnapshot], limit_bytes: int
) -> None:
    require(snapshots, "No filesystem-tier _r<rank> namespace was found")
    for name, snapshot in snapshots.items():
        require(
            snapshot.size_bytes <= limit_bytes,
            f"Capacity exceeded in {name}: {snapshot.size_bytes}>{limit_bytes}",
        )


def wait_for_quiescence(
    cache_root: Path,
    *,
    stable_seconds: float,
    timeout: float,
) -> dict[str, NamespaceSnapshot]:
    deadline = time.monotonic() + timeout
    stable_since = time.monotonic()
    previous: tuple[Any, ...] | None = None
    latest: dict[str, NamespaceSnapshot] = {}
    while time.monotonic() < deadline:
        latest = scan_namespaces(cache_root)
        signature = tuple(
            (name, snapshot.signature) for name, snapshot in sorted(latest.items())
        )
        has_files = any(snapshot.files for snapshot in latest.values())
        has_temps = any(snapshot.temp_files for snapshot in latest.values())
        if signature != previous:
            previous = signature
            stable_since = time.monotonic()
        elif has_files and not has_temps:
            if time.monotonic() - stable_since >= stable_seconds:
                return latest
        time.sleep(1)
    raise AcceptanceFailure(
        f"Filesystem tier did not quiesce within {timeout:.0f} seconds"
    )


class CapacityMonitor:
    def __init__(self, cache_root: Path, limit_bytes: int, output: Path):
        self.cache_root = cache_root
        self.limit_bytes = limit_bytes
        self.output = output
        self.stop_event = threading.Event()
        self.violations: list[str] = []
        self.high_water: dict[str, int] = {}
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)
        require(not self.thread.is_alive(), "Capacity monitor did not stop")
        require(self.error is None, f"Capacity monitor failed: {self.error}")
        require(not self.violations, "; ".join(self.violations))

    def _run(self) -> None:
        try:
            with self.output.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "utc_time",
                        "namespace",
                        "completed_files",
                        "completed_bytes",
                        "temp_files",
                        "limit_bytes",
                    ]
                )
                while not self.stop_event.is_set():
                    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
                    for name, snapshot in scan_namespaces(self.cache_root).items():
                        size = snapshot.size_bytes
                        self.high_water[name] = max(self.high_water.get(name, 0), size)
                        writer.writerow(
                            [
                                stamp,
                                name,
                                len(snapshot.files),
                                size,
                                snapshot.temp_files,
                                self.limit_bytes,
                            ]
                        )
                        if size > self.limit_bytes:
                            self.violations.append(
                                f"{name} exceeded capacity: {size}>{self.limit_bytes}"
                            )
                    handle.flush()
                    self.stop_event.wait(1)
        except Exception as exc:  # pragma: no cover - field diagnostic path
            self.error = repr(exc)


def manager_md5(container: str, manager_path: str) -> str:
    output = run_command(["docker", "exec", container, "md5sum", manager_path])
    return output.split()[0]


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], str, str]:
    require(args.cache_root.is_dir(), f"Cache root does not exist: {args.cache_root}")
    require(shutil.which("docker") is not None, "docker is required")
    require(shutil.which("findmnt") is not None, "findmnt is required")
    wait_for_health(args.base_url, args.api_key, args.health_timeout)

    fingerprint = container_fingerprint(args.container)
    require(fingerprint["running"], f"Container is not running: {fingerprint}")

    observed_md5 = manager_md5(args.container, args.manager_path)
    require(
        observed_md5 == args.expected_manager_md5,
        f"Patched manager MD5 mismatch: {observed_md5} != "
        f"{args.expected_manager_md5}",
    )

    mount = run_command(
        ["findmnt", "-J", "-T", str(args.cache_root), "-o", "TARGET,SOURCE,FSTYPE"]
    )
    disk = shutil.disk_usage(args.cache_root)
    required_free = args.limit_bytes if args.phase == "fill" else 1024**3
    require(
        disk.free >= required_free,
        f"NVMe filesystem has less than {required_free} free bytes: {disk}",
    )

    logs = docker_logs(args.container, fingerprint["started_at"])
    capacity_markers = CAPACITY_LOG_RE.findall(logs)
    require(capacity_markers, "Bounded filesystem-tier startup marker is absent")
    for used, limit, _ in capacity_markers:
        require(int(used) <= int(limit), f"Startup usage exceeds limit: {used}/{limit}")
        require(
            int(limit) == args.limit_bytes,
            f"Runtime capacity {limit} differs from expected {args.limit_bytes}",
        )

    metrics = get_metrics(args.base_url, args.api_key)
    return fingerprint, mount, metrics


def ensure_same_process(before: dict[str, Any], after: dict[str, Any]) -> None:
    for key in ("id", "started_at", "restart_count"):
        require(before[key] == after[key], f"Container {key} changed during test")
    require(after["running"], "Container is no longer running")


def check_runtime_logs(
    container: str, started_at: str, output: Path
) -> None:
    logs = docker_logs(container, started_at)
    matches = [line for line in logs.splitlines() if FATAL_LOG_RE.search(line)]
    output.write_text("\n".join(matches) + "\n")
    require(not matches, f"Fatal runtime log pattern found: {matches[0][:500]}")


def new_state_dir(requested: Path | None) -> Path:
    if requested is None:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        requested = Path.cwd() / f"nvme-kv-acceptance-{stamp}"
    requested = requested.resolve()
    require(not requested.exists(), f"State directory already exists: {requested}")
    requested.mkdir(parents=True)
    return requested


def record_exchange(
    state_dir: Path,
    sequence: int,
    role: str,
    request: dict[str, Any],
    response: dict[str, Any],
) -> None:
    exchanges = state_dir / "exchanges"
    exchanges.mkdir(exist_ok=True)
    stem = f"{sequence:03d}-{role}"
    write_json(exchanges / f"{stem}-request.json", request)
    write_json(exchanges / f"{stem}-response.json", response)


def run_fill(args: argparse.Namespace) -> None:
    state_dir = new_state_dir(args.state_dir)
    args.state_dir = state_dir
    log(f"Acceptance artifacts: {state_dir}")

    fingerprint, mount, metrics_before = preflight(args)
    write_json(
        state_dir / "run-config.json",
        {
            "base_url": args.base_url,
            "cache_root": str(args.cache_root),
            "container": args.container,
            "expected_manager_md5": args.expected_manager_md5,
            "limit_bytes": args.limit_bytes,
            "manager_path": args.manager_path,
            "max_fill_requests": args.max_fill_requests,
            "model": args.model,
            "response_token_tolerance": args.response_token_tolerance,
            "settle_seconds": args.settle_seconds,
            "target_prompt_tokens": args.target_prompt_tokens,
            "token_tolerance": args.token_tolerance,
            "turnover_multiple": args.turnover_multiple,
        },
    )
    write_json(state_dir / "container-before.json", fingerprint)
    (state_dir / "findmnt.json").write_text(mount)
    (state_dir / "metrics-before.txt").write_text(metrics_before)

    baseline = scan_namespaces(args.cache_root)
    baseline_files = sum(len(item.files) for item in baseline.values())
    require(
        args.allow_existing or baseline_files == 0,
        "Cache root already contains .bin files; use a fresh test root or pass "
        "--allow-existing for a conservative non-empty run",
    )

    monitor = CapacityMonitor(
        args.cache_root, args.limit_bytes, state_dir / "capacity-samples.csv"
    )
    monitor.start()
    response_summaries: list[dict[str, Any]] = []

    try:
        calibration_nonce = f"calibration-{uuid.uuid4()}"
        repetitions, calibrated_tokens = calibrate_repetitions(
            args, calibration_nonce
        )
        log(
            f"Calibrated {repetitions} filler repetitions to "
            f"{calibrated_tokens} chat prompt tokens"
        )

        sentinel_body = make_chat_request(
            args.model,
            build_prompt(f"eviction-sentinel-{uuid.uuid4()}", repetitions),
            f"nvme-eviction-sentinel-{uuid.uuid4()}",
        )
        response, summary = submit_prompt(args, sentinel_body)
        summary["role"] = "eviction_sentinel"
        response_summaries.append(summary)
        record_exchange(
            state_dir, len(response_summaries), summary["role"], sentinel_body, response
        )
        log(f"Sentinel completed in {summary['wall_seconds']}s")
        sentinel_after = wait_for_quiescence(
            args.cache_root,
            stable_seconds=args.settle_seconds,
            timeout=args.settle_timeout,
        )
        assert_capacity(sentinel_after, args.limit_bytes)

        payload_bytes: dict[str, int] = {}
        sentinel_paths: dict[str, set[str]] = {}
        for name, snapshot in sentinel_after.items():
            old_files = baseline.get(name, NamespaceSnapshot({}, 0)).files
            new_paths = set(snapshot.files) - set(old_files)
            payload_bytes[name] = sum(snapshot.files[path] for path in new_paths)
            sentinel_paths[name] = new_paths
            require(new_paths, f"Sentinel wrote no new block files in {name}")
        write_json(
            state_dir / "sentinel-paths.json",
            {name: sorted(paths) for name, paths in sentinel_paths.items()},
        )

        large_requests = 1
        fill_requests = 0
        current = sentinel_after
        while True:
            evicted_by_namespace = {
                name: len(paths - set(current[name].files))
                for name, paths in sentinel_paths.items()
            }
            enough_turnover = all(
                large_requests * payload_bytes[name]
                >= args.turnover_multiple * args.limit_bytes
                for name in payload_bytes
            )
            eviction_seen = all(count > 0 for count in evicted_by_namespace.values())
            if enough_turnover and eviction_seen:
                break
            require(
                fill_requests < args.max_fill_requests,
                "Maximum fill requests reached before proving both >capacity "
                "turnover and path disappearance; increase --max-fill-requests",
            )
            fill_requests += 1
            nonce = f"unique-fill-{fill_requests:03d}-{uuid.uuid4()}"
            body = make_chat_request(
                args.model,
                build_prompt(nonce, repetitions),
                f"nvme-unique-fill-{fill_requests:03d}-{uuid.uuid4()}",
            )
            response, summary = submit_prompt(args, body)
            summary["role"] = f"fill_{fill_requests:03d}"
            response_summaries.append(summary)
            record_exchange(
                state_dir, len(response_summaries), summary["role"], body, response
            )
            large_requests += 1
            current = wait_for_quiescence(
                args.cache_root,
                stable_seconds=args.settle_seconds,
                timeout=args.settle_timeout,
            )
            assert_capacity(current, args.limit_bytes)
            progress = ", ".join(
                f"{name}: bytes={current[name].size_bytes} "
                f"nominal={large_requests * payload_bytes[name]} "
                "sentinel_missing="
                f"{len(sentinel_paths[name] - set(current[name].files))}"
                for name in sorted(payload_bytes)
            )
            log(f"Fill {fill_requests}/{args.max_fill_requests}: {progress}")

        before_replay_anchor = current
        replay_body = make_chat_request(
            args.model,
            build_prompt(f"persisted-replay-anchor-{uuid.uuid4()}", repetitions),
            f"nvme-persisted-replay-{uuid.uuid4()}",
        )
        response, summary = submit_prompt(args, replay_body)
        summary["role"] = "persisted_replay_anchor"
        response_summaries.append(summary)
        record_exchange(
            state_dir, len(response_summaries), summary["role"], replay_body, response
        )
        log(f"Replay anchor completed in {summary['wall_seconds']}s")
        final_snapshot = wait_for_quiescence(
            args.cache_root,
            stable_seconds=args.settle_seconds,
            timeout=args.settle_timeout,
        )
        assert_capacity(final_snapshot, args.limit_bytes)

        replay_paths: list[str] = []
        for name, snapshot in final_snapshot.items():
            previous = before_replay_anchor.get(name, NamespaceSnapshot({}, 0))
            replay_paths.extend(sorted(set(snapshot.files) - set(previous.files)))
        require(replay_paths, "Replay anchor produced no newly retained FS blocks")
        write_json(state_dir / "replay-request.json", replay_body)
        write_json(state_dir / "replay-anchor-paths.json", replay_paths)
    finally:
        monitor.stop()

    final_fingerprint = container_fingerprint(args.container)
    ensure_same_process(fingerprint, final_fingerprint)
    wait_for_health(args.base_url, args.api_key, 30)
    metrics_after = get_metrics(args.base_url, args.api_key)
    (state_dir / "metrics-after-fill.txt").write_text(metrics_after)
    write_json(state_dir / "container-after-fill.json", final_fingerprint)
    check_runtime_logs(
        args.container,
        fingerprint["started_at"],
        state_dir / "runtime-log-findings-fill.txt",
    )

    final_snapshot = scan_namespaces(args.cache_root)
    assert_capacity(final_snapshot, args.limit_bytes)
    report = {
        "verdict": "PASS",
        "phase": "fill",
        "cache_root": str(args.cache_root),
        "limit_bytes_per_namespace": args.limit_bytes,
        "turnover_multiple": args.turnover_multiple,
        "large_requests_before_replay_anchor": large_requests,
        "payload_bytes_per_request": payload_bytes,
        "nominal_attempted_bytes": {
            name: large_requests * size for name, size in payload_bytes.items()
        },
        "sentinel_paths_missing": {
            name: len(paths - set(final_snapshot[name].files))
            for name, paths in sentinel_paths.items()
        },
        "final_namespace_bytes": {
            name: snapshot.size_bytes for name, snapshot in final_snapshot.items()
        },
        "high_water_bytes": monitor.high_water,
        "manager_md5": args.expected_manager_md5,
        "responses": response_summaries,
        "kv_store_bytes_delta": metric_sum(
            metrics_after, "vllm:kv_offload_store_bytes"
        )
        - metric_sum(metrics_before, "vllm:kv_offload_store_bytes"),
    }
    write_json(state_dir / "fill-report.json", report)

    log("PASS: bounded eviction and old-path disappearance were observed")
    log("Now restart the vLLM container normally, preserving the NVMe directory.")
    log(
        "After it is healthy, run: "
        f"uv run --no-project python {Path(__file__)} replay "
        f"--cache-root {args.cache_root} --state-dir {state_dir}"
    )


def run_replay(args: argparse.Namespace) -> None:
    require(args.state_dir is not None, "replay requires --state-dir")
    state_dir = args.state_dir.resolve()
    require(state_dir.is_dir(), f"State directory does not exist: {state_dir}")
    run_config = read_json(state_dir / "run-config.json")
    require(
        args.cache_root == Path(run_config["cache_root"]),
        "Replay --cache-root differs from the fill phase",
    )
    args.expected_manager_md5 = run_config["expected_manager_md5"]
    args.limit_bytes = run_config["limit_bytes"]
    args.manager_path = run_config["manager_path"]
    args.model = run_config["model"]
    args.response_token_tolerance = run_config["response_token_tolerance"]
    args.target_prompt_tokens = run_config["target_prompt_tokens"]
    fill_fingerprint = read_json(state_dir / "container-after-fill.json")
    replay_body = read_json(state_dir / "replay-request.json")
    replay_paths = read_json(state_dir / "replay-anchor-paths.json")

    fingerprint, mount, metrics_before = preflight(args)
    require(
        fingerprint["started_at"] != fill_fingerprint["started_at"],
        "Container StartedAt is unchanged; restart before replay so GPU/DRAM "
        "state cannot satisfy the request",
    )
    (state_dir / "findmnt-after-restart.json").write_text(mount)
    (state_dir / "metrics-before-replay.txt").write_text(metrics_before)
    write_json(state_dir / "container-before-replay.json", fingerprint)

    before_snapshot = scan_namespaces(args.cache_root)
    assert_capacity(before_snapshot, args.limit_bytes)
    resident_replay_paths = [
        path
        for path in replay_paths
        if (args.cache_root / path).is_file()
    ]
    require(
        resident_replay_paths,
        "None of the saved replay-anchor block paths survived the restart",
    )

    monitor = CapacityMonitor(
        args.cache_root,
        args.limit_bytes,
        state_dir / "capacity-samples-replay.csv",
    )
    monitor.start()
    try:
        response, response_summary = submit_prompt(args, replay_body)
        write_json(state_dir / "replay-response.json", response)
        after_snapshot = wait_for_quiescence(
            args.cache_root,
            stable_seconds=args.settle_seconds,
            timeout=args.settle_timeout,
        )
        assert_capacity(after_snapshot, args.limit_bytes)
    finally:
        monitor.stop()

    hits_before = metric_sum(metrics_before, "vllm:external_prefix_cache_hits")
    queries_before = metric_sum(
        metrics_before, "vllm:external_prefix_cache_queries"
    )
    metrics_after = ""
    hits_delta = 0.0
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        metrics_after = get_metrics(args.base_url, args.api_key)
        hits_delta = (
            metric_sum(metrics_after, "vllm:external_prefix_cache_hits")
            - hits_before
        )
        if hits_delta > 0:
            break
        time.sleep(1)
    require(hits_delta > 0, "Replay produced zero external prefix-cache hits")

    queries_delta = (
        metric_sum(metrics_after, "vllm:external_prefix_cache_queries")
        - queries_before
    )
    require(queries_delta > 0, "Replay produced zero external prefix-cache queries")
    final_fingerprint = container_fingerprint(args.container)
    ensure_same_process(fingerprint, final_fingerprint)
    wait_for_health(args.base_url, args.api_key, 30)
    check_runtime_logs(
        args.container,
        fingerprint["started_at"],
        state_dir / "runtime-log-findings-replay.txt",
    )
    (state_dir / "metrics-after-replay.txt").write_text(metrics_after)
    write_json(state_dir / "container-after-replay.json", final_fingerprint)

    report = {
        "verdict": "PASS",
        "phase": "replay",
        "response": response_summary,
        "external_prefix_cache_queries_delta": queries_delta,
        "external_prefix_cache_hits_delta": hits_delta,
        "saved_anchor_paths": len(replay_paths),
        "saved_anchor_paths_present_before_replay": len(resident_replay_paths),
        "namespace_bytes_after_replay": {
            name: snapshot.size_bytes for name, snapshot in after_snapshot.items()
        },
        "high_water_bytes": monitor.high_water,
        "container_before": fingerprint,
        "container_after": final_fingerprint,
    }
    write_json(state_dir / "replay-report.json", report)
    log(
        "PASS: persisted NVMe promotion proved by "
        f"external hits delta={hits_delta:.0f}, queries delta={queries_delta:.0f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove bounded NVMe KV eviction and persisted promotion."
    )
    parser.add_argument("phase", choices=("fill", "replay"))
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:5001")
    parser.add_argument("--model", default="GLM-5.2")
    parser.add_argument("--container", default="glm52-prod")
    parser.add_argument("--limit-bytes", type=int, default=DEFAULT_LIMIT_BYTES)
    parser.add_argument("--target-prompt-tokens", type=int, default=50_000)
    parser.add_argument("--token-tolerance", type=int, default=128)
    parser.add_argument("--response-token-tolerance", type=int, default=256)
    parser.add_argument("--turnover-multiple", type=float, default=2.0)
    parser.add_argument("--max-fill-requests", type=int, default=32)
    parser.add_argument("--settle-seconds", type=float, default=8)
    parser.add_argument("--settle-timeout", type=float, default=300)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--health-timeout", type=float, default=60)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument(
        "--expected-manager-md5", default=EXPECTED_MANAGER_MD5
    )
    parser.add_argument(
        "--manager-path",
        default=(
            "/opt/venv/lib/python3.12/site-packages/vllm/v1/kv_offload/"
            "tiering/fs/manager.py"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VLLM_API_KEY"),
        help="Defaults to VLLM_API_KEY; never written to artifacts.",
    )
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    args.cache_root = args.cache_root.resolve()
    require(args.limit_bytes > 0, "--limit-bytes must be positive")
    require(args.turnover_multiple > 1, "--turnover-multiple must be >1")
    return args


def main() -> int:
    try:
        args = parse_args()
        if args.phase == "fill":
            run_fill(args)
        else:
            run_replay(args)
        return 0
    except AcceptanceFailure as exc:
        print(f"FAIL: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
