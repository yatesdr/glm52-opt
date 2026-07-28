#!/usr/bin/env python3
"""Ordered Linux inotify monitor for a bounded filesystem KV tier.

Start this while the cache namespace is idle. Wait for the ready file before
starting the fill workload, then create the stop file after the workload has
quiesced. The monitor tracks completed ``.bin`` bytes and in-flight ``.tmp``
bytes in kernel event order.

This is a field-proof helper, not part of vLLM. It fails closed on inotify
queue overflow, baseline/final reconciliation failure, or a dynamic-directory
race that already contains cache files when its watch is installed.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import datetime as dt
import json
import os
import re
import select
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

IN_MOVED_FROM = 0x00000040
IN_MOVED_TO = 0x00000080
IN_CREATE = 0x00000100
IN_DELETE = 0x00000200
IN_MODIFY = 0x00000002
IN_CLOSE_WRITE = 0x00000008
IN_DELETE_SELF = 0x00000400
IN_MOVE_SELF = 0x00000800
IN_Q_OVERFLOW = 0x00004000
IN_IGNORED = 0x00008000
IN_ISDIR = 0x40000000
IN_NONBLOCK = os.O_NONBLOCK
IN_CLOEXEC = os.O_CLOEXEC

WATCH_MASK = (
    IN_MOVED_FROM
    | IN_MOVED_TO
    | IN_CREATE
    | IN_DELETE
    | IN_MODIFY
    | IN_CLOSE_WRITE
    | IN_DELETE_SELF
    | IN_MOVE_SELF
)
EVENT_HEADER = struct.Struct("iIII")
NAMESPACE_RE = re.compile(r"_r\d+$")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: Path, value: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def is_cache_file(path: Path) -> bool:
    return path.name.endswith(".bin") or path.name.endswith(".tmp")


def file_kind(path: Path) -> str | None:
    if path.name.endswith(".bin"):
        return "completed"
    if path.name.endswith(".tmp"):
        return "temporary"
    return None


@dataclass
class NamespaceUsage:
    completed_files: int = 0
    completed_bytes: int = 0
    temporary_files: int = 0
    temporary_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return self.completed_bytes + self.temporary_bytes


class OrderedCapacityMonitor:
    def __init__(
        self,
        cache_root: Path,
        limit_bytes: int,
        csv_path: Path,
        ready_file: Path,
        stop_file: Path,
        settle_seconds: float,
    ) -> None:
        self.cache_root = cache_root.resolve()
        self.limit_bytes = limit_bytes
        self.csv_path = csv_path
        self.ready_file = ready_file
        self.stop_file = stop_file
        self.settle_seconds = settle_seconds
        self.files: dict[Path, int] = {}
        self.watch_paths: dict[int, Path] = {}
        self.path_watches: dict[Path, int] = {}
        self.move_sizes: dict[int, int] = {}
        self.high_water: dict[str, int] = {}
        self.violations: list[dict[str, object]] = []
        self.coverage_errors: list[str] = []
        self.events_seen = 0

        libc_name = ctypes.util.find_library("c")
        if not libc_name:
            raise RuntimeError("Unable to locate libc")
        self.libc = ctypes.CDLL(libc_name, use_errno=True)
        self.libc.inotify_init1.argtypes = [ctypes.c_int]
        self.libc.inotify_init1.restype = ctypes.c_int
        self.libc.inotify_add_watch.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        self.libc.inotify_add_watch.restype = ctypes.c_int

        self.fd = self.libc.inotify_init1(IN_NONBLOCK | IN_CLOEXEC)
        if self.fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def add_watch(self, directory: Path) -> None:
        directory = directory.resolve()
        if directory in self.path_watches:
            return
        wd = self.libc.inotify_add_watch(
            self.fd, os.fsencode(directory), WATCH_MASK
        )
        if wd < 0:
            err = ctypes.get_errno()
            raise OSError(err, f"inotify_add_watch({directory}): {os.strerror(err)}")
        self.watch_paths[wd] = directory
        self.path_watches[directory] = wd

    def add_tree(self, root: Path, *, dynamic: bool = False) -> None:
        if not root.is_dir():
            return
        self.add_watch(root)
        for directory, dirnames, _ in os.walk(root):
            base = Path(directory)
            self.add_watch(base)
            for name in dirnames:
                child = base / name
                if child.is_dir():
                    self.add_watch(child)
        if dynamic:
            found = self.scan_tree(root)
            if found:
                self.coverage_errors.append(
                    "Dynamic directory already contained cache files when its "
                    f"watch was installed: {root} ({len(found)} files)"
                )
            self.files.update(found)

    def scan_tree(self, root: Path | None = None) -> dict[Path, int]:
        base = root or self.cache_root
        result: dict[Path, int] = {}
        if not base.exists():
            return result
        for path in base.rglob("*"):
            if not path.is_file() or not is_cache_file(path):
                continue
            try:
                result[path.resolve()] = path.stat().st_size
            except FileNotFoundError:
                continue
        return result

    def namespace_for(self, path: Path) -> str | None:
        try:
            first = path.relative_to(self.cache_root).parts[0]
        except (ValueError, IndexError):
            return None
        return first if NAMESPACE_RE.search(first) else None

    def usage(self) -> dict[str, NamespaceUsage]:
        result: dict[str, NamespaceUsage] = {}
        for path, size in self.files.items():
            namespace = self.namespace_for(path)
            kind = file_kind(path)
            if namespace is None or kind is None:
                continue
            usage = result.setdefault(namespace, NamespaceUsage())
            if kind == "completed":
                usage.completed_files += 1
                usage.completed_bytes += size
            else:
                usage.temporary_files += 1
                usage.temporary_bytes += size
        return result

    def record(
        self,
        writer: csv.writer,
        event: str,
        path: Path | None,
        mask: int,
    ) -> None:
        usages = self.usage()
        if not usages:
            writer.writerow(
                [self.events_seen, utc_now(), event, str(path or ""), mask]
                + ["", 0, 0, 0, 0, 0, self.limit_bytes, 0]
            )
            return
        for namespace, usage in sorted(usages.items()):
            self.high_water[namespace] = max(
                self.high_water.get(namespace, 0), usage.total_bytes
            )
            over = max(0, usage.total_bytes - self.limit_bytes)
            writer.writerow(
                [
                    self.events_seen,
                    utc_now(),
                    event,
                    str(path or ""),
                    mask,
                    namespace,
                    usage.completed_files,
                    usage.completed_bytes,
                    usage.temporary_files,
                    usage.temporary_bytes,
                    usage.total_bytes,
                    self.limit_bytes,
                    over,
                ]
            )
            if over:
                self.violations.append(
                    {
                        "event_index": self.events_seen,
                        "utc_time": utc_now(),
                        "event": event,
                        "path": str(path or ""),
                        "namespace": namespace,
                        "completed_bytes": usage.completed_bytes,
                        "temporary_bytes": usage.temporary_bytes,
                        "total_bytes": usage.total_bytes,
                        "limit_bytes": self.limit_bytes,
                    }
                )

    def update_file(
        self,
        path: Path,
        *,
        fallback_size: int | None = None,
    ) -> None:
        try:
            self.files[path] = path.stat().st_size
        except FileNotFoundError:
            if fallback_size is not None:
                self.files[path] = fallback_size

    def process_buffer(self, data: bytes, writer: csv.writer) -> None:
        offset = 0
        while offset + EVENT_HEADER.size <= len(data):
            wd, mask, cookie, name_len = EVENT_HEADER.unpack_from(data, offset)
            offset += EVENT_HEADER.size
            raw_name = data[offset : offset + name_len]
            offset += name_len
            name = os.fsdecode(raw_name.split(b"\0", 1)[0]) if name_len else ""
            self.events_seen += 1

            if mask & IN_Q_OVERFLOW:
                self.coverage_errors.append("IN_Q_OVERFLOW: ordered proof invalid")
                self.record(writer, "queue_overflow", None, mask)
                continue

            directory = self.watch_paths.get(wd)
            if directory is None:
                self.coverage_errors.append(f"Event for unknown watch descriptor {wd}")
                self.record(writer, "unknown_watch", None, mask)
                continue
            path = (directory / name).resolve() if name else directory

            if mask & IN_ISDIR:
                if mask & (IN_CREATE | IN_MOVED_TO):
                    try:
                        self.add_tree(path, dynamic=True)
                    except OSError as exc:
                        self.coverage_errors.append(str(exc))
                    self.record(writer, "directory_added", path, mask)
                elif mask & (IN_DELETE | IN_MOVED_FROM | IN_DELETE_SELF | IN_MOVE_SELF):
                    for tracked in list(self.files):
                        try:
                            tracked.relative_to(path)
                        except ValueError:
                            continue
                        self.files.pop(tracked, None)
                    self.record(writer, "directory_removed", path, mask)
                continue

            if not is_cache_file(path):
                continue

            if mask & (IN_DELETE | IN_MOVED_FROM):
                old_size = self.files.pop(path, None)
                if cookie and old_size is not None:
                    self.move_sizes[cookie] = old_size
                self.record(writer, "remove", path, mask)
                continue

            if mask & (IN_CREATE | IN_MODIFY | IN_CLOSE_WRITE | IN_MOVED_TO):
                self.update_file(path, fallback_size=self.move_sizes.pop(cookie, None))
                self.record(writer, "publish_or_resize", path, mask)

    def drain(self, writer: csv.writer, timeout: float) -> bool:
        readable, _, _ = select.select([self.fd], [], [], timeout)
        if not readable:
            return False
        while True:
            try:
                data = os.read(self.fd, 1 << 20)
            except BlockingIOError:
                break
            if not data:
                break
            self.process_buffer(data, writer)
        return True

    def establish_idle_baseline(self, writer: csv.writer) -> None:
        self.add_tree(self.cache_root)
        quiet_since = time.monotonic()
        while True:
            if self.drain(writer, 0.25):
                quiet_since = time.monotonic()
                continue
            if time.monotonic() - quiet_since < self.settle_seconds:
                continue
            first = self.scan_tree()
            time.sleep(0.25)
            self.drain(writer, 0.0)
            second = self.scan_tree()
            if first != second:
                quiet_since = time.monotonic()
                continue
            self.files = second
            temp_paths = [path for path in self.files if file_kind(path) == "temporary"]
            if temp_paths:
                raise RuntimeError(
                    f"Baseline is not quiescent: {len(temp_paths)} temporary files"
                )
            self.record(writer, "baseline", None, 0)
            return

    def run(self) -> dict[str, object]:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.ready_file.parent.mkdir(parents=True, exist_ok=True)
        self.stop_file.parent.mkdir(parents=True, exist_ok=True)
        self.ready_file.unlink(missing_ok=True)
        self.stop_file.unlink(missing_ok=True)
        columns = [
            "event_index",
            "utc_time",
            "event",
            "path",
            "mask",
            "namespace",
            "completed_files",
            "completed_bytes",
            "temporary_files",
            "temporary_bytes",
            "total_bytes",
            "limit_bytes",
            "over_bytes",
        ]
        with self.csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            self.establish_idle_baseline(writer)
            handle.flush()
            atomic_write_json(
                self.ready_file,
                {
                    "status": "READY",
                    "utc_time": utc_now(),
                    "pid": os.getpid(),
                    "cache_root": str(self.cache_root),
                    "limit_bytes": self.limit_bytes,
                    "baseline": {
                        key: vars(value) | {"total_bytes": value.total_bytes}
                        for key, value in self.usage().items()
                    },
                },
            )
            print(f"READY {self.ready_file}", flush=True)

            try:
                while not self.stop_file.exists():
                    self.drain(writer, 0.25)
                    handle.flush()
            except KeyboardInterrupt:
                self.coverage_errors.append(
                    "Interrupted without the requested stop-file handshake"
                )

            quiet_since = time.monotonic()
            while time.monotonic() - quiet_since < self.settle_seconds:
                if self.drain(writer, 0.25):
                    quiet_since = time.monotonic()
            final_scan = self.scan_tree()
            if final_scan != self.files:
                missing = sorted(
                    str(path) for path in self.files.keys() - final_scan.keys()
                )
                extra = sorted(
                    str(path) for path in final_scan.keys() - self.files.keys()
                )
                self.coverage_errors.append(
                    "Final event state does not match filesystem scan: "
                    f"missing={missing[:5]} extra={extra[:5]}"
                )
            self.files = final_scan
            self.record(writer, "final", None, 0)
            handle.flush()

        final_usage = self.usage()
        temporary_files = sum(item.temporary_files for item in final_usage.values())
        if temporary_files:
            self.coverage_errors.append(
                f"Final namespace contains {temporary_files} temporary files"
            )
        verdict = (
            "PASS"
            if not self.violations and not self.coverage_errors
            else "INCONCLUSIVE"
            if self.coverage_errors
            else "FAIL"
        )
        return {
            "verdict": verdict,
            "utc_time": utc_now(),
            "cache_root": str(self.cache_root),
            "limit_bytes": self.limit_bytes,
            "events_seen": self.events_seen,
            "high_water_bytes": self.high_water,
            "violations": self.violations,
            "coverage_errors": self.coverage_errors,
            "final": {
                key: vars(value) | {"total_bytes": value.total_bytes}
                for key, value in final_usage.items()
            },
            "csv": str(self.csv_path),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--limit-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sys.platform != "linux":
        print("ERROR: inotify monitoring requires Linux", file=sys.stderr)
        return 2
    if not args.cache_root.is_dir():
        print(f"ERROR: cache root does not exist: {args.cache_root}", file=sys.stderr)
        return 2
    if args.limit_bytes <= 0:
        print("ERROR: --limit-bytes must be positive", file=sys.stderr)
        return 2

    monitor = OrderedCapacityMonitor(
        cache_root=args.cache_root,
        limit_bytes=args.limit_bytes,
        csv_path=args.output.with_suffix(".csv"),
        ready_file=args.ready_file,
        stop_file=args.stop_file,
        settle_seconds=args.settle_seconds,
    )
    try:
        report = monitor.run()
    finally:
        monitor.close()
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
