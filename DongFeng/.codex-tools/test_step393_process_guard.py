#!/usr/bin/env python3
"""Three focused tests for the STEP393 process-enumeration race fix."""

from __future__ import annotations

import importlib.util
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("step393_process_guard", HERE / "step393_process_guard.py")
assert SPEC is not None and SPEC.loader is not None
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def stat_row(
    pid: int, pgid: int, starttime: int, state: bytes = b"S",
    comm: bytes = b"member",
) -> bytes:
    fields = [state, b"1", str(pgid).encode(), *([b"1"] * 16), str(starttime).encode()]
    return str(pid).encode() + b" (" + comm + b") " + b" ".join(fields) + b"\n"


def write_process(root: Path, pid: int, pgid: int, starttime: int, argv: tuple[bytes, ...]) -> None:
    process = root / str(pid)
    process.mkdir()
    (process / "stat").write_bytes(stat_row(pid, pgid, starttime))
    (process / "status").write_bytes(f"Name:\tmember\nNSpid:\t{pid}\t{pid}\n".encode())
    (process / "cmdline").write_bytes(b"\0".join(argv) + b"\0")


def manifest() -> dict[str, object]:
    return {
        "schema": "step358-launcher-ownership-v1",
        "port": 29950,
        "launcher_host_pid": 50,
        "launcher_starttime": 500,
        "launcher_pgid": 50,
    }


class Step393ProcessGuardTests(unittest.TestCase):
    def test_unrelated_idle_kernel_thread_is_ignored_and_never_signaled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            write_process(proc, 2, 0, 200, (b"kernel-thread",))
            (proc / "2" / "stat").write_bytes(
                stat_row(2, 0, 200, b"I", b"idle ) worker")
            )
            write_process(proc, 50, 50, 500, (b"unrelated",))
            original = guard._base._read_at

            recovered = guard.enumerate_group_identities(50, proc)
            self.assertEqual([(item.host_pid, item.starttime) for item in recovered], [(50, 500)])

            stat_reads = {"count": 0}

            def exit_during_identity(directory_fd: int, name: str, limit: int) -> bytes:
                if name == "stat":
                    stat_reads["count"] += 1
                    if stat_reads["count"] % 2:
                        return stat_row(50, 0, 0)
                    return stat_row(50, 50, 500, b"Z")
                return original(directory_fd, name, limit)

            signals = []
            with mock.patch.object(guard._base, "_read_at", side_effect=exit_during_identity):
                result = guard.safe_group_cleanup(
                    manifest(), proc_root=proc, grace_seconds=0, max_rounds=4,
                    signaler=lambda *args: signals.append(args), alive=lambda _row: False,
                    monotonic=lambda: 0.0, sleeper=lambda _value: None,
                )
            self.assertEqual(result["consecutive_empty_group_scans"], 2)
            self.assertEqual(signals, [])

    def test_live_change_unauthorized_and_persistent_instability_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            write_process(proc, 50, 50, 500, (b"unrelated",))
            original = guard._base._read_at
            stat_reads = {"count": 0}

            def changed_group(directory_fd: int, name: str, limit: int) -> bytes:
                if name == "stat":
                    stat_reads["count"] += 1
                    pgid = 50 if stat_reads["count"] == 1 else 70
                    return stat_row(50, pgid, 500)
                return original(directory_fd, name, limit)

            with mock.patch.object(guard._base, "_read_at", side_effect=changed_group), \
                    self.assertRaisesRegex(RuntimeError, "changed process group"):
                guard.enumerate_group_identities(50, proc)

            with mock.patch.object(
                guard._base, "_identity_from_open_dir",
                side_effect=RuntimeError("process identity changed while reading /proc"),
            ) as exact, self.assertRaises(guard.IdentitySnapshotUnstable):
                guard.enumerate_group_identities(50, proc)
            self.assertEqual(exact.call_count, 3)

            with mock.patch.object(
                guard._base, "_read_at", return_value=stat_row(50, 50, 0),
            ) as releasing, self.assertRaises(guard.ProcessStatSnapshotUnstable):
                guard.enumerate_group_identities(50, proc)
            self.assertEqual(releasing.call_count, 3)

            signals = []
            with self.assertRaisesRegex(RuntimeError, "owned group TERM snapshot failed"):
                guard.safe_group_cleanup(
                    manifest(), proc_root=proc, grace_seconds=0,
                    signaler=lambda *args: signals.append(args),
                    monotonic=lambda: 0.0, sleeper=lambda _value: None,
                )
            self.assertEqual(signals, [])

    def test_pgid_reuse_and_launcher_unseen_have_zero_signal_authority(self) -> None:
        launcher_argv = (
            b"timeout", b"--signal=TERM", b"--kill-after=30s", b"900s", b"docker", b"exec",
            b"-e", b"ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15",
            b"-e", b"ASCEND_CUSTOM_OPP_PATH=/shadow", b"-e", b"PYTHONPATH=/shadow",
            b"-e", b"TORCH_DEVICE_BACKEND_AUTOLOAD=0", b"mapqr-leicheng", b"bash",
            b"--noprofile", b"--norc", b"-lc",
            b"torchrun --nnodes=1 --nproc-per-node=8 --master-addr=127.0.0.1 "
            b"--master-port=29950 /diag/step377_diagnostic_math_worker.py --input-dir /in "
            b"--output-dir /out --shadow-root /shadow --installed-custom-opp /opp",
        )
        launcher = guard.ProcessIdentity(50, 500, (50,), 50, launcher_argv)
        worker_argv = (
            b"python3", b"step377_diagnostic_math_worker.py",
            b"--input-dir", b"/in", b"--output-dir", b"/out",
            b"--shadow-root", b"/shadow", b"--installed-custom-opp", b"/opp",
        )
        worker = guard.ProcessIdentity(70, 700, (70,), 50, worker_argv)
        signals = []

        sequence = iter(((launcher,), (), (worker,)))
        with mock.patch.object(guard, "enumerate_group_identities", side_effect=lambda *_a: next(sequence)), \
                self.assertRaisesRegex(RuntimeError, "owned group KILL snapshot failed"):
            guard.safe_group_cleanup(
                manifest(), grace_seconds=0,
                signaler=lambda *args: signals.append(args),
                alive=lambda _row: False,
                monotonic=lambda: 0.0, sleeper=lambda _value: None,
            )
        self.assertEqual([(item.host_pid, signum) for item, signum in signals], [(50, signal.SIGTERM)])

        signals.clear()
        with mock.patch.object(guard, "enumerate_group_identities", return_value=(worker,)), \
                self.assertRaisesRegex(RuntimeError, "ownership_unestablished"):
            guard.safe_group_cleanup(
                manifest(), grace_seconds=0, case_path=Path("/case.py"),
                signaler=lambda *args: signals.append(args),
                monotonic=lambda: 0.0, sleeper=lambda _value: None,
            )
        self.assertEqual(signals, [])


if __name__ == "__main__":
    unittest.main()
