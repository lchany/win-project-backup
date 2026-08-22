#!/usr/bin/env python3
from __future__ import annotations

import os
import hashlib
import signal
import subprocess
import sys
import tempfile
import time
import json
import contextlib
import io
import unittest
from pathlib import Path
from unittest import mock

import step377_process_guard as guard


# Sanitized from the 2026-08-22 read-only `npu-smi info` grammar check.
# Scope: process-table header, separators, and idle sentinels only; no PID/IP/credential.
REAL_IDLE_FIXTURE = """+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 0 |
| No running processes found in NPU 1 |
| No running processes found in NPU 2 |
| No running processes found in NPU 3 |
| No running processes found in NPU 4 |
| No running processes found in NPU 5 |
| No running processes found in NPU 6 |
| No running processes found in NPU 7 |
+===========================+===============+====================================================+
"""
REAL_IDLE_FIXTURE_SHA256 = "9cbe59a87cf344798642b20cdbd39ccdb0648211069629e10f834184cabfd961"


def smi_rows(pids=range(100, 108)):
    return "| NPU Chip | Process id | Process name | Memory |\n" + "\n".join(
        f"| {physical} {chip} | {pid} | python3 | 100 |"
        for (physical, chip), pid in zip(sorted(guard.BACK8_PAIRS), pids)
    )


def idle_smi():
    header = (
        "+---------------------------+---------------+----------------------------------------------------+\n"
        "| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |\n"
        "+===========================+===============+====================================================+"
    )
    rows = "\n".join(f"| No running processes found in NPU {device} |" for device in range(8))
    return header + "\n" + rows


def identities():
    worker = (b"python3", b"/diag/step377_diagnostic_math_worker.py",
              b"--input-dir", b"/in", b"--output-dir", b"/out",
              b"--shadow-root", b"/shadow", b"--installed-custom-opp", b"/opp")
    return {
        pid: guard.ProcessIdentity(pid, 1000 + pid, (pid, 200 + index), 900, worker)
        for index, pid in enumerate(range(100, 108))
    }


class Step377ProcessGuardTests(unittest.TestCase):
    def test_strict_parser_and_negative_rows(self):
        rows = guard.parse_back8_strict(smi_rows())
        self.assertEqual({row.device_id for row in rows}, set(range(8, 16)))
        cases = [
            smi_rows(range(100, 107)),
            smi_rows([100] * 8),
            smi_rows() + "\n| 4 0 | 999 | extra | 1 |",
            smi_rows().replace("| 4 0 | 100 |", "| 4 0 | zero |"),
            smi_rows().replace("Process id", "Pid"),
            "totally unrelated output",
            smi_rows() + "\n| unexpected columns |",
        ]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                guard.parse_back8_strict(value)
        self.assertEqual(guard.parse_back8_idle(idle_smi()), ())
        self.assertEqual(hashlib.sha256(REAL_IDLE_FIXTURE.encode()).hexdigest(), REAL_IDLE_FIXTURE_SHA256)
        self.assertEqual(guard.parse_back8_idle(REAL_IDLE_FIXTURE), ())
        for bad_idle in (
            idle_smi().replace("NPU 7", "NPU 3"),
            idle_smi().replace("No running processes found in NPU 7", "No processes NPU 7"),
        ):
            with self.assertRaises(RuntimeError):
                guard.parse_back8_idle(bad_idle)
        live_and_idle = smi_rows() + "\n| No running processes found in NPU 4 |"
        with self.assertRaisesRegex(RuntimeError, "both live and idle"):
            guard.parse_back8_strict(live_and_idle)

    def test_stat_parser_handles_spaces_and_right_parenthesis(self):
        suffix = [b"S"] + [b"1"] * 18 + [b"4242"] + [b"1"] * 5
        suffix[2] = b"77"
        self.assertEqual(guard.parse_stat_starttime(b"7 (name ) with spaces) " + b" ".join(suffix)), 4242)
        self.assertEqual(guard.parse_stat_identity(b"7 (name ) with spaces) " + b" ".join(suffix)), (4242, 77))
        self.assertEqual(guard.parse_stat_identity_state(b"7 (name ) with spaces) " + b" ".join(suffix)), (b"S", 4242, 77))
        for state in (b"Z", b"X"):
            dead = list(suffix); dead[0] = state
            self.assertEqual(guard.parse_stat_identity_state(b"7 (dead) " + b" ".join(dead)), (state, 4242, 77))

    def test_nspid_validation(self):
        self.assertEqual(guard.parse_nspid(b"Name:\tx\nNSpid:\t42\t7\n", 42), (42, 7))
        self.assertEqual(guard.parse_nspid(b"NSpid:  42  7 \n", 42), (42, 7))
        self.assertEqual(guard.parse_nspid(b"NSpid:\t42\v7\n", 42), (42, 7))
        self.assertEqual(guard.parse_nspid(b"NSpid:\t42\n", 42), (42,))
        for value in (
            b"Name: x\n", b"NSpid:\t\n", b"NSpid: 41 7\n", b"NSpid: 42 1\n",
            b"NSpid: 42 seven\n", b"NSpid: +42 7\n", b"NSpid: 42 7x\n",
            b"NSpid: 42 7\nNSpid:\t42\t7\n",
        ):
            with self.assertRaises(RuntimeError):
                guard.parse_nspid(value, 42)

    def test_tab_nspid_ready_binding_and_exited_cleanup_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            worker = (b"python3", b"-u", b"/diag/step377_diagnostic_math_worker.py",
                      b"--input-dir", b"/in", b"--output-dir", b"/out",
                      b"--shadow-root", b"/shadow", b"--installed-custom-opp", b"/opp")
            known = {}
            for index, pid in enumerate(range(100, 108)):
                row = proc / str(pid); row.mkdir()
                suffix = [b"S"] + [b"1"] * 24
                suffix[2] = b"900"; suffix[19] = str(1000 + pid).encode()
                (row / "cmdline").write_bytes(b"\0".join(worker) + b"\0")
                (row / "stat").write_bytes(str(pid).encode() + b" (python) " + b" ".join(suffix))
                (row / "status").write_bytes(
                    b"Name:\tpython\nNSpid:\t" + str(pid).encode() + b"\t"
                    + str(200 + index).encode() + b"\n"
                )
                known[pid] = guard.read_process_identity(pid, proc)
            ready = [{"rank": rank, "local_rank": rank, "container_pid": 200 + rank}
                     for rank in range(8)]
            binding = guard.stable_back8_binding(
                ready, lambda: smi_rows(), lambda pid: guard.read_process_identity(pid, proc)
            )
            self.assertEqual(len(binding["bindings"]), 8)
            for pid, identity in known.items():
                self.assertTrue(guard.owned_identity_alive(
                    identity, lambda current, root=proc: guard.read_process_identity(current, root)
                ))
                os.rename(proc / str(pid), proc / (str(pid) + ".exited"))
            signals = []
            guard.terminate_owned(
                known.values(),
                lambda identity: guard.owned_identity_alive(
                    identity, lambda current, root=proc: guard.read_process_identity(current, root)
                ),
                grace_seconds=0, signaler=lambda *args: signals.append(args),
                monotonic=lambda: 0.0, sleeper=lambda _value: None,
            )
            self.assertEqual(signals, [])

    def test_stable_binding_positive_and_races(self):
        known = identities()
        ready = [{"rank": rank, "local_rank": rank, "container_pid": 200 + rank} for rank in range(8)]
        result = guard.stable_back8_binding(ready, lambda: smi_rows(), known.__getitem__)
        self.assertEqual([row["device_id"] for row in result["bindings"]], list(range(8, 16)))
        samples = iter((smi_rows(), smi_rows(range(101, 109))))
        with self.assertRaisesRegex(RuntimeError, "between samples"):
            guard.stable_back8_binding(ready, lambda: next(samples), known.__getitem__)
        changed = dict(known)
        calls = {100: 0}
        def changing(pid):
            if pid == 100:
                calls[pid] += 1
                if calls[pid] > 1:
                    row = known[pid]
                    return guard.ProcessIdentity(pid, row.starttime + 1, row.nspid, row.pgid)
            return changed[pid]
        with self.assertRaisesRegex(RuntimeError, "process identity changed"):
            guard.stable_back8_binding(ready, lambda: smi_rows(), changing)

    def test_binding_rejects_container_duplicate_and_rank_swap(self):
        known = identities()
        duplicate = dict(known)
        duplicate[101] = guard.ProcessIdentity(101, 1101, (101, 200), 900)
        ready = [{"rank": rank, "local_rank": rank, "container_pid": 200 + rank} for rank in range(8)]
        with self.assertRaisesRegex(RuntimeError, "bijection"):
            guard.stable_back8_binding(ready, lambda: smi_rows(), duplicate.__getitem__)
        ready[0], ready[1] = dict(ready[0]), dict(ready[1])
        ready[0]["container_pid"], ready[1]["container_pid"] = 201, 200
        with self.assertRaisesRegex(RuntimeError, "rank/device"):
            guard.stable_back8_binding(ready, lambda: smi_rows(), known.__getitem__)
        missing = [dict(row) for row in ready]
        missing[0].pop("container_pid")
        with self.assertRaisesRegex(RuntimeError, "schema"):
            guard.stable_back8_binding(missing, lambda: smi_rows(), known.__getitem__)

    def test_exact_nul_argv_matching(self):
        path = b"/case/exact.py"
        argv = (b"python3", path, b"--port", b"29950", b"--output-dir", b"/out",
                b"--input-dir", b"/in", b"--shadow-root", b"/shadow",
                b"--installed-custom-opp", b"/opp")
        self.assertTrue(guard._argv_matches(argv, path, 29950))
        for changed in (
            (b"python3", path + b".bak", *argv[2:]),
            (*argv[:3], b"129950", *argv[4:]),
            (*argv, b"extra"),
            (*argv[:4], b"--input-dir", b"/in", b"--output-dir", b"/out", *argv[8:]),
        ):
            self.assertFalse(guard._argv_matches(changed, path, 29950))
        relative = (b"python3", b"exact.py", *argv[2:])
        self.assertTrue(guard._argv_matches(relative, path, 29950))
        inner = (b"torchrun --nnodes=1 --nproc-per-node=8 --master-addr=127.0.0.1 "
                 b"--master-port=29950 /diag/step377_diagnostic_math_worker.py --input-dir /in "
                 b"--output-dir /out --shadow-root /shadow --installed-custom-opp /opp --first-profiled-only")
        docker = (b"timeout", b"--signal=TERM", b"--kill-after=30s", b"900s", b"docker", b"exec",
                  b"-e", b"ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15",
                  b"-e", b"ASCEND_CUSTOM_OPP_PATH=/shadow", b"-e", b"PYTHONPATH=/shadow",
                  b"-e", b"TORCH_DEVICE_BACKEND_AUTOLOAD=0", b"mapqr-leicheng", b"bash",
                  b"--noprofile", b"--norc", b"-lc", inner)
        self.assertTrue(guard._argv_matches(docker, path, 29950))
        self.assertTrue(guard._approved_group_member_argv(docker[4:], path, 29950))
        self.assertFalse(guard._argv_matches((*docker[:-1], inner.replace(b"29950", b"29951")), path, 29950))

    def test_worker_argv_accepts_only_proven_optional_python_unbuffered_position(self):
        arguments = (b"--input-dir", b"/in", b"--output-dir", b"/out",
                     b"--shadow-root", b"/shadow", b"--installed-custom-opp", b"/opp")
        production = (b"python3", b"-u", b"/diag/step377_diagnostic_math_worker.py", *arguments)
        self.assertTrue(guard._worker_argv(production))
        self.assertTrue(guard._worker_argv((*production, b"--first-profiled-only")))
        for argv in (
            (b"python3", b"/diag/step377_diagnostic_math_worker.py", b"-u", *arguments),
            (b"python3", b"-u", b"-u", b"/diag/step377_diagnostic_math_worker.py", *arguments),
            (*production[:-1], b"-u"),
            (b"python3", b"/diag/step377_diagnostic_math_worker.py", *arguments, b"-u"),
        ):
            with self.subTest(argv=argv):
                self.assertFalse(guard._worker_argv(argv))
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory); row = proc / "42"; row.mkdir()
            suffix = [b"S"] + [b"1"] * 24
            suffix[2] = b"42"; suffix[19] = b"1000"
            (row / "cmdline").write_bytes(b"\0".join(production) + b"\0")
            (row / "stat").write_bytes(b"42 (python) " + b" ".join(suffix))
            (row / "status").write_bytes(b"NSpid:\t42 7\n")
            found = guard.approved_step377_rank_workers(proc)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].argv, production)

    def test_double_process_scan_rejects_change(self):
        first = (guard.ProcessIdentity(10, 20, (10, 30), 40),)
        with mock.patch.object(guard, "scan_case_processes_once", side_effect=(first, ())):
            with self.assertRaisesRegex(RuntimeError, "between samples"):
                guard.stable_case_process_scan(Path("/case"), 29950, 40)

    def test_scan_builds_argv_and_identity_from_same_open_pid_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory); (proc / "42").mkdir(); case = Path(directory) / "case.py"
            argv = (b"python3", os.fsencode(str(case.resolve())), b"--port", b"29950",
                    b"--output-dir", b"/out", b"--input-dir", b"/in",
                    b"--shadow-root", b"/shadow", b"--installed-custom-opp", b"/opp")
            identity = guard.ProcessIdentity(42, 99, (42, 7), 42, argv)
            with mock.patch.object(guard, "_matching_identity_from_open_dir", return_value=identity) as from_fd, \
                 mock.patch.object(guard, "read_process_identity") as reopened:
                self.assertEqual(guard.scan_case_processes_once(case, 29950, 42, proc), (identity,))
            from_fd.assert_called_once()
            reopened.assert_not_called()
            with mock.patch.object(guard, "_matching_identity_from_open_dir", return_value=identity):
                with self.assertRaisesRegex(RuntimeError, "process group"):
                    guard.scan_case_processes_once(case, 29950, 43, proc)

    def test_scan_skips_unrelated_malformed_stat_but_matching_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory); case = proc / "case.py"
            launch = (b"python3", os.fsencode(str(case.resolve())), b"--port", b"29950",
                      b"--output-dir", b"/out", b"--input-dir", b"/in",
                      b"--shadow-root", b"/shadow", b"--installed-custom-opp", b"/opp")
            for pid, argv in ((42, (b"unrelated",)), (43, launch)):
                row = proc / str(pid); row.mkdir()
                (row / "cmdline").write_bytes(b"\0".join(argv) + b"\0")
                (row / "stat").write_bytes(b"malformed")
                (row / "status").write_bytes(f"NSpid:\t{pid} 7\n".encode())
            with self.assertRaisesRegex(RuntimeError, "malformed proc stat"):
                guard.scan_case_processes_once(case, 29950, None, proc)
            (proc / "43" / "cmdline").write_bytes(b"unrelated\0")
            self.assertEqual(guard.scan_case_processes_once(case, 29950, None, proc), ())

    def test_matching_pid_reuse_is_detected_with_same_dirfd(self):
        argv = (b"python3", b"/case.py", b"--port", b"29950", b"--output-dir", b"/out",
                b"--input-dir", b"/in", b"--shadow-root", b"/shadow",
                b"--installed-custom-opp", b"/opp")
        def stat_row(start):
            suffix = [b"S"] + [b"1"] * 24
            suffix[2] = b"42"; suffix[19] = str(start).encode()
            return b"42 (case) " + b" ".join(suffix)
        reads = iter((b"\0".join(argv) + b"\0", stat_row(100), b"NSpid:\t42 7\n",
                      b"\0".join(argv) + b"\0", b"\0".join(argv) + b"\0", stat_row(101)))
        with mock.patch.object(guard, "_read_at", side_effect=lambda *_a: next(reads)):
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                guard._matching_identity_from_open_dir(42, 99, b"/case.py", 29950)

    def test_stable_clear_requires_two_idle_samples_and_empty_case_scan(self):
        with mock.patch.object(guard, "stable_case_process_scan", return_value=()):
            result = guard.assert_stable_clear(idle_smi, Path("/case"), 29950, 40)
        self.assertEqual((result["back8_process_count"], result["case_process_count"]), (0, 0))
        with mock.patch.object(guard, "stable_case_process_scan", return_value=()):
            with self.assertRaisesRegex(RuntimeError, "not idle"):
                guard.assert_stable_clear(lambda: smi_rows(), Path("/case"), 29950, 40)
        samples = iter((idle_smi(), idle_smi().replace("+---", "+====", 1)))
        with mock.patch.object(guard, "stable_case_process_scan", return_value=()):
            self.assertEqual(len(guard.assert_stable_clear(lambda: next(samples), Path("/case"), 29950, 40)["sample_sha256"]), 2)
        survivor = (guard.ProcessIdentity(10, 20, (10, 30), 40),)
        with mock.patch.object(guard, "stable_case_process_scan", return_value=survivor):
            with self.assertRaisesRegex(RuntimeError, "remain"):
                guard.assert_stable_clear(idle_smi, Path("/case"), 29950, 40)

    def test_pidfd_signal_revalidates_and_has_no_fallback(self):
        identity = guard.ProcessIdentity(100, 200, (100, 300), 400)
        with mock.patch.object(guard.os, "pidfd_open", None), self.assertRaisesRegex(RuntimeError, "unavailable"):
            guard.signal_owned_pidfd(identity, signal.SIGTERM, lambda _pid: identity)
        sent = []
        reads = mock.Mock(return_value=identity)
        with mock.patch.object(guard.os, "pidfd_open", return_value=9, create=True), \
             mock.patch.object(guard.signal, "pidfd_send_signal", side_effect=lambda *args: sent.append(args), create=True), \
             mock.patch.object(guard.os, "close"):
            guard.signal_owned_pidfd(identity, signal.SIGTERM, reads)
        self.assertEqual(sent[0][0:2], (9, signal.SIGTERM))
        self.assertEqual(reads.call_count, 2)
        changed = guard.ProcessIdentity(100, 201, (100, 300), 400)
        with mock.patch.object(guard.os, "pidfd_open", return_value=9, create=True), \
             mock.patch.object(guard.signal, "pidfd_send_signal", create=True) as send, \
             mock.patch.object(guard.os, "close"), \
             self.assertRaisesRegex(RuntimeError, "identity changed"):
            guard.signal_owned_pidfd(identity, signal.SIGKILL, lambda _pid: changed)
        send.assert_not_called()

    def test_pidfd_send_error_still_closes_fd(self):
        identity = guard.ProcessIdentity(100, 200, (100, 300), 400)
        with mock.patch.object(guard.os, "pidfd_open", return_value=9, create=True), \
             mock.patch.object(guard.signal, "pidfd_send_signal", side_effect=OSError("send"), create=True), \
             mock.patch.object(guard.os, "close") as close, self.assertRaisesRegex(OSError, "send"):
            guard.signal_owned_pidfd(identity, signal.SIGTERM, lambda _pid: identity)
        close.assert_called_once_with(9)

    def test_zombie_or_reused_owned_pid_is_dead_and_never_signaled(self):
        identity = guard.ProcessIdentity(100, 200, (100, 300), 400)
        exiting = ProcessLookupError("process is a zombie or dead")
        self.assertFalse(guard.owned_identity_alive(identity, mock.Mock(side_effect=exiting)))
        replacement = guard.ProcessIdentity(100, 201, (100, 301), 401)
        self.assertFalse(guard.owned_identity_alive(identity, lambda _pid: replacement))

        send = mock.Mock()
        with mock.patch.object(guard.os, "pidfd_open", return_value=9, create=True), \
             mock.patch.object(guard.signal, "pidfd_send_signal", send, create=True), \
             mock.patch.object(guard.os, "close"), self.assertRaises(ProcessLookupError):
            guard.signal_owned_pidfd(identity, signal.SIGTERM, mock.Mock(side_effect=exiting))
        send.assert_not_called()

        calls = []
        guard.terminate_owned(
            (identity,),
            lambda row: guard.owned_identity_alive(row, mock.Mock(side_effect=exiting)),
            grace_seconds=0,
            signaler=lambda row, signum: calls.append((row, signum)),
            monotonic=lambda: 0.0, sleeper=lambda _value: None,
        )
        self.assertEqual(calls, [])

    def test_non_zombie_identity_errors_remain_fail_closed_during_wait_and_verify(self):
        identity = guard.ProcessIdentity(100, 200, (100, 300), 400, (b"worker",))
        with self.assertRaisesRegex(RuntimeError, "argv drift"):
            guard.owned_identity_alive(identity, mock.Mock(side_effect=RuntimeError("argv drift")))
        clock = iter((0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0))
        with self.assertRaisesRegex(RuntimeError, "WAIT pid=100: argv drift.*VERIFY pid=100: argv drift"):
            guard.terminate_owned(
                (identity,), mock.Mock(side_effect=(True, RuntimeError("argv drift"), True,
                                                    RuntimeError("argv drift"), RuntimeError("argv drift"))),
                grace_seconds=.5, signaler=lambda _row, _sig: None,
                monotonic=lambda: next(clock), sleeper=lambda _value: None,
            )

    def test_live_argv_drift_with_same_starttime_fails_closed(self):
        identity = guard.ProcessIdentity(100, 200, (100, 300), 400, (b"worker",))
        drifted = guard.ProcessIdentity(100, 200, (100, 300), 400, (b"other",))
        with self.assertRaisesRegex(RuntimeError, "without exiting"):
            guard.owned_identity_alive(identity, lambda _pid: drifted)
        sent = mock.Mock()
        with mock.patch.object(guard.os, "pidfd_open", return_value=9, create=True), \
             mock.patch.object(guard.signal, "pidfd_send_signal", sent, create=True), \
             mock.patch.object(guard.os, "close"), self.assertRaisesRegex(RuntimeError, "identity changed"):
            guard.signal_owned_pidfd(identity, signal.SIGTERM, lambda _pid: drifted)
        sent.assert_not_called()

    def test_z_and_x_stat_states_map_to_process_lookup_without_cmdline_inference(self):
        def stat_row(state):
            suffix = [state] + [b"1"] * 24
            suffix[2] = b"42"; suffix[19] = b"100"
            return b"42 (dead) " + b" ".join(suffix)
        for state in (b"Z", b"X"):
            with self.subTest(state=state), mock.patch.object(guard, "_read_at", return_value=stat_row(state)) as reader:
                with self.assertRaises(ProcessLookupError):
                    guard._identity_from_open_dir(42, 9)
                reader.assert_called_once_with(9, "stat", 65536)

    def test_live_state_transition_is_allowed_but_transition_to_zombie_is_terminal(self):
        def stat_row(state):
            suffix = [state] + [b"1"] * 24
            suffix[2] = b"42"; suffix[19] = b"100"
            return b"42 (worker) " + b" ".join(suffix)
        argv = b"python3\0worker.py\0"
        live_reads = iter((stat_row(b"S"), b"NSpid:\t42\n", argv, argv, stat_row(b"R")))
        with mock.patch.object(guard, "_read_at", side_effect=lambda *_args: next(live_reads)):
            self.assertEqual(
                guard._identity_from_open_dir(42, 9),
                guard.ProcessIdentity(42, 100, (42,), 42, (b"python3", b"worker.py")),
            )
        zombie_reads = iter((stat_row(b"S"), b"NSpid:\t42\n", argv, argv, stat_row(b"Z")))
        with mock.patch.object(guard, "_read_at", side_effect=lambda *_args: next(zombie_reads)), \
             self.assertRaises(ProcessLookupError):
            guard._identity_from_open_dir(42, 9)

    def test_real_zombie_is_terminal_before_delayed_wait(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            identity = guard.read_process_identity(process.pid)
            guard.terminate_owned((identity,), guard.owned_identity_alive, grace_seconds=1)
            time.sleep(.05)
            self.assertFalse(guard.owned_identity_alive(identity))
            self.assertEqual(process.wait(timeout=2), -signal.SIGTERM)
        finally:
            if process.poll() is None:
                process.kill(); process.wait()

    def test_real_wrapper_and_child_cleanup_reaches_zero_before_reap(self):
        wrapper = subprocess.Popen(
            [sys.executable, "-c", "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print(p.pid,flush=True); time.sleep(30)"],
            stdout=subprocess.PIPE, text=True,
        )
        identities = ()
        try:
            child_pid = int(wrapper.stdout.readline().strip()); wrapper.stdout.close()
            child_stat = Path(f"/proc/{child_pid}/stat").read_bytes().rsplit(b")", 1)[1].split()
            self.assertEqual(int(child_stat[1]), wrapper.pid)
            identities = (guard.read_process_identity(child_pid), guard.read_process_identity(wrapper.pid))
            guard.terminate_owned(identities, guard.owned_identity_alive, grace_seconds=1)
            self.assertEqual([guard.owned_identity_alive(row) for row in identities], [False, False])
            self.assertEqual(wrapper.wait(timeout=2), -signal.SIGTERM)
        finally:
            if wrapper.stdout is not None and not wrapper.stdout.closed:
                wrapper.stdout.close()
            for identity in identities:
                try:
                    if guard.owned_identity_alive(identity):
                        guard.signal_owned_pidfd(identity, signal.SIGKILL)
                except (FileNotFoundError, ProcessLookupError):
                    pass
            if wrapper.poll() is None:
                wrapper.kill()
            wrapper.wait()

    def test_terminate_continues_natural_exit_and_aggregates_true_errors(self):
        rows = tuple(guard.ProcessIdentity(pid, pid + 10, (pid, pid + 20), 50) for pid in (100, 101, 102))
        state = {100: False, 101: True, 102: True}
        calls = []
        def signaler(identity, signum):
            calls.append((identity.host_pid, signum))
            if identity.host_pid == 101 and signum == signal.SIGTERM:
                state[101] = False
            if identity.host_pid == 102:
                raise OSError("denied")
        clock = {"value": 0.0}
        def monotonic():
            clock["value"] += 0.1
            return clock["value"]
        with self.assertRaisesRegex(RuntimeError, "TERM pid=102.*KILL pid=102"):
            guard.terminate_owned(rows, lambda row: state[row.host_pid], grace_seconds=0.05,
                                  signaler=signaler, monotonic=monotonic, sleeper=lambda _v: None)
        self.assertIn((101, signal.SIGTERM), calls)
        self.assertIn((102, signal.SIGTERM), calls)
        self.assertIn((102, signal.SIGKILL), calls)
        self.assertFalse(any(pid == 100 for pid, _signal in calls))

    def test_ownership_group_enumeration_and_grace_validation(self):
        launcher = guard.ProcessIdentity(50, 500, (50,), 50)
        child = guard.ProcessIdentity(51, 501, (51,), 50)
        manifest = {"schema": "step358-launcher-ownership-v1", "port": 29950,
                    "launcher_host_pid": 50, "launcher_starttime": 500, "launcher_pgid": 50}
        with mock.patch.object(guard, "enumerate_group_identities", side_effect=((launcher, child), (launcher, child))), \
             mock.patch.object(guard, "_docker_launcher_argv", return_value=True):
            self.assertEqual(guard._identities_from_ownership(manifest), (launcher, child))
        with mock.patch.object(guard, "enumerate_group_identities", side_effect=((child,), (child,))):
            self.assertEqual(guard._identities_from_ownership(manifest), (child,))
        with mock.patch.object(guard, "enumerate_group_identities", side_effect=((launcher,), (child,))), \
             mock.patch.object(guard, "_docker_launcher_argv", return_value=True):
            self.assertEqual(guard._identities_from_ownership(manifest), (launcher,))
            self.assertEqual(guard._identities_from_ownership(manifest), (child,))
        reused = guard.ProcessIdentity(50, 999, (50,), 50)
        with mock.patch.object(guard, "enumerate_group_identities", side_effect=((reused,), (reused,))), \
             mock.patch.object(guard, "_docker_launcher_argv", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "ownership changed"):
                guard._identities_from_ownership(manifest)
        with mock.patch.object(guard, "enumerate_group_identities", side_effect=((launcher,), (launcher,))), \
             mock.patch.object(guard, "_docker_launcher_argv", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "ownership changed"):
                guard._identities_from_ownership(manifest)
        for value in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                guard.safe_group_cleanup(manifest, grace_seconds=value)
        for value in (0, 1, 2.0):
            with self.assertRaises(ValueError):
                guard.safe_group_cleanup(manifest, max_rounds=value)

    def test_group_enumeration_skips_only_releasing_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory); (proc / "42").mkdir()
            with mock.patch.object(
                guard, "_identity_from_open_dir",
                side_effect=RuntimeError("invalid proc identity fields"),
            ):
                self.assertEqual(guard.enumerate_group_identities(50, proc), ())
                manifest = {"schema": "step358-launcher-ownership-v1", "port": 29950,
                            "launcher_host_pid": 50, "launcher_starttime": 500,
                            "launcher_pgid": 50}
                result = guard.safe_group_cleanup(
                    manifest, proc_root=proc, grace_seconds=0, max_rounds=4,
                    monotonic=lambda: 0.0, sleeper=lambda _value: None,
                )
                self.assertEqual(result["consecutive_empty_group_scans"], 2)
            with mock.patch.object(
                guard, "_identity_from_open_dir", side_effect=RuntimeError("malformed proc stat"),
            ), self.assertRaisesRegex(RuntimeError, "malformed proc stat"):
                guard.enumerate_group_identities(50, proc)

    def test_pgid_reuse_and_unknown_argv_never_gain_signal_authority(self):
        manifest = {"schema": "step358-launcher-ownership-v1", "port": 29950,
                    "launcher_host_pid": 50, "launcher_starttime": 500, "launcher_pgid": 50}
        stranger = guard.ProcessIdentity(70, 700, (70,), 50, (b"unrelated",))
        signals = []
        with mock.patch.object(guard, "enumerate_group_identities", return_value=(stranger,)):
            result = guard.safe_group_cleanup(
                manifest, grace_seconds=0, max_rounds=4,
                signaler=lambda row, signum: signals.append((row, signum)),
                alive=lambda _row: False, monotonic=lambda: 0.0, sleeper=lambda _v: None,
            )
        self.assertEqual(result["member_count"], 0)
        self.assertEqual(signals, [])

        launcher_argv = (
            b"timeout", b"--signal=TERM", b"--kill-after=30s", b"900s", b"docker", b"exec",
            b"-e", b"ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15",
            b"-e", b"ASCEND_CUSTOM_OPP_PATH=/shadow", b"-e", b"PYTHONPATH=/shadow",
            b"-e", b"TORCH_DEVICE_BACKEND_AUTOLOAD=0", b"mapqr-leicheng", b"bash",
            b"--noprofile", b"--norc", b"-lc",
            b"torchrun --nnodes=1 --nproc-per-node=8 --master-addr=127.0.0.1 --master-port=29950 /diag/step377_diagnostic_math_worker.py --input-dir /in --output-dir /out --shadow-root /shadow --installed-custom-opp /opp --first-profiled-only",
        )
        launcher = guard.ProcessIdentity(50, 500, (50,), 50, launcher_argv)
        with mock.patch.object(guard, "enumerate_group_identities", return_value=(launcher, stranger)):
            with self.assertRaisesRegex(RuntimeError, "TERM snapshot"):
                guard.safe_group_cleanup(
                    manifest, signaler=lambda row, signum: signals.append((row, signum)),
                    alive=lambda _row: False,
                )
        self.assertEqual(signals, [])

    def test_safe_group_cleanup_reenumerates_forks_and_requires_two_empty_scans(self):
        manifest = {"schema": "step358-launcher-ownership-v1", "port": 29950,
                    "launcher_host_pid": 50, "launcher_starttime": 500, "launcher_pgid": 50}
        first = guard.ProcessIdentity(50, 500, (50,), 50)
        forked = guard.ProcessIdentity(51, 501, (51,), 50)

        def run(sequence, grace=0.0):
            calls = []
            rows = iter(sequence)
            result = guard.safe_group_cleanup(
                manifest, grace_seconds=grace,
                group_reader=lambda _manifest, _root: next(rows),
                signaler=lambda identity, signum: calls.append((identity.host_pid, signum)),
                alive=lambda _identity: False,
                monotonic=lambda: 0.0, sleeper=lambda _value: None,
            )
            return result, calls

        result, calls = run(((first,), (), ()))
        self.assertEqual(result["consecutive_empty_group_scans"], 2)
        self.assertEqual(calls, [(50, signal.SIGTERM)])

        clock = {"value": -0.2}
        def advancing():
            clock["value"] += 0.2
            return clock["value"]
        rows = iter(((first,), (first, forked), (), (), ()))
        calls = []
        guard.safe_group_cleanup(
            manifest, grace_seconds=1.0, group_reader=lambda _m, _r: next(rows),
            signaler=lambda identity, signum: calls.append((identity.host_pid, signum)),
            alive=lambda _identity: False,
            monotonic=advancing, sleeper=lambda _value: None,
        )
        self.assertEqual([pid for pid, signum in calls if signum == signal.SIGTERM], [50, 51])

        result, calls = run(((first,), (first, forked), (), ()))
        self.assertEqual({pid for pid, signum in calls if signum == signal.SIGKILL}, {50, 51})

        result, calls = run(((), (), ()))
        self.assertEqual((result["member_count"], calls), (0, []))

        sleeps = []
        rows = iter(((first,), (first,), (), ()))
        ticks = iter((0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0))
        guard.safe_group_cleanup(
            manifest, grace_seconds=1.0, group_reader=lambda _m, _r: next(rows),
            signaler=lambda _row, _signal: None, alive=lambda _row: False,
            monotonic=lambda: next(ticks), sleeper=sleeps.append,
        )
        self.assertIn(0.05, sleeps)

    def test_safe_group_cleanup_reports_final_survivor(self):
        manifest = {"schema": "step358-launcher-ownership-v1", "port": 29950,
                    "launcher_host_pid": 50, "launcher_starttime": 500, "launcher_pgid": 50}
        survivor = guard.ProcessIdentity(50, 500, (50,), 50)
        clock = {"value": -0.1}
        def advancing():
            clock["value"] += 0.1
            return clock["value"]
        with self.assertRaisesRegex(RuntimeError, "SURVIVED pid=50"):
            guard.safe_group_cleanup(
                manifest, grace_seconds=0.15, group_reader=lambda _m, _r: (survivor,),
                signaler=lambda _identity, _signum: None,
                alive=lambda _identity: True,
                monotonic=advancing, sleeper=lambda _value: None,
            )

    def test_safe_group_cleanup_reader_error_alternation_and_clock_boundary(self):
        manifest = {"schema": "step358-launcher-ownership-v1", "port": 29950,
                    "launcher_host_pid": 50, "launcher_starttime": 500, "launcher_pgid": 50}
        member = guard.ProcessIdentity(50, 500, (50,), 50)
        with self.assertRaisesRegex(RuntimeError, "TERM snapshot"):
            guard.safe_group_cleanup(manifest, group_reader=lambda _m, _r: (_ for _ in ()).throw(OSError("read")))

        alternating = iter(((), (member,), (), (member,), (), (member,), ()))
        with self.assertRaisesRegex(RuntimeError, "max_rounds"):
            guard.safe_group_cleanup(
                manifest, grace_seconds=0, max_rounds=6,
                group_reader=lambda _m, _r: next(alternating), alive=lambda _row: False,
                signaler=lambda _row, _signal: None, monotonic=lambda: 0.0, sleeper=lambda _value: None,
            )

        rows = iter(((member,), (), ()))
        times = iter((0.0, 0.0, 0.0, 0.0, 0.0))
        result = guard.safe_group_cleanup(
            manifest, grace_seconds=0, max_rounds=4,
            group_reader=lambda _m, _r: next(rows), alive=lambda _row: False,
            signaler=lambda _row, _signal: None, monotonic=lambda: next(times), sleeper=lambda _value: None,
        )
        self.assertEqual(result["consecutive_empty_group_scans"], 2)

    def test_cleanup_cli_protocol_order_and_fail_closed_args(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); ownership = root / "ownership.json"; case = root / "case.py"
            value = {"schema": "step358-launcher-ownership-v1", "port": 29950,
                     "launcher_host_pid": 50, "launcher_starttime": 500, "launcher_pgid": 50}
            ownership.write_text(json.dumps(value)); case.write_text("")
            ownership_sha = hashlib.sha256(ownership.read_bytes()).hexdigest()
            self.assertEqual(guard.read_ownership_json(ownership, ownership_sha), value)
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                guard.read_ownership_json(ownership, "0" * 64)
            calls = []
            with mock.patch.object(guard, "safe_group_cleanup", side_effect=lambda *_a, **_k: calls.append("cleanup") or {"schema": "step377-owned-group-clean-v1"}), \
                 mock.patch.object(guard, "assert_stable_clear", side_effect=lambda *_a, **_k: calls.append("clear") or {"schema": "step377-stable-clear-v1"}), \
                 mock.patch.object(guard, "assert_port_free", side_effect=lambda _p: calls.append("port")), \
                 mock.patch.object(guard, "approved_step377_rank_workers", return_value=()):
                result = guard.cleanup_owned_protocol(ownership, ownership_sha, case, 29950, 0)
            self.assertEqual(calls, ["cleanup", "clear", "port"])
            self.assertEqual(result["schema"], "step377-cleanup-owned-v1")
            link = root / "link.json"; link.symlink_to(ownership)
            with self.assertRaises(RuntimeError):
                guard.read_ownership_json(link, ownership_sha)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            guard.parse_args(["snapshot-idle", "--case-path", "/case", "--port", "1"])

    def test_approved_fallback_skips_unrelated_malformed_proc_but_fails_closed_for_approved(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory); case = proc / "case.py"
            launch = (b"python3", os.fsencode(str(case.resolve())), b"--port", b"29950",
                      b"--output-dir", b"/out", b"--input-dir", b"/in",
                      b"--shadow-root", b"/shadow", b"--installed-custom-opp", b"/opp")
            row = proc / "42"; row.mkdir()
            (row / "cmdline").write_bytes(b"unrelated\0")
            (row / "stat").write_bytes(b"malformed")
            (row / "status").write_bytes(b"malformed")
            self.assertEqual(guard.approved_step377_processes(case, 29950, proc), ())
            (row / "cmdline").write_bytes(b"\0".join(launch) + b"\0")
            with self.assertRaisesRegex(RuntimeError, "malformed proc stat"):
                guard.approved_step377_processes(case, 29950, proc)

    def test_rank_ownership_fixed_identity_cleanup_and_tamper(self):
        case = Path("/case/step377_diagnostic_host_case.py")
        launcher_sha = "a" * 64
        ranks = []
        for rank, identity in enumerate(identities().values()):
            ranks.append({"rank": rank, "local_rank": rank, "host_pid": identity.host_pid,
                          "container_pid": identity.container_pid, "physical": 4 + rank // 2,
                          "chip": rank % 2, "device_id": 8 + rank,
                          "starttime": identity.starttime, "pgid": 700 + rank,
                          "nspid": list(identity.nspid),
                          "argv": [os.fsdecode(value) for value in identity.argv]})
        value = {"schema": "step377-rank-ownership-v1",
                 "launcher_ownership_sha256": launcher_sha, "gate_token_sha256": "b" * 64,
                 "case_path": str(case), "port": 29950, "ranks": ranks}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank.json"
            path.write_text(json.dumps(value))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            _manifest, fixed = guard.read_rank_ownership_json(
                path, digest, expected_launcher_sha256=launcher_sha, case_path=case, port=29950)
            self.assertEqual({row.pgid for row in fixed}, set(range(700, 708)))
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                guard.read_rank_ownership_json(path, "0" * 64,
                    expected_launcher_sha256=launcher_sha, case_path=case, port=29950)
        seen = []
        with mock.patch.object(guard, "terminate_owned", side_effect=lambda rows, *_a, **_k: seen.extend(rows)):
            guard.terminate_owned(fixed, lambda _row: False)
        self.assertEqual(len(fixed), 8)

    def test_launcher_missing_approved_worker_is_never_signaled(self):
        worker = next(iter(identities().values()))
        manifest = {"schema": "step358-launcher-ownership-v1", "port": 29950,
                    "launcher_host_pid": 50, "launcher_starttime": 500, "launcher_pgid": 50}
        signals = []
        with mock.patch.object(guard, "enumerate_group_identities", return_value=(worker,)):
            with self.assertRaisesRegex(RuntimeError, "ownership_unestablished"):
                guard.safe_group_cleanup(manifest, grace_seconds=0, case_path=Path("/case.py"),
                                         signaler=lambda *args: signals.append(args))
        self.assertEqual(signals, [])

    def test_cleanup_protocol_attempts_all_domains_after_rank_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); case = root / "case.py"; case.write_text("")
            ownership = root / "ownership.json"
            ownership.write_text(json.dumps({"schema": "step358-launcher-ownership-v1", "port": 29950,
                "launcher_host_pid": 50, "launcher_starttime": 500, "launcher_pgid": 50}))
            digest = hashlib.sha256(ownership.read_bytes()).hexdigest(); calls = []
            with mock.patch.object(guard, "read_rank_ownership_json", side_effect=RuntimeError("rank bad")), \
                 mock.patch.object(guard, "safe_group_cleanup", side_effect=lambda *_a, **_k: calls.append("launcher") or {}), \
                 mock.patch.object(guard, "assert_stable_clear", side_effect=lambda *_a, **_k: calls.append("clear") or {}), \
                 mock.patch.object(guard, "assert_port_free", side_effect=lambda *_a: calls.append("port")):
                with self.assertRaisesRegex(RuntimeError, "rank_cleanup: rank bad"):
                    guard.cleanup_owned_protocol(ownership, digest, case, 29950, 0,
                                                 rank_ownership_path=root / "rank.json",
                                                 expected_rank_ownership_sha256="b" * 64)
            self.assertEqual(calls, ["launcher", "clear", "port"])

    def test_no_rank_manifest_all_workers_exited_passes_but_live_worker_fails_without_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); case = root / "case.py"; case.write_text("")
            ownership = root / "ownership.json"
            ownership.write_text(json.dumps({"schema": "step358-launcher-ownership-v1", "port": 29950,
                "launcher_host_pid": 50, "launcher_starttime": 500, "launcher_pgid": 50}))
            digest = hashlib.sha256(ownership.read_bytes()).hexdigest()
            calls = []
            patches = (
                mock.patch.object(guard, "safe_group_cleanup", side_effect=lambda *_a, **_k: calls.append("launcher") or {}),
                mock.patch.object(guard, "assert_stable_clear", side_effect=lambda *_a, **_k: calls.append("clear") or {}),
                mock.patch.object(guard, "assert_port_free", side_effect=lambda *_a: calls.append("port")),
            )
            with patches[0], patches[1], patches[2], \
                 mock.patch.object(guard, "approved_step377_rank_workers", return_value=()), \
                 mock.patch.object(guard, "terminate_owned") as terminate:
                result = guard.cleanup_owned_protocol(ownership, digest, case, 29950, 0)
            self.assertEqual(result["rank_cleanup"], None)
            self.assertEqual(calls, ["launcher", "clear", "port"])
            terminate.assert_not_called()

            calls.clear(); base = next(iter(identities().values()))
            live = guard.ProcessIdentity(
                base.host_pid, base.starttime, base.nspid, base.pgid,
                (base.argv[0], b"-u", *base.argv[1:]),
            )
            self.assertTrue(guard._worker_argv(live.argv))
            with patches[0], patches[1], patches[2], \
                 mock.patch.object(guard, "approved_step377_rank_workers", return_value=(live,)), \
                 mock.patch.object(guard, "terminate_owned") as terminate, \
                 self.assertRaisesRegex(RuntimeError, "ownership_unestablished"):
                guard.cleanup_owned_protocol(ownership, digest, case, 29950, 0)
            self.assertEqual(calls, ["launcher", "clear", "port"])
            terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
