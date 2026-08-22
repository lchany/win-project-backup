#!/usr/bin/env python3
"""Offline mock tests for STEP385; no SSH or real build is permitted."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import step385_build_qrv2_delta2_only_remote as controller


def closure_snapshot():
    installed={"root":{"dev":1,"ino":2,"mode":493},"entries":{}}
    opc={"path":"/opc","realpath":"/opc","dev":1,"ino":3,"mode":493,"size":1,"sha256":"a"*64}
    npu={"rows":[],"device_ids":[],"host_pids":[]}
    return {"schema":"step385-closure-v2","installed_samples":[installed,installed],"opc_samples":[opc,opc],"npu_samples":[npu,npu]}


def completion_summary():
    flags={"artifact_class":"diagnostic_probe","diagnostic_only":True,"release_candidate":False,"package_forbidden":True}
    armed=controller._remote_adapter_bytes(controller.read_local_regular(controller.ADAPTER,controller.EXPECTED_INPUTS[controller.ADAPTER.name]))
    tools={"diagnostic_adapter_sha256":controller.sha256_bytes(armed),"audited_adapter_sha256":controller.EXPECTED_INPUTS[controller.AUDITED_ADAPTER.name],"base_builder_sha256":controller.EXPECTED_INPUTS[controller.BASE_BUILDER.name],"step384_patcher_sha256":controller.EXPECTED_INPUTS[controller.PATCHER.name],"v4_patcher_sha256":controller.EXPECTED_INPUTS[controller.V4_PATCHER.name]}
    entries=sorted((controller.CANDIDATE_IDENTITY+"_0_mix_aic",controller.CANDIDATE_IDENTITY+"_0_mix_aiv"))
    artifact={"object_path":"/work/o","object_size":1,"object_sha256":"a"*64,"json_path":"/work/j","json_size":1,"json_sha256":"b"*64,"opc_log_path":"/work/l","opc_log_size":1,"opc_log_sha256":"c"*64,"kernel_name":controller.CANDIDATE_IDENTITY,"bin_file_name":controller.CANDIDATE_IDENTITY,"concrete_entries":entries}
    return {"schema":"step385-summary-v2","status":"diagnostic_built_unvalidated","seal_valid":True,"policy":flags,"candidate":{"identity":controller.CANDIDATE_IDENTITY,"source_sha256":controller.CANDIDATE_SHA256,"reverse_v4_sha256":controller.REVERSE_V4_SHA256,**flags},"package":{"status":"forbidden_diagnostic_probe"},"tools":tools,"artifacts":{soc:dict(artifact) for soc in controller.SOCS},"installed_inventory_closed":True,"runtime_inventory_closed":True,"alias_bytes_equal":True,"forbidden_outputs_absent":True}


def host_container_fixture(container_id="a"*64,hostname="h"):
    cgroup="0::/docker/"+container_id+"\n"
    return {"schema":"step385-host-container-v1","container_id":container_id,"hostname":hostname,"init":{"host_pid":100,"starttime":200,"nspid":[100,1],"pgid":100,"argv":["init"]},"cgroup":cgroup,"cgroup_sha256":hashlib.sha256(cgroup.encode()).hexdigest(),"namespaces":{"pid":11,"mnt":12,"cgroup":13}}


class Step385RemoteControllerTests(unittest.TestCase):
    def test_default_gate_precedes_argparse_helper_and_backend(self) -> None:
        self.assertFalse(controller.BUILD_REMOTE_READY)
        with mock.patch.object(controller.argparse, "ArgumentParser") as parser, mock.patch.object(controller, "load_contract") as load:
            with self.assertRaisesRegex(RuntimeError, "intentionally disabled"):
                controller.main(["--dry-run"])
        parser.assert_not_called()
        load.assert_not_called()
        with mock.patch.object(controller, "load_contract") as load:
            with self.assertRaisesRegex(RuntimeError, "intentionally disabled"):
                controller.execute()
        load.assert_not_called()

    def test_locked_step384_and_dependency_closure(self) -> None:
        self.assertEqual(controller.EXPECTED_INPUTS[controller.ADAPTER.name], "c00af6a2b455c93b35c81e5133af905f460fc0ce61846470f6bb3821509e7083")
        self.assertEqual(controller.EXPECTED_INPUTS[controller.ADAPTER_TEST.name], "7258367c4173910d218f207c45a6a99973c542172b0840c63815de6019713357")
        self.assertEqual({p.name for p in controller.input_files()}, set(controller.EXPECTED_INPUTS))
        with mock.patch.object(controller, "BUILD_REMOTE_READY", True):
            controller._require_remote_ready()

    def test_step375_locked_closure_rejects_missing_tamper_and_symlink(self) -> None:
        self.assertEqual(controller.EXPECTED_INPUTS[controller.STEP375_PATCHER.name],"98a655f89ac5efedd760067fdda595d9b5fe376b1e51fdc1b12d59c727711768")
        self.assertIn(controller.STEP375_PATCHER,controller.input_files())
        missing=tuple(path for path in controller.input_files() if path!=controller.STEP375_PATCHER)
        with mock.patch.object(controller,"BUILD_REMOTE_READY",True), mock.patch.object(controller,"input_files",return_value=missing), self.assertRaisesRegex(RuntimeError,"upload closure mismatch"):
            controller._require_remote_ready()
        with self.assertRaisesRegex(RuntimeError,"SHA mismatch"):
            controller.read_local_regular(controller.STEP375_PATCHER,"0"*64)
        with tempfile.TemporaryDirectory() as directory:
            link=Path(directory)/controller.STEP375_PATCHER.name; link.symlink_to(controller.STEP375_PATCHER)
            with self.assertRaises(OSError):
                controller.read_local_regular(link,controller.EXPECTED_INPUTS[controller.STEP375_PATCHER.name])

    def test_isolated_staging_imports_step375_and_v4_from_exact_uploaded_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            staging=Path(directory)
            for source in (controller.AUDITED_ADAPTER,controller.STEP375_PATCHER,controller.V4_PATCHER):
                (staging/source.name).write_bytes(controller.read_local_regular(source,controller.EXPECTED_INPUTS[source.name]))
            code="""import hashlib,json\nfrom pathlib import Path\nimport build_qrv2_diagnostic_probe as audited\nstep375=Path(audited.diagnostic_patcher.__file__).resolve(); v4=Path(audited.diagnostic_patcher.release_v4.__file__).resolve()\nprint(json.dumps({'step375_path':str(step375),'step375_sha':hashlib.sha256(step375.read_bytes()).hexdigest(),'v4_path':str(v4),'v4_sha':hashlib.sha256(v4.read_bytes()).hexdigest()}))\n"""
            env=dict(os.environ); env.pop("PYTHONPATH",None)
            result=subprocess.run([sys.executable,"-c",code],cwd=staging,env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr); value=json.loads(result.stdout)
            self.assertEqual(value,{"step375_path":str((staging/controller.STEP375_PATCHER.name).resolve()),"step375_sha":controller.EXPECTED_INPUTS[controller.STEP375_PATCHER.name],"v4_path":str((staging/controller.V4_PATCHER.name).resolve()),"v4_sha":controller.EXPECTED_INPUTS[controller.V4_PATCHER.name]})

    def test_hash_drift_fails_closed_before_contract_import(self) -> None:
        with mock.patch.object(controller, "BUILD_REMOTE_READY", True), mock.patch.dict(controller.EXPECTED_INPUTS, {controller.PATCHER.name: "0" * 64}):
            with mock.patch.object(controller.importlib.util, "spec_from_file_location") as spec, self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                controller.load_contract()
        spec.assert_not_called()

    def test_locked_step376_factory_returns_transport_contract(self) -> None:
        with mock.patch.object(controller, "BUILD_REMOTE_READY", True):
            helper = controller.load_contract()
        self.assertTrue(callable(helper.load_remote_module))
        self.assertTrue(callable(helper.local_preflight))
        self.assertTrue(callable(helper.connect_target))

    def test_remote_adapter_copy_has_only_minimal_readiness_flip(self) -> None:
        original = controller.ADAPTER.read_bytes()
        remote = controller._remote_adapter_bytes(original)
        self.assertEqual(original.count(b"BUILD_READY = False"), 1)
        self.assertEqual(remote.count(b"BUILD_READY = True"), 1)
        self.assertEqual(remote.replace(b"BUILD_READY = True", b"BUILD_READY = False", 1), original)
        self.assertEqual(controller.sha256_file(controller.ADAPTER), controller.EXPECTED_INPUTS[controller.ADAPTER.name])

    def test_attempt_is_unique_and_never_latest(self) -> None:
        self.assertTrue(controller.ATTEMPT_NAME.startswith("step385_attempt"))
        self.assertNotIn("latest", controller.ATTEMPT_NAME.lower())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new attempt"
            script = controller._exclusive_attempt_script(str(target))
            self.assertEqual(subprocess.run(["bash", "-n"], input=script, text=True).returncode, 0)
            self.assertEqual(subprocess.run(["bash", "-c", script]).returncode, 0)
            self.assertEqual(subprocess.run(["bash", "-c", script]).returncode, 73)

    def test_dry_run_never_constructs_backend(self) -> None:
        with mock.patch.object(controller, "BUILD_REMOTE_READY", True), mock.patch.object(controller, "load_contract") as load:
            payload = controller._dry_run_payload()
        load.assert_not_called()
        self.assertEqual(payload["actions"], list(controller.DRY_RUN_ACTIONS))
        self.assertEqual(payload["forbidden"], list(controller.FORBIDDEN_ACTIONS))
        self.assertEqual(payload["socs"], list(controller.SOCS))

    def test_container_script_is_prepare_build_only_and_uses_contract_opc(self) -> None:
        contract = {"ascend_opp": "/opt/opp", "installed_cloud_root": "/installed", "opc": {"path": "/locked/opc", "sha256":"a"*64}}
        script = controller._container_script(types.SimpleNamespace(), contract, "/shared/diagnostics/attempt")
        commands = [line for line in script.splitlines() if line.startswith("python3 ")]
        self.assertEqual(len(commands), 2)
        self.assertIn(" prepare ", commands[0])
        self.assertIn(" build ", commands[1])
        self.assertIn("--opc /locked/opc", commands[1])
        for forbidden in controller.FORBIDDEN_ACTIONS:
            self.assertNotIn(" " + forbidden + " ", script)

    def test_opc_validation_rejects_alias_shapes(self) -> None:
        for value in (None, "", "relative/opc", "/bad\0opc", 1):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                controller._validated_opc({"opc": {"path": value, "sha256":"a"*64}})

    def test_upload_gate_requires_exact_names_hashes_and_readback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").write_bytes(b"payload")
            (root / "a").chmod(0o600)
            expected = {"a": hashlib.sha256(b"payload").hexdigest()}
            command = ["python3", "-c", controller._upload_gate_code(), str(root), json.dumps(expected)]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            gate=json.loads(result.stdout)
            self.assertEqual(gate["schema"], "step385-staging-gate-v1")
            self.assertEqual(subprocess.run(command+[json.dumps(gate["root"])],text=True,capture_output=True).returncode,0)
            (root / "extra").write_bytes(b"x")
            self.assertNotEqual(subprocess.run(command, text=True, capture_output=True).returncode, 0)

    def test_summary_requires_completion_seal_and_both_socs(self) -> None:
        code = controller._summary_code()
        self.assertIn("_require_completed_attempt", code)
        self.assertIn("_validate_built_artifact_closure", code)
        self.assertIn("ascend910_93", code)
        self.assertIn("ascend910b", code)

    def test_summary_run_accepts_exact_gate_then_summary(self) -> None:
        gate={"schema":"step385-staging-gate-v1","root":{"dev":1,"ino":2,"mode":448}}; summary=completion_summary(); adapter_sha=summary["tools"]["diagnostic_adapter_sha256"]; helper=mock.Mock(); helper.run.return_value=(json.dumps(gate)+"\n"+json.dumps(summary)+"\n","")
        self.assertEqual(controller._summary_run(helper,object(),"summary",gate,adapter_sha),summary)
        helper.run.assert_called_once_with(mock.ANY,"summary",timeout=controller.REMOTE_SHORT_TIMEOUT)

    def test_summary_run_rejects_all_line_protocol_drifts_and_execute_uses_it(self) -> None:
        gate={"schema":"step385-staging-gate-v1","root":{"dev":1,"ino":2,"mode":448}}; summary=completion_summary(); adapter_sha=summary["tools"]["diagnostic_adapter_sha256"]; g=json.dumps(gate); s=json.dumps(summary)
        bad_outputs=("",g+"\n",g+"\n"+g+"\n",g+"\n"+s+"\n"+s+"\n",json.dumps({**gate,"root":{"dev":9,"ino":2,"mode":448}})+"\n"+s+"\n","not-json\n"+s+"\n",g+"\nnot-json\n",g+"\n\n"+s+"\n")
        for output in bad_outputs:
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                helper=mock.Mock(); helper.run.return_value=(output,""); controller._summary_run(helper,object(),"summary",gate,adapter_sha)
        source=inspect.getsource(controller.execute); tail=source[source.index("summary_command ="):]
        self.assertIn("_summary_run(helper, target, summary_command, gate, adapter_sha256)",tail)
        self.assertNotIn("_json_run(helper, target, summary_command)",tail)

    def test_snapshot_covers_installed_process_and_opc(self) -> None:
        code = controller._snapshot_code()
        self.assertIn("installed", code)
        self.assertIn("npu_samples", code)
        self.assertIn("opc_samples", code)
        self.assertIn("opc", code)
        self.assertIn("special installed entry", code)

    def test_step387_receipt_is_same_child_terminal_success_gate(self) -> None:
        receipt=controller._receipt_code(); wrapper=controller._wrapper_code().decode(); source=inspect.getsource(controller.execute)
        for fragment in ("_validate_manifest(","_require_completed_attempt(","_validate_built_artifact_closure(","m._assert_no_release_outputs(","os.O_EXCL","build_receipt.json","ascend910_93","ascend910b","/proc/self/ns/mnt"):
            self.assertIn(fragment,receipt)
        self.assertIn('_container_script(helper, contract, remote_attempt, "build") + "\\n" + phase_gate + "\\n" + receipt_command',source)
        self.assertIn("if rc==0:",wrapper); self.assertIn("build_receipt.json",wrapper); self.assertIn("step385-build-result-v3",wrapper)

    def test_result_summary_and_postflight_revalidate_receipt_identity(self) -> None:
        waiter=controller._result_wait_code(); summary=controller._summary_code(); snapshot=controller._snapshot_code()
        for fragment in ("docker','inspect", "init_starttime", "mnt_ns", "attempt/'work'", "receipt['artifacts']", "read_state(Path(bound['path']))[1]==bound"):
            self.assertIn(fragment,waiter)
        self.assertIn("expected_receipt",snapshot); self.assertIn("attempt/'build_receipt.json'",snapshot); self.assertIn("work_state=work.lstat()",snapshot)
        self.assertIn("receipt_state",summary); self.assertIn("/proc/self/ns/mnt",summary); self.assertIn("regular_state(Path(bound['path']))[1]==bound",summary)

    def test_rc_zero_without_exact_receipt_result_is_rejected_before_postflight_success(self) -> None:
        helper=mock.Mock(); zero={"host_client":[],"container_wrapper":[],"descendant_opc":[]}
        helper.run.side_effect=[(json.dumps({"schema":"step385-build-result-v3","returncode":0}),""),(json.dumps(zero),""),(json.dumps(zero),""),(json.dumps(zero),""),(json.dumps({"closed":True}),"")]
        with self.assertRaisesRegex(RuntimeError,"receipt result mismatch"):
            controller._run_build_transaction(helper,object(),"build","snapshot",{"closed":True},"owned","cleanup")

    def test_result_waiter_rejects_receipt_and_container_toctou(self) -> None:
        def run_case(mutation):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/"attempt"; work=root/"work"; bindir=Path(directory)/"bin"; work.mkdir(parents=True); bindir.mkdir()
                paths={name:work/name for name in ("manifest","completion","object-a","json-a","log-a","object-b","json-b","log-b")}
                for name,path in paths.items(): path.write_text(name)
                def state(path):
                    value=path.stat(); data=path.read_bytes(); return {"path":str(path),"dev":value.st_dev,"ino":value.st_ino,"mode":value.st_mode&0o777,"size":value.st_size,"sha256":hashlib.sha256(data).hexdigest()}
                current_pid=os.getpid(); fields=Path(f"/proc/{current_pid}/stat").read_text().rsplit(")",1)[1].split(); start=int(fields[19]); mnt=os.stat(f"/proc/{current_pid}/ns/mnt").st_ino; cid="a"*64
                artifacts={"ascend910_93":{"object_path":state(paths["object-a"]),"json_path":state(paths["json-a"]),"opc_log_path":state(paths["log-a"])},"ascend910b":{"object_path":state(paths["object-b"]),"json_path":state(paths["json-b"]),"opc_log_path":state(paths["log-b"])}}
                root_stat=root.stat(); work_stat=work.stat(); receipt={"schema":"step387-build-receipt-v1","container":{"id":cid,"init_starttime":start,"mnt_ns":mnt},"attempt":{"path":str(root),"dev":root_stat.st_dev,"ino":root_stat.st_ino},"work":{"path":str(work),"dev":work_stat.st_dev,"ino":work_stat.st_ino},"manifest":state(paths["manifest"]),"completion":state(paths["completion"]),"artifacts":artifacts}
                receipt_path=root/"build_receipt.json"; receipt_path.write_text(json.dumps(receipt,sort_keys=True)+'\n'); receipt_path.chmod(0o600); receipt_state=state(receipt_path); expected={k:receipt_state[k] for k in ("dev","ino","mode","size","sha256")}; logs={}
                for name in ("stdout","stderr"):
                    path=root/("child_"+name+".log"); path.write_text(""); path.chmod(0o600); logs[name]=state(path)
                result=root/"build_result.json"; result.write_text(json.dumps({"schema":"step385-build-result-v3","returncode":0,"logs":logs,"receipt":expected})+'\n'); result.chmod(0o600)
                inspect_pid=1 if mutation=="container_restart" else current_pid; docker=bindir/"docker"; docker.write_text("#!/usr/bin/env python3\nimport json\nprint(json.dumps([{'Id':"+repr(cid)+",'State':{'Pid':"+str(inspect_pid)+"}}]))\n"); docker.chmod(0o755)
                passed_mnt=mnt
                if mutation=="false_rc0": result.write_text(json.dumps({"schema":"step385-build-result-v3","returncode":0})+'\n')
                elif mutation=="rc1":
                    (root/"child_stdout.log").write_text("private diagnostic body"); logs["stdout"]=state(root/"child_stdout.log"); result.write_text(json.dumps({"schema":"step385-build-result-v3","returncode":1,"logs":logs})+'\n')
                elif mutation=="receipt_tamper": receipt_path.write_text(receipt_path.read_text()+" ")
                elif mutation=="manifest_missing": paths["manifest"].unlink()
                elif mutation=="work_missing": work.rename(root/"gone")
                elif mutation=="artifact_replace": paths["object-a"].write_text("replacement")
                elif mutation=="log_replace": (root/"child_stderr.log").write_text("replacement")
                elif mutation=="log_oversize": (root/"child_stdout.log").write_bytes(b"x"*(1048576+1))
                elif mutation=="mnt_drift": passed_mnt+=1
                env={**os.environ,"PATH":str(bindir)+os.pathsep+os.environ["PATH"]}
                return subprocess.run([sys.executable,"-c",controller._result_wait_code(),str(result),"1",str(root),"box",cid,str(start),str(passed_mnt)],text=True,capture_output=True,env=env)
        valid=run_case("valid"); self.assertEqual(valid.returncode,0,valid.stderr); self.assertEqual(json.loads(valid.stdout)["returncode"],0)
        failed=run_case("rc1"); self.assertEqual(failed.returncode,0,failed.stderr); failed_value=json.loads(failed.stdout); self.assertEqual(failed_value["returncode"],1); self.assertEqual(set(failed_value["logs"]),{"stdout","stderr"}); self.assertNotIn("private diagnostic body",failed.stdout)
        for mutation in ("false_rc0","receipt_tamper","manifest_missing","work_missing","artifact_replace","log_replace","log_oversize","mnt_drift","container_restart"):
            with self.subTest(mutation=mutation): self.assertNotEqual(run_case(mutation).returncode,0)

    def test_rc1_evidence_still_runs_cleanup_and_postflight(self) -> None:
        helper=mock.Mock(); zero={"host_client":[],"container_wrapper":[],"descendant_opc":[]}; before={"closed":True}; log=lambda name:{"path":"/attempt/child_"+name+".log","dev":1,"ino":2 if name=="stdout" else 3,"mode":384,"size":4,"sha256":"a"*64}; failed={"schema":"step385-build-result-v3","returncode":1,"logs":{"stdout":log("stdout"),"stderr":log("stderr")}}
        helper.run.side_effect=[(json.dumps(failed),""),(json.dumps(zero),""),(json.dumps(zero),""),(json.dumps(zero),""),(json.dumps(before),"")]
        with self.assertRaisesRegex(RuntimeError,"STEP389 build child failed rc=1 logs=child_stdout.log,child_stderr.log"):
            controller._run_build_transaction(helper,object(),"build","snapshot",before,"owned","cleanup")
        self.assertEqual(helper.run.call_count,5); self.assertEqual(helper.run.call_args_list[-1].args[1],"snapshot")
        wrapper=controller._wrapper_code().decode(); self.assertLess(wrapper.index("os.O_EXCL",wrapper.index("stdout_path=")),wrapper.index("child=subprocess.Popen")); self.assertIn("stdout=stdout_fd,stderr=stderr_fd",wrapper); self.assertIn("<=1048576",wrapper)

    def test_failed_build_still_runs_postflight(self) -> None:
        helper = mock.Mock()
        helper.run.side_effect = [RuntimeError("build failed"), ("[]", ""), (json.dumps({"closed": True}), "")]
        with self.assertRaisesRegex(RuntimeError, "build failed"):
            controller._run_build_transaction(helper, object(), "build", "snapshot", {"closed": True}, "owned", "cleanup")
        self.assertEqual(helper.run.call_count, 3)
        self.assertEqual(helper.run.call_args_list[-1].args[1], "snapshot")

    def test_empty_build_stdout_is_deterministic_error_but_still_closes(self) -> None:
        helper=mock.Mock(); zero={"host_client":[],"container_wrapper":[],"descendant_opc":[]}; before={"closed":True}
        helper.run.side_effect=[("  \n", ""),(json.dumps(zero),""),(json.dumps(zero),""),(json.dumps(zero),""),(json.dumps(before),"")]
        with self.assertRaisesRegex(RuntimeError,"STEP388 build wait produced empty stdout"):
            controller._run_build_transaction(helper,object(),"build","snapshot",before,"owned","cleanup")
        self.assertEqual(helper.run.call_count,5); self.assertEqual(helper.run.call_args_list[-1].args[1],"snapshot")

    def test_host_wait_client_uses_setsid_wait_and_propagates_child_status(self) -> None:
        source=inspect.getsource(controller.execute)
        self.assertIn('build_command = "setsid --wait bash ',source)
        self.assertNotIn('build_command = "setsid bash ',source)
        self.assertIn('expected_client_argv = shlex.split(build_command)',source)
        self.assertNotIn('shlex.split(build_command)[1:]',source)
        result=subprocess.run(["setsid","--wait","bash","--noprofile","--norc","-lc","sleep 0.05; printf sealed; exit 7"],text=True,capture_output=True)
        self.assertEqual(result.stdout,"sealed"); self.assertEqual(result.returncode,7)

    def test_execute_mock_uses_exact_upload_and_safe_cleanup_wiring(self) -> None:
        class SFTP:
            def close(self): pass
        target = mock.Mock()
        target.open_sftp.return_value = SFTP()
        jump = mock.Mock()
        helper = mock.Mock(CONTAINER="mapqr-leicheng", EXPECTED_HOSTNAME="target")
        helper.load_remote_module.return_value = object()
        helper.local_preflight.return_value = {"shared": "/shared"}
        helper.connect_target.return_value = (jump, target)
        helper.safe_remote_path.return_value = "/shared/diagnostics/" + controller.ATTEMPT_NAME
        helper.container_probe.return_value = {"schema_version":1,"container_name":"mapqr-leicheng","inspect_container_id":"a"*64,"inspect_hostname":"h","opc":{"path":"/opc","sha256":"a"*64},"cann_version_files":[{"path":"/version.info","sha256":"b"*64}],"ascend_opp":"/opp","installed_cloud_root":"/installed"}
        snapshots = closure_snapshot()
        summary = completion_summary()
        gate={"schema":"step385-staging-gate-v1","root":{"dev":1,"ino":2,"mode":448}}
        empty_owned={"host_client":[],"container_wrapper":[],"descendant_opc":[]}
        live_owned={"host_client":[],"container_wrapper":[9],"descendant_opc":[]}
        clean={"schema":"step385-owned-clean-v1","remaining":0}
        state={"precommit":{"dev":1,"ino":2,"mode":384,"sha256":"d"*64},"start_decision":{"dev":1,"ino":3,"mode":384,"sha256":"e"*64}}
        commit={"schema":"step385-ownership-commit-v3","committed":True,"commit_state":state}
        receipt={"dev":1,"ino":4,"mode":384,"size":10,"sha256":"f"*64}; log=lambda name:{"path":"/shared/diagnostics/"+controller.ATTEMPT_NAME+"/child_"+name+".log","dev":1,"ino":5 if name=="stdout" else 6,"mode":384,"size":0,"sha256":"0"*64}; build_result={"schema":"step385-build-result-v3","returncode":0,"logs":{"stdout":log("stdout"),"stderr":log("stderr")},"receipt":receipt}
        helper.run.side_effect = [("target\n", ""), (json.dumps(gate), ""), (json.dumps(host_container_fixture()), ""), (json.dumps(snapshots), ""), (json.dumps(empty_owned), ""), ("", ""), (json.dumps(commit), ""), (json.dumps(build_result), ""), (json.dumps(live_owned), ""), (json.dumps(clean), ""), (json.dumps(empty_owned), ""), (json.dumps(empty_owned), ""), (json.dumps(snapshots), ""), (json.dumps(summary), "")]
        uploaded = []
        helper.write_remote_new.side_effect = lambda _sftp, path, data: uploaded.append((path, data))
        with mock.patch.object(controller, "BUILD_REMOTE_READY", True), mock.patch.object(controller, "load_contract", return_value=helper):
            result = controller.execute()
        self.assertTrue(result["uploaded_readback"])
        self.assertEqual(len(uploaded), len(controller.EXPECTED_INPUTS) + 2)
        adapter_payload = dict((Path(path).name, data) for path, data in uploaded)[controller.ADAPTER.name]
        self.assertIn(b"BUILD_READY = True", adapter_payload)
        all_commands = "\n".join(call.args[1] for call in helper.run.call_args_list)
        self.assertNotIn("rm ", all_commands)
        self.assertNotIn("pkill", all_commands)

    def test_contract_executes_supplied_locked_bytes_without_contract_path_loader(self) -> None:
        source=b"BUILD_READY=False\ndef load_legacy(): return 'private-bytes'\n"
        with mock.patch.object(controller,"_require_remote_ready"), mock.patch.object(controller,"CONTRACT_SHA256",hashlib.sha256(source).hexdigest()), mock.patch.object(controller.importlib.util,"spec_from_file_location") as loader:
            self.assertEqual(controller.load_contract(source),"private-bytes")
        loader.assert_not_called()

    def test_gate_rejects_root_replacement_and_mode_toctou(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory); root=parent/"staging"; root.mkdir(); item=root/"a"; item.write_bytes(b"x"); item.chmod(0o600)
            expected={"a":hashlib.sha256(b"x").hexdigest()}; base=["python3","-c",controller._upload_gate_code(),str(root),json.dumps(expected)]
            first=subprocess.run(base,text=True,capture_output=True); self.assertEqual(first.returncode,0,first.stderr); locked=json.loads(first.stdout)["root"]
            old=parent/"old"; root.rename(old); root.mkdir(); replacement=root/"a"; replacement.write_bytes(b"x"); replacement.chmod(0o600)
            self.assertNotEqual(subprocess.run(base+[json.dumps(locked)],text=True,capture_output=True).returncode,0)
            replacement.chmod(0o644)
            self.assertNotEqual(subprocess.run(base,text=True,capture_output=True).returncode,0)

    def test_strict_wrapper_and_summary_order_are_wired(self) -> None:
        wrapper=controller._wrapper_code().decode(); owned=controller._owned_code(); summary=controller._summary_code()
        self.assertIn("O_EXCL",wrapper); self.assertIn("write_all",wrapper); self.assertIn("os.fsync(parent)",wrapper); self.assertIn("NSpid:",wrapper); self.assertNotIn("start_new_session=True",wrapper)
        self.assertIn("ownership_host_lock.json",owned); self.assertIn("g.terminate_owned",owned); self.assertNotIn("pidfd_open",owned); self.assertNotIn("pidfd_send_signal",owned); self.assertNotIn("os.kill",owned); self.assertNotIn("Path(os.fsdecode(x)).name",owned)
        order=[summary.index(fragment) for fragment in ("m._validate_manifest(","m._require_completed_attempt(","m._validate_built_artifact_closure(","m._assert_no_release_outputs(")]
        self.assertEqual(order,sorted(order))

    def test_sftp_primary_error_keeps_close_error_secondary(self) -> None:
        primary=RuntimeError("upload"); close=OSError("close")
        controller._append_secondary(primary,"STEP385 SFTP close failed",close)
        notes=getattr(primary,"__notes__",getattr(primary,"cleanup_errors",()))
        self.assertTrue(any("SFTP close failed" in note for note in notes))

    def test_guard_dynamic_import_runs_in_real_python39_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); staging=root/"staging"; staging.mkdir()
            guard=staging/controller.PROCESS_GUARD.name; guard.write_bytes(controller.PROCESS_GUARD.read_bytes())
            code=controller._owned_code().replace("def allproc():","print('guard_import=PASS'); raise SystemExit(0)\ndef allproc():",1)
            result=subprocess.run([sys.executable,"-c",code,str(root),"[]","t","c","h",json.dumps(host_container_fixture("c","h")),"snapshot"],text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.strip(),"guard_import=PASS")
            self.assertIn("sys.modules[guard_name]=g",controller._snapshot_code())
            self.assertIn("del sys.modules[guard_name]",controller._snapshot_code())

    def test_wrapper_persists_regular_files_and_reaps_before_signal_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); manifest=root/"ownership.json"
            receipt=root/"build_receipt.json"; child_command="sleep 0.1; printf '{\"schema\":\"step387-build-receipt-v1\"}\\n' > "+str(receipt)+"; chmod 600 "+str(receipt)
            process=subprocess.Popen([sys.executable,"-c",controller._wrapper_code().decode(),str(manifest),"t","c",os.uname().nodename,child_command])
            deadline=time.monotonic()+5
            while not manifest.exists() and time.monotonic()<deadline: time.sleep(.02)
            ownership=json.loads(manifest.read_text()); lock=root/"ownership_host_lock.json"; lock.write_text('{"sealed":true}\n'); lock.chmod(0o600); identity=lock.stat(); lock_state={"dev":identity.st_dev,"ino":identity.st_ino,"mode":384,"sha256":hashlib.sha256(lock.read_bytes()).hexdigest()}; ack_value={"schema":"step385-start-decision-v1","status":"committed","token":"t","wrapper_nonce":ownership["nonce"],"host_lock_sha256":lock_state["sha256"],"host_lock_dev":identity.st_dev,"host_lock_ino":identity.st_ino}; ack_payload=(json.dumps(ack_value,sort_keys=True)+'\n').encode(); precommit=root/"ownership_precommit_seal.json"; precommit.write_text(json.dumps({"schema":"step385-precommit-seal-v1","token":"t","wrapper_nonce":ownership["nonce"],"lock":lock_state,"start_decision_sha256":hashlib.sha256(ack_payload).hexdigest()},sort_keys=True)+'\n'); precommit.chmod(0o600); ack=root/"ownership_start_decision.json"; ack.write_bytes(ack_payload); ack.chmod(0o600)
            result=root/"build_result.json"
            while not result.exists() and time.monotonic()<deadline: time.sleep(.02)
            self.assertTrue(result.exists()); self.assertIsNone(process.poll())
            for path in (manifest,result,receipt,lock,precommit,ack):
                self.assertTrue(path.is_file()); self.assertFalse(path.is_symlink()); self.assertEqual(path.stat().st_mode & 0o777,0o600)
            value=json.loads(result.read_text()); self.assertEqual(value["schema"],"step385-build-result-v3"); self.assertEqual(value["receipt"]["sha256"],hashlib.sha256(receipt.read_bytes()).hexdigest())
            process.terminate(); self.assertEqual(process.wait(timeout=5),0)

    def test_wrapper_term_while_child_live_does_not_abandon_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); manifest=root/"ownership.json"; result=root/"build_result.json"
            process=subprocess.Popen([sys.executable,"-c",controller._wrapper_code().decode(),str(manifest),"t","c",os.uname().nodename,"sleep 0.4"])
            deadline=time.monotonic()+5
            while not manifest.exists() and time.monotonic()<deadline: time.sleep(.01)
            self.assertTrue(manifest.exists()); process.terminate()
            self.assertEqual(process.wait(timeout=5),0); self.assertFalse(result.exists())

    def test_ack_forgery_and_host_lock_seal_tamper_never_spawn_child(self) -> None:
        for mutation in ("nonce","sha","inode"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); manifest=root/"ownership.json"; marker=root/"child-ran"
                process=subprocess.Popen([sys.executable,"-c",controller._wrapper_code().decode(),str(manifest),"t","c",os.uname().nodename,"touch "+str(marker)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                deadline=time.monotonic()+5
                while not manifest.exists() and time.monotonic()<deadline: time.sleep(.01)
                ownership=json.loads(manifest.read_text()); lock=root/"ownership_host_lock.json"; lock.write_text('{"sealed":true}\n'); lock.chmod(0o600); identity=lock.stat(); lock_state={"dev":identity.st_dev,"ino":identity.st_ino,"mode":384,"sha256":hashlib.sha256(lock.read_bytes()).hexdigest()}; ack={"schema":"step385-start-decision-v1","status":"committed","token":"t","wrapper_nonce":ownership["nonce"],"host_lock_sha256":lock_state["sha256"],"host_lock_dev":identity.st_dev,"host_lock_ino":identity.st_ino}
                if mutation=="nonce": ack["wrapper_nonce"]="0"*64
                elif mutation=="sha": ack["host_lock_sha256"]="0"*64
                else: ack["host_lock_ino"]+=1
                ack_payload=(json.dumps(ack,sort_keys=True)+'\n').encode(); precommit=root/"ownership_precommit_seal.json"; precommit.write_text(json.dumps({"schema":"step385-precommit-seal-v1","token":"t","wrapper_nonce":ownership["nonce"],"lock":lock_state,"start_decision_sha256":hashlib.sha256(ack_payload).hexdigest()},sort_keys=True)+'\n'); precommit.chmod(0o600); path=root/"ownership_start_decision.json"; path.write_bytes(ack_payload); path.chmod(0o600)
                _stdout,_stderr=process.communicate(timeout=5); self.assertNotEqual(process.returncode,0); self.assertFalse(marker.exists())

    def test_two_stage_commit_precedes_result_wait_and_requires_sets_id(self) -> None:
        source=inspect.getsource(controller.execute)
        self.assertIn('container_body = phase_gate + " && exec setsid "',source)
        self.assertLess(source.index("helper.run(target, detached"),source.index("_owned_commit_run"))
        self.assertLess(source.index("_owned_commit_run"),source.index("_run_build_transaction"))
        owned=controller._owned_code()
        self.assertIn("host_sid==wrapper==item.host_pid==item.pgid",owned)
        self.assertIn("pre-ACK SID domain is not exclusive",owned)

    def test_host_lock_precommit_start_decision_chain_is_revalidated_on_every_recovery(self) -> None:
        owned=controller._owned_code()
        self.assertIn("read_regular_state(root/'ownership_host_lock.json')",owned)
        self.assertIn("read_regular_state(root/'ownership_start_decision.json')",owned)
        self.assertIn("ownership_precommit_seal.json",owned)
        self.assertIn("precommit['lock']==lock_state",owned)
        self.assertIn("(lock_state['dev'],lock_state['ino'],lock_state['sha256'])",owned)
        self.assertGreaterEqual(owned.count("verify_commit_chain(data)"),2)

    def test_cleanup_passes_only_remaining_deadline_to_guard(self) -> None:
        owned=controller._owned_code()
        self.assertIn("remaining=deadline-time.monotonic(); assert remaining>0",owned)
        self.assertIn("grace=min(1.0,remaining/2)",owned)
        self.assertIn("assert time.monotonic()<=deadline",owned)

    def test_commit_result_seal_schema_is_exact(self) -> None:
        item={"dev":1,"ino":2,"mode":384,"sha256":"a"*64}; valid={"schema":"step385-ownership-commit-v3","committed":True,"commit_state":{"precommit":dict(item),"start_decision":dict(item)}}; helper=mock.Mock(); helper.run.return_value=(json.dumps(valid),"")
        self.assertEqual(controller._owned_commit_run(helper,object(),"commit"),valid)
        for value in ({**valid,"extra":1},{**valid,"commit_state":{**valid["commit_state"],"start_decision":{**item,"ino":0}}},{**valid,"commit_state":{**valid["commit_state"],"extra":dict(item)}}):
            helper.run.return_value=(json.dumps(value),"")
            with self.assertRaises(RuntimeError): controller._owned_commit_run(helper,object(),"commit")

    def test_start_decision_is_final_atomic_commit_point_and_recovery_is_exact(self) -> None:
        owned=controller._owned_code()
        self.assertIn("def publish_terminal_decision",owned)
        self.assertIn("os.link(temp,path.name",owned)
        self.assertIn("committed=True",owned)
        self.assertLess(owned.index("committed=True"),owned.index("return state",owned.index("committed=True")))
        self.assertLess(owned.index("persist_new(root/'ownership_precommit_seal.json'"),owned.index("publish_terminal_decision(root/'ownership_start_decision.json'"))
        state={"precommit":{"dev":1,"ino":2,"mode":384,"sha256":"a"*64},"start_decision":{"dev":1,"ino":3,"mode":384,"sha256":"b"*64}}; helper=mock.Mock(); committed={"schema":"step385-ownership-recovery-v2","status":"committed","commit_state":state}
        helper.run.return_value=(json.dumps(committed),""); self.assertEqual(controller._owned_recover_commit_run(helper,object(),"recover"),committed)
        helper.run.return_value=(json.dumps({"schema":"step385-ownership-recovery-v2","status":"aborted"}),""); self.assertEqual(controller._owned_recover_commit_run(helper,object(),"recover")["status"],"aborted")

    def test_terminal_decision_concurrency_has_exactly_one_winner(self) -> None:
        tree=ast.parse(controller._owned_code()); function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="publish_terminal_decision"); module=ast.Module(body=[function],type_ignores=[]); namespace={"os":os,"json":json,"hashlib":hashlib,"stat":__import__("stat")}; exec(compile(ast.fix_missing_locations(module),"<publisher>","exec"),namespace); publish=namespace["publish_terminal_decision"]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"ownership_start_decision.json"; nonce="a"*64; outcomes=[]; values=[{"schema":"step385-start-decision-v1","status":"aborted","token":"t","wrapper_nonce":nonce},{"schema":"step385-start-decision-v1","status":"committed","token":"t","wrapper_nonce":nonce,"host_lock_sha256":"b"*64,"host_lock_dev":1,"host_lock_ino":2}]
            def attempt(value):
                try: publish(path,value,nonce); outcomes.append("won")
                except FileExistsError: outcomes.append("lost")
            threads=[threading.Thread(target=attempt,args=(value,)) for value in values]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
            self.assertEqual(sorted(outcomes),["lost","won"]); self.assertIn(json.loads(path.read_text())["status"],("committed","aborted"))

    def test_aborted_terminal_decision_exits_wrapper_without_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); manifest=root/"ownership.json"; marker=root/"child-ran"; process=subprocess.Popen([sys.executable,"-c",controller._wrapper_code().decode(),str(manifest),"t","c",os.uname().nodename,"touch "+str(marker)])
            try:
                deadline=time.monotonic()+5; ownership=None
                while ownership is None and time.monotonic()<deadline:
                    try: ownership=json.loads(manifest.read_text())
                    except (FileNotFoundError,json.JSONDecodeError): time.sleep(.01)
                self.assertIsNotNone(ownership); decision=root/"ownership_start_decision.json"; decision.write_text(json.dumps({"schema":"step385-start-decision-v1","status":"aborted","token":"t","wrapper_nonce":ownership["nonce"]})+'\n'); decision.chmod(0o600)
                self.assertEqual(process.wait(timeout=5),0); self.assertFalse(marker.exists())
            finally:
                if process.poll() is None: process.kill(); process.wait()

    def test_replaced_client_kid_wrapper_identities_get_zero_signal_authority(self) -> None:
        spec=controller.importlib.util.spec_from_file_location("step385_test_guard",controller.PROCESS_GUARD); module=controller.importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules,{"step385_test_guard":module}): spec.loader.exec_module(module)
        identities=tuple(module.ProcessIdentity(pid,10,(pid,),pid,(b"x",)) for pid in (201,202,203)); calls=[]
        module.terminate_owned(identities,lambda _identity:False,grace_seconds=0,signaler=lambda identity,sig:calls.append((identity,sig)))
        self.assertEqual(calls,[])
        replacement=lambda identity: module.ProcessIdentity(identity.host_pid,identity.starttime+1,identity.nspid,identity.pgid,identity.argv)
        for identity in identities:
            sent=[]
            with mock.patch.object(module.os,"pidfd_open",return_value=77), mock.patch.object(module.os,"close"), mock.patch.object(module.signal,"pidfd_send_signal",side_effect=lambda *args:sent.append(args)):
                with self.assertRaisesRegex(RuntimeError,"changed after pidfd_open"):
                    module.signal_owned_pidfd(identity,module.signal.SIGTERM,identity_reader=lambda _pid,value=replacement(identity):value)
            self.assertEqual(sent,[])

    def test_local_dirfd_reader_rejects_symlink_and_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); real = root / "real"; real.write_bytes(b"x")
            self.assertEqual(controller.read_local_regular(real, hashlib.sha256(b"x").hexdigest()), b"x")
            with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                controller.read_local_regular(real, "0" * 64)
            alias = root / "alias"; alias.symlink_to(real)
            with self.assertRaises(OSError): controller.read_local_regular(alias, hashlib.sha256(b"x").hexdigest())

    def test_exact_local_validators_reject_extra_and_unstable_samples(self) -> None:
        snapshot=closure_snapshot()
        self.assertIs(controller._validate_snapshot(snapshot), snapshot)
        for mutation in ({**snapshot,"extra":1},{**snapshot,"npu_samples":[snapshot["npu_samples"][0],{"rows":[],"device_ids":[8],"host_pids":[]}]},{**snapshot,"opc_samples":[snapshot["opc_samples"][0],{**snapshot["opc_samples"][0],"size":2}]}):
            with self.assertRaises(RuntimeError): controller._validate_snapshot(mutation)
        summary=completion_summary()
        adapter_sha=summary["tools"]["diagnostic_adapter_sha256"]
        controller._validate_summary(summary,adapter_sha)
        with self.assertRaises(RuntimeError): controller._validate_summary({**summary,"extra":True},adapter_sha)

    def test_contract_validator_rejects_wrong_container_and_relative_roots(self) -> None:
        valid={"schema_version":1,"container_name":"mapqr-leicheng","inspect_container_id":"a"*64,"inspect_hostname":"host","opc":{"path":"/opc","sha256":"a"*64},"cann_version_files":[{"path":"/version.info","sha256":"b"*64}],"ascend_opp":"/opp","installed_cloud_root":"/installed"}
        self.assertIs(controller._validate_contract(valid), valid)
        for change in ({"container_name":"mapqr"},{"ascend_opp":"relative"},{"installed_cloud_root":""},{"inspect_container_id":""}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                controller._validate_contract({**valid, **change})

    def test_host_container_identity_fixture_maps_host_to_container_init(self) -> None:
        contract={"inspect_container_id":"a"*64,"inspect_hostname":"h"}; fixture=host_container_fixture()
        self.assertIs(controller._validate_host_container_identity(fixture,contract),fixture)
        bad=json.loads(json.dumps(fixture)); bad["init"]["nspid"]=[100]
        with self.assertRaisesRegex(RuntimeError,"init identity"):
            controller._validate_host_container_identity(bad,contract)
        self.assertIn("nspid[-1]==w['container_pid']",controller._owned_code())
        self.assertIn("if sample()==zero",controller._owned_code())

    def test_container_and_host_cgroup_views_are_not_compared(self) -> None:
        wrapper=controller._wrapper_code().decode(); owned=controller._owned_code()
        self.assertIn("'cgroup':cgroup",wrapper)
        self.assertIn("host_cgroup=host_contract['cgroup']",owned)
        self.assertIn("type(w['cgroup']) is str",owned)
        self.assertNotIn("host_contract['cgroup']==w['cgroup']",owned)
        self.assertNotIn("host_contract['namespaces']==w['namespaces']",owned)

    def test_cleanup_protocol_covers_fork_reparent_new_round_and_deadline(self) -> None:
        owned=controller._owned_code()
        self.assertIn("residual={pid for pid,row in procs.items() if member(row) and row[2]==host_sid}",owned)
        self.assertIn("owned descendant setsid drift",owned)
        self.assertIn("while rounds<16 and time.monotonic()<deadline",owned)
        self.assertIn("g.terminate_owned(tuple(identities)",owned)
        self.assertIn("cleanup deadline/max-rounds exceeded",owned)
        self.assertIn("for _attempt in range(4)",owned)

    def test_wrapper_manifest_schema_and_types_are_strict(self) -> None:
        owned=controller._owned_code()
        for fragment in ("set(data)=={'schema','token','nonce','container_id','container_hostname','wrapper'}","data['token']==expected_token and type(data['nonce']) is str","w['container_nspid'][-1]==w['container_pid']","all(type(x) is str for x in w['argv'])","all(type(x) is int and x>0 for x in w['namespaces'].values())"):
            self.assertIn(fragment,owned)

    def test_owned_result_schema_is_exact(self) -> None:
        valid={"host_client":[2],"container_wrapper":[],"descendant_opc":[]}; helper=mock.Mock(); helper.run.return_value=(json.dumps(valid),"")
        self.assertEqual(controller._owned_run(helper,object(),"scan"),valid)
        for value in ({}, [], {**valid,"extra":[]}):
            helper.run.return_value=(json.dumps(value),"")
            with self.assertRaises(RuntimeError): controller._owned_run(helper,object(),"scan")
        helper.run.return_value=(json.dumps({"schema":"step385-owned-clean-v1","remaining":0}),"")
        controller._owned_cleanup_run(helper,object(),"cleanup")
        helper.run.return_value=(json.dumps({"schema":"step385-owned-clean-v1","remaining":0,"extra":1}),"")
        with self.assertRaises(RuntimeError): controller._owned_cleanup_run(helper,object(),"cleanup")
        self.assertNotIn("marker not in argv",controller._owned_code())
        self.assertIn("g.terminate_owned",controller._owned_code())

    def test_primary_error_survives_owned_postflight_failures(self) -> None:
        helper=mock.Mock(); helper.run.side_effect=[TimeoutError("build-primary"),ValueError("owned-scan"),OSError("snapshot")]
        with self.assertRaises(TimeoutError) as raised:
            controller._run_build_transaction(helper,object(),"build","snapshot",{},"owned","cleanup")
        self.assertEqual(str(raised.exception),"build-primary")
        self.assertEqual(len(getattr(raised.exception,"cleanup_errors",())),2)

    def test_installed_inventory_schema_records_symlink_but_rejects_special_shape(self) -> None:
        snapshot=closure_snapshot(); sample=snapshot["installed_samples"][0]
        sample["entries"]["alias"]={"type":"SYMLINK","mode":511,"size":3,"sha256":None,"target":"dst"}
        snapshot["installed_samples"]=[sample,sample]
        controller._validate_snapshot(snapshot)
        bad=json.loads(json.dumps(snapshot)); bad["installed_samples"][0]["entries"]["alias"]["sha256"]="a"*64
        bad["installed_samples"][1]=bad["installed_samples"][0]
        with self.assertRaisesRegex(RuntimeError,"symlink"): controller._validate_snapshot(bad)

    def test_summary_rejects_tool_artifact_and_candidate_drift(self) -> None:
        for mutate in (lambda x:x["tools"].update({"base_builder_sha256":"0"*64}),lambda x:x["candidate"].update({"reverse_v4_sha256":"0"*64}),lambda x:x["artifacts"][controller.SOCS[0]].update({"object_size":0}),lambda x:x["artifacts"].update({"extra":{}})):
            value=completion_summary(); mutate(value)
            with self.assertRaises(RuntimeError): controller._validate_summary(value,completion_summary()["tools"]["diagnostic_adapter_sha256"])

    def test_upload_gate_rejects_symlink_and_staging_extra(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/"real").write_bytes(b"x"); (root/"alias").symlink_to(root/"real")
            expected={"real":hashlib.sha256(b"x").hexdigest(),"alias":hashlib.sha256(b"x").hexdigest()}
            command=["python3","-c",controller._upload_gate_code(),str(root),json.dumps(expected)]
            self.assertNotEqual(subprocess.run(command,capture_output=True,text=True).returncode,0)
            (root/"alias").unlink(); (root/"extra").write_bytes(b"z")
            self.assertNotEqual(subprocess.run(command,capture_output=True,text=True).returncode,0)


if __name__ == "__main__":
    unittest.main()
