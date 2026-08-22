#!/usr/bin/env python3
"""Build the QrV2 lifetime-fix candidate in a remote diagnostics directory."""

from __future__ import annotations

import posixpath
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from remote_exec import connect, parse_machine_info, redact


LOCAL_DIR = Path(__file__).resolve().parent
PATCHER = LOCAL_DIR / "step338_patch_qr_v2_lifetime.py"
RELATIVE_DIAG = "diagnostics/step338_mx_qr_lifetime_fix_opc_retry4_20260821"
SITE = "/home/ma-user/anaconda3/envs/PyTorch-2.7.1/lib/python3.11/site-packages/mx_driving_cloud"
EXPECTED_CANDIDATE_SHA256 = "5a4d140b8a473c3a0446d9e225431ff9f8be5e9b9f7355c5a166920e1814105b"


def main() -> int:
    info = parse_machine_info()
    remote_diag = posixpath.join(str(info["shared"]).rstrip("/"), RELATIVE_DIAG)
    jump = connect(
        str(info["jump_host"]),
        int(info["jump_port"]),
        str(info["jump_user"]),
        str(info["jump_password"]),
    )
    target = None
    try:
        channel = jump.get_transport().open_channel(
            "direct-tcpip",
            (str(info["target_host"]), int(info["target_port"])),
            ("127.0.0.1", 0),
        )
        target = connect(
            str(info["target_host"]),
            int(info["target_port"]),
            str(info["target_user"]),
            str(info["target_password"]),
            sock=channel,
        )

        quoted_diag = shlex.quote(remote_diag)
        container_preflight = """\
set -eu
OPC=/usr/local/Ascend/ascend-toolkit/latest/bin/opc
test -x "$OPC"
test -n "${ASCEND_OPP_PATH:-}"
OPP_TBE="$ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe"
test -f "$OPP_TBE/impl/util/platform_adapter.py"
export PYTHONPATH="$OPP_TBE:${PYTHONPATH:-}"
python3 -c 'import impl.util.platform_adapter; print("opp_pythonpath_preflight_ok=True")'
CUSTOM_OPP=/home/ma-user/anaconda3/envs/PyTorch-2.7.1/lib/python3.11/site-packages/mx_driving_cloud/packages/vendors/customize
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP"
python3 - <<'PY'
import ctypes
import hashlib
import os
from pathlib import Path

from tbe.common.utils import op_tiling

expected = Path(os.environ["ASCEND_CUSTOM_OPP_PATH"]) / "op_impl/ai_core/tbe/op_tiling/liboptiling.so"
resolved = [Path(path) for path in op_tiling._get_custom_opp_pathlist()]
assert resolved == [expected], (resolved, expected)
assert expected.is_file()
assert expected.stat().st_size == 919776
digest = hashlib.sha256(expected.read_bytes()).hexdigest()
assert digest == "63593c0af911550b4543e5bb70802ff6c87dba8de08cc0560930516cf646f50c"
library = ctypes.CDLL(str(expected))
assert hasattr(library, "TbeLoadSoAndSaveToRegistry")
print("custom_tiling_preflight_ok=True")
print("custom_tiling_size=919776")
print("custom_tiling_sha256=" + digest)
PY
"$OPC" --help 2>&1 | grep -F -- "--input_param" >/dev/null
"$OPC" --help 2>&1 | grep -F -- "--main_func" >/dev/null
"$OPC" --help 2>&1 | grep -F -- "--soc_version" >/dev/null
"$OPC" --help 2>&1 | grep -F -- "--op_mode" >/dev/null
echo "opc_preflight_ok=True"
        """
        preflight = f"""
set -eu
count=$(docker ps --format '{{{{.Names}}}}' | awk '$0 == "mapqr-leicheng" {{n++}} END {{print n+0}}')
[ "$count" -eq 1 ]
docker exec mapqr-leicheng bash --noprofile --norc -lc {shlex.quote(container_preflight)}
if [ -e {quoted_diag} ]; then
  echo 'diagnostic_directory_exists=True'
  exit 82
fi
mkdir -p {quoted_diag}
"""
        _, stdout, stderr = target.exec_command(preflight, timeout=120)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        status = stdout.channel.recv_exit_status()
        print(redact(out + err, info), end="" if (out + err).endswith("\n") else "\n")
        if status != 0:
            return status

        sftp = target.open_sftp()
        try:
            sftp.put(str(PATCHER), posixpath.join(remote_diag, PATCHER.name))
        finally:
            sftp.close()

        container_build = f"""\
set -eu
DIAG={quoted_diag}
SITE={shlex.quote(SITE)}
OPC=/usr/local/Ascend/ascend-toolkit/latest/bin/opc
OPP_TBE="$ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe"
export PYTHONPATH="$OPP_TBE:${{PYTHONPATH:-}}"
export ASCEND_CUSTOM_OPP_PATH="$SITE/packages/vendors/customize"
IMPL_DIR="$SITE/packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic"
KERNEL_DIR="$SITE/packages/vendors/customize/op_impl/ai_core/tbe/kernel"
installed_manifest_before=$(
  find "$IMPL_DIR" "$KERNEL_DIR" -type f \
    \( -name 'qr_v2.py' -o -name 'qr_v2.cpp' -o -iname '*qr*v2*.o' -o -iname '*qr*v2*.json' \) \
    -print0 | sort -z | xargs -0 sha256sum
)
cp "$SITE/packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.py" "$DIAG/qr_v2.py"
cp "$SITE/packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.cpp" "$DIAG/qr_v2.original.cpp"
python3 "$DIAG/{PATCHER.name}" "$DIAG/qr_v2.original.cpp" "$DIAG/qr_v2.cpp"
candidate_sha=$(sha256sum "$DIAG/qr_v2.cpp" | awk '{{print $1}}')
if [ "$candidate_sha" != {shlex.quote(EXPECTED_CANDIDATE_SHA256)} ]; then
  echo "candidate_sha_gate=False actual=$candidate_sha" >&2
  exit 92
fi
echo "candidate_sha_gate=True sha256=$candidate_sha"
mkdir "$DIAG/output" "$DIAG/debug"
python3 - <<'PY'
import json
from pathlib import Path

diag = Path({remote_diag!r})
tensor = {{
    "shape": [192, 192],
    "ori_shape": [192, 192],
    "format": "ND",
    "ori_format": "ND",
    "dtype": "float32"
}}
payload = {{
    "op_type": "QrV2",
    "op_list": [{{
        "bin_filename": "QrV2_step338_lifetime_fix",
        "inputs": [dict(tensor)],
        "outputs": [dict(tensor), dict(tensor)],
        "attrs": []
    }}]
}}
(diag / "input_param.json").write_text(
    json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8"
)
print("input_param_json_written=True")
PY
set +e
"$OPC" "$DIAG/qr_v2.py" \
  --input_param="$DIAG/input_param.json" \
  --main_func=qr_v2 \
  --bin_filename=QrV2_step338_lifetime_fix \
  --output="$DIAG/output" \
  --debug_dir="$DIAG/debug" \
  --soc_version=Ascend910_9362 \
  --op_mode=dynamic \
  --optional_input_mode=gen_placeholder \
  --optional_output_mode=gen_placeholder \
  --deterministic=false \
  --log=info
opc_rc=$?
set -e
installed_manifest_after=$(
  find "$IMPL_DIR" "$KERNEL_DIR" -type f \
    \( -name 'qr_v2.py' -o -name 'qr_v2.cpp' -o -iname '*qr*v2*.o' -o -iname '*qr*v2*.json' \) \
    -print0 | sort -z | xargs -0 sha256sum
)
if [ "$installed_manifest_before" != "$installed_manifest_after" ]; then
  echo "installed_package_hash_gate=False" >&2
  exit 93
fi
echo "installed_package_hash_gate=True"
printf '%s\\n' "$installed_manifest_after"
if [ "$opc_rc" -ne 0 ]; then
  exit "$opc_rc"
fi
EXPECTED_O="$DIAG/output/QrV2_step338_lifetime_fix.o"
EXPECTED_JSON="$DIAG/output/QrV2_step338_lifetime_fix.json"
object_count=$(find "$DIAG/output" -type f -name '*.o' -size +0c | wc -l)
json_count=$(find "$DIAG/output" -type f -name '*.json' -size +0c | wc -l)
if [ "$object_count" -ne 1 ] || [ "$json_count" -ne 1 ] || \
   [ ! -s "$EXPECTED_O" ] || [ ! -s "$EXPECTED_JSON" ]; then
  echo "opc_artifact_gate=False object_count=$object_count json_count=$json_count" >&2
  exit 91
fi
echo "opc_artifact_gate=True object_count=$object_count json_count=$json_count"
sha256sum "$EXPECTED_O" "$EXPECTED_JSON"
find "$DIAG/output" "$DIAG/debug" -maxdepth 4 -type f -printf "%P %s bytes\\n" | sort
sha256sum "$DIAG/qr_v2.original.cpp" "$DIAG/qr_v2.cpp"
"""
        command = (
            "set -eu\n"
            "docker exec mapqr-leicheng bash --noprofile --norc -lc "
            + shlex.quote(container_build)
        )
        _, stdout, stderr = target.exec_command(command, timeout=900)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        status = stdout.channel.recv_exit_status()
        print(redact(out + err, info), end="" if (out + err).endswith("\n") else "\n")
        return status
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
