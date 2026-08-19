#!/usr/bin/env bash
set -euo pipefail

wheelhouse="$STEP214_ROOT/wheelhouse"
mkdir -p "$wheelhouse"

python3 - "$wheelhouse" <<'PY'
import hashlib
import json
import pathlib
import sys
import urllib.request

wheelhouse = pathlib.Path(sys.argv[1])
metadata_url = "https://pypi.org/pypi/triton-ascend/3.2.0rc4/json"
with urllib.request.urlopen(metadata_url, timeout=30) as response:
    project = json.load(response)

matches = []
for item in project["urls"]:
    name = item["filename"]
    if "cp311-cp311" in name and "aarch64" in name and name.endswith(".whl"):
        matches.append(item)
if len(matches) != 1:
    raise SystemExit(f"expected one cp311 aarch64 wheel, found {len(matches)}")

item = matches[0]
destination = wheelhouse / item["filename"]
temporary = destination.with_suffix(destination.suffix + ".part")
with urllib.request.urlopen(item["url"], timeout=120) as response, temporary.open("wb") as output:
    while chunk := response.read(1024 * 1024):
        output.write(chunk)
digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
expected = item["digests"]["sha256"]
if digest != expected:
    temporary.unlink(missing_ok=True)
    raise SystemExit("wheel SHA256 mismatch")
temporary.replace(destination)
print("wheel_filename=" + destination.name)
print("wheel_size_bytes=" + str(destination.stat().st_size))
print("wheel_sha256=" + digest)
print("wheel_upload_time=" + item["upload_time_iso_8601"])
PY

wheel=$(find "$wheelhouse" -maxdepth 1 -type f -name 'triton_ascend-3.2.0rc4-*.whl' -print -quit)
[ -n "$wheel" ] || { echo 'wheel_missing'; exit 51; }
python3 - "$wheel" <<'PY'
import email
import pathlib
import sys
import zipfile

wheel = pathlib.Path(sys.argv[1])
with zipfile.ZipFile(wheel) as archive:
    metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
    wheel_names = [name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")]
    if len(metadata_names) != 1 or len(wheel_names) != 1:
        raise SystemExit("unexpected wheel metadata layout")
    message = email.message_from_bytes(archive.read(metadata_names[0]))
    wheel_text = archive.read(wheel_names[0]).decode("utf-8", errors="replace")
print("metadata_name=" + str(message.get("Name")))
print("metadata_version=" + str(message.get("Version")))
requires = message.get_all("Requires-Dist", [])
print("requires_dist_count=" + str(len(requires)))
for requirement in requires:
    print("requires_dist=" + requirement)
for line in wheel_text.splitlines():
    if line.startswith(("Wheel-Version:", "Root-Is-Purelib:", "Tag:")):
        print("wheel_" + line)
PY
