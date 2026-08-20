#!/usr/bin/env python3
import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info

LOCAL = Path(__file__).resolve().parent
ROOT = "diagnostics/step330_stale_q_ab_30step_back8_20260820T143600"

info = parse_machine_info()
remote = posixpath.join(str(info["shared"]).rstrip("/"), ROOT, "step330_launch_inside.sh")
jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
target = None
try:
    ch = jump.get_transport().open_channel("direct-tcpip", (str(info["target_host"]), int(info["target_port"])), ("127.0.0.1", 0))
    target = connect(str(info["target_host"]), int(info["target_port"]), str(info["target_user"]), str(info["target_password"]), sock=ch)
    sftp = target.open_sftp()
    data = (LOCAL / "step330_launch_inside.sh").read_bytes().replace(b"\r\n", b"\n")
    with sftp.open(remote, "wb") as f:
        f.write(data)
    sftp.close()
    target.exec_command(f"chmod 755 {remote}")[1].read()
    print("updated", remote)
finally:
    if target:
        target.close()
    jump.close()
