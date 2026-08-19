from __future__ import annotations

import argparse
import base64
import re
import shlex
import socket
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\Administrator\.codex\tools\python-packages")

import paramiko


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
MACHINE_INFO = ROOT / "机器IP.md"
REQUIRED_TARGET_LAST_OCTET = "42"
REQUIRED_TARGET_HOSTNAME = "yfzy-zhsc-910c-1.novalocal"


def value_after_colon(line: str) -> str:
    parts = re.split(r"[:：]", line, maxsplit=1)
    if len(parts) != 2:
        raise ValueError("expected a key/value line")
    return parts[1].strip().strip("`").strip()


def parse_machine_info() -> dict[str, object]:
    text = MACHINE_INFO.read_text(encoding="utf-8")
    jump_host = re.search(r"跳板机[\s\S]*?((?:\d{1,3}\.){3}\d{1,3})", text).group(1)
    jump_port = int(re.search(r"跳板机[\s\S]*?端口\s*[:：]\s*(\d+)", text).group(1))
    jump_user = re.search(r"跳板机[\s\S]*?账号\s*[:：]\s*(\S+)", text).group(1)
    jump_password = re.search(r"跳板机[\s\S]*?密码\s*[:：]\s*(\S+)", text).group(1)

    target_match = re.search(
        r"主机器[\s\S]*?((?:\d{1,3}\.){3}\d{1,3})\s+(\S+)\s+密码\s*[:：]\s*(\S+)",
        text,
    )
    if not target_match:
        raise ValueError("unable to parse primary NPU host line")
    target_host, target_user, target_password = target_match.groups()
    target_port = int(re.search(r"主机器[\s\S]*?端口\s*[:：]\s*(\d+)", text).group(1))

    shared = re.search(r"共同的地址\s*[:：]\s*(\S+)", text).group(1).rstrip()
    branch_match = re.search(r"([A-Za-z0-9._/-]+)分支", text)
    if not branch_match:
        raise ValueError("unable to parse target branch")
    branch = branch_match.group(1)

    return {
        "jump_host": jump_host,
        "jump_port": jump_port,
        "jump_user": jump_user,
        "jump_password": jump_password,
        "target_host": target_host,
        "target_port": target_port,
        "target_user": target_user,
        "target_password": target_password,
        "shared": shared,
        "branch": branch,
    }


def connect(host: str, port: int, user: str, password: str, sock=None):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username=user,
        password=password,
        sock=sock,
        allow_agent=False,
        look_for_keys=False,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    if host.split(".")[-1] == REQUIRED_TARGET_LAST_OCTET:
        _, stdout, stderr = client.exec_command("hostname", timeout=20)
        actual = stdout.read().decode("utf-8", errors="replace").strip()
        error = stderr.read().decode("utf-8", errors="replace").strip()
        status = stdout.channel.recv_exit_status()
        if status != 0 or actual != REQUIRED_TARGET_HOSTNAME:
            client.close()
            detail = f" status={status}" if status != 0 else ""
            if error:
                detail += " hostname_query_failed"
            raise ValueError(f"remote target hostname guard rejected connection{detail}")
    return client


def redact(text: str, info: dict[str, object]) -> str:
    values = [
        str(info[key])
        for key in (
            "jump_password",
            "target_password",
            "jump_host",
            "target_host",
            "jump_user",
            "target_user",
            "shared",
        )
    ]
    for value in sorted(set(values), key=len, reverse=True):
        if value:
            text = text.replace(value, "<redacted>")
    text = re.sub(
        r"(?im)^(\s*(?:access[_-]?key|secret[_-]?key|password|passwd|token)\s*[=:]\s*).+$",
        r"\1<redacted>",
        text,
    )
    text = re.sub(
        r"(?i)(://[^\s:/]+:)[^@\s]+(@)",
        r"\1<redacted>\2",
        text,
    )
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", choices=("jump", "npu"), required=True)
    parser.add_argument("--command-b64", required=True)
    args = parser.parse_args()

    info = parse_machine_info()
    command = base64.b64decode(args.command_b64).decode("utf-8")
    command = command.replace("{{SHARED}}", shlex.quote(str(info["shared"])))
    command = command.replace("{{BRANCH}}", shlex.quote(str(info["branch"])))

    jump = connect(
        str(info["jump_host"]),
        int(info["jump_port"]),
        str(info["jump_user"]),
        str(info["jump_password"]),
    )
    target = None
    try:
        client = jump
        if args.host == "npu":
            transport = jump.get_transport()
            channel = transport.open_channel(
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
            client = target

        _, stdout, stderr = client.exec_command(command, timeout=120)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        status = stdout.channel.recv_exit_status()
        if out:
            print(redact(out, info), end="" if out.endswith("\n") else "\n")
        if err:
            print(redact(err, info), file=sys.stderr, end="" if err.endswith("\n") else "\n")
        return status
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, socket.error, paramiko.SSHException) as exc:
        print(f"remote_exec failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
