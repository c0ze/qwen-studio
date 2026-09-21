"""User-owned desktop and Omarchy menu integration; no root privileges."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import json5

START = "// BEGIN c0ze.qwen-studio"
END = "// END c0ze.qwen-studio"


def menu_content(text: str, command: str, remove: bool = False) -> str:
    # Keep all other entries and comments byte-for-byte, and own only our block.
    text = re.sub(r"\n? *// BEGIN c0ze\.qwen-studio.*?// END c0ze\.qwen-studio\n?", "", text, flags=re.S)
    text = re.sub(r"^(\s*\{)\s*,", r"\1", text, count=1)
    data = json5.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Omarchy menu must be a JSON object")
    if remove:
        return text
    if "custom.qwen-studio" in data:
        raise ValueError("A Qwen Studio menu entry already exists outside our managed block")
    entries = {}
    if "custom" not in data:
        entries["custom"] = {"icon": "󰀻", "label": "Custom"}
    entries["custom.qwen-studio"] = {
        "parent": "custom", "icon": "󰋩", "label": "Qwen Studio",
        "description": "Local image projects, Qwen generation and editing", "action": command,
    }
    body = json.dumps(entries, indent=2, ensure_ascii=False)[1:-1]
    # Place our entry before existing properties. A trailing line comment in
    # another app's block must never swallow the separating comma.
    at = text.index("{") + 1
    separator = "," if data else ""
    block = "\n  " + START + "\n" + body + separator + "\n  " + END + "\n"
    result = text[:at] + block + text[at:]
    json5.loads(result)
    return result


def install() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninstall", action="store_true", help="remove desktop integration; keep images, models and configuration")
    args = parser.parse_args()
    home = Path.home()
    config = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config"))
    share = Path(os.environ.get("XDG_DATA_HOME", home / ".local/share"))
    bindir = home / ".local/bin"
    executable = shutil.which("qwen-studio") or str(bindir / "qwen-studio")
    opener = shutil.which("qwen-studio-open") or str(bindir / "qwen-studio-open")
    menu = config / "omarchy/extensions/omarchy-menu.jsonc"
    unit = config / "systemd/user/qwen-studio.service"
    desktop = share / "applications/qwen-studio.desktop"
    icon = share / "icons/hicolor/scalable/apps/qwen-studio.svg"
    if args.uninstall:
        subprocess.run(["systemctl", "--user", "disable", "--now", "qwen-studio.service"], check=False)
        for path in (unit, desktop, icon):
            path.unlink(missing_ok=True)
        if menu.exists():
            before = menu.read_text()
            shutil.copy2(menu, menu.with_name(menu.name + f".backup-{time.time_ns()}"))
            menu.write_text(menu_content(before, "", remove=True))
    else:
        for path in (unit, desktop, icon):
            path.parent.mkdir(parents=True, exist_ok=True)
        env = config / "qwen-studio/environment"
        env.parent.mkdir(parents=True, exist_ok=True)
        if not env.exists():
            env.write_text("# Optional settings: COMFYUI_DIR, COMFYUI_URL, QWEN_EDIT_MODEL\nQWEN_STUDIO_PORT=8790\n")
        unit.write_text("[Unit]\nDescription=Qwen Studio local image workspace\nAfter=network.target\n\n[Service]\nType=simple\n"
                        + f"ExecStart={json.dumps(executable).replace('%', '%%')}\nEnvironmentFile=-{str(env).replace('%', '%%')}\n"
                        + "Restart=on-failure\nRestartSec=3\nTimeoutStopSec=1900\n\n[Install]\nWantedBy=default.target\n")
        desktop.write_text("[Desktop Entry]\nType=Application\nName=Qwen Studio\nComment=Local image projects, generation and editing\n"
                           + f"Exec={json.dumps(opener)}\nIcon=qwen-studio\nTerminal=false\nCategories=Graphics;Photography;\nKeywords=Qwen;ComfyUI;AI;Image;\n")
        shutil.copyfile(Path(__file__).parent / "static/icon.svg", icon)
        if shutil.which("omarchy") or menu.exists():
            menu.parent.mkdir(parents=True, exist_ok=True)
            before = menu.read_text() if menu.exists() else "{}\n"
            updated = menu_content(before, json.dumps(opener))
            if menu.exists():
                shutil.copy2(menu, menu.with_name(menu.name + f".backup-{time.time_ns()}"))
            menu.write_text(updated)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(desktop.parent)], check=False)
    if shutil.which("omarchy"):
        subprocess.run(["omarchy", "menu", "refresh"], check=False)
    print("Qwen Studio desktop integration " + ("removed; your data was kept." if args.uninstall else "installed. Open Custom → Qwen Studio or launch Qwen Studio from Apps."))


def main() -> None:
    subprocess.run(["systemctl", "--user", "start", "qwen-studio.service"], check=True)
    config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "qwen-studio/environment"
    port = "8790"
    if config.exists():
        for line in config.read_text().splitlines():
            if line.startswith("QWEN_STUDIO_PORT="):
                port = str(int(line.split("=", 1)[1].strip().strip('"')))
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(0.5)
    else:
        raise SystemExit("Studio did not start. Inspect: journalctl --user -u qwen-studio.service")
    command = ["omarchy", "launch", "webapp", url] if shutil.which("omarchy") else ["xdg-open", url]
    subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
