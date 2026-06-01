#!/usr/bin/env python3
"""Configure the Claude Code extension in Antigravity for this proxy."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_BASE_URL = "https://claude-code-api.squareweb.app"
DEFAULT_MODEL = "claude-code-sonnet"
EXTENSION_ID = "anthropic.claude-code"


@dataclass
class ConfigureResult:
    claude_settings: Path
    ide_settings: list[Path]
    extension_message: str


def validate_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("A URL precisa comecar com http:// ou https://.")
    return value


def validate_token(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Informe o token da API.")
    return value


def validate_model(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Informe o modelo.")
    return value


def strip_jsonc(value: str) -> str:
    """Remove JSONC comments and trailing commas while preserving string values."""
    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(value):
        char = value[index]
        next_char = value[index + 1] if index + 1 < len(value) else ""
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == "/" and next_char == "/":
            index += 2
            while index < len(value) and value[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and next_char == "*":
            end = value.find("*/", index + 2)
            if end == -1:
                raise ValueError("Comentario JSONC de bloco nao foi fechado.")
            index = end + 2
            continue
        output.append(char)
        index += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(output))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    content = strip_jsonc(path.read_text(encoding="utf-8")).strip()
    if not content:
        return {}
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON invalido em {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"O arquivo {path} precisa conter um objeto JSON.")
    return value


def write_json_with_backup(path: Path, value: dict[str, Any], dry_run: bool = False) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = path.with_name(f"{path.name}.backup-{stamp}")
        shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    if os.name != "nt":
        path.chmod(0o600)


def proxy_environment(token: str, base_url: str, model: str) -> dict[str, str]:
    return {
        "ANTHROPIC_AUTH_TOKEN": token,
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }


def merge_claude_settings(
    path: Path, token: str, base_url: str, model: str, dry_run: bool = False
) -> None:
    data = load_json(path)
    data.setdefault("$schema", "https://json.schemastore.org/claude-code-settings.json")
    env = data.setdefault("env", {})
    if not isinstance(env, dict):
        raise ValueError(f"A chave env em {path} precisa ser um objeto JSON.")
    env.update(proxy_environment(token, base_url, model))
    write_json_with_backup(path, data, dry_run=dry_run)


def merge_ide_settings(
    path: Path, token: str, base_url: str, model: str, dry_run: bool = False
) -> None:
    data = load_json(path)
    data["claudeCode.disableLoginPrompt"] = True
    current = data.get("claudeCode.environmentVariables", [])
    if not isinstance(current, list):
        current = []
    env_by_name = {
        item["name"]: item
        for item in current
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    for name, value in proxy_environment(token, base_url, model).items():
        env_by_name[name] = {"name": name, "value": value}
    data["claudeCode.environmentVariables"] = list(env_by_name.values())
    write_json_with_backup(path, data, dry_run=dry_run)


def candidate_ide_settings() -> list[Path]:
    home = Path.home()
    system = platform.system()
    if system == "Windows":
        root = Path(os.getenv("APPDATA", home / "AppData" / "Roaming"))
    elif system == "Darwin":
        root = home / "Library" / "Application Support"
    else:
        root = Path(os.getenv("XDG_CONFIG_HOME", home / ".config"))
    names = (
        "Antigravity",
        "Antigravity IDE",
        "antigravity",
        "antigravity-ide",
        "Google Antigravity",
    )
    paths = [root / name / "User" / "settings.json" for name in names]
    if root.exists():
        paths.extend(
            directory / "User" / "settings.json"
            for directory in root.iterdir()
            if directory.is_dir() and "antigravity" in directory.name.lower()
        )
    paths = list(dict.fromkeys(paths))
    return [path for path in paths if path.parent.exists() or path.exists()]


def candidate_antigravity_commands() -> list[str]:
    found: list[str] = []
    for name in ("antigravity", "antigravity-ide", "agy"):
        command = shutil.which(name)
        if command:
            found.append(command)

    if platform.system() == "Darwin":
        for app_name in ("Antigravity", "Antigravity IDE", "Google Antigravity"):
            base = Path("/Applications") / f"{app_name}.app" / "Contents" / "Resources" / "app"
            for name in ("antigravity", "antigravity-ide"):
                candidate = base / "bin" / name
                if candidate.exists():
                    found.append(str(candidate))
    elif platform.system() == "Windows":
        local = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        for app_name in ("Antigravity", "Antigravity IDE", "Google Antigravity"):
            for name in ("antigravity.cmd", "antigravity-ide.cmd"):
                candidate = local / "Programs" / app_name / "bin" / name
                if candidate.exists():
                    found.append(str(candidate))

    return list(dict.fromkeys(found))


def install_extension() -> str:
    commands = candidate_antigravity_commands()
    if not commands:
        return (
            "Comando do Antigravity nao encontrado. A configuracao foi salva. "
            "Instale a extensao Claude Code manualmente no IDE ou rode o instalador novamente."
        )
    errors: list[str] = []
    for command in commands:
        try:
            completed = subprocess.run(
                [command, "--install-extension", EXTENSION_ID, "--force"],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{command}: {exc}")
            continue
        if completed.returncode == 0:
            return f"Extensao {EXTENSION_ID} instalada usando {command}."
        errors.append(f"{command}: codigo {completed.returncode}")
    return "Nao foi possivel instalar a extensao automaticamente: " + "; ".join(errors)


def configure(
    token: str,
    base_url: str,
    model: str,
    *,
    claude_settings: Path | None = None,
    ide_settings: list[Path] | None = None,
    should_install_extension: bool = True,
    dry_run: bool = False,
) -> ConfigureResult:
    token = validate_token(token)
    base_url = validate_base_url(base_url)
    model = validate_model(model)
    claude_path = claude_settings or Path.home() / ".claude" / "settings.json"
    merge_claude_settings(claude_path, token, base_url, model, dry_run=dry_run)

    ide_paths = candidate_ide_settings() if ide_settings is None else ide_settings
    for path in ide_paths:
        merge_ide_settings(path, token, base_url, model, dry_run=dry_run)

    if dry_run:
        extension_message = "Simulacao concluida; nenhum arquivo foi alterado."
    elif should_install_extension:
        extension_message = install_extension()
    else:
        extension_message = "Instalacao automatica da extensao ignorada."
    return ConfigureResult(claude_path, ide_paths, extension_message)


def run_gui() -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("Configurar API no Antigravity")
    root.geometry("650x430")
    root.minsize(570, 390)

    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Claude API para Antigravity", font=("TkDefaultFont", 16, "bold")).pack(
        anchor="w"
    )
    ttk.Label(
        frame,
        text=(
            "Configure o Claude Code com a URL base e a chave da sua API. "
            "Feche e abra o Antigravity depois da instalacao."
        ),
        wraplength=600,
    ).pack(anchor="w", pady=(4, 16))

    def add_field(label: str, default: str = "", secret: bool = False) -> ttk.Entry:
        ttk.Label(frame, text=label).pack(anchor="w", pady=(8, 2))
        entry = ttk.Entry(frame, show="*" if secret else "")
        entry.insert(0, default)
        entry.pack(fill="x")
        return entry

    url_entry = add_field("URL base da API", DEFAULT_BASE_URL)
    token_entry = add_field("Chave da API", secret=True)
    install_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        frame,
        text="Tentar instalar a extensao Claude Code automaticamente",
        variable=install_var,
    ).pack(anchor="w", pady=(14, 4))
    status_var = tk.StringVar(value="Pronto para configurar.")
    ttk.Label(frame, textvariable=status_var, wraplength=600).pack(anchor="w", pady=(10, 8))

    def on_configure() -> None:
        try:
            result = configure(
                token_entry.get(),
                url_entry.get(),
                DEFAULT_MODEL,
                should_install_extension=install_var.get(),
            )
        except Exception as exc:
            messagebox.showerror("Nao foi possivel configurar", str(exc))
            return
        ide_note = (
            f"{len(result.ide_settings)} arquivo(s) do IDE atualizado(s)."
            if result.ide_settings
            else "Antigravity nao detectado; execute novamente depois de instalar o IDE."
        )
        status = f"Configuracao salva em {result.claude_settings}. {ide_note} {result.extension_message}"
        status_var.set(status)
        messagebox.showinfo("Configuracao concluida", status)

    ttk.Button(frame, text="Configurar Antigravity", command=on_configure).pack(
        anchor="e", pady=(12, 0)
    )
    root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", help="Token da API. Omita para abrir a interface grafica.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--claude-settings", type=Path)
    parser.add_argument("--ide-settings", type=Path, action="append")
    parser.add_argument("--no-install-extension", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.token:
        run_gui()
        return 0
    try:
        result = configure(
            args.token,
            args.base_url,
            args.model,
            claude_settings=args.claude_settings,
            ide_settings=args.ide_settings,
            should_install_extension=not args.no_install_extension,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    print(f"Configuracao Claude Code: {result.claude_settings}")
    if result.ide_settings:
        for path in result.ide_settings:
            print(f"Configuracao Antigravity: {path}")
    else:
        print("Antigravity nao detectado. Rode novamente apos instalar o IDE.")
    print(result.extension_message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
