"""Freeze and audit the reproducible CCRO-NUBS planning baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(args: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        args, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    return {
        "command": args,
        "returncode": result.returncode,
        "output": result.stdout.strip(),
    }


def _snapshot_files(config: dict[str, Any]) -> list[Path]:
    files: set[Path] = set()
    excluded_parts = set(config.get("exclude_parts", []))
    excluded_suffixes = set(config.get("exclude_suffixes", []))
    for root_value in config["snapshot_roots"]:
        root = ROOT / root_value
        if root.is_file():
            files.add(root)
        elif root.exists():
            files.update(path for path in root.rglob("*") if path.is_file())
    for pattern in config["snapshot_globs"]:
        files.update(path for path in ROOT.glob(pattern) if path.is_file())
    return sorted(
        path for path in files
        if not any(part in excluded_parts for part in path.relative_to(ROOT).parts)
        and path.suffix not in excluded_suffixes
    )


def _environment(lock: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    packages = {}
    for name, expected in lock["packages"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        packages[name] = {"expected": str(expected), "actual": actual}
        if actual != str(expected):
            warnings.append(f"package version differs: {name}={actual}, expected={expected}")
    cmake = _command([str(Path(sys.prefix) / "bin" / "cmake"), "--version"])
    compiler = _command(["c++", "--version"])
    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        "platform": platform.platform(),
        "packages": packages,
        "cmake": cmake["output"].splitlines()[0] if cmake["returncode"] == 0 else None,
        "compiler": compiler["output"].splitlines()[0] if compiler["returncode"] == 0 else None,
    }, warnings


def run(config_path: str | Path, output_override: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _load(config_path)
    lock = _load(ROOT / config["environment_lock"])
    output = Path(output_override or config["output_dir"])
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config.yaml")

    environment, warnings = _environment(lock)
    git_head = _command(["git", "rev-parse", "HEAD"])
    git_status = _command(["git", "status", "--short"])
    status_lines = [line for line in git_status["output"].splitlines() if line.strip()]
    if status_lines:
        warnings.append(
            f"git worktree is not clean ({len(status_lines)} status entries); hashes freeze content"
        )

    files = _snapshot_files(config)
    file_hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in files}
    combined = hashlib.sha256()
    for relative, digest in file_hashes.items():
        combined.update(relative.encode("utf-8"))
        combined.update(digest.encode("ascii"))

    stages = {}
    for stage_path_value in config["required_stage_results"]:
        stage_path = ROOT / stage_path_value
        stage_name = stage_path.parent.name
        if not stage_path.exists():
            stages[stage_name] = {"exists": False, "accepted": False}
            continue
        payload = json.loads(stage_path.read_text(encoding="utf-8"))
        stages[stage_name] = {
            "exists": True,
            "accepted": bool(payload.get("accepted", False)),
            "sha256": _sha256(stage_path),
        }

    nubs_import = _command([
        sys.executable, "-c",
        "from planning import _nubs_cpp; print(_nubs_cpp.__file__)",
    ])
    tests = _command([
        sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
        str(ROOT / config["test_path"]), "-q",
    ])
    print(tests["output"])
    test_count = 0
    for token in tests["output"].replace("\n", " ").split():
        if token.isdigit():
            test_count = max(test_count, int(token))

    required_python = config["requirements"]["python_major_minor"]
    checks = {
        "git_head_readable": git_head["returncode"] == 0,
        "source_snapshot_nonempty": len(file_hashes) > 0,
        "all_stage_results_accepted": all(item["accepted"] for item in stages.values()),
        "tests_passed": tests["returncode"] == 0,
        "minimum_test_count": test_count >= int(config["requirements"]["minimum_test_count"]),
        "python_version": platform.python_version().startswith(required_python + "."),
        "nubs_extension_import": nubs_import["returncode"] == 0,
    }
    if config["requirements"].get("require_clean_git", False):
        checks["clean_git"] = not status_lines
    accepted = bool(all(checks.values()))
    manifest = {
        "accepted": accepted,
        "checks": checks,
        "warnings": warnings,
        "git": {
            "head": git_head["output"],
            "status_entry_count": len(status_lines),
            "status": status_lines,
        },
        "environment": environment,
        "source_snapshot": {
            "file_count": len(file_hashes),
            "combined_sha256": combined.hexdigest(),
            "files": file_hashes,
        },
        "stage_results": stages,
        "tests": {
            "returncode": tests["returncode"],
            "detected_test_count": test_count,
            "output": tests["output"],
        },
        "nubs_extension": nubs_import,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = {
        "accepted": accepted,
        "checks": checks,
        "warning_count": len(warnings),
        "source_file_count": len(file_hashes),
        "source_sha256": combined.hexdigest(),
        "test_count": test_count,
        "stage_acceptance": {name: item["accepted"] for name, item in stages.items()},
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "| check | result |",
        "|---|---:|",
        *[f"| {name} | {value} |" for name, value in checks.items()],
        "",
        f"Source files: **{len(file_hashes)}**  ",
        f"Source SHA-256: `{combined.hexdigest()}`  ",
        f"Warnings: **{len(warnings)}**  ",
        f"Overall: **{'PASS' if accepted else 'FAIL'}**",
        "",
    ]
    (output / "table_p0.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "ccro_p0.yaml"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps({
        "accepted": result["accepted"],
        "checks": result["checks"],
        "warnings": result["warnings"],
        "source_sha256": result["source_snapshot"]["combined_sha256"],
    }, ensure_ascii=False, indent=2))
    if not result["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
