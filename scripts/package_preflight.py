"""Preflight scanner for local EPSR submission and public reproducibility packages."""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "package_preflight_reports"

FORBIDDEN_PATH_PARTS = [
    "iac_paper_research_management",
    ".codex",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "chatgpt_research_handoff",
    "handoff",
    "AGENT_A",
    "AGENT_B",
    "DECISION_LOG",
    "MASTER_STATE",
    "CLAIM_LEDGER",
    "tmp_",
]

ALWAYS_FORBIDDEN_TEXT_PATTERNS = [
    (re.compile(r"C:\\Users\\Divine", re.IGNORECASE), "absolute local path"),
]

OBSOLETE_DATASET_VERSION_PATTERNS = [
    (re.compile(r"10\.17632/kdbtxv66n9\.[34]", re.IGNORECASE), "version-specific obsolete dataset DOI"),
    (re.compile(r"/kdbtxv66n9\.[34]", re.IGNORECASE), "version-specific obsolete dataset path"),
    (re.compile(r"\bV[34]\b|Version\s+[34]", re.IGNORECASE), "obsolete dataset version reference"),
]

NESTED_ARCHIVE_SUFFIXES = {".zip", ".7z", ".rar", ".tar", ".gz"}

PUBLIC_CLAIM_TEXT_PATTERNS = [
    (re.compile(r"identical public GitHub mirror", re.IGNORECASE), "unsafe identical GitHub mirror claim"),
    (re.compile(r"Network-aware full P/Q DMPC", re.IGNORECASE), "stale DMPC proposed-controller label"),
    (re.compile(r"full P/Q DMPC", re.IGNORECASE), "stale full P/Q DMPC label"),
    (re.compile(r"network-aware DMPC", re.IGNORECASE), "stale network-aware DMPC label"),
    (re.compile(r"network-aware P/Q DMPC", re.IGNORECASE), "stale network-aware P/Q DMPC label"),
    (re.compile(r"DMPC controller", re.IGNORECASE), "stale proposed-controller DMPC wording"),
    (re.compile(r"active-power DMPC", re.IGNORECASE), "stale active-power DMPC label"),
    (
        re.compile(r"Network-Aware Decentralized Model Predictive Control", re.IGNORECASE),
        "stale decentralized proposed-controller title",
    ),
    (re.compile(r"Decentralized MPC active-power dispatch", re.IGNORECASE), "stale decentralized controller wording"),
    (
        re.compile(
            r"(?:network-aware|proposed|full P/Q|active-power|IAC fleet).{0,80}decentralized MPC"
            r"|decentralized MPC.{0,80}(?:controller|formulation|proposed|IAC fleet)",
            re.IGNORECASE | re.DOTALL,
        ),
        "stale decentralized proposed-controller wording",
    ),
    (
        re.compile(
            r"(?:network-aware|proposed|full P/Q|active-power|IAC fleet).{0,80}distributed MPC"
            r"|distributed MPC.{0,80}(?:controller|formulation|proposed|IAC fleet)",
            re.IGNORECASE | re.DOTALL,
        ),
        "stale distributed proposed-controller wording",
    ),
    (re.compile(r"Prototype ANDES/OpenDSS", re.IGNORECASE), "stale prototype package wording"),
    (re.compile(r"Prototype explicit", re.IGNORECASE), "stale prototype T&D wording"),
    (re.compile(r"explicit T\&?D prototype", re.IGNORECASE), "stale T&D prototype wording"),
    (re.compile(r"T\&?D co-simulation prototype", re.IGNORECASE), "stale T&D prototype wording"),
    (re.compile(r"Monte Carlo uncertainty cases", re.IGNORECASE), "stale Monte Carlo uncertainty wording"),
    (re.compile(r"MC robustness", re.IGNORECASE), "stale MC robustness wording"),
    (re.compile(r"uncertainty robustness", re.IGNORECASE), "stale uncertainty robustness wording"),
    (re.compile(r"hardware validation", re.IGNORECASE), "unsupported hardware-validation wording"),
    (re.compile(r"experimental validation", re.IGNORECASE), "unsupported experimental-validation wording"),
    (re.compile(r"adaptive spatial", re.IGNORECASE), "unsupported adaptive spatial wording"),
    (re.compile(r"limiting bus phase", re.IGNORECASE), "stale limiting-bus-phase wording"),
    (re.compile(r"same command trajectory", re.IGNORECASE), "overliteral validation-trajectory wording"),
]

SCANNER_RULE_FILES = {"scripts/package_preflight.py", "package_preflight.py"}

REQUIRED_PUBLIC_FILES = [
    "README.md",
    "README_RESEARCH_DATA_EPSR.txt",
    "REVIEWER_ACCESS.md",
    "docs/experiment_suite.md",
    "requirements.txt",
    "requirements-validation.txt",
    "run_experiment_suite.py",
    "summarize_experiments.py",
    "paper/generate_paper_assets.py",
    "scripts/package_preflight.py",
]

REQUIRED_SUBMISSION_FILES = [
    "manuscript_EPSR.pdf",
    "supplementary_material_EPSR.pdf",
    "latex_source_main_EPSR_flat.zip",
    "cover_letter_EPSR.txt",
    "data_statement_EPSR.txt",
    "declarations_EPSR.txt",
    "highlights_EPSR.txt",
]


def text_entry(name: str) -> bool:
    suffix = Path(name).suffix.lower()
    return suffix in {".txt", ".md", ".tex", ".bib", ".py", ".csv", ".json", ".yml", ".yaml"}


def scan_text(name: str, data: bytes) -> list[str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = data.decode("latin-1")
        except UnicodeDecodeError:
            return []
    issues: list[str] = []
    rel_name = name.split(":")[-1].replace("\\", "/")
    suffix = Path(rel_name).suffix.lower()
    normalized = rel_name.lower()
    patterns = list(ALWAYS_FORBIDDEN_TEXT_PATTERNS)
    if normalized not in SCANNER_RULE_FILES and "/tests/" not in f"/{normalized}":
        patterns.extend(PUBLIC_CLAIM_TEXT_PATTERNS)
    for pattern, label in patterns:
        if pattern.search(text):
            issues.append(f"{name}: {label}")
    return issues


def scan_zip(path: Path, required: list[str]) -> list[str]:
    issues: list[str] = []
    with zipfile.ZipFile(path) as archive:
        names = [item.filename.replace("\\", "/") for item in archive.infolist()]
        for required_name in required:
            if required_name not in names:
                issues.append(f"{path}: missing required entry {required_name}")
        for name in names:
            for part in FORBIDDEN_PATH_PARTS:
                if part.lower() in name.lower():
                    issues.append(f"{path}: forbidden path entry {name}")
            if Path(name).name in {"manuscript_source_EPSR.tex"}:
                issues.append(f"{path}: duplicate competing manuscript source {name}")
            if required == REQUIRED_PUBLIC_FILES and Path(name).suffix.lower() in NESTED_ARCHIVE_SUFFIXES:
                issues.append(f"{path}: nested archive is not permitted: {name}")
            if text_entry(name):
                data = archive.read(name)
                issues.extend(scan_text(f"{path}:{name}", data))
                normalized = name.replace("\\", "/").lower()
                if required == REQUIRED_PUBLIC_FILES and normalized not in SCANNER_RULE_FILES and "/tests/" not in f"/{normalized}":
                    try:
                        text = data.decode("utf-8")
                    except UnicodeDecodeError:
                        text = data.decode("latin-1", errors="ignore")
                    for pattern, label in OBSOLETE_DATASET_VERSION_PATTERNS:
                        if pattern.search(text):
                            issues.append(f"{path}:{name}: {label}")
    return issues


def scan_directory(path: Path, required: list[str]) -> list[str]:
    issues: list[str] = []
    files = [item for item in path.rglob("*") if item.is_file()]
    rels = [item.relative_to(path).as_posix() for item in files]
    for required_name in required:
        if required_name not in rels:
            issues.append(f"{path}: missing required file {required_name}")
    for rel, file_path in zip(rels, files):
        for part in FORBIDDEN_PATH_PARTS:
            if part.lower() in rel.lower():
                issues.append(f"{path}: forbidden path file {rel}")
        if Path(rel).name in {"manuscript_source_EPSR.tex"}:
            issues.append(f"{path}: duplicate competing manuscript source {rel}")
        if required == REQUIRED_PUBLIC_FILES and Path(rel).suffix.lower() in NESTED_ARCHIVE_SUFFIXES:
            issues.append(f"{path}: nested archive is not permitted: {rel}")
        if text_entry(rel):
            data = file_path.read_bytes()
            issues.extend(scan_text(f"{path}:{rel}", data))
            normalized = rel.replace("\\", "/").lower()
            if required == REQUIRED_PUBLIC_FILES and normalized not in SCANNER_RULE_FILES and "/tests/" not in f"/{normalized}":
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    text = data.decode("latin-1", errors="ignore")
                for pattern, label in OBSOLETE_DATASET_VERSION_PATTERNS:
                    if pattern.search(text):
                        issues.append(f"{path}:{rel}: {label}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public-package",
        metavar="PATH",
        help="audit one extracted public reproducibility package without requiring author-workspace artifacts",
    )
    parser.add_argument("--submission-zip", default=str(ROOT / "submission_epsr.zip"))
    parser.add_argument("--flat-zip", default=str(ROOT / "submission_epsr" / "submission_files" / "latex_source_main_EPSR_flat.zip"))
    parser.add_argument("--submission-dir", default=str(ROOT / "submission_epsr" / "submission_files"))
    parser.add_argument("--public-dir", default=str(ROOT / "github_reviewer_mirror"))
    parser.add_argument(
        "--mendeley-zip",
        default=str(ROOT / "submission_epsr" / "mendeley_data" / "research_data_EPSR_reproducibility.zip"),
    )
    parser.add_argument("--report", default=str(REPORT_DIR / "package_preflight_report_A.md"))
    args = parser.parse_args()

    if args.public_package:
        checks = [("public package", Path(args.public_package).resolve(), REQUIRED_PUBLIC_FILES)]
    else:
        checks = [
            ("submission zip", Path(args.submission_zip), REQUIRED_SUBMISSION_FILES),
            ("flat latex zip", Path(args.flat_zip), ["main.tex", "supplementary_material.tex", "references.bib"]),
            ("submission dir", Path(args.submission_dir), REQUIRED_SUBMISSION_FILES),
            ("public dir", Path(args.public_dir), REQUIRED_PUBLIC_FILES),
            ("mendeley zip", Path(args.mendeley_zip), REQUIRED_PUBLIC_FILES),
        ]
    all_issues: list[str] = []
    lines = ["# Package Preflight Report", ""]
    for label, path, required in checks:
        if not path.exists():
            issues = [f"{path}: missing path"]
        elif path.is_file() and path.suffix.lower() == ".zip":
            issues = scan_zip(path, required)
        elif path.is_dir():
            issues = scan_directory(path, required)
        else:
            issues = [f"{path}: unsupported preflight target"]
        status = "PASS" if not issues else "FAIL"
        lines += [f"## {label}", "", f"Status: {status}", ""]
        if issues:
            lines.extend(f"- {issue}" for issue in issues)
            lines.append("")
            all_issues.extend(issues)
    final_status = "PASS" if not all_issues else "FAIL"
    lines.insert(2, f"Overall status: {final_status}")
    lines.insert(3, "")
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Package preflight {final_status}")
    print(f"Wrote {report}")
    if all_issues:
        for issue in all_issues:
            print(f"FAIL: {issue}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
