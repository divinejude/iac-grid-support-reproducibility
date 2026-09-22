import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "package_preflight.py"


PUBLIC_REQUIRED = [
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

SUBMISSION_REQUIRED = [
    "manuscript_EPSR.pdf",
    "supplementary_material_EPSR.pdf",
    "latex_source_main_EPSR_flat.zip",
    "cover_letter_EPSR.txt",
    "data_statement_EPSR.txt",
    "declarations_EPSR.txt",
    "highlights_EPSR.txt",
]


def write_file(path: Path, content: str = "clean\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_zip(path: Path, entries: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)


def make_clean_package(tmp_path: Path) -> dict[str, Path]:
    public_dir = tmp_path / "public"
    submission_dir = tmp_path / "submission_files"
    for rel in PUBLIC_REQUIRED:
        write_file(public_dir / rel)
    write_file(public_dir / "validate_td_cosim.py", '"""Explicit loose-coupled T&D consistency check."""\n')
    write_file(public_dir / "validate_averaged_converter.py", '"""full P/Q MPC validation."""\n')

    flat_entries = {
        "main.tex": "\\documentclass{article}\\begin{document}clean\\end{document}\n",
        "supplementary_material.tex": "\\documentclass{article}\\begin{document}clean\\end{document}\n",
        "references.bib": "",
        "td_cosim.tex": "\\caption{Explicit loose-coupled T\\&D consistency check.}\n",
    }
    build_zip(submission_dir / "latex_source_main_EPSR_flat.zip", flat_entries)

    for rel in SUBMISSION_REQUIRED:
        if rel != "latex_source_main_EPSR_flat.zip":
            write_file(submission_dir / rel)
    root_entries = {rel: "clean\n" for rel in SUBMISSION_REQUIRED if rel != "latex_source_main_EPSR_flat.zip"}
    root_entries["latex_source_main_EPSR_flat.zip"] = "placeholder\n"
    build_zip(tmp_path / "submission_epsr.zip", root_entries)

    mendeley_entries = {rel: "clean\n" for rel in PUBLIC_REQUIRED}
    mendeley_entries["validate_td_cosim.py"] = '"""Explicit loose-coupled T&D consistency check."""\n'
    mendeley_entries["paper/tables/td_cosim.tex"] = "\\caption{Explicit loose-coupled T\\&D consistency check.}\n"
    build_zip(tmp_path / "mendeley.zip", mendeley_entries)

    return {
        "submission_zip": tmp_path / "submission_epsr.zip",
        "flat_zip": submission_dir / "latex_source_main_EPSR_flat.zip",
        "submission_dir": submission_dir,
        "public_dir": public_dir,
        "mendeley_zip": tmp_path / "mendeley.zip",
        "report": tmp_path / "report.md",
    }


def run_preflight(paths: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--submission-zip",
            str(paths["submission_zip"]),
            "--flat-zip",
            str(paths["flat_zip"]),
            "--submission-dir",
            str(paths["submission_dir"]),
            "--public-dir",
            str(paths["public_dir"]),
            "--mendeley-zip",
            str(paths["mendeley_zip"]),
            "--report",
            str(paths["report"]),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_clean_minimal_package_passes_preflight(tmp_path):
    paths = make_clean_package(tmp_path)

    result = run_preflight(paths)

    assert result.returncode == 0, result.stdout + result.stderr


def test_preflight_rejects_prototype_explicit_td_caption(tmp_path):
    paths = make_clean_package(tmp_path)
    write_file(
        paths["public_dir"] / "paper" / "tables" / "td_cosim.tex",
        "\\caption{Prototype explicit T\\&D co-simulation.}\n",
    )

    result = run_preflight(paths)

    assert result.returncode != 0
    assert "stale prototype T&D wording" in result.stdout


def test_preflight_rejects_full_pq_dmpc_validation_script(tmp_path):
    paths = make_clean_package(tmp_path)
    write_file(paths["public_dir"] / "validate_averaged_converter.py", '"""full P/Q DMPC command trace."""\n')

    result = run_preflight(paths)

    assert result.returncode != 0
    assert "stale full P/Q DMPC label" in result.stdout
