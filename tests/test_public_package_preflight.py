import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "package_preflight.py"

REQUIRED_FILES = [
    "README.md",
    "README_RESEARCH_DATA_EPSR.txt",
    "REVIEWER_ACCESS.md",
    "docs/experiment_suite.md",
    "requirements.txt",
    "requirements-validation.txt",
    "run_experiment_suite.py",
    "summarize_experiments.py",
    "paper/generate_paper_assets.py",
]


def test_documented_public_preflight_passes_from_fresh_extraction(tmp_path):
    package = tmp_path / "extracted_public_package"
    for rel in REQUIRED_FILES:
        path = package / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("clean public package fixture\n", encoding="utf-8")

    packaged_script = package / "scripts" / "package_preflight.py"
    packaged_script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT, packaged_script)

    result = subprocess.run(
        [sys.executable, "scripts/package_preflight.py", "--public-package", "."],
        cwd=package,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Package preflight PASS" in result.stdout
    assert "submission_epsr" not in result.stdout
    report = package / "package_preflight_reports" / "package_preflight_report_A.md"
    assert "Overall status: PASS" in report.read_text(encoding="utf-8")
