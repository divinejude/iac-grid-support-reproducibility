"""Synchronize generated assets into the authoritative EPSR manuscript tree.

Authoritative manuscript tree:
    submission_epsr/manuscript/

The `paper/` directory is an intermediate generation workspace. This script
copies only assets imported by the active manuscript or supplement, verifies
source/destination hashes, and writes an auditable manifest.
"""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "submission_epsr" / "manuscript"
PAPER = ROOT / "paper"
HEADROOM = ROOT / "experiments" / "stage01_headroom_ablation_A"
REPORT_DIR = ROOT / "iac_paper_research_management" / "stage_05_reproducibility"


@dataclass(frozen=True)
class Asset:
    manuscript_asset: str
    generator: str
    input_source: str
    generated_path: Path
    final_path: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def imported_assets(tex_path: Path) -> set[str]:
    text = tex_path.read_text(encoding="utf-8")
    figures = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text))
    tables = set(re.findall(r"\\input\{(tables/[^}]+)\}", text))
    return figures | tables


def build_manifest() -> list[Asset]:
    imported = imported_assets(MANUSCRIPT / "main.tex") | imported_assets(MANUSCRIPT / "supplementary_material.tex")
    assets: list[Asset] = []
    for name in sorted(imported):
        if name.startswith("tables/"):
            filename = Path(name).name
            if filename == "headroom_tradeoff_ablation.tex":
                source = HEADROOM / filename
                generator = "python analyze_headroom_tradeoff.py"
                input_source = "experiments/stage01_headroom_ablation_A"
            else:
                source = PAPER / "tables" / filename
                generator = "python paper/generate_paper_assets.py or validation/calibration script"
                input_source = "frozen experiment outputs and validation metrics"
            dest = MANUSCRIPT / "tables" / filename
        else:
            filename = Path(name).name
            if filename.startswith("headroom_tradeoff_ablation"):
                source = HEADROOM / filename
                generator = "python analyze_headroom_tradeoff.py"
                input_source = "experiments/stage01_headroom_ablation_A"
            else:
                source = PAPER / "figures" / filename
                generator = "python paper/generate_paper_assets.py or validation/calibration script"
                input_source = "frozen experiment outputs and validation metrics"
            dest = MANUSCRIPT / "figures" / filename
        assets.append(Asset(name, generator, input_source, source, dest))
    return assets


def write_reports(rows: list[dict[str, str]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "asset_sync_manifest_A.csv"
    md_path = REPORT_DIR / "asset_sync_manifest_A.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "manuscript_asset",
                "generator",
                "input",
                "generated_path",
                "final_imported_path",
                "sha256",
                "verified",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Authoritative Manuscript Asset Sync Manifest - Agent A",
        "",
        "| Manuscript asset | Generator | Input | Generated path | Final imported path | Hash/content verified? |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {manuscript_asset} | {generator} | {input} | {generated_path} | {final_imported_path} | {verified} |".format(
                **row
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    mismatched: list[str] = []
    for asset in build_manifest():
        if not asset.generated_path.exists():
            missing.append(str(asset.generated_path.relative_to(ROOT)))
            continue
        asset.final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(asset.generated_path, asset.final_path)
        src_hash = sha256(asset.generated_path)
        dst_hash = sha256(asset.final_path)
        verified = src_hash == dst_hash
        if not verified:
            mismatched.append(str(asset.final_path.relative_to(ROOT)))
        rows.append(
            {
                "manuscript_asset": asset.manuscript_asset,
                "generator": asset.generator,
                "input": asset.input_source,
                "generated_path": str(asset.generated_path.relative_to(ROOT)).replace("\\", "/"),
                "final_imported_path": str(asset.final_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": dst_hash,
                "verified": "YES" if verified else "NO",
            }
        )
    write_reports(rows)
    if missing or mismatched:
        if missing:
            print("Missing generated assets:")
            for item in missing:
                print(f"  {item}")
        if mismatched:
            print("Hash mismatches:")
            for item in mismatched:
                print(f"  {item}")
        raise SystemExit(1)
    print(f"Synchronized and verified {len(rows)} manuscript assets.")
    print(f"Wrote {REPORT_DIR / 'asset_sync_manifest_A.md'}")


if __name__ == "__main__":
    main()
