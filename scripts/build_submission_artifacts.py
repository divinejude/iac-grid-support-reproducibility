"""Build local EPSR submission artifacts from the authoritative manuscript tree."""

from __future__ import annotations

import hashlib
import re
import shutil
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission_epsr"
MANUSCRIPT = SUBMISSION / "manuscript"
SUBMISSION_FILES = SUBMISSION / "submission_files"
REPORT_DIR = ROOT / "iac_paper_research_management" / "stage_05_reproducibility"
FLAT_ZIP = SUBMISSION_FILES / "latex_source_main_EPSR_flat.zip"
ROOT_ZIP = ROOT / "submission_epsr.zip"


TEXT_FILES = [
    "additional_comments_EPSR.txt",
    "cover_letter_EPSR.txt",
    "data_statement_EPSR.txt",
    "declarations_EPSR.txt",
    "highlights_EPSR.txt",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flat_tex(text: str) -> str:
    text = text.replace(r"\graphicspath{{figures/}}", r"\graphicspath{{./}}")
    text = re.sub(r"\\input\{tables/([^}]+)\}", r"\\input{\1}", text)
    return text


def imported_assets(tex_path: Path) -> set[str]:
    text = tex_path.read_text(encoding="utf-8")
    figures = {Path(match).name for match in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)}
    tables = {Path(match).name for match in re.findall(r"\\input\{tables/([^}]+)\}", text)}
    return figures | tables


def write_docx_from_text(text_path: Path, docx_path: Path) -> None:
    paragraphs = text_path.read_text(encoding="utf-8").splitlines()
    body = []
    for paragraph in paragraphs:
        if paragraph.strip():
            body.append(f"<w:p><w:r><w:t>{escape(paragraph)}</w:t></w:r></w:p>")
        else:
            body.append("<w:p/>")
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(body)}<w:sectPr/></w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/></Relationships>'
    )
    docx_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document_xml)


def build_flat_zip() -> list[str]:
    assets = imported_assets(MANUSCRIPT / "main.tex") | imported_assets(MANUSCRIPT / "supplementary_material.tex")
    entries: dict[str, bytes] = {
        "main.tex": flat_tex((MANUSCRIPT / "main.tex").read_text(encoding="utf-8")).encode("utf-8"),
        "supplementary_material.tex": flat_tex((MANUSCRIPT / "supplementary_material.tex").read_text(encoding="utf-8")).encode(
            "utf-8"
        ),
        "references.bib": (MANUSCRIPT / "references.bib").read_bytes(),
    }
    for name in sorted(assets):
        source = MANUSCRIPT / "tables" / name
        if not source.exists():
            source = MANUSCRIPT / "figures" / name
        if not source.exists():
            raise FileNotFoundError(f"Missing manuscript asset for flat archive: {name}")
        entries[name] = source.read_bytes()
    with zipfile.ZipFile(FLAT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(entries):
            archive.writestr(name, entries[name])
    return sorted(entries)


def build_root_zip() -> list[str]:
    files = [
        "additional_comments_EPSR.txt",
        "cover_letter_EPSR.txt",
        "data_statement_EPSR.txt",
        "declarations_EPSR.docx",
        "declarations_EPSR.txt",
        "graphical_abstract_EPSR.pdf",
        "graphical_abstract_EPSR.png",
        "highlights_EPSR.txt",
        "latex_source_main_EPSR_flat.zip",
        "manuscript_EPSR.pdf",
        "supplementary_material_EPSR.pdf",
    ]
    with zipfile.ZipFile(ROOT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            source = SUBMISSION_FILES / name
            if not source.exists():
                raise FileNotFoundError(source)
            archive.write(source, arcname=name)
    return files


def main() -> None:
    SUBMISSION_FILES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MANUSCRIPT / "main.pdf", SUBMISSION_FILES / "manuscript_EPSR.pdf")
    shutil.copy2(MANUSCRIPT / "supplementary_material.pdf", SUBMISSION_FILES / "supplementary_material_EPSR.pdf")
    write_docx_from_text(SUBMISSION_FILES / "declarations_EPSR.txt", SUBMISSION_FILES / "declarations_EPSR.docx")
    flat_entries = build_flat_zip()
    root_entries = build_root_zip()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Local Submission Artifact Build - Agent A",
        "",
        f"- Authoritative manuscript tree: `{MANUSCRIPT.relative_to(ROOT)}`",
        f"- Manuscript PDF: `{(SUBMISSION_FILES / 'manuscript_EPSR.pdf').relative_to(ROOT)}` "
        f"`{sha256(SUBMISSION_FILES / 'manuscript_EPSR.pdf')}`",
        f"- Supplementary PDF: `{(SUBMISSION_FILES / 'supplementary_material_EPSR.pdf').relative_to(ROOT)}` "
        f"`{sha256(SUBMISSION_FILES / 'supplementary_material_EPSR.pdf')}`",
        f"- Flat source ZIP: `{FLAT_ZIP.relative_to(ROOT)}` with {len(flat_entries)} flat entries "
        f"`{sha256(FLAT_ZIP)}`",
        f"- Root submission ZIP: `{ROOT_ZIP.relative_to(ROOT)}` with {len(root_entries)} entries `{sha256(ROOT_ZIP)}`",
        "",
        "## Flat ZIP Entries",
        "",
    ]
    lines.extend(f"- `{entry}`" for entry in flat_entries)
    (REPORT_DIR / "submission_artifact_build_A.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Built {FLAT_ZIP}")
    print(f"Built {ROOT_ZIP}")
    print(f"Wrote {REPORT_DIR / 'submission_artifact_build_A.md'}")


if __name__ == "__main__":
    main()
