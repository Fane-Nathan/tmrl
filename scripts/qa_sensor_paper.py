"""Read-only structural checks for the anonymous sensor pilot manuscript."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader


parser = argparse.ArgumentParser()
parser.add_argument("pdf", type=Path)
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
reader = PdfReader(args.pdf)
pages = [p.extract_text() or "" for p in reader.pages]
text = "\n".join(pages)
flat = re.sub(r"\s+", " ", text)
forbidden = ["Felix", "Surjodinoto", "Keandre", "Delroy", "Samuel Philip", "BINUS", "Bina Nusantara", "D:\\Project", "wandb.ai/"]
found = [s for s in forbidden if s.casefold() in text.casefold()]
assert not found, found
assert not reader.metadata.get("/Author", ""), reader.metadata
assert 6 <= len(pages) <= 15, len(pages)
assert "AI Usage Declaration" in text and "References" in text
assert text.rfind("AI Usage Declaration") < text.rfind("References")
assert "??" not in text and "[?]" not in text
fonts = {}


def inspect_font(font):
    font = font.get_object()
    children = font.get("/DescendantFonts", [])
    if children:
        for child in children:
            inspect_font(child)
        return
    name = str(font.get("/BaseFont", "unnamed"))
    descriptor = font.get("/FontDescriptor")
    embedded = bool(descriptor and any(k in descriptor.get_object() for k in ["/FontFile", "/FontFile2", "/FontFile3"]))
    fonts[name] = embedded


def inspect_resources(resources):
    resources = resources.get_object()
    for font in resources.get("/Font", {}).get_object().values() if "/Font" in resources else []:
        inspect_font(font)
    for obj in resources.get("/XObject", {}).get_object().values() if "/XObject" in resources else []:
        obj = obj.get_object()
        if "/Resources" in obj:
            inspect_resources(obj["/Resources"])


for page in reader.pages:
    inspect_resources(page["/Resources"])
assert fonts and all(fonts.values()), fonts
bib = (args.source / "references.bib").read_text(encoding="utf-8")
years = [int(v) for v in re.findall(r"year\s*=\s*\{(\d{4})\}", bib)]
recent = sum(2022 <= y <= 2026 for y in years)
assert recent / len(years) >= 0.7
tex = "\n".join(p.read_text(encoding="utf-8") for p in args.source.glob("*.tex"))
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
cited = set(k.strip() for match in re.findall(r"\\cite\{([^}]+)\}", tex) for k in match.split(","))
assert keys == cited, {"uncited": sorted(keys - cited), "missing": sorted(cited - keys)}
report = {
    "pdf": str(args.pdf), "sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
    "pages": len(pages), "author_metadata_blank": True, "identity_search_passed": True,
    "fonts_embedded": fonts, "references": len(years), "recent_references": recent,
    "recent_fraction": recent / len(years), "all_bibliography_entries_cited": True,
    "ai_declaration_before_references": True, "unresolved_reference_markers": False,
    "visual_review": "PENDING: inspect rendered final pages before delivery",
}
args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
args.output.with_suffix(".txt").write_text("\n\n".join(f"PAGE {i+1}\n{p}" for i, p in enumerate(pages)), encoding="utf-8")
print(json.dumps(report, indent=2))
