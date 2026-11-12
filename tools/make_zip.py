"""Build <team_name>_submission.zip in the layout required by the Amazon ML Challenge 2026.

Run from the repository root after the pipeline has written output/*.tsv:

    python tools/make_zip.py [team_name]
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
team = sys.argv[1] if len(sys.argv) > 1 else "team"
out = ROOT / f"{team}_submission.zip"

# (file on disk, path inside the zip)
files = [(ROOT / "output" / "matching_results.tsv", "output/matching_results.tsv"),
         (ROOT / "output" / "candidate_pairs.tsv", "output/candidate_pairs.tsv"),
         (ROOT / "docs" / "SOLUTION.md", "Documentation_template.md"),
         (ROOT / "README.md", "code/business_entity_resolution/README.md"),
         (ROOT / "requirements.txt", "code/business_entity_resolution/requirements.txt")]
src = sorted((ROOT / "src").glob("*.py")) + sorted((ROOT / "src").glob("*.sh"))
files += [(p, f"code/business_entity_resolution/src/{p.name}") for p in src]

with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for f, arc in files:
        assert f.exists(), f"missing {f}"
        z.write(f, arc)
        print("added", arc)
print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
