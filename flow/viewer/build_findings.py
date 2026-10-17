"""Build the findings page: inject results/findings_data.json into viewer/findings.html.
python viewer/build_findings_data.py && python viewer/build_findings.py <out.html> [template.html, default findings.html]
"""
import sys
from pathlib import Path

here = Path(__file__).parent
s = (here / (sys.argv[2] if len(sys.argv) > 2 else "archive/findings.html")).read_text()
js = (here.parent / "results" / "findings_data.json").read_text().replace("</", "<\\/")
s = s.replace("__FDATA__", js)
Path(sys.argv[1]).write_text(s)
print(f"{len(s) / 1e6:.2f} MB ->", sys.argv[1])
