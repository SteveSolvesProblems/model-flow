"""Build the Model Flow post: inject results/findings_data.json into viewer/post.html.
python viewer/build_findings_data.py && python viewer/build_post.py <out.html>
"""
import sys
from pathlib import Path

here = Path(__file__).parent
s = (here / "post.html").read_text()
js = (here.parent / "results" / "findings_data.json").read_text().replace("</", "<\\/")
Path(sys.argv[1]).write_text(s.replace("__FDATA__", js))
print(f"{len(s) / 1e6:.2f} MB ->", sys.argv[1])
