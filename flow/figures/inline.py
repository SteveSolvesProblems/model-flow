"""Inline a figure SVG into a page, replacing the inline SVG whose aria-label starts with the given text. The page's
existing <svg> opening tag (width, role, aria-label) is kept, with the new viewBox; the palette colours from style.py become the page's CSS
colour variables, so the figure follows the light and dark themes.

python inline.py ../../publish/docs/figures/rotation_profile.svg ../viewer/post.html "Damage to predictions"
"""
import re
import sys

from style import CSS_VARS


def main(svg_path, html_path, label):
    s = open(html_path).read()
    i = s.index(f'aria-label="{label}')
    a, b = s.rindex("<svg", 0, i), s.index("</svg>", i) + len("</svg>")
    opening = s[a:s.index(">", a) + 1]
    new = open(svg_path).read()
    new = new[new.index("<svg"):new.rindex("</svg>") + len("</svg>")]
    vb = re.search(r'viewBox="[^"]*"', new[:new.index(">")]).group(0)
    opening = re.sub(r'viewBox="[^"]*"', vb, opening)                  # the figure's own size, if it changed
    body = new[new.index(">") + 1:]
    for hexc, var in CSS_VARS.items():
        body = re.sub(re.escape(hexc), f"var({var})", body, flags=re.I)
    s = s[:a] + opening + body + s[b:]
    s = s.replace('<div class="scroll" style="background:#fff;border-radius:4px">' + opening, '<div class="scroll">' + opening)
    open(html_path, "w").write(s)
    print("inlined", svg_path, "into", html_path)


if __name__ == "__main__":
    main(*sys.argv[1:4])
