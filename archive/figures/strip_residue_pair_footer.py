# Strip the bottom footer (N/A legend + Pearson/MAE stats line) from the
# archived residue-pair matrix SVG and crop the canvas to the panel content.
# Source of paper Fig 4 (film fold 9075 run, r=0.839 / MAE=5.0pp / 274 pairs).
import re

SRC = r"D:\UserData\Desktop\Github-project\Dbond2\DBond\figures\residue_pair_film_fold9075\residue_pair_matrix.svg"
DST = r"D:\UserData\Desktop\Github-project\Dbond2\DBond\figures\residue_pair_film_fold9075\residue_pair_matrix_nofooter.svg"
CROP_Y = "554"  # lowest legitimate text = xlabel baseline 542.35; +12pt pad

src = open(SRC, encoding="utf-8").read()


def cut_group(s: str, marker: str) -> str:
    """Remove the balanced <g ...> block that starts at `marker`."""
    start = s.find(marker)
    assert start != -1, f"marker not found: {marker}"
    depth, i = 0, start
    token = re.compile(r"<g[ >]|</g>")
    for m in token.finditer(s, start):
        depth += 1 if m.group(0).startswith("<g") else -1
        if depth == 0:
            end = m.end()
            break
    else:
        raise AssertionError("unbalanced group: " + marker)
    return s[:start] + s[end:]


for marker in ('<g id="legend_1">', '<g id="text_452">'):
    src = cut_group(src, marker)

src, n1 = re.subn(r'viewBox="0 0 2079\.582656 664\.696706"',
                  f'viewBox="0 0 2079.582656 {CROP_Y}"', src)
src, n2 = re.subn(r'height="664\.696706pt"', f'height="{CROP_Y}pt"', src)
assert n1 == 1 and n2 == 1, (n1, n2)

for w in ("Pearson", "no observation", "legend_1"):
    assert w not in src, f"residual footer content: {w}"

open(DST, "w", encoding="utf-8").write(src)
print("written:", DST, "| size:", len(src))
