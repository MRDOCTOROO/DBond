import html
import uuid
from pathlib import Path

OUT = Path(r"D:\UserData\Desktop\graph-structures-reference-replica.drawio")
W, H = 1720, 1080

cells = []
_next = 2


def esc(s):
    return html.escape(s, quote=True)


def add_cell(value, style, x, y, w, h, parent="1", vertex=True):
    global _next
    cid = str(_next)
    _next += 1
    vflag = ' vertex="1"' if vertex else ''
    cells.append(
        f'<mxCell id="{cid}" value="{esc(value)}" style="{style}"{vflag} parent="{parent}">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
        f'</mxCell>'
    )
    return cid


def add_rect(value, x, y, w, h, fill="#ffffff", stroke="#d9d9d9", rounded=1, fs=22, bold=False, align="center", valign="middle", extra=""):
    style = (
        f"rounded={rounded};whiteSpace=wrap;html=1;arcSize=8;fillColor={fill};strokeColor={stroke};"
        f"fontFamily=Times New Roman;fontSize={fs};fontColor=#111111;align={align};verticalAlign={valign};"
        f"spacing=4;{extra}"
    )
    if bold:
        style += "fontStyle=1;"
    return add_cell(value, style, x, y, w, h)


def add_text(value, x, y, w, h, fs=22, bold=False, align="center", color="#111111"):
    style = (
        f"text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;rounded=0;"
        f"fontFamily=Times New Roman;fontSize={fs};fontColor={color};align={align};verticalAlign=middle;"
    )
    if bold:
        style += "fontStyle=1;"
    return add_cell(value, style, x, y, w, h)


def add_node(label, x, y, r=27, fill="#f8fbff", stroke="#23466d"):
    style = (
        f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor={fill};strokeColor={stroke};strokeWidth=2;"
        f"fontFamily=Times New Roman;fontSize=24;fontStyle=2;fontColor=#111111;align=center;verticalAlign=middle;"
    )
    return add_cell(label, style, x - r, y - r, 2 * r, 2 * r)


def add_line(x1, y1, x2, y2, color, width=2, dashed=False, dash="6 6", dotted=False, curved=False, waypoints=None, end=False):
    global _next
    cid = str(_next)
    _next += 1
    style = f"endArrow={'none' if not end else 'classic'};html=1;rounded=0;strokeWidth={width};strokeColor={color};"
    if dashed:
        style += f"dashed=1;dashPattern={dash};"
    if dotted:
        style += "dashed=1;dashPattern=1 4;"
    if curved:
        style += "curved=1;"
    points = ""
    if waypoints:
        pts = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in waypoints)
        points = f'<Array as="points">{pts}</Array>'
    cells.append(
        f'<mxCell id="{cid}" value="" style="{style}" edge="1" parent="1">'
        f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>'
        f'<mxPoint x="{x2}" y="{y2}" as="targetPoint"/>{points}</mxGeometry></mxCell>'
    )
    return cid


def panel(x, y, w, h, title, fill, stroke):
    add_rect("", x, y, w, h, fill="#ffffff", stroke="#d3d3d3", rounded=1, fs=1, extra="shadow=0;")
    add_rect(title, x, y, w, 80, fill=fill, stroke=stroke, rounded=1, fs=24, bold=True)


def graph_nodes(xs, y, labels):
    ids = []
    for x, lab in zip(xs, labels):
        ids.append(add_node(lab, x, y))
    return ids


def seq_edges(xs, y, color="#3366cc"):
    for a, b in zip(xs[:4], xs[1:4]):
        add_line(a + 27, y, b - 27, y, color, 3)
    add_line(xs[4] + 30, y, xs[5] - 27, y, color, 3)


def arc(x1, x2, y, height, color, dashed=False, dash="8 6", width=2.2):
    mid = (x1 + x2) / 2
    add_line(x1, y - 22, x2, y - 22, color, width, dashed=dashed, dash=dash, curved=True, waypoints=[(mid, y - height)])


def bottom_arc(x1, x2, y, depth, color, dashed=False, dash="12 6 4 6", width=2.2):
    mid = (x1 + x2) / 2
    add_line(x1, y + 22, x2, y + 22, color, width, dashed=dashed, dash=dash, curved=True, waypoints=[(mid, y + depth)])


# Panels
panel(8, 8, 565, 538, "(a) Sequence Graph<br>(d<sub>max</sub> = 1)", "#eef7ff", "#d8e8f8")
panel(573, 8, 565, 538, "(b) Distance Graph<br>(d<sub>max</sub> &gt; 1)", "#eefaf0", "#d8ead7")
panel(1138, 8, 574, 538, "(c) Hybrid Graph<br>(Distance + Long-range)", "#eefaf0", "#d8ead7")
panel(8, 560, 970, 512, "(d) Global Node (for all graphs)", "#f6f1ff", "#e4d9fa")
panel(980, 560, 732, 512, "(e) Edge Type Legend", "#f6f1ff", "#e4d9fa")

labels = ["v<sub>1</sub>", "v<sub>2</sub>", "v<sub>3</sub>", "v<sub>4</sub>", "v<sub>L−1</sub>", "v<sub>L</sub>"]
blue = "#3366cc"
green = "#55b95a"
orange = "#f3a629"
purple = "#9b3fd2"
stroke = "#23466d"

# Panel a
xsa = [50, 132, 214, 296, 446, 530]
ytop = 250
graph_nodes(xsa, ytop, labels)
seq_edges(xsa, ytop, blue)
add_text("⋯", 355, 232, 55, 36, fs=28, bold=True)
add_text("Only adjacent residues connected", 94, 344, 390, 34, fs=24)
add_text("Edges: type 0 (sequence)", 148, 388, 285, 34, fs=24)

# Panel b
xsb = [620, 702, 784, 866, 1010, 1090]
graph_nodes(xsb, ytop, labels)
seq_edges(xsb, ytop, stroke)
for a, b, h in [(620, 702, 75), (702, 784, 75), (784, 866, 75), (620, 784, 90), (702, 866, 90), (866, 1010, 75)]:
    arc(a, b, ytop, h, green)
for a, b, h in [(620, 866, 132), (702, 1010, 150), (784, 1090, 128), (866, 1090, 75)]:
    arc(a, b, ytop, h, orange, dashed=True, dash="8 6")
add_text("⋯", 930, 232, 55, 36, fs=28, bold=True)
add_text("Connect residues with folding-adjusted<br>distance <i>d</i><sub>ij</sub> ≤ <i>d</i><sub>max</sub>", 642, 338, 435, 58, fs=24)
add_line(675, 428, 750, 428, green, 3)
add_text("type 1 (short): <i>d</i><sub>ij</sub> ≤ 2", 765, 410, 265, 38, fs=22, align="left")
add_line(675, 474, 750, 474, orange, 3, dashed=True, dash="8 6")
add_text("type 2 (long): <i>d</i><sub>ij</sub> &gt; 2", 765, 456, 280, 38, fs=22, align="left")

# Panel c
xsc = [1188, 1270, 1352, 1434, 1584, 1670]
graph_nodes(xsc, ytop, labels)
seq_edges(xsc, ytop, stroke)
for a, b, h in [(1188, 1270, 75), (1270, 1352, 75), (1352, 1434, 75), (1188, 1352, 90), (1270, 1434, 90), (1434, 1584, 75), (1584, 1670, 75)]:
    arc(a, b, ytop, h, green)
for a, b, h in [(1188, 1434, 132), (1270, 1584, 150), (1352, 1670, 128), (1434, 1670, 75)]:
    arc(a, b, ytop, h, orange, dashed=True, dash="8 6")
for a, b in [(1188, 1584), (1270, 1670)]:
    arc(a, b, ytop, 98, purple, dashed=True, dash="12 6 4 6")
    bottom_arc(a, b, ytop, 100, purple, dashed=True, dash="12 6 4 6")
add_text("⋯", 1505, 232, 55, 36, fs=28, bold=True)
add_text("Distance edges + long-range edges", 1238, 340, 420, 36, fs=24)
add_line(1210, 405, 1285, 405, green, 3)
add_text("type 1 (short): <i>d</i><sub>ij</sub> ≤ 2", 1300, 386, 290, 38, fs=22, align="left")
add_line(1210, 450, 1285, 450, orange, 3, dashed=True, dash="8 6")
add_text("type 2 (long): <i>d</i><sub>ij</sub> &gt; 2", 1300, 431, 300, 38, fs=22, align="left")
add_line(1210, 495, 1285, 495, purple, 3, dashed=True, dash="12 6 4 6")
add_text("type 3 (long-range): |<i>i</i> − <i>j</i>| = <i>k</i>Δ", 1300, 476, 390, 38, fs=22, align="left")

# Panel d
xsd = [52, 132, 212, 292, 410, 490]
yd = 925
graph_nodes(xsd, yd, labels)
seq_edges(xsd, yd, stroke)
add_text("⋯", 330, 907, 55, 36, fs=28, bold=True)
vgx, vgy = 300, 720
add_node("v<sub>g</sub>", vgx, vgy, r=28, fill="#e9d5ff", stroke="#7e3fb7")
for x in xsd:
    add_line(vgx, vgy + 28, x, yd - 27, "#b06adf", 1.8, dotted=True)
add_text("Connected to all residues (type 4: global)", 78, 990, 385, 34, fs=22)
# formula callout
add_rect("Global node representation:", 470, 666, 495, 70, fill="#f4edff", stroke="#d7c5f0", fs=23, bold=True, align="left")
add_rect("<b>h</b><sub>g</sub><sup>(0)</sup> = <b>e</b><sub><i>learn</i></sub> + <b>MLP</b><sub>g</sub> ([<b>h</b><sup><i>state</i></sup>; <b>h</b><sup><i>env</i></sup>])", 470, 736, 495, 92, fill="#ffffff", stroke="#d7c5f0", fs=28, bold=True, align="left")
add_rect("where<br>• <b>e</b><sub><i>learn</i></sub> : learnable global embedding<br>• <b>h</b><sup><i>state</i></sup> : encoded precursor ion state<br>• <b>h</b><sup><i>env</i></sup> : encoded experimental environment<br>• <b>MLP</b><sub>g</sub>(·) : MLP with ReLU and Dropout", 525, 828, 440, 205, fill="#ffffff", stroke="#d7c5f0", fs=20, align="left")
add_line(vgx + 28, vgy - 18, 525, 666, "#b06adf", 1.5, dashed=True, dash="4 4", waypoints=[(335, 666), (470, 666)])
add_line(vgx + 28, vgy + 15, 525, 828, "#b06adf", 1.5, dashed=True, dash="4 4", waypoints=[(385, 828), (470, 828)])

# Panel e table
x0, y0 = 990, 618
colw = [98, 165, 162, 295]
rowh = [50, 78, 78, 78, 78, 78]
headers = ["Type", "Name", "Visual", "Definition"]
xx = x0
for i, htxt in enumerate(headers):
    add_rect(htxt, xx, y0, colw[i], rowh[0], fill="#f3f1f6", stroke="#d6d6d6", fs=21, bold=True)
    xx += colw[i]
rows = [
    ["0", "sequence<br>(backbone)", "", "Adjacent residues<br>(|<i>i</i> − <i>j</i>| = 1)<br>(Sequence Graph only)"],
    ["1", "distance<br>(short)", "", "Folding-adjusted distance<br><i>d</i><sub>ij</sub> ≤ 2"],
    ["2", "distance<br>(long)", "", "Folding-adjusted distance<br><i>d</i><sub>ij</sub> &gt; 2 and ≤ <i>d</i><sub>max</sub>"],
    ["3", "long-range", "", "|<i>i</i> − <i>j</i>| = <i>k</i>Δ<br>(<i>k</i> = 1, . . . , <i>K</i>)"],
    ["4", "global", "", "Connected to global node <i>v</i><sub>g</sub><br>(All graphs)"],
]
colors = [blue, green, orange, purple, purple]
dashs = [None, None, "8 6", "12 6 4 6", "1 5"]
cy = y0 + rowh[0]
for r, row in enumerate(rows):
    xx = x0
    for c, txt in enumerate(row):
        add_rect(txt, xx, cy, colw[c], rowh[r + 1], fill="#ffffff", stroke="#d6d6d6", fs=20 if c != 3 else 19, bold=(c == 0), align="center" if c != 3 else "center")
        xx += colw[c]
    ly = cy + rowh[r + 1] / 2
    add_line(x0 + colw[0] + colw[1] + 33, ly, x0 + colw[0] + colw[1] + colw[2] - 33, ly, colors[r], 3, dashed=bool(dashs[r]), dash=dashs[r] or "")
    cy += rowh[r + 1]

# XML
root = "\n".join(cells)
xml = f'''<mxfile host="app.diagrams.net" modified="2026-07-14T00:00:00.000Z" agent="ZCode" version="24.7.17" type="device">
  <diagram id="{uuid.uuid4().hex[:12]}" name="Page-1">
    <mxGraphModel dx="1720" dy="1080" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="1" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {root}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''
OUT.write_text(xml, encoding="utf-8")
print(OUT)
