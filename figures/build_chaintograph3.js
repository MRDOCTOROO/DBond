// Fig. 2 — sequence -> residue graph construction (pure construction figure).
// Division of labour with the architecture figure (panels_v3): this figure owns
// graph construction only (residue nodes, backbone/peptide-bond edges, distance
// window edges, global node); the architecture figure owns the model pipeline.
// Edge-type numbering matches the code: 0 = sequence backbone (Sequence Graph
// ablation only), 1 = near (|i-j|=1), 2 = remote (2<=|i-j|<=R), 4 = global.
const pptxgen = require("pptxgenjs");

const PX_W = 1808, PX_H = 624;
const W_IN = 13.33, H_IN = 7.5;                 // LAYOUT_WIDE
const S = W_IN / PX_W;
const OY = (H_IN - PX_H * S) / 2;
const PX = (px) => px * S;
const PY = (py) => py * S + OY;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
const s = pres.addSlide();
s.background = { color: "FFFFFF" };

const GREEN = { line: "059669", head: "047857", tint: "ECFDF5" };
const PURPLE = { line: "7C3AED", head: "5B21B6", tint: "F5F3FF" };
const BLUE = { line: "2563EB", head: "1D4ED8", tint: "EFF6FF" };

const T = (x, y, w, h, text, o = {}) => s.addText(text, {
  x: PX(x), y: PY(y), w: PX(w), h: PX(h), margin: 0, valign: "middle",
  align: o.align || "center", fontFace: "Arial", fontSize: o.fs || 11,
  bold: !!o.bold, italic: !!o.italic, color: o.color || "1E293B",
});
const R = (x, y, w, h, o = {}) => s.addShape(pres.ShapeType.roundRect, {
  x: PX(x), y: PY(y), w: PX(w), h: PX(h), rectRadius: 0.02,
  fill: { color: o.fill || "FFFFFF" },
  line: { color: o.line || "94A3B8", width: o.lw || 1, dashType: o.dash ? "dash" : "solid" },
});
const OVAL = (x, y, w, h, o = {}) => s.addShape(pres.ShapeType.ellipse, {
  x: PX(x), y: PY(y), w: PX(w), h: PX(h),
  fill: { color: o.fill || "FFFFFF" },
  line: { color: o.line || "94A3B8", width: o.lw || 1, dashType: o.dash ? "dash" : "solid" },
});
const AR = (x1, y1, x2, y2, o = {}) => {
  const line = { color: o.color || "475569", width: o.lw || 1.25 };
  if (o.dash) line.dashType = "dash";
  if (o.dot) line.dashType = "sysDot";
  if (o.arrow !== false) line.endArrowType = "triangle";
  s.addShape(pres.ShapeType.line, {
    x: PX(Math.min(x1, x2)), y: PY(Math.min(y1, y2)),
    w: PX(Math.abs(x2 - x1)) || 0.01, h: PX(Math.abs(y2 - y1)) || 0.01,
    flipH: x2 < x1, flipV: y2 < y1, line,
  });
};
const curve = (x1, y1, x2, y2, cx, cy, o = {}) => {
  const n = 16, col = o.color || PURPLE.line, lw = o.lw || 1;
  const lineOpts = { color: col, lw, arrow: false };
  if (o.dash) lineOpts.dash = true;
  if (o.dot) lineOpts.dashType = "sysDot";
  for (let i = 0; i < n; i++) {
    const t0 = i / n, t1 = (i + 1) / n;
    const q = (a, b, c, t) => (1 - t) ** 2 * a + 2 * (1 - t) * t * b + t ** 2 * c;
    AR(q(x1, cx, x2, t0), q(y1, cy, y2, t0), q(x1, cx, x2, t1), q(y1, cy, y2, t1), lineOpts);
  }
};

// ---------- top: linear peptide sequence ----------
T(64, 36, 400, 26, "Linear peptide sequence P", { align: "left", bold: true, fs: 12, color: BLUE.head });
const seq = ["D", "A", "K", "B", "D", "E", "K", "B"];
const SQ = 44, GAP = 26, X0 = 200, YSQ = 76;
seq.forEach((ch, i) => {
  const x = X0 + i * (SQ + GAP);
  R(x, YSQ, SQ, SQ, { fill: "FFFFFF", line: BLUE.line, lw: 1.25 });
  T(x, YSQ, SQ, SQ, ch, { bold: true, fs: 12, color: BLUE.head });
});
// termini
T(X0 - 120, YSQ, 110, SQ, "H\u2083N\u207A", { fs: 11, color: "475569" });
T(X0 + seq.length * (SQ + GAP) + 6, YSQ, 110, SQ, "COO\u207B", { fs: 11, color: "475569" });
// backbone connectors + bond-site diamonds (all peptide bonds = prediction targets)
for (let i = 0; i < seq.length - 1; i++) {
  const xa = X0 + i * (SQ + GAP) + SQ, xb = X0 + (i + 1) * (SQ + GAP);
  AR(xa, YSQ + SQ / 2, xb, YSQ + SQ / 2, { color: GREEN.line, lw: 1.25, arrow: false });
  const d = 17;
  s.addShape(pres.ShapeType.diamond, {
    x: PX((xa + xb) / 2 - d / 2), y: PY(YSQ + SQ / 2 - d / 2), w: PX(d), h: PX(d),
    fill: { color: "D97706" }, line: { color: "B45309", width: 0.75 },
  });
}
T(X0, YSQ + SQ + 8, seq.length * (SQ + GAP), 22,
  "\u25C6 peptide bond (bond-cleavage prediction target)", { fs: 9.5, italic: true, color: "B45309" });

// ---------- transform arrow ----------
T(700, 160, 420, 26, "transform to residue graph", { italic: true, fs: 11, color: "475569" });
AR(904, 146, 904, 214, { lw: 2 });

// ---------- (a) Sequence Graph ----------
const panelA = { x: 100, y: 224, w: 720, h: 240 };
R(panelA.x, panelA.y, panelA.w, panelA.h, { fill: GREEN.tint, line: GREEN.line, dash: true });
T(panelA.x + 16, panelA.y + 4, 420, 30, "(a) Sequence Graph", { align: "left", bold: true, fs: 12, color: GREEN.head });
T(panelA.x + 16, panelA.y + 34, 560, 24, "adjacent residues only \u00B7 window R = 1", { align: "left", fs: 10, color: "475569" });
const na = 6, na_x0 = panelA.x + 90, na_gap = 96, na_y = panelA.y + 120;
const nodeA = (i) => [na_x0 + i * na_gap, na_y];
for (let i = 0; i < na; i++) {
  const [x, y] = nodeA(i);
  OVAL(x, y, 48, 48, { fill: GREEN.line, line: "047857" });
  T(x, y, 48, 48, i === na - 1 ? "v_L" : "v" + (i + 1), { bold: true, fs: 10.5, color: "FFFFFF" });
  if (i < na - 2) AR(x + 48, y + 24, x + na_gap, y + 24, { color: GREEN.line, lw: 1.75, arrow: false });
}
// ellipsis + last node
T(nodeA(na - 1)[0] - 34, na_y, 30, 48, "\u22EF", { fs: 14, color: "475569" });
T(panelA.x + 30, na_y + 76, 660, 24, "\u2014\u2014 type 0 backbone edge (sequence adjacency; sequence-strategy graphs only)",
  { align: "left", fs: 9.5, color: GREEN.head });

// ---------- (b) Distance Graph ----------
const panelB = { x: 940, y: 224, w: 760, h: 240 };
R(panelB.x, panelB.y, panelB.w, panelB.h, { fill: GREEN.tint, line: GREEN.line, dash: true });
T(panelB.x + 16, panelB.y + 4, 420, 30, "(b) Distance Graph", { align: "left", bold: true, fs: 12, color: GREEN.head });
T(panelB.x + 16, panelB.y + 34, 620, 24, "backbone + distance window \u00B7 R = 5 (|i \u2212 j| \u2264 5)", { align: "left", fs: 10, color: "475569" });
const nb = 6, nb_x0 = panelB.x + 90, nb_gap = 106, nb_y = panelB.y + 150;
const v_g = { x: panelB.x + 300, y: panelB.y + 52 };
OVAL(v_g.x, v_g.y, 48, 48, { fill: "EDE9FE", line: PURPLE.line });
T(v_g.x, v_g.y, 48, 48, "v_g", { bold: true, fs: 10.5, color: PURPLE.head });
const nodeB = (i) => [nb_x0 + i * nb_gap, nb_y];
for (let i = 0; i < nb; i++) {
  const [x, y] = nodeB(i);
  OVAL(x, y, 48, 48, { fill: GREEN.line, line: "047857" });
  T(x, y, 48, 48, i === nb - 1 ? "v_L" : "v" + (i + 1), { bold: true, fs: 10.5, color: "FFFFFF" });
  if (i < nb - 2) AR(x + 48, y + 24, x + nb_gap, y + 24, { color: GREEN.line, lw: 1.75, arrow: false });
}
T(nodeB(nb - 1)[0] - 38, nb_y, 30, 48, "\u22EF", { fs: 14, color: "475569" });
// type 2 remote arcs (|i-j| = 2 and 3)
curve(nodeB(0)[0] + 24, nb_y, nodeB(2)[0] + 24, nb_y, (nodeB(0)[0] + nodeB(2)[0]) / 2 + 24, nb_y - 46);
curve(nodeB(2)[0] + 24, nb_y, nodeB(4)[0] + 24, nb_y, (nodeB(2)[0] + nodeB(4)[0]) / 2 + 24, nb_y - 46);
curve(nodeB(1)[0] + 24, nb_y, nodeB(4)[0] + 24, nb_y, (nodeB(1)[0] + nodeB(4)[0]) / 2 + 24, nb_y - 70);
// type 4 global dotted curves from v_g
curve(v_g.x + 24, v_g.y + 48, nodeB(0)[0] + 24, nb_y, v_g.x - 60, nb_y - 40, { dot: true, lw: 0.9 });
curve(v_g.x + 24, v_g.y + 48, nodeB(2)[0] + 24, nb_y, v_g.x + 24, nb_y - 10, { dot: true, lw: 0.9 });
curve(v_g.x + 24, v_g.y + 48, nodeB(5)[0] + 24, nb_y, v_g.x + 140, nb_y - 50, { dot: true, lw: 0.9 });

// ---------- legend band ----------
R(100, 488, 1600, 96, { fill: "FFFFFF", line: "CBD5E1", dash: true });
T(116, 494, 200, 26, "Legend", { align: "left", bold: true, fs: 12, color: "64748B" });
AR(116, 532, 156, 532, { color: GREEN.line, lw: 2, arrow: false });
T(164, 520, 330, 26, "type 0 / type 1 near \u00B7 |i \u2212 j| = 1", { align: "left", fs: 9.5, color: "475569" });
AR(540, 532, 580, 532, { color: PURPLE.line, lw: 1.75, arrow: false, dash: true });
T(588, 520, 380, 26, "type 2 remote \u00B7 2 \u2264 |i \u2212 j| \u2264 R", { align: "left", fs: 9.5, color: "475569" });
AR(1010, 532, 1050, 532, { color: PURPLE.line, lw: 1.5, arrow: false, dot: true });
T(1058, 520, 380, 26, "type 4 global \u00B7 v_g \u2194 residues", { align: "left", fs: 9.5, color: "475569" });
T(1440, 520, 250, 26, "window radius R = 5", { align: "left", fs: 9.5, bold: true, color: GREEN.head });

pres.writeFile({ fileName: "figures/chaintograph3.pptx" })
  .then(() => console.log("written: figures/chaintograph3.pptx"));
