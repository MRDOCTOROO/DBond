// DBond-GT minimal architecture (paper style, cf. DBond / DBond-AF figures):
// two dashed panels, white boxes, thick vertical arrows, small input squares.
const pptxgen = require("pptxgenjs");

const PX_W = 1620, PX_H = 640;
const W_IN = 13.33, H_IN = 7.5;
const S = W_IN / PX_W;
const OY = (H_IN - PX_H * S) / 2;
const PX = (px) => px * S;
const PY = (py) => py * S + OY;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
const s = pres.addSlide();
s.background = { color: "FFFFFF" };

const INK = "334155", BOX = "FFFFFF", BOXL = "5A6B87";
const A_FILL = "FDF2F4", A_LINE = "E09AA6", A_TXT = "9D3B4A";   // left panel
const B_FILL = "EAF1FB", B_LINE = "8FAEE0", B_TXT = "2E5CA8";   // right panel

const R = (x, y, w, h, o = {}) => s.addShape(pres.ShapeType.roundRect, {
  x: PX(x), y: PY(y), w: PX(w), h: PX(h), rectRadius: 0.015,
  fill: { color: o.fill || BOX },
  line: { color: o.line || BOXL, width: o.lw || 0.75, dashType: o.dash ? "dash" : "solid" },
});
const T = (x, y, w, h, text, o = {}) => s.addText(text, {
  x: PX(x), y: PY(y), w: PX(w), h: PX(h), margin: 0, valign: "middle",
  align: o.align || "center", fontFace: "Arial", fontSize: o.fs || 11,
  bold: !!o.bold, italic: !!o.italic, color: o.color || INK,
});
const AR = (x1, y1, x2, y2, o = {}) => s.addShape(pres.ShapeType.line, {
  x: PX(Math.min(x1, x2)), y: PY(Math.min(y1, y2)),
  w: PX(Math.abs(x2 - x1)) || 0.01, h: PX(Math.abs(y2 - y1)) || 0.01,
  flipH: x2 < x1, flipV: y2 < y1,
  line: Object.assign({ color: INK, width: o.lw || 1.75 },
    o.arrow === false ? {} : { endArrowType: "triangle" }),
});
const SQ = (x, y, c, w = 20) => R(x, y, w, w, { fill: c, line: "FFFFFF", lw: 0.5 });

// ---------- panels ----------
R(40, 40, 780, 560, { fill: A_FILL, line: A_LINE, dash: true, lw: 1 });
T(62, 52, 400, 30, "Graph encoder", { align: "left", bold: true, fs: 13, color: A_TXT });
R(860, 40, 720, 560, { fill: B_FILL, line: B_LINE, dash: true, lw: 1 });
T(882, 52, 460, 30, "Key-level prediction", { align: "left", bold: true, fs: 13, color: B_TXT });

// ---------- left: sequence path ----------
["2E7D6B", "3E9C86", "2E7D6B", "E0B25A", "D98E4A", "7FC8B5"].forEach((c, i) => SQ(84 + i * 26, 112, c));
T(84, 138, 220, 22, "sequence", { italic: true, fs: 9, color: "64748B" });
AR(160, 162, 160, 192, { lw: 2 });
R(60, 192, 200, 52); T(60, 192, 200, 52, "Amino Acid Embedding", { bold: true, fs: 10.5 });
AR(160, 244, 160, 274, { lw: 2 });
R(60, 274, 200, 52); T(60, 274, 200, 52, "+ Position Encoding", { bold: true, fs: 10.5 });
AR(160, 326, 160, 356, { lw: 2 });
R(60, 356, 200, 52, { line: "D97706" }); T(60, 356, 200, 52, "FiLM (\u03B3, \u03B2)", { bold: true, fs: 10.5 });

// ---------- left: condition path ----------
SQ(520, 112, "C9D3E0"); SQ(546, 112, "C9D3E0"); SQ(572, 112, "6B7A90");
T(500, 138, 140, 22, "state / env", { italic: true, fs: 9, color: "64748B" });
AR(560, 162, 560, 192, { lw: 2 });
R(460, 192, 200, 52); T(460, 192, 200, 52, "Numerical Embedding", { bold: true, fs: 10.5 });
AR(560, 244, 560, 382, { lw: 2, arrow: false });
AR(560, 382, 264, 382, { lw: 2 });

// ---------- left: message passing ----------
AR(160, 408, 160, 438, { lw: 2 });
R(60, 438, 600, 58); T(60, 438, 600, 58, "3 \u00D7 GCN   +   2 \u00D7 GAT", { bold: true, fs: 12 });
T(60, 500, 600, 22, "residues as nodes \u00B7 distance-window edges (|i \u2212 j| \u2264 5) \u00B7 global node", { italic: true, fs: 9, color: "64748B" });

// ---------- cross-panel arrow ----------
AR(660, 467, 800, 467, { lw: 2, arrow: false });
AR(800, 467, 800, 218, { lw: 2, arrow: false });
AR(800, 218, 896, 218, { lw: 2 });

// ---------- right: bond features ----------
["A8C6E8", "A8C6E8", "A8C6E8", "E0B25A", "E0B25A", "7FC8B5", "7FC8B5", "7FC8B5"].forEach((c, i) => SQ(900 + i * 24, 112, c, 18));
T(900, 138, 400, 22, "bond features \u00B7 h\u1D62, h\u2C7C, e\u1D62\u2C7C, 15-d theory", { italic: true, fs: 9, color: "64748B" });
AR(1030, 162, 1030, 192, { lw: 2 });
R(900, 192, 340, 52); T(900, 192, 340, 52, "Concat & Flatten", { bold: true, fs: 10.5 });
AR(1070, 244, 1070, 274, { lw: 2 });
R(900, 274, 340, 52); T(900, 274, 340, 52, "MLP", { bold: true, fs: 10.5 });
AR(1070, 326, 1070, 356, { lw: 2 });
R(900, 356, 340, 52); T(900, 356, 340, 52, "Sigmoid", { bold: true, fs: 10.5 });
AR(1070, 408, 1070, 438, { lw: 2 });
["4CAF7D", "4CAF7D", "E0B25A", "D98E4A", "4CAF7D", "4CAF7D", "D98E4A"].forEach((c, i) => SQ(940 + i * 26, 446, c));
T(940, 476, 260, 22, "p\u1D62 per bond  (\u03C4 = 0.5)", { italic: true, fs: 9, color: "64748B" });

pres.writeFile({ fileName: "figures/dbond_gt_architecture_minimal.pptx" })
  .then(() => console.log("written"));
