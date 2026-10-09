// DBond-GT architecture — v4 "premium" card style (reference-inspired).
// Design language: white cards with soft shadows on light canvas, accent-colored
// headers, 3D-stacked layer blocks, small glyphs (chips / attention grid / bars),
// thick color-coded flow arrows, NO legend band (numbering lives in the paper's
// graph-strategy table; type tokens intentionally omitted).
// All coordinates in absolute inches (no px canvas -> no PY offset trap).
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
const s = pres.addSlide();
s.background = { color: "F8FAFC" };

const INK = "0F172A", SLATE = "475569", MUTED = "64748B", BORDER = "E2E8F0";
const BLUE = { a: "2563EB", h: "1E3A8A", t: "EFF6FF" };
const TEAL = { a: "0D9488", h: "134E4A", t: "F0FDFA" };
const VIO  = { a: "7C3AED", h: "4C1D95", t: "F5F3FF" };
const AMB  = { a: "D97706", h: "78350F", t: "FFFBEB" };
const F = "Segoe UI";
const sh = () => ({ type: "outer", color: "94A3B8", blur: 7, offset: 2, angle: 90, opacity: 0.28 });
const shSoft = () => ({ type: "outer", color: "94A3B8", blur: 4, offset: 1, angle: 90, opacity: 0.2 });

const T = (x, y, w, h, text, o = {}) => s.addText(text, {
  x, y, w, h, margin: 0, valign: o.valign || "middle", align: o.align || "center",
  fontFace: F, fontSize: o.fs || 10, bold: !!o.bold, italic: !!o.italic,
  color: o.color || INK, charSpacing: o.cs || 0,
});
const RR = (x, y, w, h, o = {}) => s.addShape(pres.ShapeType.roundRect, {
  x, y, w, h, rectRadius: o.r ?? 0.07,
  fill: { color: o.fill || "FFFFFF" },
  line: o.line === null ? { type: "none" } : { color: o.line || BORDER, width: o.lw || 1 },
  shadow: o.shadow === false ? undefined : (o.soft ? shSoft() : sh()),
});
const LN = (x1, y1, x2, y2, o = {}) => {
  const line = { color: o.color || SLATE, width: o.lw || 2 };
  if (o.dash) line.dashType = "sysDash";
  if (o.dot) line.dashType = "sysDot";
  if (o.arrow !== false) line.endArrowType = o.head || "triangle";
  if (o.begin) line.beginArrowType = "triangle";
  s.addShape(pres.ShapeType.line, {
    x: Math.min(x1, x2), y: Math.min(y1, y2),
    w: Math.abs(x2 - x1) || 0.005, h: Math.abs(y2 - y1) || 0.005,
    flipH: x2 < x1, flipV: y2 < y1, line,
  });
};
const curve = (x1, y1, x2, y2, cx, cy, o = {}) => {
  const n = 10;
  for (let i = 0; i < n; i++) {
    const t0 = i / n, t1 = (i + 1) / n;
    const q = (a, b, c, t) => (1 - t) ** 2 * a + 2 * (1 - t) * t * b + t ** 2 * c;
    LN(q(x1, cx, x2, t0), q(y1, cy, y2, t0), q(x1, cx, x2, t1), q(y1, cy, y2, t1),
      { color: o.color || VIO.a, lw: o.lw || 1.1, arrow: false, dash: o.dash, dot: o.dot });
  }
};
const card = (x, y, w, h, title, accent) => {
  RR(x, y, w, h, { fill: "FFFFFF" });
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h: 0.42, rectRadius: 0.07, fill: { color: accent.t },
    line: { color: accent.t, width: 0 },
  });
  s.addShape(pres.ShapeType.rect, { x, y: y + 0.28, w, h: 0.16, fill: { color: accent.t }, line: { type: "none" } });
  T(x + 0.14, y + 0.05, w - 0.28, 0.36, title, { align: "left", bold: true, fs: 12.5, color: accent.h, cs: 1 });
};
// 3D-stacked block: darker offset layers behind a white top card
const stack3d = (x, y, w, h, depth = 0.09) => {
  for (let i = 2; i >= 1; i--) {
    RR(x + depth * i, y + depth * i, w, h, { fill: "E2E8F0", line: null, r: 0.06, shadow: false });
  }
};

// =============================== layout =====================================
const Y = 1.15, H = 3.62;                    // main row (vertically balanced)
const xIn = 0.42, wIn = 2.02;                // input
const xGr = 2.78, wGr = 2.46;                // graph
const xEn = 5.58, wEn = 2.12;                // encoding
const xBk = 7.98, wBk = 2.34;                // backbone
const xHd = 10.6, wHd = 2.42;                // head + output
const MID = Y + H / 2;

// ------------------------------ 1 · input -----------------------------------
card(xIn, Y, wIn, H, "Input", BLUE);
T(xIn + 0.13, Y + 0.5, wIn - 0.26, 0.24, "sequence", { align: "left", fs: 9, bold: true, color: MUTED, cs: 1 });
const chips = [["D", "BFDBFE", "1D4ED8"], ["A", "BBF7D0", "166534"], ["K", "FDE68A", "854D0E"], ["B", "FBCFE8", "9D174D"]];
chips.forEach(([ch, f, c], i) => {
  const cxs = xIn + 0.16 + i * 0.42;
  RR(cxs, Y + 0.78, 0.34, 0.34, { fill: f, line: null, r: 0.05, soft: true });
  T(cxs, Y + 0.78, 0.34, 0.34, ch, { bold: true, fs: 11, color: c });
});
T(xIn + 0.16 + 4 * 0.42, Y + 0.78, 0.3, 0.34, "⋯", { fs: 12, color: MUTED });
LN(xIn + 0.16, Y + 1.02, xIn + 0.16 + 4 * 0.42, Y + 1.02, { color: "CBD5E1", lw: 1, arrow: false });
T(xIn + 0.13, Y + 1.36, wIn - 0.26, 0.24, "conditions", { align: "left", fs: 9, bold: true, color: MUTED, cs: 1 });
[["charge", BLUE], ["pep_mass", TEAL], ["NCE", VIO]].forEach(([t, a], i) => {
  const py = Y + 1.66 + i * 0.5;
  RR(xIn + 0.16, py, wIn - 0.32, 0.4, { fill: a.t, line: a.a, lw: 0.75, r: 0.1, soft: true });
  T(xIn + 0.16, py, wIn - 0.32, 0.4, t, { fs: 10, bold: true, color: a.h });
});

// --------------------------- 2 · residue graph ------------------------------
card(xGr, Y, wGr, H, "Residue graph", TEAL);
const gx = xGr + wGr / 2;
// global node
const gy = Y + 0.72;
s.addShape(pres.ShapeType.ellipse, { x: gx - 0.19, y: gy, w: 0.38, h: 0.38, fill: { color: VIO.t }, line: { color: VIO.a, width: 1 }, shadow: shSoft() });
T(gx - 0.19, gy, 0.38, 0.38, "v_g", { bold: true, fs: 9, color: VIO.h });
// residues
const ry = Y + 1.98, rs = 0.4, gap = 0.52, rxs = [0, 1, 2, 3].map(i => gx - (3 * gap + rs) / 2 + i * gap);
rxs.forEach((rx, i) => {
  s.addShape(pres.ShapeType.ellipse, { x: rx, y: ry, w: rs, h: rs, fill: { color: TEAL.a }, line: { color: "0F766E", width: 1 }, shadow: shSoft() });
  T(rx, ry, rs, rs, "D A K B"[i * 2], { bold: true, fs: 10.5, color: "FFFFFF" });
});
// backbone edges + bond diamonds
for (let i = 0; i < 3; i++) LN(rxs[i] + rs, ry + rs / 2, rxs[i + 1], ry + rs / 2, { color: TEAL.a, lw: 1.6, arrow: false });
rxs.slice(0, 3).forEach((rx, i) => {
  const mx = (rx + rs + rxs[i + 1]) / 2;
  s.addShape(pres.ShapeType.diamond, { x: mx - 0.045, y: ry + rs / 2 - 0.045, w: 0.09, h: 0.09, fill: { color: AMB.a }, line: null });
});
// remote arcs (distance window) + global dotted
curve(rxs[0] + rs / 2, ry, rxs[2] + rs / 2, ry, (rxs[0] + rxs[2]) / 2 + rs / 2, ry - 0.34, { dash: true });
curve(rxs[1] + rs / 2, ry, rxs[3] + rs / 2, ry, (rxs[1] + rxs[3]) / 2 + rs / 2, ry - 0.34, { dash: true });
curve(gx, gy + 0.38, rxs[0] + rs / 2, ry, gx - 0.55, ry - 0.3, { dot: true, lw: 0.9 });
curve(gx, gy + 0.38, rxs[3] + rs / 2, ry, gx + 0.55, ry - 0.3, { dot: true, lw: 0.9 });
// window note (no type numbers)
T(xGr + 0.1, Y + 2.62, wGr - 0.2, 0.26, "distance window  R = 5  (|i − j| ≤ 5)", { fs: 9.5, color: TEAL.h, bold: true });
T(xGr + 0.1, Y + 2.92, wGr - 0.2, 0.5,
  "◆ peptide bond = prediction target\ndashed = window edges · dotted = global", { fs: 8.5, color: MUTED, valign: "top" });

// ----------------------------- 3 · encoding ---------------------------------
card(xEn, Y, wEn, H, "Encoding", VIO);
const encRows = [
  ["Node  h\u1D62 \u2208 \u211D\u00B2\u2075\u2076", "AA \u2295 pos \u2295 physchem \u2295 state/env"],
  ["Edge  e\u1D62\u2C7C \u2208 \u211D\u00B2\u2075\u2076", "type \u2295 distance \u2295 attrs"],
  ["Global node", "emb \u2295 condition proj"],
];
encRows.forEach(([t, sub], i) => {
  const ey = Y + 0.62 + i * 0.98;
  RR(xEn + 0.16, ey, wEn - 0.32, 0.84, { fill: VIO.t, line: null, r: 0.06, soft: true });
  T(xEn + 0.26, ey + 0.08, wEn - 0.52, 0.34, t, { align: "left", bold: true, fs: 10, color: VIO.h });
  T(xEn + 0.26, ey + 0.4, wEn - 0.52, 0.36, sub, { align: "left", fs: 8.5, color: SLATE });
});

// -------------------------- 4 · message passing -----------------------------
card(xBk, Y, wBk, H, "Message passing", BLUE);
stack3d(xBk + 0.18, Y + 0.56, wBk - 0.36, 0.78);
RR(xBk + 0.18, Y + 0.56, wBk - 0.36, 0.78, { fill: BLUE.t, line: BLUE.a, lw: 0.75, r: 0.06 });
T(xBk + 0.18, Y + 0.62, wBk - 0.36, 0.34, "3 \u00D7 gated residual GCN", { bold: true, fs: 10.5, color: BLUE.h });
T(xBk + 0.18, Y + 0.94, wBk - 0.36, 0.3, "neighborhood aggregation", { fs: 8, color: SLATE });
LN(xBk + wBk / 2, Y + 1.42, xBk + wBk / 2, Y + 1.66, { color: BLUE.a, lw: 1.4 });
stack3d(xBk + 0.18, Y + 1.74, wBk - 0.36, 0.94);
RR(xBk + 0.18, Y + 1.74, wBk - 0.36, 0.94, { fill: BLUE.t, line: BLUE.a, lw: 0.75, r: 0.06 });
T(xBk + 0.18, Y + 1.8, wBk - 0.36, 0.34, "2 \u00D7 edge-aware GAT", { bold: true, fs: 10.5, color: BLUE.h });
// attention glyph: 5x5 mini-grid
for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) {
  const op = 0.15 + 0.75 * Math.abs(Math.sin((r * 5 + c) * 1.7));
  const cell = 0.11;
  const ax = xBk + wBk / 2 - (5 * cell + 4 * 0.025) / 2 + c * (cell + 0.025);
  const ay = Y + 2.16 + r * (cell + 0.025);
  s.addShape(pres.ShapeType.rect, { x: ax, y: ay, w: cell, h: cell, fill: { color: BLUE.a, transparency: Math.round((1 - op) * 100) }, line: null });
}
T(xBk + 0.18, Y + 2.96, wBk - 0.36, 0.3, "4-head attention \u00B7 residual + LayerNorm", { fs: 8, color: SLATE });

// --------------------------- 5 · bond head / out ----------------------------
card(xHd, Y, wHd, H, "Bond head \u2192 output", VIO);
RR(xHd + 0.14, Y + 0.54, wHd - 0.28, 0.62, { fill: VIO.t, line: null, r: 0.05, soft: true });
T(xHd + 0.2, Y + 0.56, wHd - 0.4, 0.58,
  "[ h\u1D62 \u2016 h\u1D62\u208A\u2081 \u2016 e\u1D62\u2C7C \u2016 h\u1D62\u2212h\u1D62\u208A\u2081\n  \u2016 h\u1D62\u2299h\u1D62\u208A\u2081 \u2016 theory ]  (6\u00D7256)", { fs: 8.2, color: VIO.h, valign: "middle" });
LN(xHd + wHd / 2, Y + 1.16, xHd + wHd / 2, Y + 1.36, { color: SLATE, lw: 1.4 });
RR(xHd + 0.34, Y + 1.38, wHd - 0.68, 0.42, { fill: "FFFFFF", line: VIO.a, lw: 0.75, r: 0.08, soft: true });
T(xHd + 0.34, Y + 1.38, wHd - 0.68, 0.42, "MLP 1536 \u2192 256 \u2192 1", { fs: 9.5, bold: true });
LN(xHd + wHd / 2, Y + 1.8, xHd + wHd / 2, Y + 2.0, { color: SLATE, lw: 1.4 });
// probability bars + threshold (line drawn exactly at p = 0.5 level)
const barY = Y + 3.06, barH = 0.72, tauY = barY - 0.5 * barH;
LN(xHd + 0.34, tauY, xHd + wHd - 0.34, tauY, { color: "CBD5E1", lw: 1, dash: true, arrow: false });
T(xHd + 0.32, tauY - 0.26, 0.9, 0.2, "\u03C4 = 0.5", { fs: 7.5, color: MUTED, align: "left" });
const probs = [0.24, 0.58, 0.35, 0.78, 0.42, 0.66, 0.3];
probs.forEach((p, i) => {
  const bh = p * barH, bw = 0.17, bx = xHd + 0.42 + i * 0.29;
  s.addShape(pres.ShapeType.roundRect, {
    x: bx, y: barY - bh, w: bw, h: bh, rectRadius: 0.02,
    fill: { color: p > 0.5 ? AMB.a : TEAL.a }, line: null, shadow: shSoft(),
  });
});
T(xHd + 0.14, barY + 0.06, wHd - 0.28, 0.28, "p\u1D62 per peptide bond", { fs: 9, bold: true, color: INK });

// ------------------------------ FiLM branch ---------------------------------
const fx = xEn - 0.1, fw = (xBk + wBk) - fx + 0.1, fy = Y + H + 0.5, fh = 0.98;
RR(fx, fy, fw, fh, { fill: AMB.t, line: AMB.a, lw: 1 });
T(fx + 0.16, fy + 0.08, 1.3, 0.3, "FiLM", { align: "left", bold: true, fs: 12, color: AMB.h });
T(fx + 0.16, fy + 0.4, fw - 0.32, 0.5,
  "[charge, NCE] \u2192 MLP \u2192 (\u03B3, \u03B2):   h \u2190 (1+\u03B3)\u2299h + \u03B2   (zero-init)", { align: "left", fs: 10, color: AMB.h });
// arrows: conditions -> FiLM (down), FiLM -> packed nodes before MP (up, arrowhead at top)
LN(xIn + wIn / 2, Y + H, fx + 0.35, fy + 0.1, { color: AMB.a, lw: 1.6, dash: true });
LN(xBk + wBk / 2, fy, xBk + wBk / 2, Y + H + 0.02, { color: AMB.a, lw: 1.6, dash: true });
T(xBk + wBk / 2 + 0.1, fy - 0.34, 1.4, 0.24, "modulates all nodes", { fs: 8, italic: true, color: AMB.h, align: "left" });

// --------------------------- main flow arrows --------------------------------
const arrowY = MID + 0.55;
[[xIn + wIn, xGr], [xGr + wGr, xEn], [xEn + wEn, xBk], [xBk + wBk, xHd]].forEach(([a, b]) => {
  LN(a + 0.04, arrowY, b - 0.04, arrowY, { color: SLATE, lw: 2.4, head: "triangle" });
});

pres.writeFile({ fileName: "figures/dbond_gt_architecture_v4.pptx" })
  .then(() => console.log("written: figures/dbond_gt_architecture_v4.pptx"));
