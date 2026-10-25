// DBond-GT architecture — panel-style figure v3 (native editable PowerPoint shapes).
// v3 changes (2026-09-28, review round):
//   - panel 2 label synced to window-radius semantics: "R = 5 (|i-j| <= 5)"
//   - panel 2 in-panel edge-type text lines removed (Fig 2 owns construction detail; legend band keeps one copy)
//   - panel 1 "excluded: intensity, scan_num, rt" box removed (moved to caption)
//   - panel 6 "Key feature (+15-d theory)" -> "Key feature" (theory detail moved to caption)
//   - legend band: 15-d theory detail line removed (moved to caption)
const pptxgen = require("pptxgenjs");

const PX_W = 1808, PX_H = 624;          // figure size in px
const W_IN = 13.33, H_IN = 7.5;         // LAYOUT_WIDE
const S = W_IN / PX_W;                  // px -> inch
const OY = (H_IN - PX_H * S) / 2;       // vertical centring
const PX = (px) => px * S;
const PY = (py) => py * S + OY;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
const s = pres.addSlide();
s.background = { color: "FFFFFF" };

const C = {
  blue:   { line: "2563EB", head: "1D4ED8", tint: "EFF6FF" },
  green:  { line: "059669", head: "047857", tint: "ECFDF5" },
  purple: { line: "7C3AED", head: "5B21B6", tint: "F5F3FF" },
  amber:  { line: "D97706", head: "B45309", tint: "FFFBEB" },
};
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
  const n = 16, col = o.color || "7C3AED", lw = o.lw || 1;
  for (let i = 0; i < n; i++) {
    const t0 = i / n, t1 = (i + 1) / n;
    const q = (a, b, c, t) => (1 - t) ** 2 * a + 2 * (1 - t) * t * b + t ** 2 * c;
    AR(q(x1, cx, x2, t0), q(y1, cy, y2, t0), q(x1, cx, x2, t1), q(y1, cy, y2, t1), { color: col, lw, arrow: false });
  }
};

const panel = (x, y, w, h, title, col) => {
  R(x, y, w, h, { fill: col.tint, line: col.line, dash: true });
  T(x + 14, y + 2, w - 20, 30, title, { align: "left", bold: true, fs: 11.5, color: col.head });
};

// ---------- 1 Input ----------
panel(40, 40, 248, 408, "1   Input", C.blue);
T(64, 104, 200, 28, "Sequence P", { align: "left", bold: true, fs: 11, color: "1E3A8A" });
R(64, 136, 200, 40, { fill: "FFFFFF", line: C.blue.line });
T(64, 136, 200, 40, "D   A   K   B   \u22EF", { bold: true, fs: 12 });
T(64, 208, 200, 24, "3 synthesis conditions", { align: "left", bold: true, fs: 10.5, color: "1E3A8A" });
R(64, 244, 200, 44, { fill: "DBEAFE", line: C.blue.line });   T(64, 244, 200, 44, "charge", { fs: 11 });
R(64, 300, 200, 44, { fill: "D1FAE5", line: C.green.line });  T(64, 300, 200, 44, "pep_mass", { fs: 11 });
R(64, 356, 200, 44, { fill: "EDE9FE", line: C.purple.line }); T(64, 356, 200, 44, "NCE", { fs: 11 });

// ---------- 2 Graph construction (detail in Fig. 2) ----------
panel(296, 40, 352, 380, "2   Graph construction", C.green);
R(456, 104, 44, 44, { fill: "EDE9FE", line: C.purple.line });
T(456, 104, 44, 44, "v_g", { bold: true, fs: 10, color: "5B21B6" });
const res = [[320, "D"], [392, "A"], [464, "K"], [536, "B"]];
res.forEach(([x, ch]) => {
  R(x, 216, 44, 44, { fill: C.green.line, line: "047857" });
  T(x, 216, 44, 44, ch, { bold: true, fs: 11, color: "FFFFFF" });
});
T(336, 300, 288, 24, "R = 5 \u2192 |i \u2212 j| \u2264 5", { fs: 10 });
AR(res[0][0] + 44, 238, res[1][0], 238, { color: "047857", lw: 1.5, arrow: false });
AR(res[1][0] + 44, 238, res[2][0], 238, { color: "047857", lw: 1.5, arrow: false });
AR(res[2][0] + 44, 238, res[3][0], 238, { color: "047857", lw: 1.5, arrow: false });
curve(res[0][0] + 22, 216, res[2][0] + 22, 216, (res[0][0] + res[2][0]) / 2 + 22, 186);   // type 2 remote, |i-j|=2
curve(res[1][0] + 22, 216, res[3][0] + 22, 216, (res[1][0] + res[3][0]) / 2 + 22, 176);   // type 2 remote, |i-j|=2
curve(res[0][0] + 22, 216, res[3][0] + 22, 216, (res[0][0] + res[3][0]) / 2 + 22, 150);   // type 2 remote, |i-j|=3
curve(478, 148, res[0][0] + 22, 216, 386, 178, { lw: 0.75 });   // type 4 global
curve(478, 148, res[1][0] + 22, 216, 430, 178, { lw: 0.75 });
curve(478, 148, res[2][0] + 22, 216, 526, 178, { lw: 0.75 });
curve(478, 148, res[3][0] + 22, 216, 570, 178, { lw: 0.75 });

// ---------- 3 Encoding ----------
panel(608, 40, 320, 360, "3   Encoding", C.purple);
R(632, 104, 272, 88, { fill: "FFFFFF", line: C.purple.line });
T(632, 104, 272, 88, "Node encoding: AA \u2295 pos \u2295 physchem", { fs: 10, color: "312E81" });
R(632, 208, 272, 88, { fill: "FFFFFF", line: C.purple.line });
T(632, 208, 272, 88, "Edge encoding: type \u2295 dist \u2295 attrs", { fs: 10, color: "312E81" });
R(632, 312, 272, 88, { fill: "FFFFFF", line: C.purple.line });
T(632, 312, 272, 88, "Global node emb. + condition vector", { fs: 10, color: "312E81" });

// ---------- 4 FiLM conditioning ----------
panel(920, 40, 248, 380, "4   FiLM conditioning", C.amber);
R(944, 104, 200, 44, { fill: "DBEAFE", line: C.blue.line });
T(944, 104, 200, 44, "[ charge, NCE ]", { bold: true, fs: 10 });
R(984, 176, 120, 48, { fill: "FECACA", line: "DC2626" });
T(984, 176, 120, 48, "MLP", { bold: true, fs: 10.5, color: "7F1D1D" });
R(944, 260, 80, 40, { fill: "FECACA", line: "DC2626" });
T(944, 260, 80, 40, "\u03B3", { bold: true, fs: 11, color: "7F1D1D" });
R(1064, 260, 80, 40, { fill: "FECACA", line: "DC2626" });
T(1064, 260, 80, 40, "\u03B2", { bold: true, fs: 11, color: "7F1D1D" });
R(944, 332, 200, 88, { fill: "FEF3C7", line: C.amber.line });
T(944, 332, 200, 88, "h \u2190 (1 + \u03B3) \u00B7 h + \u03B2", { bold: true, fs: 11, color: "7C2D12" });
AR(1044, 148, 1044, 176); AR(1044, 224, 984, 260); AR(1044, 224, 1104, 260);
AR(984, 300, 984, 332); AR(1104, 300, 1104, 332);

// ---------- 5 Message passing ----------
panel(1208, 40, 248, 344, "5   Message passing", C.blue);
R(1232, 104, 200, 104, { fill: "DBEAFE", line: C.blue.line });
T(1232, 104, 200, 104, "3 \u00D7 GCN", { bold: true, fs: 11.5, color: "1E3A8A" });
R(1232, 224, 200, 104, { fill: "DBEAFE", line: C.blue.line });
T(1232, 224, 200, 104, "2 \u00D7 GAT", { bold: true, fs: 11.5, color: "1E3A8A" });
T(1232, 344, 200, 26, "(residual + LayerNorm)", { fs: 9.5, color: "475569" });
AR(1332, 208, 1332, 224);

// ---------- 6 Prediction ----------
panel(1496, 40, 256, 392, "6   Prediction", C.purple);
R(1520, 104, 208, 80, { fill: "FFFFFF", line: C.purple.line });
T(1520, 104, 208, 80, "Key feature", { fs: 10, color: "312E81" });
R(1544, 212, 160, 44, { fill: "EDE9FE", line: C.purple.line });
T(1544, 212, 160, 44, "MLP", { bold: true, fs: 10.5, color: "312E81" });
R(1544, 276, 160, 44, { fill: "EDE9FE", line: C.purple.line });
T(1544, 276, 160, 44, "Sigmoid", { bold: true, fs: 10.5, color: "312E81" });
R(1520, 344, 208, 88, { fill: "F5F3FF", line: C.purple.line });
T(1520, 344, 208, 88, "Bond probability p_i", { fs: 10, color: "312E81" });
AR(1624, 184, 1624, 212); AR(1624, 256, 1624, 276); AR(1624, 320, 1624, 344);

// ---------- cross-panel arrows ----------
AR(240, 156, 456, 118, { lw: 1.5 });
AR(580, 238, 632, 148, { lw: 1.5 });
AR(904, 356, 944, 126, { lw: 1.5 });
AR(1144, 376, 1232, 156, { lw: 1.5 });
AR(1432, 328, 1520, 144, { lw: 1.5 });

// ---------- legend band ----------
R(40, 462, 1696, 124, { fill: "FFFFFF", line: "CBD5E1", dash: true });
T(56, 468, 300, 26, "Legend", { align: "left", bold: true, fs: 12, color: "64748B" });
T(64, 502, 300, 26, "Residue node \u00B7 Global node v_g", { fs: 9, align: "left" });
AR(380, 515, 420, 515, { color: "059669", lw: 2, arrow: false });
T(426, 502, 260, 26, "type 1 near (sequence backbone)", { fs: 8.5, align: "left" });
AR(700, 515, 740, 515, { color: "7C3AED", lw: 1.75, arrow: false, dash: true });
T(746, 502, 250, 26, "type 2 remote (distance window)", { fs: 8.5, align: "left" });
AR(1010, 515, 1050, 515, { color: "7C3AED", lw: 1.5, arrow: false, dot: true });
T(1056, 502, 150, 26, "type 4 global", { fs: 8.5, align: "left" });
T(64, 536, 440, 26, "FiLM conditioning (charge, NCE) \u2192 (\u03B3, \u03B2)", { fs: 9, align: "left", color: C.amber.head });
T(520, 536, 560, 26, "Output: p_i (per bond) \u00B7 \u03C4 = 0.5 \u00B7 5-seed average", { fs: 9, align: "left", color: C.purple.head });

pres.writeFile({ fileName: "figures/dbond_gt_architecture_panels_v3.pptx" })
  .then(() => console.log("written: figures/dbond_gt_architecture_panels_v3.pptx"));
