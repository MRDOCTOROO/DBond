// DBond-GT architecture — v5 "two-stream" premium style (Simonyan&Zisserman-inspired).
// Metaphor (faithful to the model): sequence/graph stream + condition stream
// converge into the FiLM fusion box, feeding message passing, then the bond head.
// Visual language: dotted-border stream boxes, 3D slab stacks (feature maps),
// glossy fusion box, bold charcoal arrows, minimal labels, no legend, no type ids.
// Absolute-inch coordinates (no canvas scaling -> no PY offset trap).
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
const s = pres.addSlide();
s.background = { color: "FFFFFF" };

const INK = "1E293B", NAVY = "1E3A8A", CHAR = "334155", MUTED = "64748B";
const AMB = { h: "92400E" };
const TEAL = { a: "0D9488" };
const F = "Segoe UI";

const T = (x, y, w, h, text, o = {}) => s.addText(text, {
  x, y, w, h, margin: 0, valign: o.valign || "middle", align: o.align || "center",
  fontFace: F, fontSize: o.fs || 11, bold: !!o.bold, italic: !!o.italic,
  color: o.color || INK, charSpacing: o.cs || 0,
});
const RR = (x, y, w, h, o = {}) => s.addShape(pres.ShapeType.roundRect, {
  x, y, w, h, rectRadius: o.r ?? 0.06,
  fill: { color: o.fill || "FFFFFF" },
  line: o.line === null ? { type: "none" } : { color: o.line || "CBD5E1", width: o.lw ?? 1, dashType: o.dash },
  shadow: o.shadow,
});
const LN = (x1, y1, x2, y2, o = {}) => {
  const line = { color: o.color || CHAR, width: o.lw || 3 };
  if (o.dash) line.dashType = "sysDash";
  if (o.dot) line.dashType = "sysDot";
  if (o.arrow !== false) line.endArrowType = "triangle";
  s.addShape(pres.ShapeType.line, {
    x: Math.min(x1, x2), y: Math.min(y1, y2),
    w: Math.abs(x2 - x1) || 0.005, h: Math.abs(y2 - y1) || 0.005,
    flipH: x2 < x1, flipV: y2 < y1, line,
  });
};
// 3D slab stack: back copies darker->brighter toward front
const slab = (x, y, w, h, cols, dx = 0.09, dy = 0.07) => {
  const n = cols.length;
  cols.forEach((c, i) => {
    const off = n - 1 - i;
    s.addShape(pres.ShapeType.parallelogram, {
      x: x + dx * off, y: y + dy * off, w, h,
      fill: { color: c }, line: { color: "FFFFFF", width: 0.75 },
      shadow: i === 0 ? { type: "outer", color: "94A3B8", blur: 5, offset: 2, angle: 45, opacity: 0.3 } : undefined,
    });
  });
  return { cx: x + w / 2 + dx * (n - 1) / 2, cy: y + h / 2 + dy * (n - 1) / 2 };
};
// glossy two-tone box (fake gradient)
const gloss = (x, y, w, h, dark, mid) => {
  RR(x, y, w, h, { fill: dark, line: null, r: 0.09, shadow: { type: "outer", color: "64748B", blur: 6, offset: 2, angle: 90, opacity: 0.35 } });
  RR(x + 0.02, y + 0.02, w - 0.04, h * 0.46, { fill: mid, line: null, r: 0.08 });
};

// ================================ layout ====================================
const TOP_Y = 0.85, TOP_H = 2.5;      // graph stream box
const BOT_Y = 4.15, BOT_H = 2.5;      // condition stream box
const MID = 3.6;

// ------------------------------ input ---------------------------------------
const ix = 0.5, iy = MID - 1.2;
// depth frames behind (stack effect), then a front row of letter cards
[2, 1].forEach(k => {
  RR(ix + 0.1 * k, iy + 0.08 * k, 2.28, 0.62, { fill: "FFFFFF", line: "CBD5E1", lw: 1, r: 0.08 });
});
["D", "A", "K", "B"].forEach((ch, i) => {
  const cols = { D: ["BFDBFE", "1D4ED8"], A: ["BBF7D0", "166534"], K: ["FDE68A", "854D0E"], B: ["FBCFE8", "9D174D"] }[ch];
  RR(ix + i * 0.57, iy, 0.5, 0.62, { fill: cols[0], line: "FFFFFF", lw: 1, r: 0.05, shadow: { type: "outer", color: "94A3B8", blur: 3, offset: 1, angle: 45, opacity: 0.2 } });
  T(ix + i * 0.57, iy, 0.5, 0.62, ch, { bold: true, fs: 14, color: cols[1] });
});
T(ix - 0.05, iy + 0.92, 2.4, 0.3, "peptide sequence", { fs: 12, bold: true, color: NAVY });
[["charge", "2563EB"], ["pep_mass", "0D9488"], ["NCE", "7C3AED"]].forEach(([t, c], i) => {
  const py = iy + 1.5 + i * 0.42;
  RR(ix + 0.02, py, 1.3, 0.34, { fill: "F8FAFC", line: c, lw: 0.75, r: 0.12 });
  T(ix + 0.02, py, 1.3, 0.34, t, { fs: 9, bold: true, color: c });
});
T(ix + 0.02, iy + 2.82, 1.6, 0.28, "conditions", { fs: 9, italic: true, color: MUTED });

// --------------------- dotted stream boxes ----------------------------------
const SBX = 2.95, SBW = 4.1;
const streamBox = (y, h) => RR(SBX, y, SBW, h, { fill: "FFFFFF", line: CHAR, lw: 2.5, dash: "sysDot", r: 0.05 });
streamBox(TOP_Y, TOP_H);
streamBox(BOT_Y, BOT_H);

// --- top: graph stream ---
T(SBX + 0.15, TOP_Y + 0.1, SBW - 0.3, 0.32, "Graph stream", { align: "left", bold: true, fs: 13, color: NAVY });
const g1x = SBX + 0.3, mgy = TOP_Y + 1.65, mgx = g1x + 0.72;
s.addShape(pres.ShapeType.ellipse, { x: mgx - 0.14, y: mgy - 0.98, w: 0.28, h: 0.28, fill: { color: "EDE9FE" }, line: { color: "7C3AED", width: 1 } });
T(mgx - 0.14, mgy - 0.98, 0.28, 0.28, "v", { bold: true, fs: 8, color: "5B21B6" });
const mrx = [0, 1, 2, 3].map(i => g1x + i * 0.32);
mrx.forEach((rx, i) => {
  s.addShape(pres.ShapeType.ellipse, { x: rx, y: mgy, w: 0.26, h: 0.26, fill: { color: TEAL.a }, line: { color: "0F766E", width: 0.75 } });
  if (i < 3) LN(rx + 0.26, mgy + 0.13, rx + 0.32, mgy + 0.13, { color: "0F766E", lw: 1.2, arrow: false });
});
LN(mgx, mgy - 0.7, mrx[0] + 0.13, mgy, { color: "A78BFA", lw: 0.9, arrow: false, dot: true });
LN(mgx, mgy - 0.7, mrx[3] + 0.13, mgy, { color: "A78BFA", lw: 0.9, arrow: false, dot: true });
T(g1x - 0.1, mgy + 0.36, 1.7, 0.26, "residue graph", { fs: 9.5, color: "0F766E", bold: true });
slab(SBX + 2.3, TOP_Y + 0.85, 0.85, 0.62, ["C4B5FD", "A78BFA", "8B5CF6", "7C3AED"]);
T(SBX + 2.2, TOP_Y + 1.68, 1.7, 0.26, "node/edge encoders", { fs: 9.5, color: "5B21B6", bold: true });
T(SBX + 2.2, TOP_Y + 1.94, 1.7, 0.26, "h \u2208 \u211D\u00B2\u2075\u2076", { fs: 9, color: MUTED });
LN(g1x + 1.35, mgy + 0.1, SBX + 2.25, TOP_Y + 1.15, { lw: 2.5 });
LN(SBX + SBW - 0.5, TOP_Y + 1.1, SBX + SBW - 0.08, TOP_Y + 1.1, { lw: 2.5 });

// --- bottom: condition stream ---
T(SBX + 0.15, BOT_Y + 0.1, SBW - 0.3, 0.32, "Condition stream", { align: "left", bold: true, fs: 13, color: AMB.h });
[["charge", "2563EB"], ["NCE", "7C3AED"]].forEach(([t, c], i) => {
  const py = BOT_Y + 0.72 + i * 0.58;
  RR(SBX + 0.3, py, 1.05, 0.44, { fill: "F8FAFC", line: c, lw: 1, r: 0.12 });
  T(SBX + 0.3, py, 1.05, 0.44, t, { fs: 10, bold: true, color: c });
});
LN(SBX + 1.42, BOT_Y + 1.15, SBX + 1.85, BOT_Y + 1.15, { lw: 2.5 });
slab(SBX + 1.95, BOT_Y + 0.75, 0.92, 0.78, ["FDE68A", "FCD34D", "FBBF24", "F59E0B"]);
T(SBX + 1.9, BOT_Y + 1.68, 1.7, 0.26, "FiLM MLP", { fs: 9.5, color: AMB.h, bold: true });
T(SBX + 1.9, BOT_Y + 1.94, 1.7, 0.26, "\u03B3, \u03B2 \u2208 \u211D\u00B2\u2075\u2076 (each)", { fs: 9, color: MUTED });
LN(SBX + SBW - 0.5, BOT_Y + 1.15, SBX + SBW - 0.08, BOT_Y + 1.15, { lw: 2.5 });

// --------------------- FiLM glossy fusion box -------------------------------
const FX = 7.55, FW = 1.5, FH = 1.25, FY = MID - FH / 2;
gloss(FX, FY, FW, FH, "1E3A8A", "3B82F6");
T(FX, FY + 0.14, FW, 0.45, "FiLM", { bold: true, fs: 16, color: "FFFFFF" });
T(FX + 0.05, FY + 0.6, FW - 0.1, 0.5, "h \u2190 (1+\u03B3)\u2299h + \u03B2", { fs: 8.5, color: "DBEAFE" });
LN(SBX + SBW + 0.06, TOP_Y + 1.1, FX - 0.06, FY + 0.35, { lw: 3 });
LN(SBX + SBW + 0.06, BOT_Y + 1.15, FX - 0.06, FY + FH - 0.35, { lw: 3 });
LN(ix + 1.45, MID - 0.55, SBX - 0.06, TOP_Y + 1.1, { lw: 3 });
LN(ix + 0.78, iy + 2.78, SBX - 0.06, BOT_Y + 1.15, { lw: 3 });
// pep_mass 注入路径：经节点/边/全局节点特征进入图流（teal 与左上 chips 中
// pep_mass 同色，图注说明其经节点、边与全局节点编码注入）。
LN(ix + 1.34, iy + 2.09, SBX - 0.06, TOP_Y + 2.45, { lw: 2, color: "0D9488" });

// --------------------- message passing dotted box ---------------------------
const MX = 9.4, MW = 2.0, MY = MID - 1.25, MH = 2.5;
RR(MX, MY, MW, MH, { fill: "FFFFFF", line: CHAR, lw: 2.5, dash: "sysDot", r: 0.05 });
T(MX + 0.12, MY + 0.08, MW - 0.24, 0.3, "Message passing", { align: "left", bold: true, fs: 12, color: NAVY });
slab(MX + 0.22, MY + 0.52, 1.0, 0.52, ["93C5FD", "60A5FA", "3B82F6"]);
T(MX + 0.1, MY + 1.1, MW - 0.2, 0.26, "3 \u00D7 gated residual GCN", { fs: 9, bold: true, color: "1D4ED8" });
LN(MX + MW / 2, MY + 1.36, MX + MW / 2, MY + 1.54, { lw: 2 });
slab(MX + 0.22, MY + 1.58, 1.0, 0.52, ["93C5FD", "60A5FA", "3B82F6"]);
T(MX + 0.1, MY + 2.16, MW - 0.2, 0.26, "2 \u00D7 edge-aware GAT", { fs: 9, bold: true, color: "1D4ED8" });
LN(FX + FW + 0.06, MID, MX - 0.06, MID, { lw: 3 });

// --------------------- bond head + output -----------------------------------
const BX = 11.75, BW = 1.35;
slab(BX, MID - 0.85, 1.05, 0.62, ["C4B5FD", "A78BFA", "7C3AED"]);
T(BX - 0.12, MID - 0.12, BW + 0.2, 0.26, "bond head (+theory)", { fs: 9.5, bold: true, color: "5B21B6" });
T(BX - 0.12, MID + 0.14, BW + 0.2, 0.26, "MLP \u2192 Sigmoid \u2192 p\u1D62", { fs: 8.5, color: MUTED });
LN(MX + MW + 0.06, MID, BX - 0.06, MID, { lw: 3 });
// probability bars
const barY = MID + 1.35, barH = 0.85, bw = 0.14;
[0.62, 0.35, 0.78].forEach((p, i) => {
  const bh = p * barH, bx = BX + 0.05 + i * 0.19;
  s.addShape(pres.ShapeType.roundRect, {
    x: bx, y: barY - bh, w: bw, h: bh, rectRadius: 0.015,
    fill: { color: p > 0.5 ? "D97706" : TEAL.a }, line: null,
    shadow: { type: "outer", color: "94A3B8", blur: 3, offset: 1, angle: 90, opacity: 0.25 },
  });
});
LN(BX + 0.02, barY - 0.5 * barH, BX + BW - 0.05, barY - 0.5 * barH, { color: "CBD5E1", lw: 1, arrow: false, dash: true });
T(BX - 0.12, barY + 0.08, BW + 0.2, 0.28, "p\u1D62 per peptide bond", { fs: 9.5, bold: true, color: INK });

pres.writeFile({ fileName: "figures/dbond_gt_architecture_v5.pptx" })
  .then(() => console.log("written: figures/dbond_gt_architecture_v5.pptx"));
