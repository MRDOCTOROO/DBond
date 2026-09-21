// workflow_mesure — Figure 1: mirror-image peptide data storage & sequence-optimization workflow
// Pure-vector rebuild (no embedded rasters). One shared geometry model emits:
//   workflow_mesure.drawio  (editable draw.io, native cells)
//   workflow_mesure.svg     (standalone vector, Desktop-free export)
//   workflow_mesure.pptx    (native editable PowerPoint shapes)
// All keywords are verbatim from the paper pipeline (README: 原始数据→二进制编码→候选序列生成→
// DBond预测筛选→最优序列选择→物理合成→MS/MS测序→数据恢复闭环).
const fs = require("fs");
const path = require("path");

// ---------------------------------------------------------------- palette
const C = {
  blue: "#1E6BB8", greenD: "#4F7F3A", green: "#4E9A47", red: "#C00000",
  gold: "#BF9000", purple: "#7030A0", teal: "#12766E",
  ink: "#1A1A1A", axis: "#333333", gray: "#7F7F7F", frame: "#666666",
  barBlue: "#4472C4", barRed: "#C00000",
  richHdr: "#D6E8CE", richLine: "#538135", sparseHdr: "#F5CBCB",
  bead: ["#9DC3E6", "#E06666", "#F6B26B", "#FFD966", "#93C47D", "#B4A7D6", "#76C7C0", "#D5A6BD"],
  screen: "#101820", termGreen: "#7CFC98", metal: "#D6DEE8", metalD: "#33475D",
  liquid: "#BDD7EE", glass: "#5B7C99", base: "#B7C0CC",
};

const F = "Arial";
const MONO = "Courier New";

// ---------------------------------------------------------------- shared model
// element kinds: rect, tri, ellipse, seg, txt  (array order = paint order)
const M = [];
const R = (x, y, w, h, fill, stroke, sw = 1, o = {}) => { const el = { k: "rect", x, y, w, h, fill, stroke, sw, ...o }; M.push(el); return el; };
const E = (cx, cy, rx, ry, fill, stroke, sw = 1) => { const el = { k: "ellipse", cx, cy, rx, ry, fill, stroke, sw }; M.push(el); return el; };
const S = (x1, y1, x2, y2, stroke, sw = 1, o = {}) => { const el = { k: "seg", x1, y1, x2, y2, stroke, sw, ...o }; M.push(el); return el; };
// txt: block box; lines = [{t, bold, size, color, mono}]; rot 270 = vertical (reads bottom-up)
const T = (x, y, w, h, lines, o = {}) => { const el = { k: "txt", x, y, w, h, lines, align: o.align || "center", rot: o.rot || 0 }; M.push(el); return el; };

// ---------------------------------------------------------------- geometry helpers
function card(x, y, w, h, color, num, titleLines) {
  const hdrH = titleLines.length > 1 ? 44 + (titleLines.length - 2) * 17 : 40;
  const outer = R(x, y, w, h, "#FFFFFF", color, 2);
  R(x, y, w, hdrH, color, color, 1);
  const title = titleLines.map((t, i) => ({ t: i === 0 ? `${num}\u2002\u2002${t}` : t, bold: true, size: titleLines.length > 2 ? 13 : 14, color: "#FFFFFF" }));
  T(x, y, w, hdrH, title);
  return outer;
}

function beadRow(x0, cy, r, letters, gap, opt = {}) {
  const d = 2 * r;
  if (!opt.noLine) S(x0 - 3, cy, x0 + (letters.length - 1) * gap + d + 1, cy, "#999999", 1.2);
  letters.forEach((ch, i) => {
    const cx = x0 + i * gap + r;
    E(cx - r, cy - r, r, r, C.bead[i % C.bead.length], "#666666", 0.8);
    if (ch) T(cx - r, cy - r, d, d, [{ t: ch, bold: true, size: opt.fs || 11, color: "#1A1A1A" }]);
  });
}

function cleaveMark(x, yTop, yBot) {
  S(x, yTop, x, yBot, C.barRed, 1.5, { dash: "4 3" });
}

// spectrum: bars sit on baseline; axes optional captions
function spectrum(bx, by, plotW, top, bars, o = {}) {
  const barW = o.barW || 7, step = o.step || 27;
  S(bx, top, bx, by, C.axis, 1.2);
  S(bx, by, bx + plotW, by, C.axis, 1.2);
  if (o.intensity !== false)
    T(bx - 24, top + 30, 16, by - top - 60, [{ t: "Intensity", size: o.axFs || 10, color: C.ink }], { rot: 270 });
  if (o.mz !== false)
    T(bx + plotW - 38, by + 5, 40, 16, [{ t: "m/z", size: o.axFs || 10, color: C.ink }], { align: "left" });
  bars.forEach((b, i) => {
    const x = bx + 25 + i * step;
    R(x, by - b.h, barW, b.h, b.c, b.c, 0.5);
    if (b.lab)
      T(x - 9, by - b.h - 15, barW + 18, 12, [{ t: b.lab, size: 9, color: b.c, bold: true }]);
  });
}

// ---------------------------------------------------------------- step cards
card(30, 40, 240, 300, C.blue, "1", ["Raw Binary Data"]);
T(30, 105, 240, 105, [
  { t: "01010110", mono: true, size: 15 }, { t: "10110101", mono: true, size: 15 },
  { t: "01011011", mono: true, size: 15 }, { t: "···", mono: true, size: 15 },
]);
T(30, 268, 240, 50, [{ t: "Arbitrary binary files", size: 12.5 }, { t: "(text, images, videos, etc.)", size: 12.5 }]);

card(310, 40, 240, 300, C.greenD, "2", ["Encoding Mapping"]);
T(310, 95, 240, 190, [
  { t: "Binary → Amino Acid", mono: true, size: 13.5, bold: true },
  { t: "", size: 6 },
  { t: " 00  →  A", mono: true, size: 13.5 }, { t: " 01  →  R", mono: true, size: 13.5 },
  { t: " 10  →  N", mono: true, size: 13.5 }, { t: " 11  →  D", mono: true, size: 13.5 },
  { t: "", size: 6 }, { t: "⋮", size: 14 },
]);

card(590, 40, 260, 300, C.green, "3", ["Candidate", "Mirror Peptide Sequences"]);
T(590, 92, 260, 130, [
  { t: "Seq 1: H-Lys-Ala-Arg-Gly-...", bold: true, size: 12.5 },
  { t: "Seq 2: H-Gly-Asn-Leu-Tyr-...", bold: true, size: 12.5 },
  { t: "Seq 3: H-Arg-Val-Asn-Ala-...", bold: true, size: 12.5 },
  { t: "···", size: 12.5 },
  { t: "Seq N: H-Leu-Gly-Lys-Val-...", bold: true, size: 12.5 },
]);
beadRow(618, 288, 8, ["", "", "", "", "", "", "", "", ""], 26, { letters: false, noLine: false });

card(890, 30, 270, 320, C.red, "4", ["Peptide Bond", "Cleavage Prediction", "(Critical Step)"]);
R(955, 130, 140, 95, "#FFFFFF", "#3B3B3B", 2);
{
  const hs = [28, 48, 38, 62, 52, 72, 42, 58, 32];
  hs.forEach((h, i) => R(975 + i * 13, 217 - h, 6, h, i % 3 === 2 ? C.barRed : C.barBlue, i % 3 === 2 ? C.barRed : C.barBlue, 0.5));
}
R(1018, 225, 12, 12, "#3B3B3B", "#3B3B3B");
R(993, 237, 62, 6, "#3B3B3B", "#3B3B3B", 1, { rx: 2 });
T(890, 262, 270, 66, [
  { t: "Predict peptide bond", size: 12.5 }, { t: "cleavage probabilities", size: 12.5 }, { t: "and theoretical spectra", size: 12.5 },
]);

card(1200, 40, 250, 300, C.gold, "5", ["Optimal Sequence", "Selection"]);
T(1215, 95, 220, 165, [
  { t: "Evaluation Metrics", bold: true, size: 13 },
  { t: "", size: 7 },
  { t: "✔ High cleavage yield", size: 12.5 }, { t: "(predictability)", size: 12.5 },
  { t: "✔ Sequence stability", size: 12.5 },
  { t: "✔ Chemical synthesizability", size: 12.5 },
  { t: "✔ Length constraint", size: 12.5 },
  { t: "···", size: 12.5 },
], { align: "left" });
T(1200, 268, 250, 45, [{ t: "Select the optimal", size: 12.5 }, { t: "mirror peptide sequence", size: 12.5 }]);

card(1490, 40, 240, 300, C.purple, "6", ["Chemical", "Synthesis"]);
R(1592, 106, 16, 18, "#FFFFFF", C.glass, 1.5);
M.push({ k: "tri", x: 1566, y: 122, w: 58, h: 62, fill: C.liquid, stroke: C.glass, sw: 1.5, down: false });
{
  const ys = [152, 144, 138, 144, 152];
  ys.forEach((y, i) => E(1652 + i * 15 - 6, y - 6, 6, 6, C.bead[(i + 2) % C.bead.length], "#666666", 0.8));
}
T(1490, 250, 240, 62, [{ t: "Solid-phase peptide", size: 12.5 }, { t: "synthesis and", size: 12.5 }, { t: "purification", size: 12.5 }]);

card(1490, 470, 240, 300, C.teal, "7", ["Tandem Mass", "Spectrometry (MS/MS)"]);
R(1520, 548, 180, 70, "#EDF1F6", C.metalD, 1.5, { rx: 4 });
R(1520, 532, 52, 86, C.metal, C.metalD, 1.5);
R(1592, 534, 46, 14, C.metal, C.metalD, 1.5);
R(1652, 558, 40, 28, "#FFFFFF", C.metalD, 1.2);
{
  const hs = [8, 14, 10, 16];
  hs.forEach((h, i) => R(1658 + i * 8, 580 - h, 4, h, i % 2 ? C.barRed : C.barBlue, i % 2 ? C.barRed : C.barBlue, 0.5));
}
spectrum(1530, 712, 160, 652, [
  { h: 22, c: C.barBlue, lab: "y7" }, { h: 40, c: C.barRed, lab: "b3" },
  { h: 30, c: C.barBlue, lab: "y2" }, { h: 48, c: C.barRed, lab: "b7" },
  { h: 24, c: C.barBlue, lab: "y4" },
], { step: 30, barW: 5, intensity: false, mz: false });
T(1700, 716, 30, 16, [{ t: "m/z", size: 10 }], { align: "left" });
T(1490, 736, 240, 26, [{ t: "Acquire MS/MS spectra", size: 12.5 }]);

card(270, 470, 240, 300, C.blue, "8", ["Data Recovery"]);
R(320, 540, 140, 90, "#FFFFFF", "#3B3B3B", 2);
R(326, 546, 128, 78, C.screen, C.screen, 1);
T(326, 546, 128, 78, [
  { t: "01010110", mono: true, size: 10.5, color: C.termGreen },
  { t: "10110101", mono: true, size: 10.5, color: C.termGreen },
  { t: "01011011", mono: true, size: 10.5, color: C.termGreen },
  { t: "···", mono: true, size: 10.5, color: C.termGreen },
]);
R(305, 630, 170, 10, C.base, "#8A94A6", 1, { rx: 3 });
T(270, 668, 240, 70, [
  { t: "Spectrum decoding", size: 12.5 }, { t: "and error correction", size: 12.5 },
  { t: "to recover the", size: 12.5 }, { t: "original binary data", size: 12.5 },
]);

// ---------------------------------------------------------------- closed loop + detail frame
const loopBox = R(75, 520, 150, 200, "none", "#555555", 1.5, { rx: 12, dash: "6 4" });
loopBox.id = "loop";
T(75, 520, 150, 200, [
  { t: "Closed-loop", bold: true, size: 12.5 }, { t: "design-and-readout", bold: true, size: 12.5 }, { t: "workflow", bold: true, size: 12.5 },
]);

// Fragmentation Pattern Comparison callout
T(560, 428, 860, 28, [{ t: "Fragmentation Pattern Comparison (Example)", bold: true, size: 15 }]);
const frameBox = R(560, 470, 860, 420, "none", C.frame, 1.5, { dash: "8 5" });
frameBox.id = "frame";

// --- rich fragmentation panel
R(580, 495, 410, 60, C.richHdr, C.richLine, 1.5);
T(580, 495, 410, 60, [
  { t: "Rich Fragmentation (Easy to Sequence)", bold: true, size: 13.5 },
  { t: "More cleavage sites, stronger signals,", size: 11.5 },
  { t: "better sequence coverage", size: 11.5 },
]);
R(580, 555, 410, 315, "#FFFFFF", C.richLine, 1.5);
T(594, 578, 32, 16, [{ t: "H—", size: 11.5, bold: true }], { align: "left" });
beadRow(628, 586, 12, ["A", "L", "R", "G", "N", "V", "K", "K"], 30, { fs: 12 });
[655, 685, 745, 805].forEach((x) => cleaveMark(x, 570, 602));
T(848, 578, 60, 16, [{ t: "—···", size: 11.5, bold: true }], { align: "left" });
T(580, 608, 410, 20, [{ t: "b/y ions: abundant", bold: true, size: 12.5 }]);
spectrum(640, 800, 335, 642, [
  { h: 55, c: C.barBlue, lab: "y3" }, { h: 85, c: C.barRed, lab: "b2" },
  { h: 45, c: C.barBlue, lab: "y5" }, { h: 105, c: C.barRed, lab: "b4" },
  { h: 65, c: C.barBlue, lab: "y7" }, { h: 92, c: C.barRed, lab: "b3" },
  { h: 50, c: C.barBlue, lab: "y9" }, { h: 118, c: C.barRed, lab: "b8" },
  { h: 78, c: C.barBlue, lab: "y5" }, { h: 60, c: C.barRed, lab: "b4" },
  { h: 98, c: C.barBlue, lab: "y2" }, { h: 40, c: C.barRed, lab: "b9" },
]);
T(635, 830, 300, 36, [
  { t: "High signal, high coverage,", bold: true, size: 13 },
  { t: "easy to sequence ✅", bold: true, size: 13 },
]);

// --- sparse fragmentation panel
R(1010, 495, 390, 60, C.sparseHdr, C.red, 1.5);
T(1010, 495, 390, 60, [
  { t: "Sparse Fragmentation (Hard to Sequence)", bold: true, size: 13.5 },
  { t: "Fewer cleavage sites, weaker signals,", size: 11.5 },
  { t: "poor sequence coverage", size: 11.5 },
]);
R(1010, 555, 390, 315, "#FFFFFF", C.red, 1.5);
T(1024, 578, 32, 16, [{ t: "H—", size: 11.5, bold: true }], { align: "left" });
beadRow(1058, 586, 12, ["A", "V", "L", "I", "F", "W", "K"], 30, { fs: 12 });
[1113, 1203].forEach((x) => cleaveMark(x, 570, 602));
T(1252, 578, 60, 16, [{ t: "—···", size: 11.5, bold: true }], { align: "left" });
T(1010, 608, 390, 20, [{ t: "b/y ions: sparse", bold: true, size: 12.5 }]);
spectrum(1048, 800, 330, 642, [
  { h: 60, c: C.barBlue, lab: "y7" }, { h: 90, c: C.barRed, lab: "b3" },
  { h: 50, c: C.barBlue, lab: "y2" }, { h: 110, c: C.barRed, lab: "b7" },
], { step: 70, barW: 8 });
T(1045, 830, 320, 36, [
  { t: "Low signal, low coverage,", bold: true, size: 13 },
  { t: "difficult to sequence ❌", bold: true, size: 13 },
]);

// ---------------------------------------------------------------- edges (draw.io cell edges + svg/pptx arrows)
const midY = 190;
const XS = { c1: [30, 270], c2: [310, 550], c3: [590, 850], c4: [890, 1160], c5: [1200, 1450], c6: [1490, 1730] };
const ARROW = "endArrow=classic;endSize=6;strokeColor=#1A1A1A;strokeWidth=2;html=1;";
const DASH_ARROW = "endArrow=classic;endSize=6;strokeColor=#555555;strokeWidth=1.5;dashed=1;dashPattern=6 4;html=1;";
const edges = [
  { id: "e12", src: "c1", dst: "c2" }, { id: "e23", src: "c2", dst: "c3" },
  { id: "e34", src: "c3", dst: "c4" }, { id: "e45", src: "c4", dst: "c5" },
  { id: "e56", src: "c5", dst: "c6" },
  { id: "e67", src: "c6", dst: "c7", exit: [0.5, 1], entry: [0.5, 0] },
  { id: "e7f", src: "c7", dst: "frame", exit: [0, 0.5], entry: [1, 0.357] },
  { id: "ef8", src: "frame", dst: "c8", exit: [0, 0.357], entry: [1, 0.5] },
  { id: "eloop1", src: "loop", dst: "c1", exit: [0.5, 0], entry: [0.5, 1], dash: true },
  { id: "eloop2", src: "c8", dst: "loop", exit: [0, 0.5], entry: [1, 0.5], dash: true },
];
// zoom callout lines (c4 bottom corners -> frame top corners), dashed, no arrowheads
S(890, 350, 560, 470, C.gray, 1.3, { dash: "6 4" });
S(1160, 350, 1420, 470, C.gray, 1.3, { dash: "6 4" });

// explicit id map for card outer rects (set after model build; find() returns first match = outer card)
const idMap = {
  c1: (el) => el.k === "rect" && el.x === 30 && el.y === 40 && el.h === 300,
  c2: (el) => el.k === "rect" && el.x === 310 && el.y === 40 && el.h === 300,
  c3: (el) => el.k === "rect" && el.x === 590 && el.y === 40 && el.h === 300,
  c4: (el) => el.k === "rect" && el.x === 890 && el.y === 30,
  c5: (el) => el.k === "rect" && el.x === 1200 && el.y === 40 && el.h === 300,
  c6: (el) => el.k === "rect" && el.x === 1490 && el.y === 40 && el.h === 300,
  c7: (el) => el.k === "rect" && el.x === 1490 && el.y === 470,
  c8: (el) => el.k === "rect" && el.x === 270 && el.y === 470,
};
for (const [want, pred] of Object.entries(idMap)) {
  const el = M.find(pred);
  if (el) el.id = want; else console.error("WARN: no element for id", want);
}

// ================================================================ draw.io emitter
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

function txtHtml(el) {
  return el.lines
    .map((l) => {
      const t = esc(l.t || "\u00A0");
      const fam = l.mono ? `font-family:${MONO}` : `font-family:${F}`;
      const col = l.color ? `color:${l.color}` : "color:#1A1A1A";
      const body = l.bold ? `<b>${t}</b>` : t;
      return `<span style="${fam};font-size:${l.size || 12}px;${col}">${body}</span>`;
    })
    .join("<br>");
}

function emitDrawio() {
  const cells = [];
  for (const el of M) {
    if (el.k === "seg") {
      cells.push(`<mxCell id="n${cells.length}_seg" value="" style="endArrow=none;strokeColor=${el.stroke};strokeWidth=${el.sw};${el.dash ? `dashed=1;dashPattern=${el.dash};` : ""}html=1;" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="${el.x1}" y="${el.y1}" as="sourcePoint"/><mxPoint x="${el.x2}" y="${el.y2}" as="targetPoint"/></mxGeometry></mxCell>`);
      continue;
    }
    const id = el.id || `n${cells.length}`;
    let style = "";
    if (el.k === "rect")
      style = `rounded=${el.rx ? 1 : 0};whiteSpace=wrap;html=1;fillColor=${el.fill};strokeColor=${el.stroke};strokeWidth=${el.sw};${el.dash ? `dashed=1;dashPattern=${el.dash};` : ""}${el.rx ? `arcSize=${Math.round((el.rx / Math.min(el.w, el.h)) * 100)};` : ""}`;
    else if (el.k === "tri")
      style = `shape=triangle;direction=${el.down ? "south" : "north"};whiteSpace=wrap;html=1;fillColor=${el.fill};strokeColor=${el.stroke};strokeWidth=${el.sw};`;
    else if (el.k === "ellipse")
      style = `ellipse;html=1;fillColor=${el.fill};strokeColor=${el.stroke};strokeWidth=${el.sw};`;
    else if (el.k === "txt")
      style = `text;html=1;align=${el.align};verticalAlign=middle;fontFamily=${F};fontColor=#1A1A1A;${el.rot === 270 ? "horizontal=0;" : ""}`;
    const gx = el.k === "ellipse" ? el.cx - el.rx : el.x;
    const gy = el.k === "ellipse" ? el.cy - el.ry : el.y;
    const gw = el.k === "ellipse" ? el.rx * 2 : el.w;
    const gh = el.k === "ellipse" ? el.ry * 2 : el.h;
    const value = el.k === "txt" ? esc(txtHtml(el)) : "";
    cells.push(`<mxCell id="${id}" value="${value}" style="${style}" vertex="1" parent="1"><mxGeometry x="${gx}" y="${gy}" width="${gw}" height="${gh}" as="geometry"/></mxCell>`);
  }
  for (const e of edges) {
    const st = (e.dash ? DASH_ARROW : ARROW)
      + (e.exit ? `exitX=${e.exit[0]};exitY=${e.exit[1]};exitDx=0;exitDy=0;` : "")
      + (e.entry ? `entryX=${e.entry[0]};entryY=${e.entry[1]};entryDx=0;entryDy=0;` : "");
    cells.push(`<mxCell id="${e.id}" value="" style="${st}" edge="1" parent="1" source="${e.src}" target="${e.dst}"><mxGeometry relative="1" as="geometry"/></mxCell>`);
  }
  return `<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" agent="build_workflow_figure.js" version="24.7.7" type="device">
  <diagram id="dbond-workflow" name="Figure-1 Workflow">
    <mxGraphModel dx="1600" dy="900" grid="1" gridSize="8" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1770" pageHeight="930" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        ${cells.join("\n        ")}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
`;
}

// ================================================================ SVG emitter (Desktop-free vector export)
function emitSvg() {
  const P = [];
  P.push(`<svg xmlns="http://www.w3.org/2000/svg" width="1770" height="930" viewBox="0 0 1770 930" font-family="${F}">`);
  P.push(`<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#1A1A1A"/></marker><marker id="ahg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#555555"/></marker></defs>`);
  P.push(`<rect width="1770" height="930" fill="#FFFFFF"/>`);
  for (const el of M) {
    if (el.k === "rect") {
      const st = el.stroke ? ` stroke="${el.stroke}" stroke-width="${el.sw}"${el.dash ? ` stroke-dasharray="${el.dash}"` : ""}` : "";
      P.push(`<rect x="${el.x}" y="${el.y}" width="${el.w}" height="${el.h}" fill="${el.fill}"${st}${el.rx ? ` rx="${el.rx}"` : ""}/>`);
    } else if (el.k === "tri") {
      const pts = el.down ? `${el.x},${el.y} ${el.x + el.w},${el.y} ${el.x + el.w / 2},${el.y + el.h}` : `${el.x + el.w / 2},${el.y} ${el.x},${el.y + el.h} ${el.x + el.w},${el.y + el.h}`;
      P.push(`<polygon points="${pts}" fill="${el.fill}" stroke="${el.stroke}" stroke-width="${el.sw}"/>`);
    } else if (el.k === "ellipse") {
      P.push(`<ellipse cx="${el.cx}" cy="${el.cy}" rx="${el.rx}" ry="${el.ry}" fill="${el.fill}" stroke="${el.stroke}" stroke-width="${el.sw}"/>`);
    } else if (el.k === "seg") {
      P.push(`<line x1="${el.x1}" y1="${el.y1}" x2="${el.x2}" y2="${el.y2}" stroke="${el.stroke}" stroke-width="${el.sw}"${el.dash ? ` stroke-dasharray="${el.dash}"` : ""}/>`);
    } else if (el.k === "txt") {
      const lh = Math.max(...el.lines.map((l) => (l.size || 12) * 1.25));
      const boxH = el.lines.length * lh;
      const y0 = el.y + el.h / 2 - boxH / 2 + lh * 0.8;
      const anchor = el.align === "center" ? "middle" : "start";
      const tx = el.align === "center" ? el.x + el.w / 2 : el.x + 2;
      el.lines.forEach((l, i) => {
        if (!l.t) return;
        const weight = l.bold ? ` font-weight="bold"` : "";
        const fam = l.mono ? ` font-family="${MONO}"` : "";
        const tr = el.rot === 270 ? ` transform="rotate(-90 ${tx} ${el.y + el.h / 2})"` : "";
        P.push(`<text x="${tx}" y="${y0 + i * lh}" text-anchor="${anchor}"${weight}${fam} font-size="${l.size || 12}" fill="${l.color || C.ink}"${tr}>${esc(l.t)}</text>`);
      });
    }
  }
  const arrow = (x1, y1, x2, y2, color, w, dash) =>
    P.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#${color}" stroke-width="${w}"${dash ? ` stroke-dasharray="6 4"` : ""} marker-end="url(#${color === "555555" ? "ahg" : "ah"})"/>`);
  [["c1", "c2"], ["c2", "c3"], ["c3", "c4"], ["c4", "c5"], ["c5", "c6"]]
    .forEach(([a, b]) => arrow(XS[a][1] + 2, midY, XS[b][0] - 4, midY, "1A1A1A", 2));
  arrow(1610, 342, 1610, 466, "1A1A1A", 2);
  arrow(1488, 620, 1426, 620, "1A1A1A", 2);
  arrow(560, 620, 516, 620, "1A1A1A", 2);
  arrow(150, 518, 150, 344, "555555", 1.5, true);
  arrow(268, 620, 229, 620, "555555", 1.5, true);
  P.push(`</svg>`);
  return P.join("\n");
}

// ================================================================ PPTX emitter (native shapes)
async function emitPptx(outPath) {
  const pptxgen = require("pptxgenjs");
  const pres = new pptxgen();
  const SW = 18.5, SCALE = SW / 1770, PT = SCALE * 72; // px -> inch / px -> pt
  pres.defineLayout({ name: "WF", width: SW, height: +(930 * SCALE).toFixed(2) });
  pres.layout = "WF";
  pres.author = "DBond";
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  const IN = (v) => +(v * SCALE).toFixed(3);

  for (const el of M) {
    if (el.k === "rect") {
      const opts = {
        x: IN(el.x), y: IN(el.y), w: IN(el.w), h: IN(el.h),
        fill: el.fill === "none" ? { color: "FFFFFF", transparency: 100 } : { color: el.fill },
      };
      if (el.stroke) opts.line = { color: el.stroke, width: el.sw * PT, dashType: el.dash ? "dash" : "solid" };
      if (el.rx) { opts.rectRadius = IN(el.rx); }
      s.addShape(el.rx ? pres.shapes.ROUNDED_RECTANGLE : pres.shapes.RECTANGLE, opts);
    } else if (el.k === "tri") {
      s.addShape(pres.shapes.ISOSCELES_TRIANGLE, {
        x: IN(el.x), y: IN(el.y), w: IN(el.w), h: IN(el.h),
        fill: { color: el.fill }, line: { color: el.stroke, width: el.sw * PT }, flipV: !!el.down,
      });
    } else if (el.k === "ellipse") {
      s.addShape(pres.shapes.OVAL, {
        x: IN(el.cx - el.rx), y: IN(el.cy - el.ry), w: IN(el.rx * 2), h: IN(el.ry * 2),
        fill: { color: el.fill }, line: { color: el.stroke, width: el.sw * PT },
      });
    } else if (el.k === "seg") {
      s.addShape(pres.shapes.LINE, {
        x: IN(Math.min(el.x1, el.x2)), y: IN(Math.min(el.y1, el.y2)),
        w: IN(Math.abs(el.x2 - el.x1)), h: IN(Math.abs(el.y2 - el.y1)),
        flipH: el.x2 < el.x1, flipV: el.y2 < el.y1,
        line: { color: el.stroke, width: el.sw * PT, dashType: el.dash ? "dash" : "solid" },
      });
    } else if (el.k === "txt") {
      const runs = el.lines.map((l) => ({
        text: l.t || " ",
        options: {
          fontSize: +((l.size || 12) * PT).toFixed(1), bold: !!l.bold,
          color: l.color || C.ink, fontFace: l.mono ? MONO : F, breakLine: true,
        },
      }));
      s.addText(runs, {
        x: IN(el.x), y: IN(el.y), w: IN(el.w), h: IN(el.h),
        align: el.align, valign: "middle", margin: 0, wrap: true,
        ...(el.rot === 270 ? { vert: "vert270" } : {}),
      });
    }
  }
  const arrow = (x1, y1, x2, y2, color, wPx, dash) =>
    s.addShape(pres.shapes.LINE, {
      x: IN(Math.min(x1, x2)), y: IN(Math.min(y1, y2)), w: IN(Math.abs(x2 - x1)), h: IN(Math.abs(y2 - y1)),
      flipH: x2 < x1, flipV: y2 < y1,
      line: { color, width: wPx * PT, dashType: dash ? "dash" : "solid", endArrowType: "triangle" },
    });
  [["c1", "c2"], ["c2", "c3"], ["c3", "c4"], ["c4", "c5"], ["c5", "c6"]]
    .forEach(([a, b]) => arrow(XS[a][1] + 2, midY, XS[b][0] - 3, midY, "1A1A1A", 2));
  arrow(1610, 342, 1610, 466, "1A1A1A", 2);
  arrow(1488, 620, 1426, 620, "1A1A1A", 2);
  arrow(560, 620, 516, 620, "1A1A1A", 2);
  arrow(150, 518, 150, 344, "555555", 1.5, true);
  arrow(268, 620, 229, 620, "555555", 1.5, true);

  await pres.writeFile({ fileName: outPath });
}

// ================================================================ run
const dir = __dirname;
fs.writeFileSync(path.join(dir, "workflow_mesure.drawio"), emitDrawio());
fs.writeFileSync(path.join(dir, "workflow_mesure.svg"), emitSvg());
emitPptx(path.join(dir, "workflow_mesure.pptx")).then(() => console.log("OK: drawio + svg + pptx written"));
