// DBond-GT — minimal architecture figure (reference-style vertical flow)
// All modules are native PowerPoint shapes — fully editable.
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "DBond";

// palette: teal primary (graph / message passing), amber accent (conditioning / theory)
const T900="134E4A", T800="115E59", T700="0F766E", T600="0D9488", T200="99F6E4", T100="CCFBF1", T50="F0FDFA";
const A900="78350F", A600="D97706", A50="FFFBEB";
const TEXT="1E293B", MUTED="64748B", ARROW="475569", DASH="94A3B8", PANEL="F8FAFC";
const F = "Segoe UI";
const shadow = () => ({ type:"outer", color:"000000", blur:6, offset:2, angle:45, opacity:0.13 });

const s = pres.addSlide();
s.background = { color: "FFFFFF" };

// ---------------- header ----------------
s.addText("DBond-GT: Residue-Graph Network for Peptide Bond-Cleavage Prediction",
  { x:0.45, y:0.2, w:12.4, h:0.4, fontSize:19, bold:true, color:T900, fontFace:F, margin:0 });
s.addText("pre-synthesis setting  ·  one residue graph per peptide  ·  hidden dim 256  ·  dropout 0.1",
  { x:0.45, y:0.62, w:12.4, h:0.26, fontSize:11, color:MUTED, fontFace:F, margin:0 });

// ---------------- helpers ----------------
const block = (x,y,w,h,fill,border,dash=false) => s.addShape(pres.shapes.ROUNDED_RECTANGLE,
  { x, y, w, h, fill:{color:fill}, line: dash ? {color:border,width:1, dashType:"dash"} : {color:border, width:1.25},
    rectRadius:0.05, shadow: dash ? undefined : shadow() });

const boxText = (x,y,w,h,lines,valign="middle") => s.addText(lines,
  { x:x+0.13, y, w:w-0.26, h, fontFace:F, valign, margin:0 });

const line = (x1,y1,x2,y2,opts={}) => s.addShape(pres.shapes.LINE, {
  x:Math.min(x1,x2), y:Math.min(y1,y2), w:Math.abs(x2-x1), h:Math.abs(y2-y1),
  flipV: y2 < y1, flipH: x2 < x1, line: opts });

const arr = (x1,y1,x2,y2) => line(x1,y1,x2,y2,{ color:ARROW, width:2.25, endArrowType:"triangle" });

// spine geometry
const SX = 4.35, SW = 4.6, CX = SX + SW/2;   // 4.35 – 8.95, centre 6.65

// ---------------- 1. output ----------------
block(SX, 0.96, SW, 0.56, "FFFFFF", T600);
boxText(SX, 0.96, 3.25, 0.56, [
  { text:"Per-bond cleavage probability p\u1D62", options:{ bold:true, color:T900, fontSize:12.5, breakLine:true } },
  { text:"sigmoid  ·  threshold \u03C4 = 0.5", options:{ color:MUTED, fontSize:10.5 } },
]);
// probability bars (right side of the output box)
[0.13,0.24,0.18,0.30,0.21,0.15].forEach((h,i) => {
  s.addShape(pres.shapes.RECTANGLE, { x:7.85+i*0.19, y:1.38-h, w:0.13, h,
    fill:{ color: h > 0.22 ? A600 : T600 } });
});

// ---------------- 2. bond head ----------------
block(SX, 1.68, SW, 0.84, T50, T600);
boxText(SX, 1.68, SW, 0.84, [
  { text:"Bond prediction head (per peptide bond i)", options:{ bold:true, color:T900, fontSize:12.5, breakLine:true } },
  { text:"[h\u1D62 \u2016 h\u1D62\u208A\u2081 \u2016 e\u1D62\u2C7C \u2016 h\u1D62\u2212h\u1D62\u208A\u2081 \u2016 h\u1D62\u2299h\u1D62\u208A\u2081 \u2016 t\u1D62]   (6 \u00D7 256 = 1536)", options:{ color:TEXT, fontSize:11, breakLine:true } },
  { text:"\u2192 MLP 1536 \u2192 256 \u2192 ReLU \u2192 Dropout(0.1) \u2192 logit", options:{ color:MUTED, fontSize:10.5 } },
]);

// ---------------- 3. message passing ----------------
block(SX, 2.68, SW, 1.5, PANEL, T200);
s.addText("message passing \u00D7 5 layers  ·  hidden 256  ·  dropout 0.1",
  { x:SX, y:2.74, w:SW, h:0.24, fontSize:11, color:MUTED, fontFace:F, align:"center", margin:0 });
const inner = (y,t) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x:SX+0.4, y, w:SW-0.8, h:0.5,
    fill:{color:T100}, line:{color:T600, width:0.75}, rectRadius:0.04 });
  s.addText(t, { x:SX+0.4, y, w:SW-0.8, h:0.5, fontSize:12.5, bold:true, color:T900,
    fontFace:F, align:"center", valign:"middle", margin:0 });
};
inner(3.04, "2 \u00D7 edge-aware GAT  (4 heads)");
inner(3.62, "3 \u00D7 gated residual GCN");

// ---------------- 4. FiLM ----------------
block(SX, 4.32, SW, 0.72, A50, A600);
boxText(SX, 4.32, SW, 0.72, [
  { text:"FiLM conditioning (before message passing)", options:{ bold:true, color:A900, fontSize:12.5, breakLine:true } },
  { text:"h \u2190 (1 + \u03B3) \u2299 h + \u03B2   (zero-init: identity at start)", options:{ color:TEXT, fontSize:11, breakLine:true } },
  { text:"(\u03B3, \u03B2) \u2190 MLP(2 \u2192 64 \u2192 512) of [0.1\u00B7charge, 0.01\u00B7NCE]", options:{ color:MUTED, fontSize:10.5 } },
]);

// ---------------- 5. node encoder ----------------
block(SX, 5.2, SW, 0.94, T50, T600);
boxText(SX, 5.2, SW, 0.94, [
  { text:"Node encoder  \u2192  h \u2208 \u211D\u00B2\u2075\u2076", options:{ bold:true, color:T900, fontSize:12.5, breakLine:true } },
  { text:"AA emb (64) + position (32) + physicochemistry (32)", options:{ color:TEXT, fontSize:11, breakLine:true } },
  { text:"+ state MLP (charge, pep_mass) + env MLP (NCE)", options:{ color:TEXT, fontSize:11, breakLine:true } },
  { text:"\u2192 Linear + LayerNorm + ReLU", options:{ color:MUTED, fontSize:10.5 } },
]);

// ---------------- 6. input row ----------------
const chip = (x,w,title,sub) => {
  block(x, 6.34, w, 0.62, T50, T600);
  boxText(x, 6.34, w, 0.62, [
    { text:title, options:{ bold:true, color:T900, fontSize:11.5, breakLine:true } },
    { text:sub, options:{ color:MUTED, fontSize:9.5 } },
  ]);
};
chip(SX,        1.55, "sequence P", "D-AA alphabet (24)");
chip(SX+1.65,   1.45, "state",      "charge · pep_mass");
chip(SX+3.2,    1.40, "env",        "NCE");

// ---------------- left column: training note ----------------
block(0.45, 0.98, 3.5, 1.3, PANEL, DASH, true);
boxText(0.45, 0.98, 3.5, 1.3, [
  { text:"Training", options:{ bold:true, color:TEXT, fontSize:11.5, breakLine:true } },
  { text:"condition-group folding (\u2248 7,643 groups) + q soft labels, group_uniform weights", options:{ color:MUTED, fontSize:10, breakLine:true } },
  { text:"AdamW lr 1e-4 · cosine + warmup 40 · early stop 25 · batch 128 groups", options:{ color:MUTED, fontSize:10 } },
]);

// ---------------- left column: edge encoder ----------------
block(0.45, 3.32, 3.5, 1.1, T50, T600);
boxText(0.45, 3.32, 3.5, 1.1, [
  { text:"Edge encoder  \u2192  e \u2208 \u211D\u00B2\u2075\u2076", options:{ bold:true, color:T900, fontSize:12.5, breakLine:true } },
  { text:"type + distance + raw attributes", options:{ color:TEXT, fontSize:11, breakLine:true } },
  { text:"[d, charge, pep_mass, NCE] \u2192 Linear + LN + ReLU", options:{ color:MUTED, fontSize:10.5 } },
]);

// ---------------- left column: residue graph panel ----------------
block(0.45, 4.62, 3.5, 1.95, T50, T200);
s.addText("Residue graph (one per peptide)",
  { x:0.55, y:4.68, w:3.3, h:0.24, fontSize:12, bold:true, color:T900, fontFace:F, align:"center", margin:0 });
s.addText([
  { text:"chain edges = peptide bonds (\u25C6)", options:{ color:MUTED, fontSize:9.5, breakLine:true } },
  { text:"distance edges |i\u2212j| \u2264 5  ·  global node v\u1D67 \u2194 all", options:{ color:MUTED, fontSize:9.5 } },
], { x:0.55, y:4.94, w:3.3, h:0.34, fontFace:F, align:"center", margin:0 });

// global node + global edges (dashed)
s.addShape(pres.shapes.OVAL, { x:1.55, y:5.34, w:0.28, h:0.28, fill:{color:T800} });
s.addText("G", { x:1.55, y:5.34, w:0.28, h:0.28, fontSize:10, bold:true, color:"FFFFFF",
  fontFace:F, align:"center", valign:"middle", margin:0 });
line(1.69, 5.62, 1.69, 5.90, { color:DASH, width:1.25, dashType:"dash" });
line(1.58, 5.60, 0.89, 5.90, { color:DASH, width:1.25, dashType:"dash" });
line(1.80, 5.60, 2.49, 5.90, { color:DASH, width:1.25, dashType:"dash" });

// residues + chain edges
["P","E","P","T","I"].forEach((c,i) => {
  const x = 0.75 + i*0.4;
  s.addShape(pres.shapes.OVAL, { x, y:5.90, w:0.28, h:0.28, fill:{color:T600} });
  s.addText(c, { x, y:5.90, w:0.28, h:0.28, fontSize:10, bold:true, color:"FFFFFF",
    fontFace:F, align:"center", valign:"middle", margin:0 });
});
[1.03, 1.43, 1.83, 2.23].forEach(x =>
  line(x, 6.04, x+0.12, 6.04, { color:T700, width:1.5 }));
// bond-site marker
s.addShape(pres.shapes.DIAMOND, { x:1.445, y:5.995, w:0.09, h:0.09, fill:{color:A600} });
// one distance edge (skip connection), dashed U
line(0.89, 6.20, 0.89, 6.34, { color:DASH, width:1.25, dashType:"dash" });
line(0.89, 6.34, 1.69, 6.34, { color:DASH, width:1.25, dashType:"dash" });
line(1.69, 6.20, 1.69, 6.34, { color:DASH, width:1.25, dashType:"dash" });

// ---------------- right column: theory features ----------------
block(9.35, 1.68, 3.55, 1.4, A50, A600);
boxText(9.35, 1.68, 3.55, 1.4, [
  { text:"Theory features t\u1D62 (15-d)", options:{ bold:true, color:A900, fontSize:12.5, breakLine:true } },
  { text:"prefix / suffix mass · b\u2081\u207A/b\u2082\u207A/y\u2081\u207A/y\u2082\u207A m/z · relative position · flanking residue masses · H\u2082O / NH\u2083 loss flags · Pro context", options:{ color:TEXT, fontSize:10.5, breakLine:true } },
  { text:"sequence-derived only \u2192 Linear(15 \u2192 256)", options:{ color:MUTED, fontSize:10 } },
]);

// ---------------- right column: masked inputs ----------------
block(9.35, 3.32, 3.55, 1.1, "FFFFFF", DASH, true);
boxText(9.35, 3.32, 3.55, 1.1, [
  { text:"\u00D7 intensity · scan_num · rt", options:{ bold:true, color:MUTED, fontSize:11.5, breakLine:true } },
  { text:"masked at runtime (pre-synthesis setting)", options:{ color:MUTED, fontSize:10, breakLine:true } },
  { text:"effective inputs: sequence + charge + pep_mass + NCE + theory features", options:{ color:MUTED, fontSize:10 } },
]);

// ---------------- right column: inference ----------------
block(9.35, 5.2, 3.55, 0.94, A50, A600);
boxText(9.35, 5.2, 3.55, 0.94, [
  { text:"Inference — DBond-GT-Ens", options:{ bold:true, color:A900, fontSize:11.5, breakLine:true } },
  { text:"5 seeds (42–46) · probability averaging", options:{ color:TEXT, fontSize:10, breakLine:true } },
  { text:"dual checkpoint: F1-selected (main tables) · \u03C1-selected (ranking)", options:{ color:MUTED, fontSize:10 } },
]);

// ---------------- arrows ----------------
arr(5.10, 6.34, 5.10, 6.14);   // sequence  -> node encoder
arr(6.70, 6.34, 6.70, 6.14);   // state     -> node encoder
arr(8.20, 6.34, 8.20, 6.14);   // env       -> node encoder
arr(3.95, 5.60, 4.35, 5.60);   // graph     -> node encoder
arr(CX,   5.20, CX,   5.04);   // node enc  -> FiLM
arr(CX,   4.32, CX,   4.18);   // FiLM      -> message passing
arr(CX,   2.68, CX,   2.52);   // MP        -> bond head
arr(CX,   1.68, CX,   1.52);   // bond head -> output
arr(9.35, 2.10, 8.95, 2.10);   // theory    -> bond head
arr(3.95, 3.87, 4.72, 3.87);   // edge enc  -> message passing

pres.writeFile({ fileName: "dbond_gt_architecture_simple.pptx" }).then(() => console.log("written: dbond_gt_architecture_simple.pptx"));
