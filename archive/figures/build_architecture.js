// DBond-GT architecture figure — native editable PowerPoint shapes
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "DBond";

// palette: teal primary (graph/MS), amber accent (conditioning / theory)
const T900="134E4A", T800="115E59", T700="0F766E", T600="0D9488", T200="99F6E4", T100="CCFBF1", T50="F0FDFA";
const A900="78350F", A700="B45309", A600="D97706", A50="FFFBEB";
const TEXT="1E293B", MUTED="64748B", ARROW="475569", DASH="94A3B8";
const F = "Segoe UI";
const shadow = () => ({ type:"outer", color:"000000", blur:6, offset:2, angle:45, opacity:0.13 });

const s = pres.addSlide();
s.background = { color: "FFFFFF" };

// ---------- title ----------
s.addText("DBond-GT: Graph Transformer for Peptide Bond-Cleavage Prediction",
  { x:0.45, y:0.22, w:9.3, h:0.42, fontSize:20, bold:true, color:T900, fontFace:F, margin:0 });
s.addText("pre-synthesis setting\ngroup-folded 5-fold CV",
  { x:9.85, y:0.24, w:3.0, h:0.55, fontSize:12, color:MUTED, fontFace:F, align:"right", margin:0 });

// ---------- column headers ----------
const hdr = (x,w,t) => s.addText(t, { x, y:0.82, w, h:0.28, fontSize:12.5, bold:true, color:MUTED, fontFace:F, charSpacing:2, margin:0 });
hdr(0.45, 2.55, "INPUTS & GRAPH");
hdr(3.5, 2.6, "ENCODERS");
hdr(6.6, 2.7, "MESSAGE PASSING");
hdr(9.8, 3.08, "PREDICTION");

// ---------- col 1: graph illustration panel ----------
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x:0.45, y:1.15, w:2.55, h:1.95, fill:{color:T50}, line:{color:T200,width:0.75}, rectRadius:0.05 });
// global node G
s.addShape(pres.shapes.OVAL, { x:1.52, y:1.38, w:0.4, h:0.4, fill:{color:T800} });
s.addText("G", { x:1.52, y:1.38, w:0.4, h:0.4, fontSize:12, bold:true, color:"FFFFFF", fontFace:F, align:"center", valign:"middle", margin:0 });
// global edges (dashed) G -> A / C / E
s.addShape(pres.shapes.LINE, { x:0.78, y:1.78, w:0.94, h:0.55, flipH:true, line:{color:DASH, width:1.25, dashType:"dash"} });
s.addShape(pres.shapes.LINE, { x:1.72, y:1.78, w:0,    h:0.55, line:{color:DASH, width:1.25, dashType:"dash"} });
s.addShape(pres.shapes.LINE, { x:1.72, y:1.78, w:0.94, h:0.55, line:{color:DASH, width:1.25, dashType:"dash"} });
// distance edges (dashed, skip connections)
[[0.86,0.78],[1.33,0.78],[1.80,0.78]].forEach(([x0,w]) =>
  s.addShape(pres.shapes.LINE, { x:x0, y:2.27, w:w, h:0, line:{color:DASH, width:1.25, dashType:"dash"} }));
// sequence edges (solid)
[[0.95,0.13],[1.42,0.13],[1.89,0.13],[2.36,0.13]].forEach(([x0,w]) =>
  s.addShape(pres.shapes.LINE, { x:x0, y:2.5, w:w, h:0, line:{color:T700, width:1.5} }));
// residue circles A C D E F
const aa = ["A","K","L","G","F"];
aa.forEach((c,i) => {
  const x = 0.61 + i*0.47;
  s.addShape(pres.shapes.OVAL, { x, y:2.33, w:0.34, h:0.34, fill:{color:T600} });
  s.addText(c, { x, y:2.33, w:0.34, h:0.34, fontSize:12, bold:true, color:"FFFFFF", fontFace:F, align:"center", valign:"middle", margin:0 });
});
// bond-site marker (amber diamond between C and D)
s.addShape(pres.shapes.RECTANGLE, { x:1.905, y:2.445, w:0.11, h:0.11, rotate:45, fill:{color:A600} });

s.addText("Bidirectional sequence edges + distance edges (d_max=6, effective |i\u2212j|\u22645) + global node;  \u25C6 = bond site (i, i+1)",
  { x:0.45, y:3.16, w:2.55, h:0.68, fontSize:12, color:MUTED, fontFace:F, margin:0 });

// ---------- input / condition boxes ----------
const block = (x,y,w,h,fill,border) => s.addShape(pres.shapes.ROUNDED_RECTANGLE,
  { x, y, w, h, fill:{color:fill}, line:{color:border, width:1.25}, rectRadius:0.05, shadow:shadow() });

block(0.45, 3.95, 2.55, 1.0, T50, T600);
s.addText([
  { text:"Input (per condition group)", options:{ bold:true, color:T900, fontSize:12.5, breakLine:true } },
  { text:"sequence + charge, pep_mass, NCE", options:{ color:TEXT, fontSize:12, breakLine:true } },
  { text:"(pre-synthesis: no intensity / scan_num / rt)", options:{ color:MUTED, fontSize:11 } },
], { x:0.58, y:3.95, w:2.32, h:1.0, fontFace:F, valign:"middle", margin:0 });

block(0.45, 5.5, 4.15, 1.0, A50, A600);
s.addText([
  { text:"Conditioning signal", options:{ bold:true, color:A900, fontSize:13, breakLine:true } },
  { text:"c = [charge, NCE] \u2014 observable before synthesis", options:{ color:TEXT, fontSize:12.5 } },
], { x:0.62, y:5.5, w:3.85, h:1.0, fontFace:F, valign:"middle", margin:0 });

// ---------- col 2: encoders ----------
const enc = (y,h,title,body) => {
  block(3.5, y, 2.6, h, T50, T600);
  const lines = [{ text:title, options:{ bold:true, color:T900, fontSize:14, breakLine:true } }];
  body.forEach((t,i) => lines.push({ text:t, options:{ color:TEXT, fontSize:12, breakLine:i < body.length-1 } }));
  s.addText(lines, { x:3.62, y, w:2.38, h, fontFace:F, valign:"middle", margin:0 });
};
enc(1.15, 1.5, "Node Encoder", ["AA emb (64) \u2295 position (32) \u2295", "physicochem. (32) \u2295 state MLP (charge, pep_mass)", "\u2295 env MLP (NCE) \u2192 Linear+LN \u2192 h\u2208\u211D\u00B2\u2075\u2076"]);
enc(2.85, 1.15, "Edge Encoder", ["type emb \u2295 distance emb \u2295 raw attrs (4-d: d, q, m, NCE)", "\u2192 e\u2208\u211D\u00B2\u2075\u2076"]);
enc(4.2, 1.0, "Global Node", ["learnable emb \u2295 proj(state \u2295 env),", "linked to all residues"]);

// ---------- col 3: message passing ----------
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x:6.6, y:1.15, w:2.7, h:3.5, fill:{color:"FFFFFF"}, line:{color:T700, width:1.5}, rectRadius:0.06, shadow:shadow() });
s.addText("Message Passing", { x:6.6, y:1.28, w:2.7, h:0.3, fontSize:14, bold:true, color:T900, fontFace:F, align:"center", margin:0 });
const inner = (y,t,fs) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x:6.85, y, w:2.2, h:0.62, fill:{color:T100}, line:{color:T600, width:0.75}, rectRadius:0.04 });
  s.addText(t, { x:6.85, y, w:2.2, h:0.62, fontSize:fs||12.5, bold:true, color:T900, fontFace:F, align:"center", valign:"middle", margin:0 });
};
inner(1.75, "3 \u00D7 Residual GCN");
inner(2.55, "2 \u00D7 Edge-gated GAT");
inner(3.35, "global node in every layer", 12);
s.addText("(γ, β) modulate node features here", { x:6.7, y:4.12, w:2.5, h:0.4, fontSize:12, italic:true, color:MUTED, fontFace:F, align:"center", margin:0 });

// FiLM box (bottom band)
block(5.0, 5.5, 2.2, 1.0, A50, A600);
s.addText([
  { text:"FiLM", options:{ bold:true, color:A900, fontSize:13, breakLine:true } },
  { text:"MLP(2\u219264\u2192512) \u2192 (γ, β);", options:{ color:TEXT, fontSize:12, breakLine:true } },
  { text:"h \u2190 (1+γ)\u2299h + β;  zero-init", options:{ color:TEXT, fontSize:12 } },
], { x:5.1, y:5.5, w:2.0, h:1.0, fontFace:F, valign:"middle", margin:0 });

// ---------- col 4: bond head / theory / output ----------
block(9.8, 1.15, 3.08, 1.75, T50, T600);
s.addText([
  { text:"Bond Prediction Head", options:{ bold:true, color:T900, fontSize:14, breakLine:true } },
  { text:"for each bond (i, j=i+1), concat", options:{ color:TEXT, fontSize:12, breakLine:true } },
  { text:"[ h\u1D62 \u2016 h\u2C7C \u2016 e\u1D62\u2C7C \u2016 h\u1D62\u2212h\u2C7C \u2016 h\u1D62\u2299h\u2C7C \u2016 theory ]", options:{ color:TEXT, fontSize:12, breakLine:true } },
  { text:"(6\u00D7256=1536) \u2192 MLP \u2192 logit", options:{ color:TEXT, fontSize:12 } },
], { x:9.95, y:1.15, w:2.82, h:1.75, fontFace:F, valign:"middle", margin:0 });

block(10.6, 3.15, 2.28, 1.4, A50, A600);
s.addText([
  { text:"Theory features", options:{ bold:true, color:A900, fontSize:13, breakLine:true } },
  { text:"15-d per bond: b/y m/z, flanking masses, H\u2082O/NH\u2083 loss, Pro context \u2192 proj 256", options:{ color:TEXT, fontSize:12 } },
], { x:10.72, y:3.15, w:2.06, h:1.4, fontFace:F, valign:"middle", margin:0 });

block(9.8, 4.8, 3.08, 1.62, "FFFFFF", T600);
s.addText("Per-bond cleavage probability", { x:9.95, y:4.9, w:2.85, h:0.28, fontSize:13, bold:true, color:T900, fontFace:F, margin:0 });
// threshold dashed line + label
s.addShape(pres.shapes.LINE, { x:10.05, y:5.52, w:2.6, h:0, line:{color:DASH, width:1, dashType:"dash"} });
s.addText("\u03C4", { x:12.6, y:5.26, w:0.25, h:0.24, fontSize:12, color:MUTED, fontFace:F, margin:0 });
// probability bars
[0.28,0.55,0.38,0.75,0.45,0.30,0.62].forEach((h,i) => {
  s.addShape(pres.shapes.RECTANGLE, { x:10.15+i*0.3, y:5.98-h, w:0.17, h, fill:{color: h>0.7 ? A600 : T600} });
});
s.addText("inference: 5-seed probability averaging", { x:9.95, y:6.02, w:2.85, h:0.26, fontSize:12, color:MUTED, fontFace:F, margin:0 });

// ---------- arrows ----------
const arr = (x1,y1,x2,y2,opts={}) => s.addShape(pres.shapes.LINE, {
  x:Math.min(x1,x2), y:Math.min(y1,y2), w:Math.abs(x2-x1), h:Math.abs(y2-y1),
  flipV:(x2>x1 && y2<y1) || (x2===x1 && y2<y1), flipH:(x2<x1 && y2>y1),
  line:{ color:ARROW, width:2.25, endArrowType:"triangle", ...opts },
});
arr(3.0,1.9, 3.5,1.9);      // graph -> node encoder
arr(3.0,4.35, 3.5,3.45);    // input -> edge encoder
arr(3.0,4.75, 3.5,4.7);     // input -> global node
arr(6.1,1.9, 6.6,1.9);      // node enc -> MP
arr(6.1,3.42, 6.6,3.42);    // edge enc -> MP
arr(6.1,4.7, 6.6,4.7);      // global -> MP
arr(7.0,5.5, 7.0,4.65);     // FiLM -> MP (up)
arr(4.6,6.0, 5.0,6.0);      // condition -> FiLM
arr(9.3,2.0, 9.8,2.0);      // MP -> head
arr(10.2,2.9, 10.2,4.8);    // head -> output
arr(11.74,3.15, 11.74,2.9); // theory -> head

// ---------- bottom legend ----------
s.addShape(pres.shapes.LINE, { x:0.45, y:6.9, w:0.4, h:0, line:{color:T700, width:2} });
s.addText("sequence edge", { x:0.92, y:6.76, w:1.5, h:0.28, fontSize:12, color:MUTED, fontFace:F, margin:0 });
s.addShape(pres.shapes.LINE, { x:2.55, y:6.9, w:0.4, h:0, line:{color:DASH, width:1.5, dashType:"dash"} });
s.addText("distance / global edge", { x:3.02, y:6.76, w:2.0, h:0.28, fontSize:12, color:MUTED, fontFace:F, margin:0 });
s.addShape(pres.shapes.RECTANGLE, { x:5.2, y:6.835, w:0.13, h:0.13, rotate:45, fill:{color:A600} });
s.addText("bond site (i, i+1)", { x:5.45, y:6.76, w:1.6, h:0.28, fontSize:12, color:MUTED, fontFace:F, margin:0 });
s.addShape(pres.shapes.OVAL, { x:7.25, y:6.81, w:0.18, h:0.18, fill:{color:T600} });
s.addText("residue", { x:7.5, y:6.76, w:0.9, h:0.28, fontSize:12, color:MUTED, fontFace:F, margin:0 });
s.addShape(pres.shapes.OVAL, { x:8.65, y:6.79, w:0.22, h:0.22, fill:{color:T800} });
s.addText("global node", { x:8.95, y:6.76, w:1.3, h:0.28, fontSize:12, color:MUTED, fontFace:F, margin:0 });
s.addText("all modules are native PowerPoint shapes \u2014 fully editable", { x:10.5, y:6.76, w:2.4, h:0.5, fontSize:12, italic:true, color:DASH, fontFace:F, align:"right", margin:0 });

pres.writeFile({ fileName: "dbond_gt_architecture.pptx" }).then(() => console.log("written"));
