const fs = require('fs');

const OUT = 'D:/UserData/Desktop/graph-structures-code-consistent.drawio';
const W = 1720, H = 1080;
let cells = [];
let nextId = 2;

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function cell(value, style, x, y, w, h, vertex = true) {
  const id = String(nextId++);
  cells.push(`<mxCell id="${id}" value="${esc(value)}" style="${style}"${vertex ? ' vertex="1"' : ''} parent="1"><mxGeometry x="${x}" y="${y}" width="${w}" height="${h}" as="geometry"/></mxCell>`);
  return id;
}
function rect(value, x, y, w, h, fill, stroke, fs = 22, bold = false, align = 'center', extra = '') {
  let style = `rounded=1;whiteSpace=wrap;html=1;arcSize=8;fillColor=${fill};strokeColor=${stroke};fontFamily=Times New Roman;fontSize=${fs};fontColor=#111111;align=${align};verticalAlign=middle;spacing=4;${extra}`;
  if (bold) style += 'fontStyle=1;';
  return cell(value, style, x, y, w, h);
}
function text(value, x, y, w, h, fs = 22, bold = false, align = 'center', color = '#111111') {
  let style = `text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;rounded=0;fontFamily=Times New Roman;fontSize=${fs};fontColor=${color};align=${align};verticalAlign=middle;`;
  if (bold) style += 'fontStyle=1;';
  return cell(value, style, x, y, w, h);
}
function node(label, x, y, r = 27, fill = '#f8fbff', stroke = '#23466d') {
  return cell(label, `ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=${fill};strokeColor=${stroke};strokeWidth=2;fontFamily=Times New Roman;fontSize=24;fontStyle=2;fontColor=#111111;align=center;verticalAlign=middle;`, x - r, y - r, 2 * r, 2 * r);
}
function line(x1, y1, x2, y2, color, width = 2, dashed = false, dash = '6 6', dotted = false, waypoints = []) {
  const id = String(nextId++);
  let style = `endArrow=none;html=1;rounded=0;strokeWidth=${width};strokeColor=${color};`;
  if (waypoints.length) style += 'curved=1;';
  if (dashed) style += `dashed=1;dashPattern=${dash};`;
  if (dotted) style += 'dashed=1;dashPattern=1 5;';
  const pts = waypoints.length ? `<Array as="points">${waypoints.map(p => `<mxPoint x="${p[0]}" y="${p[1]}"/>`).join('')}</Array>` : '';
  cells.push(`<mxCell id="${id}" value="" style="${style}" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="${x1}" y="${y1}" as="sourcePoint"/><mxPoint x="${x2}" y="${y2}" as="targetPoint"/>${pts}</mxGeometry></mxCell>`);
  return id;
}
function panel(x, y, w, h, title, fill, stroke) {
  rect('', x, y, w, h, '#ffffff', '#d3d3d3', 1, false, 'center', 'shadow=0;');
  rect(title, x, y, w, 80, fill, stroke, 24, true);
}
function graphNodes(xs, y, labels) { xs.forEach((x, i) => node(labels[i], x, y)); }
function seqEdges(xs, y, color) { [[0,1],[1,2],[2,3],[4,5]].forEach(([i,j]) => line(xs[i]+27, y, xs[j]-27, y, color, 3)); }
function arc(x1, x2, y, height, color, dashed = false, dash = '8 6', lower = false) {
  const mid = (x1 + x2) / 2;
  if (lower) line(x1, y + 25, x2, y + 25, color, 2.3, dashed, dash, false, [[mid, y + height]]);
  else line(x1, y - 25, x2, y - 25, color, 2.3, dashed, dash, false, [[mid, y - height]]);
}
function adjustedDistance(seqDist) { return seqDist * (1.0 + 0.2 * Math.sin(seqDist * 0.5)); }
function drawDistanceEdges(xs, y, maxSeqSpan = 4) {
  // Mimic graph_builder.py: distance = seq_dist * (1 + 0.2 * sin(seq_dist * 0.5));
  // type 1 if distance <= 2, otherwise type 2.
  const seqPos = [1, 2, 3, 4, 5, 6];
  for (let i = 0; i < xs.length; i++) {
    for (let j = i + 1; j < xs.length; j++) {
      const seqDist = Math.abs(seqPos[j] - seqPos[i]);
      if (seqDist > maxSeqSpan) continue;
      const dij = adjustedDistance(seqDist);
      const isShort = dij <= 2.0;
      const span = j - i;
      if (span === 1) {
        line(xs[i] + 27, y, xs[j] - 27, y, isShort ? green : orange, 3, !isShort, '8 6');
      } else {
        const h = 55 + span * 22 + (i % 2) * 12;
        arc(xs[i], xs[j], y, h, isShort ? green : orange, !isShort, '8 6');
      }
    }
  }
}
function drawHybridLongRange(xs, y) {
  // Hybrid Graph = Distance edges + forced sparse long-range edges (type 3).
  // The real code uses long_range_stride and long_range_hops; in this schematic we show two stride-like long-range links.
  [[0,4], [1,5]].forEach(([i,j]) => {
    arc(xs[i], xs[j], y, 82, purple, true, '12 6 4 6');
    arc(xs[i], xs[j], y, 92, purple, true, '12 6 4 6', true);
  });
}

const labels = ['v<sub>1</sub>', 'v<sub>2</sub>', 'v<sub>3</sub>', 'v<sub>4</sub>', 'v<sub>L−1</sub>', 'v<sub>L</sub>'];
const blue = '#3366cc', green = '#55b95a', orange = '#f3a629', purple = '#9b3fd2', stroke = '#23466d';

panel(8, 8, 565, 538, '(a) Sequence Graph<br>(d<sub>max</sub> = 1)', '#eef7ff', '#d8e8f8');
panel(573, 8, 565, 538, '(b) Distance Graph<br>(d<sub>max</sub> &gt; 1)', '#eefaf0', '#d8ead7');
panel(1138, 8, 574, 538, '(c) Hybrid Graph<br>(Distance + Long-range)', '#eefaf0', '#d8ead7');
panel(8, 560, 970, 512, '(d) Global Node (for all graphs)', '#f6f1ff', '#e4d9fa');
panel(980, 560, 732, 512, '(e) Edge Type Legend', '#f6f1ff', '#e4d9fa');

const ytop = 250;
const xsa = [50, 132, 214, 296, 446, 530];
graphNodes(xsa, ytop, labels); seqEdges(xsa, ytop, blue);
text('⋯', 355, 232, 55, 36, 28, true);
text('Only adjacent residues connected', 94, 344, 390, 34, 24);
text('Edges: type 0 (sequence)', 148, 388, 285, 34, 24);

const xsb = [620, 702, 784, 866, 1010, 1090];
graphNodes(xsb, ytop, labels); drawDistanceEdges(xsb, ytop, 4);
text('⋯', 930, 232, 55, 36, 28, true);
text('Connect residues with folding-adjusted<br>distance <i>d</i><sub>ij</sub> ≤ <i>d</i><sub>max</sub>', 642, 338, 435, 58, 24);
line(675, 428, 750, 428, green, 3); text('type 1 (short): <i>d</i><sub>ij</sub> ≤ 2', 765, 410, 265, 38, 22, false, 'left');
line(675, 474, 750, 474, orange, 3, true, '8 6'); text('type 2 (long): <i>d</i><sub>ij</sub> &gt; 2', 765, 456, 280, 38, 22, false, 'left');

const xsc = [1188, 1270, 1352, 1434, 1584, 1670];
graphNodes(xsc, ytop, labels); drawDistanceEdges(xsc, ytop, 4); drawHybridLongRange(xsc, ytop);
text('⋯', 1505, 232, 55, 36, 28, true);
text('Distance edges + long-range edges', 1238, 340, 420, 36, 24);
line(1210, 405, 1285, 405, green, 3); text('type 1 (short): <i>d</i><sub>ij</sub> ≤ 2', 1300, 386, 290, 38, 22, false, 'left');
line(1210, 450, 1285, 450, orange, 3, true, '8 6'); text('type 2 (long): <i>d</i><sub>ij</sub> &gt; 2', 1300, 431, 300, 38, 22, false, 'left');
line(1210, 495, 1285, 495, purple, 3, true, '12 6 4 6'); text('type 3 (long-range): |<i>i</i> − <i>j</i>| = <i>k</i>Δ', 1300, 476, 390, 38, 22, false, 'left');

const xsd = [52, 132, 212, 292, 410, 490], yd = 925;
graphNodes(xsd, yd, labels); seqEdges(xsd, yd, stroke); text('⋯', 330, 907, 55, 36, 28, true);
const vgx = 300, vgy = 720; node('v<sub>g</sub>', vgx, vgy, 28, '#e9d5ff', '#7e3fb7');
xsd.forEach(x => line(vgx, vgy + 28, x, yd - 27, '#b06adf', 1.8, false, '', true));
text('Connected to all residues (type 4: global)', 78, 990, 385, 34, 22);
rect('Global node representation:', 470, 666, 495, 70, '#f4edff', '#d7c5f0', 23, true, 'left');
rect('<b>h</b><sub>g</sub><sup>(0)</sup> = <b>e</b><sub><i>learn</i></sub> + <b>MLP</b><sub>g</sub> ([<b>h</b><sup><i>state</i></sup>; <b>h</b><sup><i>env</i></sup>])', 470, 736, 495, 92, '#ffffff', '#d7c5f0', 28, true, 'left');
rect('where<br>• <b>e</b><sub><i>learn</i></sub> : learnable global embedding<br>• <b>h</b><sup><i>state</i></sup> : encoded precursor ion state<br>• <b>h</b><sup><i>env</i></sup> : encoded experimental environment<br>• <b>MLP</b><sub>g</sub>(·) : MLP with ReLU and Dropout', 525, 828, 440, 205, '#ffffff', '#d7c5f0', 20, false, 'left');
line(vgx + 28, vgy - 18, 525, 666, '#b06adf', 1.5, true, '4 4', false, [[335,666],[470,666]]);
line(vgx + 28, vgy + 15, 525, 828, '#b06adf', 1.5, true, '4 4', false, [[385,828],[470,828]]);

const x0 = 990, y0 = 618, colw = [98,165,162,295], rowh = [50,78,78,78,78,78];
['Type','Name','Visual','Definition'].forEach((h,i) => rect(h, x0 + colw.slice(0,i).reduce((a,b)=>a+b,0), y0, colw[i], rowh[0], '#f3f1f6', '#d6d6d6', 21, true));
const rows = [
  ['0','sequence<br>(backbone)','Adjacent residues<br>(|<i>i</i> − <i>j</i>| = 1)<br>(Sequence Graph only)'],
  ['1','distance<br>(short)','Folding-adjusted distance<br><i>d</i><sub>ij</sub> ≤ 2'],
  ['2','distance<br>(long)','Folding-adjusted distance<br><i>d</i><sub>ij</sub> &gt; 2 and ≤ <i>d</i><sub>max</sub>'],
  ['3','long-range','|<i>i</i> − <i>j</i>| = <i>k</i>Δ<br>(<i>k</i> = 1, . . . , <i>K</i>)'],
  ['4','global','Connected to global node <i>v</i><sub>g</sub><br>(All graphs)']
];
const colors = [blue, green, orange, purple, purple], dashs = [null, null, '8 6', '12 6 4 6', '1 5'];
let cy = y0 + rowh[0];
rows.forEach((r, idx) => {
  rect(r[0], x0, cy, colw[0], rowh[idx+1], '#ffffff', '#d6d6d6', 20, true);
  rect(r[1], x0 + colw[0], cy, colw[1], rowh[idx+1], '#ffffff', '#d6d6d6', 20);
  rect('', x0 + colw[0] + colw[1], cy, colw[2], rowh[idx+1], '#ffffff', '#d6d6d6', 1);
  rect(r[2], x0 + colw[0] + colw[1] + colw[2], cy, colw[3], rowh[idx+1], '#ffffff', '#d6d6d6', 19);
  const ly = cy + rowh[idx+1] / 2;
  line(x0 + colw[0] + colw[1] + 33, ly, x0 + colw[0] + colw[1] + colw[2] - 33, ly, colors[idx], 3, Boolean(dashs[idx]), dashs[idx] || '');
  cy += rowh[idx+1];
});

const xml = `<mxfile host="app.diagrams.net" modified="2026-07-14T00:00:00.000Z" agent="ZCode" version="24.7.17" type="device"><diagram id="graph-code-consistent" name="Page-1"><mxGraphModel dx="1720" dy="1080" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="${W}" pageHeight="${H}" math="1" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>${cells.join('')}</root></mxGraphModel></diagram></mxfile>`;
fs.writeFileSync(OUT, xml, 'utf8');
console.log(OUT);
