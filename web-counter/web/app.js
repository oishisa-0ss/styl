(() => {
'use strict';

const API = (() => {
  const a = new URLSearchParams(location.search).get('api') || '';
  return /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(a) ? a : '';
})();
const MAX_SIDE = 2560;
const W = 1800;
const CAND_MIN = 0.15;
const MAXZ = 8;
const REVIEW_BAND = 0.15;
const REVEAL_MS = 1100;
const LS = 140;
const INSIDE = 1.06;
const C = { ai: '#2F6FEB', add: '#E8702A', off: '#9AA5A0' };
const FONT = '"BIZ UDPGothic","Hiragino Sans","Yu Gothic UI","Meiryo",sans-serif';
const VERB = { ai: '除外', off: '数に戻す', addc: '追加を取り消す', user: '追加を取り消す', cand: '数に入れる', snap: '追加', new: '追加' };
const RESULT = { ai: C.off, off: C.ai, addc: C.off, user: C.off, cand: C.add, snap: C.add, new: C.add, outside: C.off };

const $ = id => document.getElementById(id);
const stage = $('stage'), cv = $('cv'), ctx = cv.getContext('2d');
const mini = $('mini'), mctx = mini.getContext('2d');
const loupeWrap = $('loupeWrap'), lcv = $('loupe'), lctx = lcv.getContext('2d');
const RM = matchMedia('(prefers-reduced-motion: reduce)');
const motion = () => !RM.matches;
const fine = matchMedia('(hover: hover) and (pointer: fine)');
let framed = true;
try { framed = window.self !== window.top; } catch (_) { framed = true; }
const ease = t => 1 - Math.pow(1 - t, 3);
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
const buzz = ms => { try { if (navigator.vibrate) navigator.vibrate(ms); } catch (_) {} };
const tipText = (kind, how) => kind === 'outside' ? 'シャーレの外です' : `${how}${VERB[kind]}`;

const settings = { conf: 0.40, showCand: false, antsUntil: 0 };
let cur = null;
let job = null;
let busy = false;
let mode = 'edit';
let size = 0, dpr = 1, fitS = 1;
const view = { s: 1, tx: 0, ty: 0 };
const dv = { s: 1, tx: 0, ty: 0 };
let dishEdit = null, dishTween = null;
let trans = null;
let cam = null, fx = [], floats = [], pulses = [];
let hover = null, aim = null, review = null;
let rafId = 0;

function cropOf(sm, d) {
  const s = Math.min(2 * d.r * 1.04, sm.w, sm.h);
  return { l: clamp(d.cx - s / 2, 0, sm.w - s), t: clamp(d.cy - s / 2, 0, sm.h - s), s };
}
function inCrop(b, cr, d) {
  const cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2;
  if (cx < cr.l || cy < cr.t || cx > cr.l + cr.s || cy > cr.t + cr.s) return false;
  return (cx - d.cx) ** 2 + (cy - d.cy) ** 2 <= (d.r * INSIDE) ** 2;
}

function applyDish(st, d) {
  st.dish = { cx: d.cx, cy: d.cy, r: d.r };
  const cr = st.crop = cropOf(st.sm, d);
  const k = W / cr.s;
  const c = st.img || document.createElement('canvas');
  c.width = c.height = W;
  const g = c.getContext('2d');
  g.imageSmoothingQuality = 'high';
  g.drawImage(st.photo, cr.l, cr.t, cr.s, cr.s, 0, 0, W, W);
  st.img = c;
  st.thumb = makeThumb(c);
  st.dishSq = { cx: (d.cx - cr.l) * k, cy: (d.cy - cr.t) * k, r: d.r * k };
  st.ai = [];
  st.sm.boxes.forEach((b, i) => {
    if (!inCrop(b, cr, d)) return;
    st.ai.push({ id: 'a' + i, x1: (b[0] - cr.l) * k, y1: (b[1] - cr.t) * k, x2: (b[2] - cr.l) * k, y2: (b[3] - cr.t) * k, s: b[4] });
  });
  st.user = []; st.ov = new Map(); st.undo = []; st.redo = []; st.seq = 0;
}

function cat(st, b) {
  const byAi = b.s >= settings.conf;
  const o = st.ov.get(b.id);
  if (byAi) return o === false ? 'off' : 'ai';
  if (o === true) return 'addc';
  return b.s >= CAND_MIN ? 'cand' : 'hid';
}
function tally(st) {
  let ai = 0, off = 0, add = st.user.length, cand = 0;
  for (const b of st.ai) {
    const c = cat(st, b);
    if (c === 'ai') ai++;
    else if (c === 'off') { ai++; off++; }
    else if (c === 'addc') add++;
    else if (c === 'cand') cand++;
  }
  return { ai, off, add, cand, total: ai - off + add, edited: off + add > 0 };
}
function medianSide(st) {
  const v = st.ai.filter(b => b.s >= settings.conf).map(b => Math.max(b.x2 - b.x1, b.y2 - b.y1)).sort((a, b) => a - b);
  return v.length ? v[v.length >> 1] : 36;
}
const counted = b => { const c = cat(cur, b); return c === 'ai' || c === 'addc'; };

function applyAct(a, reverse) {
  const st = cur;
  if (a.k === 'ov') {
    const v = reverse ? a.prev : a.next;
    if (v === undefined) st.ov.delete(a.id); else st.ov.set(a.id, v);
  } else if (a.k === 'addU') {
    if (reverse) st.user = st.user.filter(u => u !== a.box); else st.user.push(a.box);
  } else if (a.k === 'delU') {
    if (reverse) st.user.splice(a.idx, 0, a.box); else st.user = st.user.filter(u => u !== a.box);
  } else if (a.k === 'reset') {
    if (reverse) { st.ov = new Map(a.prevOv); st.user = a.prevUser.slice(); }
    else { st.ov = new Map(); st.user = []; }
  }
}
function doAct(a) { applyAct(a, false); cur.undo.push(a); cur.redo.length = 0; update(); }
function undo() { if (mode !== 'edit') return; const a = cur.undo.pop(); if (!a) return; applyAct(a, true); cur.redo.push(a); update(); toast('1つ前に戻しました'); }
function redo() { if (mode !== 'edit') return; const a = cur.redo.pop(); if (!a) return; applyAct(a, false); cur.undo.push(a); update(); toast('やり直しました'); }

function addFx(type, b, color) {
  if (!motion()) return;
  fx.push({ type, b, color, t0: performance.now(), dur: type === 'add' ? 380 : type === 'gone' ? 320 : 420 });
  requestRender();
}
function addFloat(b, d) {
  if (!motion() || !b || !d) return;
  floats.push({ ix: (b.x1 + b.x2) / 2, iy: b.y1, text: d > 0 ? '+1' : '−1', color: d > 0 ? C.add : '#4B5868', t0: performance.now() });
  requestRender();
}

function pick(ix, iy, minT, sc) {
  const st = cur, tol = 3 / sc;
  let best = null, bestD = Infinity;
  const consider = (b, kind) => {
    const w = b.x2 - b.x1, h = b.y2 - b.y1;
    const px = Math.max(tol, (minT - w) / 2), py = Math.max(tol, (minT - h) / 2);
    if (ix < b.x1 - px || ix > b.x2 + px || iy < b.y1 - py || iy > b.y2 + py) return;
    const d = Math.hypot(ix - (b.x1 + b.x2) / 2, iy - (b.y1 + b.y2) / 2) / Math.max(w, h, 1);
    if (d < bestD) { bestD = d; best = { kind, b }; }
  };
  for (const b of st.ai) {
    const c = cat(st, b);
    if (c === 'ai' || c === 'off' || c === 'addc' || (c === 'cand' && settings.showCand)) consider(b, c);
  }
  for (const u of st.user) consider(u, 'user');
  if (best) return best;
  const d = st.dishSq;
  if ((ix - d.cx) ** 2 + (iy - d.cy) ** 2 > (d.r * INSIDE) ** 2) return { kind: 'outside' };
  let snap = null, sd = Infinity;
  const pad = 4 / sc;
  for (const b of st.ai) {
    const c = cat(st, b);
    if (c !== 'hid' && c !== 'cand') continue;
    if (ix < b.x1 - pad || ix > b.x2 + pad || iy < b.y1 - pad || iy > b.y2 + pad) continue;
    const dd = Math.hypot(ix - (b.x1 + b.x2) / 2, iy - (b.y1 + b.y2) / 2);
    if (dd < sd) { sd = dd; snap = b; }
  }
  if (snap) return { kind: 'snap', b: snap };
  const h = medianSide(st) / 2;
  const cx = clamp(ix, h, W - h), cy = clamp(iy, h, W - h);
  return { kind: 'new', box: { x1: cx - h, y1: cy - h, x2: cx + h, y2: cy + h, s: 1 } };
}
function act(t) {
  const st = cur;
  if (t.kind === 'outside') { toast('シャーレの外には追加できません。範囲がずれていたら「範囲を直す」へ'); return; }
  const prev = t.b ? st.ov.get(t.b.id) : undefined;
  let d = 0, b = t.b;
  switch (t.kind) {
    case 'ai': doAct({ k: 'ov', id: b.id, prev, next: false }); addFx('off', b); d = -1; break;
    case 'off': doAct({ k: 'ov', id: b.id, prev, next: undefined }); addFx('on', b); d = 1; break;
    case 'addc': doAct({ k: 'ov', id: b.id, prev, next: undefined }); addFx('gone', b, C.add); d = -1; break;
    case 'cand': case 'snap': doAct({ k: 'ov', id: b.id, prev, next: true }); addFx('add', b); d = 1; break;
    case 'user': doAct({ k: 'delU', box: b, idx: st.user.indexOf(b) }); addFx('gone', b, C.add); d = -1; break;
    case 'new': b = { ...t.box, id: 'u' + (++st.seq) }; doAct({ k: 'addU', box: b }); addFx('add', b); d = 1; break;
  }
  addFloat(b, d);
  buzz(10);
}
function handleTap(sx, sy, type) {
  const ix = (sx - view.tx) / view.s, iy = (sy - view.ty) / view.s;
  if (ix < 0 || iy < 0 || ix > W || iy > W) return;
  act(pick(ix, iy, (type === 'mouse' ? 14 : 32) / view.s, view.s));
  if (motion()) pulses.push({ x: sx, y: sy, t0: performance.now() });
  if (type === 'mouse') setHover({ x: sx, y: sy });
  requestRender();
}

function clampView() {
  view.s = clamp(view.s, fitS, fitS * MAXZ);
  const span = W * view.s;
  if (span <= size + 0.5) { view.tx = (size - span) / 2; view.ty = (size - span) / 2; }
  else {
    view.tx = clamp(view.tx, size - span, 0);
    view.ty = clamp(view.ty, size - span, 0);
  }
  const z = view.s / fitS;
  $('zlv').textContent = '×' + (z < 1.05 ? '1' : z.toFixed(1));
  $('zfit').disabled = z < 1.01;
  $('zout').disabled = z < 1.01;
  $('zin').disabled = z > MAXZ - 0.01;
}
const viewCenter = () => ({ cx: (size / 2 - view.tx) / view.s, cy: (size / 2 - view.ty) / view.s });
function setView(s, cx, cy) {
  view.s = clamp(s, fitS, fitS * MAXZ);
  view.tx = size / 2 - cx * view.s; view.ty = size / 2 - cy * view.s;
  clampView();
}
function animateView(s, cx, cy, dur = 380) {
  if (!motion() || !size) { setView(s, cx, cy); requestRender(); return; }
  const c = viewCenter();
  cam = { s0: view.s, s1: clamp(s, fitS, fitS * MAXZ), cx0: c.cx, cy0: c.cy, cx1: cx, cy1: cy, t0: performance.now(), dur };
  requestRender();
}
function stepCam(now) {
  const p = Math.min(1, (now - cam.t0) / cam.dur), e = ease(p);
  setView(cam.s0 * Math.pow(cam.s1 / cam.s0, e), cam.cx0 + (cam.cx1 - cam.cx0) * e, cam.cy0 + (cam.cy1 - cam.cy0) * e);
  if (p >= 1) { cam = null; return false; }
  return true;
}
function zoomAt(x, y, s) {
  const ix = (x - view.tx) / view.s, iy = (y - view.ty) / view.s;
  view.s = clamp(s, fitS, fitS * MAXZ);
  view.tx = x - ix * view.s; view.ty = y - iy * view.s;
  clampView(); requestRender();
}

const dvMin = () => size / Math.max(cur.sm.w, cur.sm.h);
function clampDv(v) {
  const f = dvMin();
  v.s = clamp(v.s, f, f * 8);
  const sw = cur.sm.w * v.s, sh = cur.sm.h * v.s;
  v.tx = sw <= size ? (size - sw) / 2 : clamp(v.tx, size - sw, 0);
  v.ty = sh <= size ? (size - sh) / 2 : clamp(v.ty, size - sh, 0);
  return v;
}
function dvWhole() { return clampDv({ s: dvMin(), tx: 0, ty: 0 }); }
function dvAround(d) {
  const s = clamp(size * 0.78 / (2 * d.r), dvMin(), dvMin() * 8);
  return clampDv({ s, tx: size / 2 - d.cx * s, ty: size / 2 - d.cy * s });
}
const rectIn = (v, cr) => ({ x: cr.l * v.s + v.tx, y: cr.t * v.s + v.ty, z: cr.s * v.s });
const knobOf = d => ({ x: d.cx + d.r * Math.SQRT1_2, y: d.cy + d.r * Math.SQRT1_2 });

function measure() {
  const w = stage.getBoundingClientRect().width;
  if (!w) return;
  const rel = size ? view.s / fitS : 1;
  const c = size ? viewCenter() : { cx: W / 2, cy: W / 2 };
  size = w; dpr = Math.min(window.devicePixelRatio || 1, 3);
  cv.width = Math.round(size * dpr); cv.height = Math.round(size * dpr);
  lcv.width = lcv.height = Math.round(LS * dpr);
  mini.width = mini.height = Math.round(84 * dpr);
  fitS = size / W;
  setView(fitS * rel, c.cx, c.cy);
  if (cur && mode === 'dish' && dishEdit) Object.assign(dv, dvAround(dishEdit));
  render();
}

function requestRender() { if (!rafId) rafId = requestAnimationFrame(render); }
function rrect(g, x, y, w, h, r) { if (g.roundRect) g.roundRect(x, y, w, h, r); else g.rect(x, y, w, h); }
function render() {
  rafId = 0;
  if (!size) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);
  if (!cur || !cur.img) { if (job) drawWaiting(); if (!mini.hidden) mini.hidden = true; return; }
  const now = performance.now();
  let live;
  if (trans) live = renderTrans(now);
  else if (mode === 'dish') live = renderDish(now);
  else live = renderEdit(now);
  if ((mode !== 'edit' || trans) && !mini.hidden) mini.hidden = true;
  if (live || trans) requestRender();
}

function drawWaiting() {
  const c = job.canvas, s = size / Math.max(c.width, c.height);
  ctx.fillStyle = '#111516'; ctx.fillRect(0, 0, size, size);
  ctx.drawImage(c, (size - c.width * s) / 2, (size - c.height * s) / 2, c.width * s, c.height * s);
}
function dimOutside(cx, cy, r, a) {
  if (a <= 0) return;
  ctx.beginPath(); ctx.rect(0, 0, size, size); ctx.arc(cx, cy, r, 0, Math.PI * 2, true);
  ctx.fillStyle = `rgba(5,9,13,${a})`; ctx.fill('evenodd');
}
function dishLine(cx, cy, r, q, strong, now) {
  const a0 = -Math.PI / 2, a1 = a0 + Math.PI * 2 * q;
  ctx.save();
  ctx.lineCap = 'round';
  ctx.beginPath(); ctx.arc(cx, cy, r, a0, a1);
  ctx.lineWidth = strong ? 5 : 3.5; ctx.strokeStyle = 'rgba(8,14,20,.35)'; ctx.stroke();
  if (strong && q >= 1) { ctx.setLineDash([9, 7]); ctx.lineDashOffset = motion() ? -(now / 45) % 16 : 0; }
  else if (!strong) ctx.setLineDash([6, 6]);
  ctx.beginPath(); ctx.arc(cx, cy, r, a0, a1);
  ctx.lineWidth = strong ? 2.5 : 1.5; ctx.strokeStyle = strong ? '#fff' : 'rgba(255,255,255,.7)'; ctx.stroke();
  if (q < 1) { ctx.setLineDash([]); ctx.beginPath(); ctx.arc(cx + r * Math.cos(a1), cy + r * Math.sin(a1), 4.5, 0, Math.PI * 2); ctx.fillStyle = C.ai; ctx.fill(); }
  ctx.restore();
}
function chip(text, y) {
  ctx.font = `700 13px ${FONT}`;
  const w = ctx.measureText(text).width + 24;
  ctx.fillStyle = 'rgba(11,18,25,.82)';
  ctx.beginPath(); rrect(ctx, (size - w) / 2, y, w, 30, 15); ctx.fill();
  ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, size / 2, y + 15);
  ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';
}

function renderTrans(now) {
  const st = cur, tr = trans;
  const p = tr.dur ? Math.min(1, (now - tr.t0) / tr.dur) : 1, e = ease(p);
  const r = { x: tr.from.x + (tr.to.x - tr.from.x) * e, y: tr.from.y + (tr.to.y - tr.from.y) * e, z: tr.from.z + (tr.to.z - tr.from.z) * e };
  const cr = tr.crop || st.crop, d = tr.dish || st.dish;
  const sc = r.z / cr.s, tx = r.x - cr.l * sc, ty = r.y - cr.t * sc;
  ctx.fillStyle = '#111516'; ctx.fillRect(0, 0, size, size);
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(st.photo, tx, ty, st.sm.w * sc, st.sm.h * sc);
  const cx = d.cx * sc + tx, cy = d.cy * sc + ty, rr = d.r * sc;
  dimOutside(cx, cy, rr * INSIDE, tr.dim[0] + (tr.dim[1] - tr.dim[0]) * e);
  dishLine(cx, cy, rr, tr.draw ? e : 1, true, now);
  if (tr.label) chip(tr.label, size - 46);
  if (p >= 1) { const then = tr.then; trans = null; if (then) then(); }
  return true;
}

function renderDish(now) {
  const st = cur, sm = st.sm;
  let live = false;
  if (dishTween) {
    const p = Math.min(1, (now - dishTween.t0) / dishTween.dur), e = ease(p);
    const a = dishTween.from, b = dishTween.to;
    dishEdit = { cx: a.cx + (b.cx - a.cx) * e, cy: a.cy + (b.cy - a.cy) * e, r: a.r + (b.r - a.r) * e };
    if (p >= 1) dishTween = null; else live = true;
    updateDishPanel();
  }
  const d = dishEdit;
  ctx.fillStyle = '#111516'; ctx.fillRect(0, 0, size, size);
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(st.photo, dv.tx, dv.ty, sm.w * dv.s, sm.h * dv.s);
  const X = v => v * dv.s + dv.tx, Y = v => v * dv.s + dv.ty;
  const cx = X(d.cx), cy = Y(d.cy), r = d.r * dv.s;
  dimOutside(cx, cy, r * INSIDE, 0.5);
  const cr = cropOf(sm, d);
  ctx.setLineDash([4, 5]); ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(255,255,255,.4)';
  ctx.strokeRect(X(cr.l), Y(cr.t), cr.s * dv.s, cr.s * dv.s); ctx.setLineDash([]);
  for (const b of sm.boxes) {
    if (b[4] < settings.conf) continue;
    const inside = inCrop(b, cr, d);
    ctx.beginPath(); ctx.arc(X((b[0] + b[2]) / 2), Y((b[1] + b[3]) / 2), inside ? 3 : 2.2, 0, Math.PI * 2);
    ctx.fillStyle = inside ? C.ai : 'rgba(255,255,255,.55)'; ctx.fill();
  }
  dishLine(cx, cy, r, 1, true, now);
  if (motion()) live = true;
  ctx.lineWidth = 2; ctx.strokeStyle = '#fff';
  ctx.beginPath(); ctx.moveTo(cx - 9, cy); ctx.lineTo(cx + 9, cy); ctx.moveTo(cx, cy - 9); ctx.lineTo(cx, cy + 9); ctx.stroke();
  const k = knobOf(d), kx = X(k.x), ky = Y(k.y);
  ctx.beginPath(); ctx.arc(kx, ky, 13, 0, Math.PI * 2);
  ctx.fillStyle = '#fff'; ctx.shadowColor = 'rgba(0,0,0,.45)'; ctx.shadowBlur = 8; ctx.fill(); ctx.shadowBlur = 0;
  ctx.lineWidth = 2.5; ctx.strokeStyle = '#2E7D5B'; ctx.stroke();
  ctx.beginPath(); ctx.lineWidth = 2; ctx.lineCap = 'round';
  ctx.moveTo(kx - 5, ky - 5); ctx.lineTo(kx + 5, ky + 5);
  ctx.moveTo(kx + 5, ky + 5); ctx.lineTo(kx + 5, ky + 1); ctx.moveTo(kx + 5, ky + 5); ctx.lineTo(kx + 1, ky + 5);
  ctx.moveTo(kx - 5, ky - 5); ctx.lineTo(kx - 5, ky - 1); ctx.moveTo(kx - 5, ky - 5); ctx.lineTo(kx - 1, ky - 5);
  ctx.stroke(); ctx.lineCap = 'butt';
  return live;
}

function renderEdit(now) {
  let live = false;
  if (cam) live = stepCam(now);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(cur.img, view.tx, view.ty, W * view.s, W * view.s);
  const ds = cur.dishSq;
  const dcx = ds.cx * view.s + view.tx, dcy = ds.cy * view.s + view.ty, dr = ds.r * view.s;
  dimOutside(dcx, dcy, dr * INSIDE, 0.35);
  dishLine(dcx, dcy, dr, 1, false, now);

  let scanY = Infinity;
  if (cur.reveal) {
    if (!isFinite(cur.reveal.t0)) scanY = -1;
    else {
      const p = Math.min(1, (now - cur.reveal.t0) / REVEAL_MS);
      if (p >= 1) { cur.reveal = null; update({ silent: true }); }
      else { scanY = ease(p) * (W + 90); live = true; }
    }
  }
  const lw = view.s / fitS >= 2 ? 2.5 : 2;
  fx = fx.filter(f => now - f.t0 < f.dur);
  const fxOf = new Map(fx.map(f => [f.b, f]));
  if (fx.length) live = true;
  const ants = settings.showCand && now < settings.antsUntil;
  if (ants) live = true;
  let seen = 0;
  for (const b of cur.ai) {
    const c = cat(cur, b);
    if (c === 'hid' || (c === 'cand' && !settings.showCand) || b.y1 > scanY) continue;
    if (c === 'ai' || c === 'addc') seen++;
    const a = scanY === Infinity ? 1 : Math.min(1, (scanY - b.y1) / 90);
    drawBox(b, c === 'addc' ? 'add' : c, lw, fxOf.get(b), now, a, ants);
  }
  for (const u of cur.user) drawBox(u, 'add', lw, fxOf.get(u), now, 1, false);
  for (const f of fx) if (f.type === 'gone') drawGhost(f, now, lw);
  if (scanY !== Infinity && scanY >= 0) { drawScan(scanY); setCount(seen, true); }
  if (review) { drawSpotlight(now); if (motion()) live = true; }
  if (hover && !aim && !review) drawHover(lw);
  if (drawFloats(now)) live = true;
  if (drawPulses(now)) live = true;
  drawMini();
  return live;
}
function screenRect(b, g = 1.5) {
  const s = view.s;
  return { x: b.x1 * s + view.tx - g, y: b.y1 * s + view.ty - g, w: (b.x2 - b.x1) * s + 2 * g, h: (b.y2 - b.y1) * s + 2 * g };
}
function drawBox(b, kind, lw, f, now, alpha, ants) {
  let { x, y, w, h } = screenRect(b);
  if (x > size || y > size || x + w < 0 || y + h < 0) return;
  let p = 1;
  if (f) {
    p = Math.min(1, (now - f.t0) / f.dur);
    const k = f.type === 'add' ? 1 + 0.35 * (1 - ease(p)) : (f.type === 'off' || f.type === 'on') ? 1 + 0.18 * Math.sin(Math.PI * p) : 1;
    if (k !== 1) { const cx = x + w / 2, cy = y + h / 2; w *= k; h *= k; x = cx - w / 2; y = cy - h / 2; }
  }
  const col = kind === 'add' ? C.add : kind === 'off' ? C.off : C.ai;
  const lwk = kind === 'cand' ? 1.5 : lw;
  let dash = kind === 'off' ? [5, 3] : kind === 'cand' ? [2, 3] : [];
  if (f && f.type === 'add' && p < 1) { const L = 2 * (w + h); dash = [L * ease(p), L]; }
  ctx.globalAlpha = alpha;
  ctx.setLineDash(dash);
  ctx.lineDashOffset = kind === 'cand' && ants ? -(now / 60) % 5 : 0;
  ctx.lineWidth = lwk + 2; ctx.strokeStyle = 'rgba(10,16,22,.35)'; ctx.strokeRect(x, y, w, h);
  ctx.lineWidth = lwk; ctx.strokeStyle = col; ctx.strokeRect(x, y, w, h);
  ctx.setLineDash([]); ctx.lineDashOffset = 0;
  if (f && f.type === 'off' && p < 1) { ctx.globalAlpha = alpha * (1 - p); ctx.strokeStyle = C.ai; ctx.strokeRect(x, y, w, h); ctx.globalAlpha = alpha; }
  if (w >= 18 && (kind === 'add' || kind === 'off')) badge(ctx, x + w, y, kind, col, f ? ease(p) : 1);
  ctx.globalAlpha = 1;
}
function badge(g, cx, cy, kind, col, p) {
  const r = 6.5 * (0.35 + 0.65 * p);
  g.beginPath(); g.arc(cx, cy, r, 0, Math.PI * 2); g.fillStyle = col; g.fill();
  if (p < 0.4) return;
  g.beginPath(); g.lineWidth = 1.8; g.strokeStyle = '#fff'; g.lineCap = 'round';
  if (kind === 'add') { g.moveTo(cx - 3.2, cy); g.lineTo(cx + 3.2, cy); g.moveTo(cx, cy - 3.2); g.lineTo(cx, cy + 3.2); }
  else { g.moveTo(cx - 2.6, cy - 2.6); g.lineTo(cx + 2.6, cy + 2.6); g.moveTo(cx + 2.6, cy - 2.6); g.lineTo(cx - 2.6, cy + 2.6); }
  g.stroke(); g.lineCap = 'butt';
}
function drawGhost(f, now, lw) {
  const p = Math.min(1, (now - f.t0) / f.dur);
  let { x, y, w, h } = screenRect(f.b);
  const k = 1 - 0.35 * ease(p), cx = x + w / 2, cy = y + h / 2;
  w *= k; h *= k; x = cx - w / 2; y = cy - h / 2;
  ctx.globalAlpha = 1 - p; ctx.lineWidth = lw; ctx.strokeStyle = f.color; ctx.strokeRect(x, y, w, h); ctx.globalAlpha = 1;
}
function drawScan(scanY) {
  const y = scanY * view.s + view.ty;
  const g = ctx.createLinearGradient(0, y - 70, 0, y);
  g.addColorStop(0, 'rgba(200,215,216,0)'); g.addColorStop(1, 'rgba(200,215,216,.30)');
  ctx.fillStyle = g; ctx.fillRect(0, y - 70, size, 70);
  ctx.fillStyle = 'rgba(235,242,242,.9)'; ctx.fillRect(0, y - 1, size, 2);
}
function drawSpotlight(now) {
  const b = review.items[review.i];
  if (!b) return;
  const r = screenRect(b, 16);
  ctx.save();
  ctx.beginPath(); ctx.rect(0, 0, size, size); rrect(ctx, r.x, r.y, r.w, r.h, 12);
  ctx.fillStyle = 'rgba(5,9,13,.58)'; ctx.fill('evenodd');
  const k = motion() ? 0.5 + 0.5 * Math.sin(now / 260) : 1;
  ctx.beginPath(); rrect(ctx, r.x, r.y, r.w, r.h, 12);
  ctx.lineWidth = 2; ctx.strokeStyle = `rgba(255,255,255,${0.4 + 0.5 * k})`; ctx.stroke();
  ctx.restore();
}
function drawHover(lw) {
  const t = hover.t, col = RESULT[t.kind];
  if (t.kind === 'outside') return;
  ctx.save();
  if (t.b) {
    const r = screenRect(t.b, 3);
    ctx.shadowColor = col; ctx.shadowBlur = 12; ctx.lineWidth = lw + 1; ctx.strokeStyle = col; ctx.strokeRect(r.x, r.y, r.w, r.h);
  } else {
    const r = screenRect(t.box);
    ctx.setLineDash([4, 3]); ctx.globalAlpha = 0.9; ctx.lineWidth = lw; ctx.strokeStyle = C.add; ctx.strokeRect(r.x, r.y, r.w, r.h);
    ctx.setLineDash([]); if (r.w >= 18) badge(ctx, r.x + r.w, r.y, 'add', C.add, 1);
  }
  ctx.restore();
}
function drawFloats(now) {
  floats = floats.filter(f => now - f.t0 < 850);
  ctx.font = `700 16px ${FONT}`; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
  for (const f of floats) {
    const p = (now - f.t0) / 850;
    const x = f.ix * view.s + view.tx, y = f.iy * view.s + view.ty - 6 - 26 * ease(p);
    ctx.globalAlpha = p < 0.15 ? p / 0.15 : 1 - (p - 0.15) / 0.85;
    ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(255,255,255,.92)'; ctx.strokeText(f.text, x, y);
    ctx.fillStyle = f.color; ctx.fillText(f.text, x, y);
  }
  ctx.globalAlpha = 1; ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';
  return floats.length > 0;
}
function drawPulses(now) {
  pulses = pulses.filter(p => now - p.t0 < 320);
  for (const p of pulses) {
    const k = (now - p.t0) / 320;
    ctx.beginPath(); ctx.arc(p.x, p.y, 10 + 22 * k, 0, Math.PI * 2);
    ctx.globalAlpha = 1 - k; ctx.lineWidth = 2.5; ctx.strokeStyle = '#fff'; ctx.stroke();
  }
  ctx.globalAlpha = 1;
  return pulses.length > 0;
}
function drawMini() {
  const show = view.s / fitS > 1.25 && !!cur.thumb;
  if (mini.hidden === show) mini.hidden = !show;
  if (!show) return;
  const m = mini.width;
  mctx.setTransform(1, 0, 0, 1, 0, 0);
  mctx.drawImage(cur.thumb, 0, 0, m, m);
  const k = m / W;
  const x = -view.tx / view.s * k, y = -view.ty / view.s * k, w = size / view.s * k;
  mctx.fillStyle = 'rgba(0,0,0,.38)';
  mctx.beginPath(); mctx.rect(0, 0, m, m); mctx.rect(x, y, w, w); mctx.fill('evenodd');
  mctx.lineWidth = 2 * dpr; mctx.strokeStyle = '#BFE3CF'; mctx.strokeRect(x, y, w, w);
}

function startAim(id) {
  if (!gest || gest.mode !== 'tap' || pts.size !== 1) return;
  const q = pts.get(id);
  if (!q) return;
  gest.mode = 'aim';
  aim = { sx: q.x, sy: q.y, t: null };
  loupeWrap.hidden = false;
  updateAim();
  buzz(8);
}
function updateAim() {
  const ix = clamp((aim.sx - view.tx) / view.s, 0, W), iy = clamp((aim.sy - view.ty) / view.s, 0, W);
  const Lz = clamp(Math.max(view.s * 2.5, fitS * 4), fitS, fitS * 14);
  const R = LS / 2 / Lz;
  const t = pick(ix, iy, 20 / Lz, Lz);
  aim.t = t;
  const g = lctx;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, LS, LS);
  g.save();
  g.beginPath(); g.arc(LS / 2, LS / 2, LS / 2, 0, Math.PI * 2); g.clip();
  g.fillStyle = '#111516'; g.fillRect(0, 0, LS, LS);
  g.drawImage(cur.img, ix - R, iy - R, 2 * R, 2 * R, 0, 0, LS, LS);
  const X = v => (v - (ix - R)) * Lz, Y = v => (v - (iy - R)) * Lz;
  const box = (b, col, dash, lw) => {
    if (b.x2 < ix - R || b.x1 > ix + R || b.y2 < iy - R || b.y1 > iy + R) return;
    g.setLineDash(dash); g.lineWidth = lw; g.strokeStyle = col;
    g.strokeRect(X(b.x1) - 2, Y(b.y1) - 2, (b.x2 - b.x1) * Lz + 4, (b.y2 - b.y1) * Lz + 4);
  };
  for (const b of cur.ai) {
    const c = cat(cur, b);
    if (c === 'ai') box(b, C.ai, [], 2);
    else if (c === 'off') box(b, C.off, [5, 3], 2);
    else if (c === 'addc') box(b, C.add, [], 2);
    else if (c === 'cand' && settings.showCand) box(b, C.ai, [2, 3], 1.5);
  }
  for (const u of cur.user) box(u, C.add, [], 2);
  g.setLineDash([]);
  if (t.kind !== 'outside') {
    const col = RESULT[t.kind];
    g.shadowColor = col; g.shadowBlur = 10;
    if (t.b) box(t.b, col, [], 3.5); else box(t.box, C.add, [4, 3], 2.5);
    g.shadowBlur = 0; g.setLineDash([]);
  }
  const c = LS / 2;
  g.lineCap = 'round';
  for (const [w, s] of [[4, 'rgba(10,16,22,.5)'], [2, '#fff']]) {
    g.lineWidth = w; g.strokeStyle = s; g.beginPath();
    g.moveTo(c - 13, c); g.lineTo(c - 5, c); g.moveTo(c + 5, c); g.lineTo(c + 13, c);
    g.moveTo(c, c - 13); g.lineTo(c, c - 5); g.moveTo(c, c + 5); g.lineTo(c, c + 13);
    g.stroke();
  }
  g.restore();
  $('loupeLabel').textContent = tipText(t.kind, '離すと');
  const left = clamp(aim.sx - LS / 2, 6, size - LS - 6);
  let top = aim.sy - LS - 72;
  if (top < 6) top = Math.min(aim.sy + 48, size - LS - 36);
  loupeWrap.style.transform = `translate(${left}px, ${top}px)`;
}
function endAim() { aim = null; loupeWrap.hidden = true; }

const pts = new Map();
let gest = null;
const local = e => { const r = cv.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
const pair = () => { const [a, b] = [...pts.values()]; return { d: Math.hypot(a.x - b.x, a.y - b.y) || 1, m: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 } }; };

cv.addEventListener('pointerdown', e => {
  if (e.pointerType === 'mouse' && e.button !== 0) return;
  if (trans || busy || !cur) return;
  cv.setPointerCapture(e.pointerId);
  cam = null; hover = null; hideTip();
  const p = local(e);
  pts.set(e.pointerId, { x: p.x, y: p.y, x0: p.x, y0: p.y, t0: performance.now(), type: e.pointerType });
  if (mode === 'dish') { dishDown(p); requestRender(); return; }
  if (pts.size === 1) {
    gest = { mode: 'tap', multi: false, tx0: view.tx, ty0: view.ty, hold: 0 };
    if (e.pointerType !== 'mouse') { const id = e.pointerId; gest.hold = setTimeout(() => startAim(id), 260); }
  } else if (pts.size === 2) {
    if (gest) clearTimeout(gest.hold);
    endAim();
    const { d, m } = pair();
    gest = { mode: 'pinch', multi: true, d0: d, m0: m, s0: view.s, tx0: view.tx, ty0: view.ty, hold: 0 };
  }
  requestRender();
});
cv.addEventListener('pointermove', e => {
  const q = pts.get(e.pointerId);
  if (!q) { if (e.pointerType === 'mouse' && !pts.size) { if (mode === 'dish') dishHover(local(e)); else setHover(local(e)); } return; }
  if (!gest) return;
  const p = local(e); q.x = p.x; q.y = p.y;
  if (mode === 'dish') { dishMove(q); return; }
  if (gest.mode === 'pinch' && pts.size >= 2) {
    const { d, m } = pair();
    const s = clamp(gest.s0 * d / gest.d0, fitS, fitS * MAXZ);
    const ix = (gest.m0.x - gest.tx0) / gest.s0, iy = (gest.m0.y - gest.ty0) / gest.s0;
    view.s = s; view.tx = m.x - ix * s; view.ty = m.y - iy * s;
    clampView(); requestRender();
  } else if (gest.mode === 'aim') {
    aim.sx = q.x; aim.sy = q.y; updateAim(); requestRender();
  } else if (pts.size === 1) {
    const th = q.type === 'mouse' ? 5 : 10;
    if (gest.mode === 'tap' && Math.hypot(q.x - q.x0, q.y - q.y0) > th) { gest.mode = 'pan'; clearTimeout(gest.hold); }
    if (gest.mode === 'pan') {
      view.tx = gest.tx0 + (q.x - q.x0); view.ty = gest.ty0 + (q.y - q.y0);
      clampView(); requestRender();
    }
  }
});
function endPointer(e) {
  const q = pts.get(e.pointerId);
  if (!q) return;
  pts.delete(e.pointerId);
  if (mode === 'dish') { dishUp(e, q); requestRender(); return; }
  if (gest) clearTimeout(gest.hold);
  if (gest && gest.mode === 'aim') {
    if (e.type === 'pointerup' && aim && aim.t) act(aim.t);
    endAim();
  } else if (e.type === 'pointerup' && gest && gest.mode === 'tap' && !gest.multi && pts.size === 0 && performance.now() - q.t0 < 700) {
    handleTap(q.x, q.y, q.type);
  }
  if (pts.size === 0) gest = null;
  else if (pts.size === 1 && gest && gest.mode === 'pinch') {
    const r = [...pts.values()][0]; r.x0 = r.x; r.y0 = r.y;
    gest = { mode: 'pan', multi: true, tx0: view.tx, ty0: view.ty, hold: 0 };
  }
  requestRender();
}
cv.addEventListener('pointerup', endPointer);
cv.addEventListener('pointercancel', endPointer);
cv.addEventListener('pointerleave', e => { if (e.pointerType === 'mouse' && !pts.size) { hover = null; hideTip(); cv.style.cursor = ''; requestRender(); } });
cv.addEventListener('wheel', e => {
  e.preventDefault();
  if (trans || busy || !cur) return;
  const p = local(e), k = Math.exp(-e.deltaY * 0.0018);
  if (mode === 'dish') { dvZoomAt(p.x, p.y, dv.s * k); return; }
  cam = null;
  zoomAt(p.x, p.y, view.s * k);
  setHover(p);
}, { passive: false });
cv.addEventListener('contextmenu', e => e.preventDefault());
document.addEventListener('gesturestart', e => e.preventDefault());
mini.addEventListener('pointerdown', e => {
  e.stopPropagation();
  const r = mini.getBoundingClientRect();
  animateView(view.s, (e.clientX - r.left) / r.width * W, (e.clientY - r.top) / r.height * W, 300);
});
$('zin').addEventListener('click', () => { const c = viewCenter(); animateView(view.s * 1.7, c.cx, c.cy, 260); });
$('zout').addEventListener('click', () => { const c = viewCenter(); animateView(view.s / 1.7, c.cx, c.cy, 260); });
$('zfit').addEventListener('click', () => animateView(fitS, W / 2, W / 2));

function setHover(p) {
  if (!cur || !size || review || mode !== 'edit' || trans) return;
  const ix = (p.x - view.tx) / view.s, iy = (p.y - view.ty) / view.s;
  if (ix < 0 || iy < 0 || ix > W || iy > W) { hover = null; hideTip(); requestRender(); return; }
  const t = pick(ix, iy, 14 / view.s, view.s);
  hover = { t };
  showTip(p, tipText(t.kind, 'クリックで'));
  requestRender();
}
function showTip(p, text) {
  const tip = $('tip');
  tip.textContent = text;
  tip.hidden = false;
  const x = Math.min(p.x + 14, size - tip.offsetWidth - 6), y = Math.min(p.y + 18, size - tip.offsetHeight - 6);
  tip.style.transform = `translate(${x}px, ${y}px)`;
}
function hideTip() { $('tip').hidden = true; }

let dg = null;
function dishHit(p) {
  const d = dishEdit, k = knobOf(d);
  const kx = k.x * dv.s + dv.tx, ky = k.y * dv.s + dv.ty;
  const cx = d.cx * dv.s + dv.tx, cy = d.cy * dv.s + dv.ty;
  if (Math.hypot(p.x - kx, p.y - ky) < 28) return 'resize';
  if (Math.hypot(p.x - cx, p.y - cy) < d.r * dv.s) return 'move';
  return 'out';
}
function dishDown(p) {
  dishTween = null;
  if (pts.size >= 2) {
    const { d, m } = pair();
    dg = { mode: 'pinch', d0: d, m0: m, s0: dv.s, tx0: dv.tx, ty0: dv.ty };
    return;
  }
  const hit = dishHit(p);
  gest = { mode: hit };
  dg = { mode: hit === 'out' ? 'tap' : hit, x0: p.x, y0: p.y, cx0: dishEdit.cx, cy0: dishEdit.cy, tx0: dv.tx, ty0: dv.ty };
}
function dishMove(q) {
  if (!dg) return;
  const sm = cur.sm;
  if (dg.mode === 'pinch' && pts.size >= 2) {
    const { d, m } = pair();
    const ix = (dg.m0.x - dg.tx0) / dg.s0, iy = (dg.m0.y - dg.ty0) / dg.s0;
    const s = clamp(dg.s0 * d / dg.d0, dvMin(), dvMin() * 8);
    Object.assign(dv, clampDv({ s, tx: m.x - ix * s, ty: m.y - iy * s }));
  } else if (dg.mode === 'move') {
    dishEdit.cx = clamp(dg.cx0 + (q.x - dg.x0) / dv.s, 0, sm.w);
    dishEdit.cy = clamp(dg.cy0 + (q.y - dg.y0) / dv.s, 0, sm.h);
    updateDishPanel();
  } else if (dg.mode === 'resize') {
    const px = (q.x - dv.tx) / dv.s, py = (q.y - dv.ty) / dv.s;
    const m = Math.min(sm.w, sm.h);
    dishEdit.r = clamp(Math.hypot(px - dishEdit.cx, py - dishEdit.cy), m * 0.1, m * 0.6);
    updateDishPanel();
  } else if (dg.mode === 'tap' || dg.mode === 'pan') {
    const th = q.type === 'mouse' ? 5 : 10;
    if (dg.mode === 'tap' && Math.hypot(q.x - q.x0, q.y - q.y0) > th) dg.mode = 'pan';
    if (dg.mode === 'pan') Object.assign(dv, clampDv({ s: dv.s, tx: dg.tx0 + (q.x - q.x0), ty: dg.ty0 + (q.y - q.y0) }));
  }
  requestRender();
}
function dishUp(e, q) {
  if (dg && dg.mode === 'tap' && e.type === 'pointerup' && pts.size === 0) {
    const to = { cx: clamp((q.x - dv.tx) / dv.s, 0, cur.sm.w), cy: clamp((q.y - dv.ty) / dv.s, 0, cur.sm.h), r: dishEdit.r };
    tweenDish(to);
    buzz(8);
  }
  if (pts.size === 0) { dg = null; gest = null; }
  else if (dg && dg.mode === 'pinch') { const r = [...pts.values()][0]; dg = { mode: 'pan', x0: r.x, y0: r.y, tx0: dv.tx, ty0: dv.ty }; }
}
function dishHover(p) {
  const hit = dishHit(p);
  cv.style.cursor = hit === 'resize' ? 'nwse-resize' : hit === 'move' ? 'move' : 'pointer';
  showTip(p, hit === 'resize' ? 'ドラッグで大きさを変える' : hit === 'move' ? 'ドラッグで円を動かす' : 'クリックで円をここへ');
}
function dvZoomAt(x, y, s) {
  const ix = (x - dv.tx) / dv.s, iy = (y - dv.ty) / dv.s;
  s = clamp(s, dvMin(), dvMin() * 8);
  Object.assign(dv, clampDv({ s, tx: x - ix * s, ty: y - iy * s }));
  requestRender();
}
function tweenDish(to) {
  if (!motion()) { dishEdit = { ...to }; updateDishPanel(); requestRender(); return; }
  dishTween = { from: { ...dishEdit }, to, t0: performance.now(), dur: 260 };
  requestRender();
}
function updateDishPanel() {
  if (mode !== 'dish' || !dishEdit) return;
  $('dishN').textContent = '';
}

function enterDish() {
  if (mode !== 'edit' || review || trans || !cur) return;
  mode = 'dish';
  dishEdit = { ...cur.dish }; dishTween = null;
  hover = null; hideTip(); cam = null;
  const target = dvAround(dishEdit);
  const from = { x: view.tx, y: view.ty, z: W * view.s };
  trans = { from, to: rectIn(target, cur.crop), t0: performance.now(), dur: motion() ? 420 : 0, dim: [0.35, 0.5],
    then: () => { Object.assign(dv, target); requestRender(); } };
  syncUI(); updateDishPanel();
  $('dishOk').focus();
  requestRender();
}

async function confirmDish() {
  if (mode !== 'dish' || trans || busy) return;
  const st = cur, d = dishTween ? { ...dishTween.to } : { ...dishEdit };
  const changed = Math.abs(d.cx - st.dish.cx) > 0.5 || Math.abs(d.cy - st.dish.cy) > 0.5 || Math.abs(d.r - st.dish.r) > 0.5;
  if (!changed) { leaveDish(null); return; }
  let j;
  try { j = await countWithBusy(d, 'この範囲で数えています…'); } catch (e) { toast(e.message); return; }
  if (cur !== st || mode !== 'dish') return;
  st.sm.boxes = j.boxes;
  leaveDish({ cx: j.dish[0], cy: j.dish[1], r: j.dish[2] });
}

function leaveDish(newDish) {
  if (mode !== 'dish' || trans) return;
  const st = cur, changed = !!newDish;
  const hadEdits = st.ov.size > 0 || st.user.length > 0;
  if (changed) applyDish(st, newDish);
  const from = rectIn(dv, st.crop);
  mode = 'edit'; dishEdit = null; dishTween = null; dg = null; gest = null;
  hideTip(); cv.style.cursor = '';
  setView(fitS, W / 2, W / 2);
  fx = []; floats = [];
  if (changed && motion()) { st.reveal = { t0: Infinity }; setCount(0, true); }
  syncUI();
  update({ silent: true });
  trans = { from, to: { x: 0, y: 0, z: size }, t0: performance.now(), dur: motion() ? 460 : 0, dim: [0.5, 0.35],
    then: () => { if (st.reveal && !isFinite(st.reveal.t0)) st.reveal.t0 = performance.now(); requestRender(); } };
  if (changed) toast(hadEdits ? 'この範囲で数え直しました（手で直した分はリセットしました）' : 'この範囲で数え直しました');
  requestRender();
}
$('dishBtn').addEventListener('click', enterDish);
$('dishOk').addEventListener('click', confirmDish);
$('dishCancel').addEventListener('click', () => { if (!busy) leaveDish(null); });
$('dishAuto').addEventListener('click', () => { if (mode === 'dish') tweenDish({ ...cur.dishAuto }); });

function syncUI() {
  const dish = mode === 'dish', has = !!cur;
  $('start').hidden = has || !!job;
  $('newBtn').hidden = !has;
  $('dishBtn').hidden = !has || dish || !!review;
  $('zoomBar').hidden = !has || dish;
  for (const id of ['candBtn', 'rvBtn', 'save']) $(id).disabled = !has;
  document.body.classList.toggle('is-busy', busy);
  $('dockMain').hidden = dish || !!review;
  $('dockReview').hidden = !review;
  $('dockDish').hidden = !dish;
  $('dishHelp').textContent = fine.matches
    ? '円の中をドラッグで移動、右下の●で大きさ、円の外をクリックするとそこへ移動します。'
    : '円の中をドラッグで移動、右下の●で大きさ、円の外をタップするとそこへ移動します。';
}

const odo = $('odo');
let shown = null;
function setCount(n, silent) {
  const dash = odo.querySelector('.dash');
  if (dash) dash.remove();
  const str = String(n);
  while (odo.children.length < str.length) {
    const dg2 = document.createElement('span'); dg2.className = 'dg';
    const strip = document.createElement('span'); strip.className = 'strip';
    strip.innerHTML = '<span>0</span><span>1</span><span>2</span><span>3</span><span>4</span><span>5</span><span>6</span><span>7</span><span>8</span><span>9</span>';
    dg2.appendChild(strip); odo.insertBefore(dg2, odo.firstChild);
  }
  while (odo.children.length > str.length) odo.removeChild(odo.firstChild);
  [...str].forEach((ch, i) => odo.children[i].firstChild.style.setProperty('--d', ch));
  if (!silent && shown !== null && n !== shown && motion()) {
    const el = document.createElement('span');
    el.className = 'delta ' + (n > shown ? 'up' : 'down');
    el.textContent = (n > shown ? '+' : '−') + Math.abs(n - shown);
    $('deltaHost').appendChild(el);
    el.addEventListener('animationend', () => el.remove());
  }
  if (shown !== n) $('totalSr').textContent = `大腸菌数 ${n}個`;
  shown = n;
}
function drawHist(t) {
  const bins = new Array(19).fill(0);
  for (const b of cur.ai) { const i = Math.floor((b.s - 0.05) / 0.05 + 1e-9); if (i >= 0) bins[Math.min(18, i)]++; }
  const max = Math.max(...bins, 1);
  let h = '';
  bins.forEach((n, i) => {
    if (!n) return;
    const lo = 0.05 + i * 0.05, bh = Math.max(2, n / max * 42);
    const cls = lo >= settings.conf - 1e-9 ? 'on' : lo >= CAND_MIN - 1e-9 ? 'cand' : 'low';
    h += `<rect class="${cls}" x="${i * 10 + 1}" y="${46 - bh}" width="8" height="${bh}" rx="1.5"/>`;
  });
  const x = (settings.conf - 0.05) / 0.05 * 10;
  h += `<line class="th" x1="${x}" x2="${x}" y1="0" y2="46"/>`;
  $('hist').innerHTML = h;
  $('histSum').innerHTML = `数える <b>${t.ai}</b> ／ 候補 <b>${t.cand}</b>`;
}

function showEmpty() {
  odo.innerHTML = '<span class="dash">—</span>';
  shown = null;
  $('totalSr').textContent = '';
  for (const id of ['nAi', 'nOff', 'nAdd']) $(id).textContent = id === 'nAi' ? '0' : id === 'nOff' ? '−0' : '＋0';
  $('edited').hidden = true; $('undo').disabled = true; $('redo').disabled = true;
  $('candN').hidden = true; $('rvBadge').hidden = true;
  $('hist').innerHTML = ''; $('histSum').textContent = '';
}
function update(o = {}) {
  if (!cur) { showEmpty(); syncUI(); return; }
  const t = tally(cur);
  if (!cur.reveal) setCount(t.total, o.silent);
  $('nAi').textContent = t.ai;
  $('nOff').textContent = '−' + t.off;
  $('nAdd').textContent = '＋' + t.add;
  $('edited').hidden = !t.edited;
  $('undo').disabled = !cur.undo.length;
  $('redo').disabled = !cur.redo.length;
  $('candN').textContent = t.cand; $('candN').hidden = !t.cand;
  $('candBtn').setAttribute('aria-pressed', String(settings.showCand));
  const rn = reviewItems().length;
  $('rvBadge').textContent = rn; $('rvBadge').hidden = !rn;
  $('confV').textContent = settings.conf.toFixed(2);
  drawHist(t);
  if (review) updateReviewUI();
  updateDishPanel();
  requestRender();
}
let toastTimer = 0;
function toast(msg) {
  const el = $('toast');
  el.textContent = msg; el.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove('show'), 2400);
}

function reviewItems() {
  const hi = settings.conf + REVIEW_BAND, row = 180;
  const cy = b => (b.y1 + b.y2) / 2, cx = b => (b.x1 + b.x2) / 2;
  return cur.ai.filter(b => b.s >= CAND_MIN && b.s < hi)
    .sort((a, b) => (Math.floor(cy(a) / row) - Math.floor(cy(b) / row)) || (cx(a) - cx(b)));
}
function startReview() {
  if (mode !== 'edit' || trans) return;
  const items = reviewItems();
  if (!items.length) { toast('見直しが必要な枠はありません'); return; }
  review = { items, i: 0, done: new Set() };
  hover = null; hideTip();
  syncUI();
  focusReview();
  $('rvYes').focus();
}
function focusReview() {
  const b = review.items[review.i];
  const side = Math.max(b.x2 - b.x1, b.y2 - b.y1);
  animateView(clamp(size * 0.2 / side, fitS * 3, fitS * MAXZ), (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2, 420);
  updateReviewUI();
}
function updateReviewUI() {
  const b = review.items[review.i];
  $('rvI').textContent = review.i + 1;
  $('rvN').textContent = review.items.length;
  $('rvFill').style.width = (review.done.size / review.items.length * 100) + '%';
  $('rvScore').textContent = `確信度 ${b.s.toFixed(2)}`;
  const c = counted(b);
  $('rvYes').setAttribute('aria-pressed', String(c));
  $('rvNo').setAttribute('aria-pressed', String(!c));
  $('rvPrev').disabled = review.i === 0;
  $('rvNext').disabled = review.i >= review.items.length - 1;
}
function decide(yes) {
  if (!review) return;
  const b = review.items[review.i];
  if (counted(b) !== yes) {
    const byAi = b.s >= settings.conf;
    const next = yes ? (byAi ? undefined : true) : (byAi ? false : undefined);
    doAct({ k: 'ov', id: b.id, prev: cur.ov.get(b.id), next });
    addFx(yes ? (byAi ? 'on' : 'add') : (byAi ? 'off' : 'gone'), b, C.add);
    addFloat(b, yes ? 1 : -1);
    buzz(10);
  }
  review.done.add(b.id);
  updateReviewUI();
  const at = review.i;
  setTimeout(() => {
    if (!review || review.i !== at) return;
    if (at >= review.items.length - 1) finishReview(); else stepReview(1);
  }, motion() ? 280 : 0);
}
function stepReview(d) {
  const ni = review.i + d;
  if (ni < 0 || ni >= review.items.length) return;
  review.i = ni; focusReview();
}
function finishReview() { const n = review.items.length; endReview(); toast(`見直しが終わりました（${n}件）`); }
function endReview(quiet) {
  review = null;
  syncUI();
  if (!quiet) animateView(fitS, W / 2, W / 2, 420);
  update({ silent: true });
}
$('rvBtn').addEventListener('click', startReview);
$('rvYes').addEventListener('click', () => decide(true));
$('rvNo').addEventListener('click', () => decide(false));
$('rvPrev').addEventListener('click', () => stepReview(-1));
$('rvNext').addEventListener('click', () => stepReview(1));
$('rvExit').addEventListener('click', () => endReview());

$('undo').addEventListener('click', undo);
$('redo').addEventListener('click', redo);
$('candBtn').addEventListener('click', () => {
  settings.showCand = !settings.showCand;
  if (settings.showCand) settings.antsUntil = performance.now() + 1500;
  update({ silent: true });
  toast(settings.showCand ? 'AIが迷った枠を点線で表示しています。タップで数に入ります' : '候補を隠しました');
});
$('conf').addEventListener('input', e => { settings.conf = Number(e.target.value); update(); });
$('reset').addEventListener('click', () => {
  if (!cur || mode !== 'edit') return;
  if (!cur.ov.size && !cur.user.length) { toast('まだ手で直した所はありません'); return; }
  doAct({ k: 'reset', prevOv: [...cur.ov], prevUser: cur.user.slice() });
  toast('AIの結果に戻しました。「元に戻す」で取り消せます');
});

async function renderOutput() {
  try { await Promise.all([document.fonts.load(`700 64px ${FONT}`), document.fonts.load(`400 30px ${FONT}`)]); } catch (_) {}
  const c = document.createElement('canvas'); c.width = W; c.height = W;
  const g = c.getContext('2d');
  g.drawImage(cur.img, 0, 0, W, W);
  const ds = cur.dishSq;
  g.setLineDash([16, 12]); g.lineWidth = 3; g.strokeStyle = 'rgba(255,255,255,.8)';
  g.beginPath(); g.arc(ds.cx, ds.cy, ds.r, 0, Math.PI * 2); g.stroke(); g.setLineDash([]);
  const box = (b, col, dash) => {
    const x = b.x1 - 2, y = b.y1 - 2, w = b.x2 - b.x1 + 4, h = b.y2 - b.y1 + 4;
    g.setLineDash(dash);
    g.lineWidth = 5; g.strokeStyle = 'rgba(10,16,22,.35)'; g.strokeRect(x, y, w, h);
    g.lineWidth = 3; g.strokeStyle = col; g.strokeRect(x, y, w, h);
    g.setLineDash([]);
  };
  for (const b of cur.ai) {
    const k = cat(cur, b);
    if (k === 'ai') box(b, C.ai, []);
    else if (k === 'off') box(b, C.off, [10, 6]);
    else if (k === 'addc') box(b, C.add, []);
  }
  for (const u of cur.user) box(u, C.add, []);
  const t = tally(cur);
  const parts = Object.fromEntries(new Intl.DateTimeFormat('ja-JP', {
    timeZone: 'Asia/Tokyo', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
  }).formatToParts(new Date()).map(p => [p.type, p.value]));
  const lines = [
    { t: `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}`, f: `400 30px ${FONT}`, c: '#44526A', h: 44 },
    { t: `大腸菌　${t.total} 個`, f: `700 64px ${FONT}`, c: '#15212C', h: 82 },
    { t: `AI ${t.ai} − 除外 ${t.off} ＋ 追加 ${t.add}${t.edited ? '（手修正あり）' : ''}`, f: `400 30px ${FONT}`, c: '#15212C', h: 46 },
    { t: `01_XMG_s・conf ${settings.conf.toFixed(2)}・入力 ${cur.imgsz || 1280}`, f: `400 26px ${FONT}`, c: '#56657A', h: 40 },
  ];
  const legend = [['AI検出', C.ai, []], ['追加', C.add, []], ['除外', C.off, [8, 5]]];
  g.textBaseline = 'top';
  let wmax = 0;
  for (const l of lines) { g.font = l.f; wmax = Math.max(wmax, g.measureText(l.t).width); }
  g.font = `400 26px ${FONT}`;
  wmax = Math.max(wmax, legend.reduce((a, [n]) => a + 34 + g.measureText(n).width + 26, 0));
  const px = 28, py = 22;
  const pw = wmax + px * 2, ph = py * 2 + lines.reduce((a, l) => a + l.h, 0) + 40;
  g.fillStyle = 'rgba(255,255,255,.88)';
  g.beginPath(); rrect(g, 24, 24, pw, ph, 16); g.fill();
  let y = 24 + py;
  for (const l of lines) { g.font = l.f; g.fillStyle = l.c; g.fillText(l.t, 24 + px, y); y += l.h; }
  let x = 24 + px;
  g.font = `400 26px ${FONT}`;
  for (const [name, col, dash] of legend) {
    g.setLineDash(dash); g.lineWidth = 4; g.strokeStyle = col; g.strokeRect(x + 2, y + 4, 22, 22); g.setLineDash([]);
    g.fillStyle = '#15212C'; g.fillText(name, x + 34, y + 2);
    x += 34 + g.measureText(name).width + 26;
  }
  const foot = '検出結果は参考値です。点線の円はシャーレとして数えた範囲です。';
  g.font = `400 24px ${FONT}`;
  const fw = g.measureText(foot).width;
  g.fillStyle = 'rgba(255,255,255,.8)'; g.fillRect(24, W - 64, fw + 28, 40);
  g.fillStyle = '#44526A'; g.fillText(foot, 38, W - 56);
  const blob = await new Promise(res => c.toBlob(res, 'image/jpeg', 0.9));
  return { blob, name: `rksi_${parts.year}${parts.month}${parts.day}_${parts.hour}${parts.minute}_${t.total}個.jpg` };
}
let lastFocus = null, outUrl = null, outFile = null;
async function openSheet() {
  if (!cur || mode !== 'edit' || trans || busy) return;
  lastFocus = document.activeElement;
  const out = await renderOutput();
  if (outUrl) URL.revokeObjectURL(outUrl);
  outUrl = URL.createObjectURL(out.blob);
  outFile = new File([out.blob], out.name, { type: 'image/jpeg' });
  $('outImg').src = outUrl;
  $('outName').textContent = out.name;
  const dl = $('outDl');
  dl.hidden = framed;
  if (!framed) { dl.href = outUrl; dl.download = out.name; }
  let canShare = false;
  try { canShare = !framed && !!navigator.canShare && navigator.canShare({ files: [outFile] }); } catch (_) {}
  $('outShare').hidden = !canShare;
  $('outNote').textContent = framed ? 'このページでは保存できません。' : '';
  $('sheet').hidden = false;
  $('sheetClose').focus();
}
$('outShare').addEventListener('click', async () => {
  try { await navigator.share({ files: [outFile], title: '大腸菌数' }); }
  catch (e) { if (e && e.name !== 'AbortError') toast('共有できませんでした。「保存」をお使いください'); }
});
function closeSheet() { $('sheet').hidden = true; if (lastFocus && lastFocus.focus) lastFocus.focus(); }
$('save').addEventListener('click', openSheet);
$('sheetClose').addEventListener('click', closeSheet);
$('sheet').addEventListener('click', e => { if (e.target === $('sheet')) closeSheet(); });

const COACH = [
  { t: '枠をタップすると除外', p: 'AIが間違えて囲んだ枠をタップすると、灰色の点線になって数から外れます。もう一度タップすると戻ります。' },
  { t: '何もない所をタップすると追加', p: 'AIが見落としたコロニーの上をタップすると、青い枠で追加されます。' },
  { t: '2本指で拡大', p: '密集した所は2本指で広げるか、右上の＋で拡大してから直すと確実です。' },
  { t: '長押しで虫めがね', p: '指で隠れて見えないときは長押しします。虫めがねを見ながら指をずらして狙い、離すと決まります。' },
  { t: '円がずれていたら「範囲を直す」', p: 'シャーレは自動で見つけます。円がずれていたら写真の左上の「範囲を直す」を押し、円をドラッグして合わせ、右下の●で大きさを変えます。' },
];
let coachI = 0;
function showCoach(i) {
  coachI = i;
  document.querySelectorAll('#coach .art').forEach((a, k) => { a.hidden = k !== i; });
  $('coachT').textContent = COACH[i].t;
  $('coachP').textContent = COACH[i].p;
  [...$('dots').children].forEach((d, k) => d.classList.toggle('on', k === i));
  $('coachPrev').disabled = i === 0;
  $('coachNext').textContent = i === COACH.length - 1 ? 'はじめる' : '次へ';
}
function openCoach() { lastFocus = document.activeElement; $('coach').hidden = false; showCoach(0); $('coachNext').focus(); }
function closeCoach() { $('coach').hidden = true; if (lastFocus && lastFocus.focus) lastFocus.focus(); }
$('helpBtn').addEventListener('click', openCoach);
$('coachPrev').addEventListener('click', () => showCoach(Math.max(0, coachI - 1)));
$('coachNext').addEventListener('click', () => { if (coachI >= COACH.length - 1) closeCoach(); else showCoach(coachI + 1); });
$('coach').addEventListener('click', e => { if (e.target === $('coach')) closeCoach(); });

document.addEventListener('keydown', e => {
  if (!$('coach').hidden) {
    if (e.key === 'Escape') closeCoach();
    else if (e.key === 'ArrowRight') $('coachNext').click();
    else if (e.key === 'ArrowLeft' && coachI > 0) showCoach(coachI - 1);
    return;
  }
  if (!$('sheet').hidden) { if (e.key === 'Escape') closeSheet(); return; }
  if (e.target.closest && e.target.closest('input')) return;
  if (!cur || busy) return;
  if (mode === 'dish') {
    if (e.key === 'Enter') { e.preventDefault(); confirmDish(); return; }
    if (e.key === 'Escape') { if (!busy) leaveDish(null); return; }
    const step = dishEdit.r * (e.shiftKey ? 0.05 : 0.01);
    const mv = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
    if (mv) { e.preventDefault(); dishEdit.cx += mv[0] * step; dishEdit.cy += mv[1] * step; }
    else if (e.key === '+' || e.key === ';') dishEdit.r *= 1.02;
    else if (e.key === '-') dishEdit.r /= 1.02;
    else return;
    updateDishPanel(); requestRender();
    return;
  }
  const mod = e.ctrlKey || e.metaKey, key = e.key.toLowerCase();
  if (mod && key === 'z' && !e.shiftKey) { e.preventDefault(); undo(); return; }
  if (mod && (key === 'y' || (key === 'z' && e.shiftKey))) { e.preventDefault(); redo(); return; }
  if (review) {
    if (e.key === 'Enter') { e.preventDefault(); decide(true); }
    else if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); decide(false); }
    else if (e.key === 'ArrowRight') stepReview(1);
    else if (e.key === 'ArrowLeft') stepReview(-1);
    else if (e.key === 'Escape') endReview();
    return;
  }
  if (mod) return;
  const c = viewCenter();
  if (e.key === '+' || e.key === ';') animateView(view.s * 1.7, c.cx, c.cy, 260);
  else if (e.key === '-') animateView(view.s / 1.7, c.cx, c.cy, 260);
  else if (e.key === '0') animateView(fitS, W / 2, W / 2);
});

function loadImg(src) {
  return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src; });
}
function makeThumb(img) {
  const t = document.createElement('canvas'); t.width = t.height = 256;
  t.getContext('2d').drawImage(img, 0, 0, 256, 256);
  return t;
}

function playIntro(st) {
  const from = rectIn(dvWhole(), st.crop);
  st.reveal = { t0: Infinity };
  setCount(0, true);
  trans = { from, to: from, t0: performance.now(), dur: 750, draw: true, dim: [0, 0.35], label: 'シャーレを自動で見つけました',
    then: () => {
      trans = { from, to: { x: 0, y: 0, z: size }, t0: performance.now(), dur: 650, dim: [0.35, 0.35],
        then: () => { if (st.reveal && !isFinite(st.reveal.t0)) st.reveal.t0 = performance.now(); requestRender(); } };
    } };
}

async function prepPhoto(file) {
  const url = URL.createObjectURL(file);
  try {
    const im = await loadImg(url);
    const k = Math.min(1, MAX_SIDE / Math.max(im.naturalWidth, im.naturalHeight));
    const w = Math.round(im.naturalWidth * k), h = Math.round(im.naturalHeight * k);
    const c = document.createElement('canvas'); c.width = w; c.height = h;
    const g = c.getContext('2d');
    g.imageSmoothingQuality = 'high';
    g.drawImage(im, 0, 0, w, h);
    const blob = await new Promise(res => c.toBlob(res, 'image/jpeg', 0.9));
    if (!blob) throw new Error('encode');
    return { canvas: c, blob, w, h };
  } finally { URL.revokeObjectURL(url); }
}
async function callApi(blob, dish) {
  const fd = new FormData();
  fd.append('image', blob, 'photo.jpg');
  if (dish) { fd.append('cx', dish.cx.toFixed(1)); fd.append('cy', dish.cy.toFixed(1)); fd.append('r', dish.r.toFixed(1)); }
  const ctrl = new AbortController(), timer = setTimeout(() => ctrl.abort(), 60000);
  let r;
  try { r = await fetch(`${API}/api/count`, { method: 'POST', body: fd, signal: ctrl.signal }); }
  catch (e) {
    throw new Error(e.name === 'AbortError' ? '時間がかかりすぎました。電波の良い所でもう一度お試しください。'
      : '通信できませんでした。電波の状態を確かめて、もう一度お試しください。');
  } finally { clearTimeout(timer); }
  if (!r.ok) {
    let msg = '';
    try { msg = (await r.json()).detail; } catch (_) {}
    throw new Error(typeof msg === 'string' && msg ? msg : `数えられませんでした（${r.status}）。もう一度お試しください。`);
  }
  return r.json();
}
function setBusy(on, text) {
  busy = on;
  $('busy').hidden = !on;
  if (text) $('busyText').textContent = text;
  syncUI();
  requestRender();
}
async function countWithBusy(dish, text) {
  setBusy(true, text);
  const slow = setTimeout(() => { $('busyText').textContent = 'AIを起動しています。最初の1回は少し時間がかかります…'; }, 3500);
  try { return await callApi(job.blob, dish); }
  finally { clearTimeout(slow); setBusy(false); }
}
function showErr(msg) { $('errText').textContent = msg; $('err').hidden = false; }
function hideErr() { $('err').hidden = true; }

async function startWith(file) {
  if (busy) return;
  resetAll();
  try { job = await prepPhoto(file); }
  catch (_) { job = null; syncUI(); showErr('写真を読み込めませんでした。別の写真を選んでください。'); return; }
  syncUI(); requestRender();
  await runFirstCount();
}
async function runFirstCount() {
  hideErr();
  let j;
  try { j = await countWithBusy(null, 'シャーレを探しています…'); }
  catch (e) { showErr(e.message); return; }
  const st = { sm: { w: j.w, h: j.h, boxes: j.boxes }, photo: job.canvas, img: null, thumb: null, reveal: null,
    dishAuto: { cx: j.dish[0], cy: j.dish[1], r: j.dish[2] } };
  st.imgsz = j.imgsz;
  applyDish(st, st.dishAuto);
  cur = st; mode = 'edit'; hover = null; fx = []; floats = []; cam = null; gest = null;
  setView(fitS, W / 2, W / 2);
  syncUI();
  if (!j.found) {
    update({ silent: true });
    enterDish();
    toast('シャーレを見つけられませんでした。円を合わせて「この範囲で数える」を押してください');
    return;
  }
  if (motion() && size) playIntro(st);
  update({ silent: true });
}
function resetAll() {
  review = null;
  cur = null; job = null; mode = 'edit'; trans = null; dishEdit = null; dishTween = null; dg = null; gest = null;
  hover = null; fx = []; floats = []; pulses = []; cam = null;
  hideTip(); endAim(); hideErr();
  update();
  requestRender();
}
let newArmed = 0;
$('newBtn').addEventListener('click', () => {
  if (busy) return;
  const edited = cur && (cur.ov.size || cur.user.length);
  if (edited && Date.now() - newArmed > 3000) { newArmed = Date.now(); toast('もう一度押すと、今の結果を消して新しい写真にします'); return; }
  newArmed = 0;
  resetAll();
});
for (const id of ['camIn', 'fileIn']) {
  $(id).addEventListener('change', e => {
    const f = e.target.files && e.target.files[0];
    e.target.value = '';
    if (f) startWith(f);
  });
}
$('sampleBtn').addEventListener('click', async () => {
  try { const r = await fetch('samples/sample.jpg'); if (!r.ok) throw new Error(); startWith(await r.blob()); }
  catch (_) { showErr('サンプルの写真を読み込めませんでした。'); }
});
$('retryBtn').addEventListener('click', () => { if (job) runFirstCount(); else resetAll(); });
$('errBack').addEventListener('click', resetAll);
stage.addEventListener('dragover', e => { if (!cur && !busy) e.preventDefault(); });
stage.addEventListener('drop', e => {
  if (cur || busy) return;
  e.preventDefault();
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) startWith(f);
});
function setHint() {
  $('hintLine').innerHTML = fine.matches
    ? '<span><b>クリック</b>除外・追加</span><span><b>ホイール</b>拡大</span><span><b>ドラッグ</b>移動</span>'
    : '<span><b>タップ</b>除外・追加</span><span><b>長押し</b>虫めがね</span><span><b>2本指</b>拡大</span>';
  syncUI();
}
setHint();
if (fine.addEventListener) fine.addEventListener('change', setHint);
measure();
new ResizeObserver(measure).observe(stage);
update();
})();
