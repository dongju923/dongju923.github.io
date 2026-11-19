/* 두 시각화 페이지(Transformer / LLaVA)가 공유하는 UI 유틸리티 */
(function (global) {
  'use strict';

  const $ = (s, root) => (root || document).querySelector(s);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (n) => Number(n).toLocaleString('en-US');
  const vis = (ch) => (ch === ' ' ? '␣' : ch === '\n' ? '⏎' : ch === '\t' ? '⇥' : ch);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const frame = () => new Promise((r) => requestAnimationFrame(() => r()));

  function short(n) {
    if (n >= 1e12) return (n / 1e12).toFixed(2) + 'T';
    if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return String(n);
  }
  function num(x) {
    if (!Number.isFinite(x)) return String(x);
    if (x === 0) return '0';
    const a = Math.abs(x);
    return a >= 1e-3 && a < 1e4 ? x.toPrecision(4) : x.toExponential(2);
  }
  function bytes(n) {
    const u = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return n.toFixed(i ? 2 : 0) + ' ' + u[i];
  }

  function hmColors() {
    const cs = getComputedStyle(document.documentElement);
    const p = (n) => cs.getPropertyValue(n).split(',').map((x) => +x.trim());
    return { zero: p('--hm-zero'), neg: p('--hm-neg'), pos: p('--hm-pos'), seq: p('--hm-seq') };
  }

  function tooltipEl() {
    let t = $('#tooltip');
    if (!t) {
      t = document.createElement('div');
      t.id = 'tooltip';
      t.hidden = true;
      document.body.appendChild(t);
    }
    return t;
  }

  /**
   * 행렬 히트맵. data 는 길이 rows*cols 의 배열.
   *   mode  : 'div' = 0 중심 발산형(파랑↔빨강), 'seq' = [0, max] 순차형
   *   masked(r, c) 가 true 인 칸은 빗금 배경으로 비워 둔다 (causal mask 등)
   *   onCell(r, c) 를 주면 클릭 가능해진다
   */
  function heatmap({ data, rows, cols, rowLabels, colLabels, colGroup, mode = 'div', caption = '', masked, onCell, maxHeight = 360 }) {
    const MAXR = 256, MAXC = 1024;
    const R = Math.min(rows, MAXR), C = Math.min(cols, MAXC);
    let maxAbs = 0;
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      const v = Math.abs(data[r * cols + c]);
      if (v > maxAbs) maxAbs = v;
    }
    if (mode === 'seq') maxAbs = Math.max(maxAbs, 1e-12);
    const rowH = Math.max(2, Math.min(16, Math.floor(maxHeight / R)));

    const wrap = document.createElement('div');
    wrap.className = 'hm';
    const trunc = rows > R || cols > C ? ` · 앞 ${R}×${C} 만 표시` : '';
    wrap.innerHTML = `<div class="hm-cap"><span>${esc(caption)}</span><span class="mono">${rows} × ${cols}${trunc}</span></div>`;

    const body = document.createElement('div');
    body.className = 'hm-body';
    if (rowLabels && rowH >= 10) {
      const lab = document.createElement('div');
      lab.className = 'hm-labels';
      lab.innerHTML = rowLabels.slice(0, R).map((s) => `<div style="height:${rowH}px">${esc(s)}</div>`).join('');
      body.appendChild(lab);
    }
    const cw = document.createElement('div');
    cw.className = 'hm-canvas';
    const cv = document.createElement('canvas');
    cv.width = C; cv.height = R;
    cv.style.height = R * rowH + 'px';
    if (masked) cv.classList.add('masked-bg');
    if (onCell) cv.style.cursor = 'crosshair';
    const ctx = cv.getContext('2d');
    const img = ctx.createImageData(C, R);
    const col = hmColors();
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      const o = (r * C + c) * 4;
      if (masked && masked(r, c)) { img.data[o + 3] = 0; continue; }
      const v = data[r * cols + c];
      const t = maxAbs ? Math.min(1, Math.abs(v) / maxAbs) : 0;
      const tgt = mode === 'seq' ? col.seq : v < 0 ? col.neg : col.pos;
      img.data[o] = col.zero[0] + (tgt[0] - col.zero[0]) * t;
      img.data[o + 1] = col.zero[1] + (tgt[1] - col.zero[1]) * t;
      img.data[o + 2] = col.zero[2] + (tgt[2] - col.zero[2]) * t;
      img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    cw.appendChild(cv);
    if (colGroup && colGroup < C && C / colGroup <= 64) {
      for (let k = colGroup; k < C; k += colGroup) {
        const s = document.createElement('div');
        s.className = 'hm-sep';
        s.style.left = (100 * k / C) + '%';
        cw.appendChild(s);
      }
    }
    const cellAt = (e) => {
      const rect = cv.getBoundingClientRect();
      return [
        Math.min(R - 1, Math.max(0, Math.floor((e.clientY - rect.top) / rect.height * R))),
        Math.min(C - 1, Math.max(0, Math.floor((e.clientX - rect.left) / rect.width * C))),
      ];
    };
    cv.addEventListener('mousemove', (e) => {
      const [r, c] = cellAt(e);
      const tt = tooltipEl();
      const rl = rowLabels ? rowLabels[r] : r;
      const cl = colLabels ? colLabels[c] : c;
      const isMask = masked && masked(r, c);
      tt.textContent = `row ${rl}\ncol ${cl}${colGroup ? `  (head ${Math.floor(c / colGroup)}, dim ${c % colGroup})` : ''}\n${isMask ? 'masked' : num(data[r * cols + c])}`;
      tt.hidden = false;
      tt.style.left = e.clientX + 14 + 'px';
      tt.style.top = e.clientY + 12 + 'px';
    });
    cv.addEventListener('mouseleave', () => { tooltipEl().hidden = true; });
    if (onCell) cv.addEventListener('click', (e) => onCell(...cellAt(e)));
    body.appendChild(cw);
    wrap.appendChild(body);

    const z = `rgb(${col.zero.join(',')})`;
    const scale = document.createElement('div');
    if (mode === 'seq') {
      scale.innerHTML = `<div class="hm-scale" style="background:linear-gradient(90deg,${z},rgb(${col.seq.join(',')}))"></div>
        <div class="hm-scale-lbl"><span>0</span><span>${num(maxAbs)}</span></div>`;
    } else {
      scale.innerHTML = `<div class="hm-scale" style="background:linear-gradient(90deg,rgb(${col.neg.join(',')}),${z},rgb(${col.pos.join(',')}))"></div>
        <div class="hm-scale-lbl"><span>−${num(maxAbs)}</span><span>0</span><span>+${num(maxAbs)}</span></div>`;
    }
    wrap.appendChild(scale);
    return wrap;
  }

  function statLine(arr, stats) {
    const s = stats(arr);
    return `<div class="statline"><span>mean ${num(s.mean)}</span><span>std ${num(s.std)}</span><span>min ${num(s.min)}</span><span>max ${num(s.max)}</span><span>‖·‖₂ ${num(s.norm)}</span></div>`;
  }

  global.TUI = { $, esc, fmt, short, num, bytes, vis, sleep, frame, heatmap, statLine, hmColors, tooltipEl };
})(window);
