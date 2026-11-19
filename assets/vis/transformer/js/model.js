/*
 * Llama 스타일 Decoder-only Transformer — 순수 JS 구현
 *   input → Embedding → [RMSNorm → Q/K/V → RoPE → Causal Attention(GQA) → o_proj → +residual
 *                        → RMSNorm → gate/up → SiLU(gate)⊙up → down_proj → +residual] × L
 *         → RMSNorm → LM head → Softmax/CrossEntropy
 *
 * 모든 활성값은 batch=1 기준 2D 텐서 [T, C] 로 저장한다.
 * 각 연산은 forward 시 backward 클로저를 tape 에 기록하고, backward() 가 이를 역순으로 실행한다.
 * Linear 가중치는 PyTorch 규약을 따른다: W ∈ [out, in], y = x Wᵀ
 */
(function (global) {
  'use strict';

  const Arr = Float64Array;

  // ---------------------------------------------------------------- 난수
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function makeRandn(seed) {
    const rand = mulberry32(seed);
    let spare = null;
    return function () {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let u = 0;
      while (u === 0) u = rand();
      const v = rand();
      const m = Math.sqrt(-2 * Math.log(u));
      spare = m * Math.sin(2 * Math.PI * v);
      return m * Math.cos(2 * Math.PI * v);
    };
  }

  // ---------------------------------------------------------------- Tensor
  class Tensor {
    constructor(rows, cols, data) {
      this.rows = rows;
      this.cols = cols;
      this.data = data || new Arr(rows * cols);
      this.grad = null;
    }
    ensureGrad() {
      if (!this.grad) this.grad = new Arr(this.data.length);
      return this.grad;
    }
  }

  // ---------------------------------------------------------------- 연산 (forward + backward)

  // y[t] = W[ids[t]]
  function embedding(tape, ids, W) {
    const T = ids.length, D = W.cols;
    const Y = new Tensor(T, D);
    for (let t = 0; t < T; t++) {
      const src = ids[t] * D, dst = t * D;
      for (let j = 0; j < D; j++) Y.data[dst + j] = W.data[src + j];
    }
    tape.push(() => {
      if (!Y.grad) return;
      const dW = W.ensureGrad();
      for (let t = 0; t < T; t++) {
        const src = ids[t] * D, dst = t * D;
        for (let j = 0; j < D; j++) dW[src + j] += Y.grad[dst + j];
      }
    });
    return Y;
  }

  // y = x Wᵀ,  X [T, in], W [out, in] → Y [T, out]
  function linear(tape, X, W) {
    const T = X.rows, I = X.cols, O = W.rows;
    if (W.cols !== I) throw new Error(`linear: shape mismatch [${T},${I}] x [${O},${W.cols}]ᵀ`);
    const x = X.data, w = W.data;
    const Y = new Tensor(T, O), y = Y.data;
    for (let t = 0; t < T; t++) {
      const xo = t * I;
      for (let o = 0; o < O; o++) {
        const wo = o * I;
        let s = 0;
        for (let i = 0; i < I; i++) s += x[xo + i] * w[wo + i];
        y[t * O + o] = s;
      }
    }
    tape.push(() => {
      if (!Y.grad) return;
      const dy = Y.grad, dx = X.ensureGrad(), dw = W.ensureGrad();
      for (let t = 0; t < T; t++) {
        const xo = t * I;
        for (let o = 0; o < O; o++) {
          const g = dy[t * O + o];
          if (g === 0) continue;
          const wo = o * I;
          for (let i = 0; i < I; i++) {
            dx[xo + i] += g * w[wo + i];
            dw[wo + i] += g * x[xo + i];
          }
        }
      }
    });
    return Y;
  }

  // y = x / sqrt(mean(x²) + eps) * γ
  function rmsnorm(tape, X, G, eps) {
    const T = X.rows, D = X.cols, x = X.data, g = G.data;
    const Y = new Tensor(T, D);
    const invRms = new Arr(T), xhat = new Arr(T * D);
    for (let t = 0; t < T; t++) {
      const o = t * D;
      let ss = 0;
      for (let j = 0; j < D; j++) ss += x[o + j] * x[o + j];
      const r = 1 / Math.sqrt(ss / D + eps);
      invRms[t] = r;
      for (let j = 0; j < D; j++) {
        const xh = x[o + j] * r;
        xhat[o + j] = xh;
        Y.data[o + j] = xh * g[j];
      }
    }
    tape.push(() => {
      if (!Y.grad) return;
      const dy = Y.grad, dx = X.ensureGrad(), dg = G.ensureGrad();
      for (let t = 0; t < T; t++) {
        const o = t * D;
        let dot = 0;
        for (let j = 0; j < D; j++) {
          dg[j] += dy[o + j] * xhat[o + j];
          dot += dy[o + j] * g[j] * xhat[o + j];
        }
        dot /= D;
        for (let j = 0; j < D; j++) {
          dx[o + j] += (dy[o + j] * g[j] - xhat[o + j] * dot) * invRms[t];
        }
      }
    });
    return Y;
  }

  // Rotary embedding (Llama rotate_half 방식). X [T, nHeads*hd]
  function rope(tape, X, nHeads, hd, theta) {
    const T = X.rows, C = X.cols, half = hd / 2, x = X.data;
    const cos = new Arr(T * half), sin = new Arr(T * half);
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < half; i++) {
        const ang = t * Math.pow(theta, -2 * i / hd);
        cos[t * half + i] = Math.cos(ang);
        sin[t * half + i] = Math.sin(ang);
      }
    }
    const Y = new Tensor(T, C), y = Y.data;
    for (let t = 0; t < T; t++) {
      for (let h = 0; h < nHeads; h++) {
        const b = t * C + h * hd;
        for (let i = 0; i < half; i++) {
          const c = cos[t * half + i], s = sin[t * half + i];
          const x1 = x[b + i], x2 = x[b + i + half];
          y[b + i] = x1 * c - x2 * s;
          y[b + i + half] = x2 * c + x1 * s;
        }
      }
    }
    tape.push(() => {
      if (!Y.grad) return;
      const dy = Y.grad, dx = X.ensureGrad();
      for (let t = 0; t < T; t++) {
        for (let h = 0; h < nHeads; h++) {
          const b = t * C + h * hd;
          for (let i = 0; i < half; i++) {
            const c = cos[t * half + i], s = sin[t * half + i];
            const d1 = dy[b + i], d2 = dy[b + i + half];
            dx[b + i] += d1 * c + d2 * s;
            dx[b + i + half] += -d1 * s + d2 * c;
          }
        }
      }
    });
    return Y;
  }

  // Causal scaled-dot-product attention with GQA.
  // Q [T, H*hd], K,V [T, KV*hd] → O [T, H*hd],  probs [H, T, T]
  function attention(tape, Q, K, V, H, KV, hd) {
    const T = Q.rows, QC = H * hd, KC = KV * hd, grp = H / KV, scale = 1 / Math.sqrt(hd);
    const q = Q.data, k = K.data, v = V.data;
    const O = new Tensor(T, QC), o = O.data;
    const P = new Arr(H * T * T);
    for (let h = 0; h < H; h++) {
      const qo = h * hd, ko = Math.floor(h / grp) * hd;
      for (let t = 0; t < T; t++) {
        const row = (h * T + t) * T;
        let max = -Infinity;
        for (let j = 0; j <= t; j++) {
          let s = 0;
          for (let d = 0; d < hd; d++) s += q[t * QC + qo + d] * k[j * KC + ko + d];
          s *= scale;
          P[row + j] = s;
          if (s > max) max = s;
        }
        let sum = 0;
        for (let j = 0; j <= t; j++) { const e = Math.exp(P[row + j] - max); P[row + j] = e; sum += e; }
        for (let j = 0; j <= t; j++) P[row + j] /= sum;
        for (let d = 0; d < hd; d++) {
          let acc = 0;
          for (let j = 0; j <= t; j++) acc += P[row + j] * v[j * KC + ko + d];
          o[t * QC + qo + d] = acc;
        }
      }
    }
    tape.push(() => {
      if (!O.grad) return;
      const dO = O.grad, dQ = Q.ensureGrad(), dK = K.ensureGrad(), dV = V.ensureGrad();
      const dP = new Arr(T);
      for (let h = 0; h < H; h++) {
        const qo = h * hd, ko = Math.floor(h / grp) * hd;
        for (let t = 0; t < T; t++) {
          const row = (h * T + t) * T;
          let rs = 0;
          for (let j = 0; j <= t; j++) {
            let s = 0;
            for (let d = 0; d < hd; d++) {
              const g = dO[t * QC + qo + d];
              s += g * v[j * KC + ko + d];
              dV[j * KC + ko + d] += P[row + j] * g;
            }
            dP[j] = s;
            rs += s * P[row + j];
          }
          for (let j = 0; j <= t; j++) {
            const dS = P[row + j] * (dP[j] - rs) * scale;
            if (dS === 0) continue;
            for (let d = 0; d < hd; d++) {
              dQ[t * QC + qo + d] += dS * k[j * KC + ko + d];
              dK[j * KC + ko + d] += dS * q[t * QC + qo + d];
            }
          }
        }
      }
    });
    return { out: O, probs: P };
  }

  // SiLU(x) = x·σ(x)
  function silu(tape, X) {
    const n = X.data.length, x = X.data;
    const Y = new Tensor(X.rows, X.cols), sig = new Arr(n);
    for (let i = 0; i < n; i++) {
      const s = 1 / (1 + Math.exp(-x[i]));
      sig[i] = s;
      Y.data[i] = x[i] * s;
    }
    tape.push(() => {
      if (!Y.grad) return;
      const dx = X.ensureGrad();
      for (let i = 0; i < n; i++) dx[i] += Y.grad[i] * sig[i] * (1 + x[i] * (1 - sig[i]));
    });
    return Y;
  }

  function mul(tape, A, B) {
    const n = A.data.length, Y = new Tensor(A.rows, A.cols);
    for (let i = 0; i < n; i++) Y.data[i] = A.data[i] * B.data[i];
    tape.push(() => {
      if (!Y.grad) return;
      const dA = A.ensureGrad(), dB = B.ensureGrad();
      for (let i = 0; i < n; i++) {
        dA[i] += Y.grad[i] * B.data[i];
        dB[i] += Y.grad[i] * A.data[i];
      }
    });
    return Y;
  }

  function add(tape, A, B) {
    const n = A.data.length, Y = new Tensor(A.rows, A.cols);
    for (let i = 0; i < n; i++) Y.data[i] = A.data[i] + B.data[i];
    tape.push(() => {
      if (!Y.grad) return;
      const dA = A.ensureGrad(), dB = B.ensureGrad();
      for (let i = 0; i < n; i++) { dA[i] += Y.grad[i]; dB[i] += Y.grad[i]; }
    });
    return Y;
  }

  // 평균 cross-entropy. target < 0 인 위치는 무시.
  function crossEntropy(tape, logits, targets) {
    const T = logits.rows, V = logits.cols, x = logits.data;
    const probs = new Arr(T * V);
    let loss = 0, n = 0;
    for (let t = 0; t < T; t++) {
      const o = t * V;
      let max = -Infinity;
      for (let v = 0; v < V; v++) if (x[o + v] > max) max = x[o + v];
      let sum = 0;
      for (let v = 0; v < V; v++) { const e = Math.exp(x[o + v] - max); probs[o + v] = e; sum += e; }
      for (let v = 0; v < V; v++) probs[o + v] /= sum;
      const tgt = targets[t];
      if (tgt >= 0) { loss -= x[o + tgt] - max - Math.log(sum); n++; }
    }
    const L = new Tensor(1, 1);
    L.data[0] = n ? loss / n : 0;
    tape.push(() => {
      if (!L.grad || n === 0) return;
      const g = L.grad[0] / n, dx = logits.ensureGrad();
      for (let t = 0; t < T; t++) {
        const tgt = targets[t];
        if (!(tgt >= 0)) continue; // forward 의 조건과 정확히 동일해야 한다 (undefined 포함)
        const o = t * V;
        for (let v = 0; v < V; v++) dx[o + v] += g * (probs[o + v] - (v === tgt ? 1 : 0));
      }
    });
    return { loss: L, probs, count: n };
  }

  // ---------------------------------------------------------------- 설정 / 파라미터
  const INT_KEYS = ['vocab_size', 'hidden_dim', 'num_layers', 'num_heads', 'num_kv_heads', 'head_dim', 'intermediate_size'];

  function validateConfig(c) {
    const errs = [];
    for (const k of INT_KEYS) {
      if (!Number.isInteger(c[k]) || c[k] < 1) errs.push(`${k} 는 1 이상의 정수여야 합니다.`);
    }
    if (Number.isInteger(c.head_dim) && c.head_dim % 2 !== 0) errs.push('RoPE 를 적용하려면 head_dim 이 짝수여야 합니다.');
    if (Number.isInteger(c.num_heads) && Number.isInteger(c.num_kv_heads) && c.num_kv_heads >= 1 && c.num_heads % c.num_kv_heads !== 0) {
      errs.push('GQA: num_heads 는 num_kv_heads 로 나누어 떨어져야 합니다.');
    }
    if (!(c.rope_theta > 0)) errs.push('rope_theta 는 양수여야 합니다.');
    if (!(c.rms_eps > 0)) errs.push('rms_norm_eps 는 양수여야 합니다.');
    return errs;
  }

  function countParams(c) {
    const V = c.vocab_size, D = c.hidden_dim, L = c.num_layers, H = c.num_heads,
      KV = c.num_kv_heads, hd = c.head_dim, I = c.intermediate_size;
    const layer = {
      input_layernorm: D,
      q_proj: D * H * hd,
      k_proj: D * KV * hd,
      v_proj: D * KV * hd,
      o_proj: H * hd * D,
      post_attention_layernorm: D,
      gate_proj: D * I,
      up_proj: D * I,
      down_proj: I * D,
    };
    const perLayer = Object.values(layer).reduce((a, b) => a + b, 0);
    const embed = V * D;
    const lm_head = c.tie_word_embeddings ? 0 : V * D;
    const final_norm = D;
    const groups = {
      embed,
      attn: (layer.q_proj + layer.k_proj + layer.v_proj + layer.o_proj) * L,
      mlp: (layer.gate_proj + layer.up_proj + layer.down_proj) * L,
      norm: 2 * D * L + final_norm,
      lm_head,
    };
    return { total: embed + perLayer * L + final_norm + lm_head, layer, perLayer, groups };
  }

  function initParams(c, seed) {
    const randn = makeRandn(seed);
    const std = c.init_std;
    const W = (r, k) => {
      const t = new Tensor(r, k);
      for (let i = 0; i < t.data.length; i++) t.data[i] = randn() * std;
      return t;
    };
    const ones = (n) => { const t = new Tensor(1, n); t.data.fill(1); return t; };
    const D = c.hidden_dim, H = c.num_heads, KV = c.num_kv_heads, hd = c.head_dim, I = c.intermediate_size;
    const embed = W(c.vocab_size, D);
    const layers = [];
    for (let l = 0; l < c.num_layers; l++) {
      layers.push({
        attn_norm: ones(D),
        q_proj: W(H * hd, D),
        k_proj: W(KV * hd, D),
        v_proj: W(KV * hd, D),
        o_proj: W(D, H * hd),
        mlp_norm: ones(D),
        gate_proj: W(I, D),
        up_proj: W(I, D),
        down_proj: W(D, I),
      });
    }
    return {
      embed,
      layers,
      final_norm: ones(D),
      lm_head: c.tie_word_embeddings ? embed : W(c.vocab_size, D),
      tied: !!c.tie_word_embeddings,
    };
  }

  // HuggingFace 이름 규칙으로 (이름, 텐서) 목록. tied 인 경우 lm_head 는 중복이므로 제외.
  function paramList(p) {
    const out = [['model.embed_tokens.weight', p.embed]];
    p.layers.forEach((L, l) => {
      const pre = `model.layers.${l}.`;
      out.push(
        [pre + 'input_layernorm.weight', L.attn_norm],
        [pre + 'self_attn.q_proj.weight', L.q_proj],
        [pre + 'self_attn.k_proj.weight', L.k_proj],
        [pre + 'self_attn.v_proj.weight', L.v_proj],
        [pre + 'self_attn.o_proj.weight', L.o_proj],
        [pre + 'post_attention_layernorm.weight', L.mlp_norm],
        [pre + 'mlp.gate_proj.weight', L.gate_proj],
        [pre + 'mlp.up_proj.weight', L.up_proj],
        [pre + 'mlp.down_proj.weight', L.down_proj],
      );
    });
    out.push(['model.norm.weight', p.final_norm]);
    if (!p.tied) out.push(['lm_head.weight', p.lm_head]);
    return out;
  }

  // ---------------------------------------------------------------- 모델 forward / backward
  function forward(c, p, ids, targets) {
    const tape = [];
    const acts = {};
    const attnProbs = [];
    const H = c.num_heads, KV = c.num_kv_heads, hd = c.head_dim;

    let x = embedding(tape, ids, p.embed);
    acts.embed = x;
    for (let l = 0; l < c.num_layers; l++) {
      const w = p.layers[l], pre = `L${l}.`;
      const h1 = rmsnorm(tape, x, w.attn_norm, c.rms_eps); acts[pre + 'attn_norm'] = h1;
      const q = linear(tape, h1, w.q_proj); acts[pre + 'q_proj'] = q;
      const k = linear(tape, h1, w.k_proj); acts[pre + 'k_proj'] = k;
      const v = linear(tape, h1, w.v_proj); acts[pre + 'v_proj'] = v;
      const qr = rope(tape, q, H, hd, c.rope_theta); acts[pre + 'rope_q'] = qr;
      const kr = rope(tape, k, KV, hd, c.rope_theta); acts[pre + 'rope_k'] = kr;
      const at = attention(tape, qr, kr, v, H, KV, hd);
      acts[pre + 'attn'] = at.out;
      attnProbs.push(at.probs);
      const o = linear(tape, at.out, w.o_proj); acts[pre + 'o_proj'] = o;
      x = add(tape, x, o); acts[pre + 'resid1'] = x;
      const h2 = rmsnorm(tape, x, w.mlp_norm, c.rms_eps); acts[pre + 'mlp_norm'] = h2;
      const g = linear(tape, h2, w.gate_proj); acts[pre + 'gate_proj'] = g;
      const u = linear(tape, h2, w.up_proj); acts[pre + 'up_proj'] = u;
      const a = silu(tape, g); acts[pre + 'act'] = a;
      const m = mul(tape, a, u); acts[pre + 'mul'] = m;
      const dn = linear(tape, m, w.down_proj); acts[pre + 'down_proj'] = dn;
      x = add(tape, x, dn); acts[pre + 'resid2'] = x;
    }
    const xf = rmsnorm(tape, x, p.final_norm, c.rms_eps); acts.final_norm = xf;
    const logits = linear(tape, xf, p.lm_head); acts.lm_head = logits;
    const ce = crossEntropy(tape, logits, targets);
    acts.loss = ce.loss;
    return { tape, acts, attnProbs, probs: ce.probs, loss: ce.loss.data[0], lossTensor: ce.loss, count: ce.count, backwardDone: false };
  }

  function zeroGrads(p) {
    for (const [, t] of paramList(p)) t.grad = null;
  }

  function backward(fw, p) {
    if (fw.backwardDone) throw new Error('backward 는 forward 결과마다 한 번만 실행할 수 있습니다.');
    zeroGrads(p);
    fw.lossTensor.ensureGrad()[0] = 1;
    for (let i = fw.tape.length - 1; i >= 0; i--) fw.tape[i]();
    fw.backwardDone = true;
  }

  // ---------------------------------------------------------------- 옵티마이저
  class Adam {
    constructor(lr) {
      this.lr = lr; this.b1 = 0.9; this.b2 = 0.999; this.eps = 1e-8;
      this.t = 0; this.state = new Map();
    }
    step(list) {
      this.t++;
      const { b1, b2, eps, lr } = this;
      const bc1 = 1 - Math.pow(b1, this.t), bc2 = 1 - Math.pow(b2, this.t);
      for (const [, p] of list) {
        if (!p.grad) continue;
        let s = this.state.get(p);
        if (!s) { s = { m: new Arr(p.data.length), v: new Arr(p.data.length) }; this.state.set(p, s); }
        for (let i = 0; i < p.data.length; i++) {
          const g = p.grad[i];
          s.m[i] = b1 * s.m[i] + (1 - b1) * g;
          s.v[i] = b2 * s.v[i] + (1 - b2) * g * g;
          p.data[i] -= lr * (s.m[i] / bc1) / (Math.sqrt(s.v[i] / bc2) + eps);
        }
      }
    }
  }

  // ---------------------------------------------------------------- 토크나이저 (문자 단위 toy)
  // 처음 등장한 순서대로 문자에 id 를 부여. vocab_size 를 넘으면 mod 로 겹친다.
  class CharTokenizer {
    constructor(V) { this.V = V; this.map = new Map(); this.chars = []; }
    encode(text) {
      return Array.from(text).map((ch) => {
        if (!this.map.has(ch)) { this.map.set(ch, this.chars.length); this.chars.push(ch); }
        return this.map.get(ch) % this.V;
      });
    }
    decode(id) {
      const hits = [];
      for (let i = id; i < this.chars.length; i += this.V) hits.push(this.chars[i]);
      return hits;
    }
  }

  function tensorStats(a) {
    const n = a.length;
    if (!n) return { mean: 0, std: 0, min: 0, max: 0, norm: 0 };
    let s = 0, ss = 0, min = Infinity, max = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = a[i];
      s += v; ss += v * v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const mean = s / n;
    return { mean, std: Math.sqrt(Math.max(0, ss / n - mean * mean)), min, max, norm: Math.sqrt(ss) };
  }

  const api = {
    Tensor, ops: { embedding, linear, rmsnorm, rope, attention, silu, mul, add, crossEntropy },
    validateConfig, countParams, initParams, paramList, forward, backward, zeroGrads,
    Adam, CharTokenizer, tensorStats,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else global.TModel = api;
})(typeof window !== 'undefined' ? window : globalThis);
