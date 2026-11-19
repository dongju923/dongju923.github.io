/*
 * LLaVA (Large Language and Vision Assistant) — 순수 JS 구현
 *
 *   image ──▶ CLIP ViT vision tower ──▶ feature select(-2, CLS 제거) ──▶ projector(mlp2x_gelu)
 *                                                                          │
 *   text  ──▶ embed_tokens ─────────────────────────────────────────┐      ▼
 *                                                                   └─▶ [텍스트 | 이미지 | 텍스트] 임베딩 결합
 *                                                                          │
 *                                                            Llama decoder ▼ ──▶ lm_head ──▶ CE loss (answer 토큰만)
 *
 * model.js 의 Tensor / 연산(rmsnorm, rope, causal attention, silu, …)을 그대로 재사용하고,
 * ViT 에 필요한 연산(patchify, LayerNorm, GELU, 양방향 attention, concat/slice)을 여기서 추가한다.
 * 모든 연산은 forward 시 backward 클로저를 tape 에 기록한다.
 */
(function (global) {
  'use strict';

  const M = (typeof module !== 'undefined' && module.exports) ? require('./model.js') : global.TModel;
  const Tensor = M.Tensor;
  const Arr = Float64Array;
  const ops = M.ops;

  // ---------------------------------------------------------------- 추가 연산

  // 이미지 [S, S*C] (행 = y, 열 = x*C + c) → 패치 [N, C*P*P]  (PyTorch Conv2d 평탄화 순서)
  function patchify(tape, img, S, P, C) {
    const G = S / P, N = G * G, D = C * P * P;
    const Y = new Tensor(N, D);
    const idx = new Int32Array(N * D);
    for (let py = 0; py < G; py++) for (let px = 0; px < G; px++) {
      const n = py * G + px;
      for (let c = 0; c < C; c++) for (let i = 0; i < P; i++) for (let j = 0; j < P; j++) {
        const dst = n * D + c * P * P + i * P + j;
        const src = (py * P + i) * (S * C) + (px * P + j) * C + c;
        idx[dst] = src;
        Y.data[dst] = img.data[src];
      }
    }
    tape.push(() => {
      if (!Y.grad) return;
      const dx = img.ensureGrad();
      for (let i = 0; i < idx.length; i++) dx[idx[i]] += Y.grad[i];
    });
    return Y;
  }

  // y = x Wᵀ + b   (b 는 없을 수 있음)
  function linearB(tape, X, W, B) {
    const Y = ops.linear(tape, X, W);
    if (!B) return Y;
    const O = Y.cols, T = Y.rows;
    for (let t = 0; t < T; t++) for (let o = 0; o < O; o++) Y.data[t * O + o] += B.data[o];
    tape.push(() => {
      if (!Y.grad) return;
      const db = B.ensureGrad();
      for (let t = 0; t < T; t++) for (let o = 0; o < O; o++) db[o] += Y.grad[t * O + o];
    });
    return Y;
  }

  function concatRows(tape, parts) {
    const cols = parts[0].cols;
    const rows = parts.reduce((s, p) => s + p.rows, 0);
    const Y = new Tensor(rows, cols);
    let off = 0;
    for (const p of parts) { Y.data.set(p.data, off * cols); off += p.rows; }
    tape.push(() => {
      if (!Y.grad) return;
      let o = 0;
      for (const p of parts) {
        const dp = p.ensureGrad();
        for (let i = 0; i < p.data.length; i++) dp[i] += Y.grad[o * cols + i];
        o += p.rows;
      }
    });
    return Y;
  }

  function sliceRows(tape, X, start, end) {
    const cols = X.cols, rows = end - start;
    const Y = new Tensor(rows, cols);
    Y.data.set(X.data.subarray(start * cols, end * cols));
    tape.push(() => {
      if (!Y.grad) return;
      const dx = X.ensureGrad();
      for (let i = 0; i < Y.grad.length; i++) dx[start * cols + i] += Y.grad[i];
    });
    return Y;
  }

  // y = (x − μ) / √(σ² + ε) · γ + β   (평균을 빼는 표준 LayerNorm: CLIP 이 사용)
  function layernorm(tape, X, G, B, eps) {
    const T = X.rows, D = X.cols, x = X.data;
    const Y = new Tensor(T, D);
    const xhat = new Arr(T * D), invStd = new Arr(T);
    for (let t = 0; t < T; t++) {
      const o = t * D;
      let mean = 0;
      for (let j = 0; j < D; j++) mean += x[o + j];
      mean /= D;
      let varr = 0;
      for (let j = 0; j < D; j++) { const d = x[o + j] - mean; varr += d * d; }
      varr /= D;
      const inv = 1 / Math.sqrt(varr + eps);
      invStd[t] = inv;
      for (let j = 0; j < D; j++) {
        const xh = (x[o + j] - mean) * inv;
        xhat[o + j] = xh;
        Y.data[o + j] = xh * G.data[j] + B.data[j];
      }
    }
    tape.push(() => {
      if (!Y.grad) return;
      const dy = Y.grad, dx = X.ensureGrad(), dg = G.ensureGrad(), db = B.ensureGrad();
      for (let t = 0; t < T; t++) {
        const o = t * D;
        let m1 = 0, m2 = 0;
        for (let j = 0; j < D; j++) {
          const dyh = dy[o + j] * G.data[j];
          dg[j] += dy[o + j] * xhat[o + j];
          db[j] += dy[o + j];
          m1 += dyh;
          m2 += dyh * xhat[o + j];
        }
        m1 /= D; m2 /= D;
        for (let j = 0; j < D; j++) {
          dx[o + j] += (dy[o + j] * G.data[j] - m1 - xhat[o + j] * m2) * invStd[t];
        }
      }
    });
    return Y;
  }

  // kind: 'quick' = x·σ(1.702x)  (OpenAI CLIP),  'tanh' = GELU tanh 근사 (projector)
  function gelu(tape, X, kind) {
    const n = X.data.length, x = X.data;
    const Y = new Tensor(X.rows, X.cols), d = new Arr(n);
    if (kind === 'quick') {
      for (let i = 0; i < n; i++) {
        const s = 1 / (1 + Math.exp(-1.702 * x[i]));
        Y.data[i] = x[i] * s;
        d[i] = s * (1 + 1.702 * x[i] * (1 - s));
      }
    } else {
      const k = Math.sqrt(2 / Math.PI);
      for (let i = 0; i < n; i++) {
        const v = x[i], u = k * (v + 0.044715 * v * v * v), t = Math.tanh(u);
        Y.data[i] = 0.5 * v * (1 + t);
        d[i] = 0.5 * (1 + t) + 0.5 * v * (1 - t * t) * k * (1 + 3 * 0.044715 * v * v);
      }
    }
    tape.push(() => {
      if (!Y.grad) return;
      const dx = X.ensureGrad();
      for (let i = 0; i < n; i++) dx[i] += Y.grad[i] * d[i];
    });
    return Y;
  }

  // 마스크 없는(양방향) multi-head attention. Q,K,V [T, H*hd]
  function attentionFull(tape, Q, K, V, H, hd) {
    const T = Q.rows, C = H * hd, scale = 1 / Math.sqrt(hd);
    const q = Q.data, k = K.data, v = V.data;
    const O = new Tensor(T, C), o = O.data;
    const P = new Arr(H * T * T);
    for (let h = 0; h < H; h++) {
      const off = h * hd;
      for (let t = 0; t < T; t++) {
        const row = (h * T + t) * T;
        let max = -Infinity;
        for (let j = 0; j < T; j++) {
          let s = 0;
          for (let d = 0; d < hd; d++) s += q[t * C + off + d] * k[j * C + off + d];
          s *= scale;
          P[row + j] = s;
          if (s > max) max = s;
        }
        let sum = 0;
        for (let j = 0; j < T; j++) { const e = Math.exp(P[row + j] - max); P[row + j] = e; sum += e; }
        for (let j = 0; j < T; j++) P[row + j] /= sum;
        for (let d = 0; d < hd; d++) {
          let acc = 0;
          for (let j = 0; j < T; j++) acc += P[row + j] * v[j * C + off + d];
          o[t * C + off + d] = acc;
        }
      }
    }
    tape.push(() => {
      if (!O.grad) return;
      const dO = O.grad, dQ = Q.ensureGrad(), dK = K.ensureGrad(), dV = V.ensureGrad();
      const dP = new Arr(T);
      for (let h = 0; h < H; h++) {
        const off = h * hd;
        for (let t = 0; t < T; t++) {
          const row = (h * T + t) * T;
          let rs = 0;
          for (let j = 0; j < T; j++) {
            let s = 0;
            for (let d = 0; d < hd; d++) {
              const g = dO[t * C + off + d];
              s += g * v[j * C + off + d];
              dV[j * C + off + d] += P[row + j] * g;
            }
            dP[j] = s;
            rs += s * P[row + j];
          }
          for (let j = 0; j < T; j++) {
            const dS = P[row + j] * (dP[j] - rs) * scale;
            if (dS === 0) continue;
            for (let d = 0; d < hd; d++) {
              dQ[t * C + off + d] += dS * k[j * C + off + d];
              dK[j * C + off + d] += dS * q[t * C + off + d];
            }
          }
        }
      }
    });
    return { out: O, probs: P };
  }

  // ---------------------------------------------------------------- 설정 / 파라미터
  const VIS_INT = ['image_size', 'patch_size', 'vision_hidden', 'vision_layers', 'vision_heads'];

  function numPatches(c) {
    const g = c.image_size / c.patch_size;
    return g * g;
  }

  function validateConfig(c) {
    const errs = M.validateConfig(c);
    for (const k of VIS_INT) if (!Number.isInteger(c[k]) || c[k] < 1) errs.push(`${k} 는 1 이상의 정수여야 합니다.`);
    if (Number.isInteger(c.image_size) && Number.isInteger(c.patch_size) && c.patch_size >= 1 && c.image_size % c.patch_size !== 0) {
      errs.push('image_size 는 patch_size 로 나누어 떨어져야 합니다.');
    }
    if (Number.isInteger(c.vision_hidden) && Number.isInteger(c.vision_heads) && c.vision_heads >= 1 && c.vision_hidden % c.vision_heads !== 0) {
      errs.push('vision_hidden 은 vision_heads 로 나누어 떨어져야 합니다.');
    }
    if (!(c.vision_mlp_ratio > 0)) errs.push('vision_mlp_ratio 는 양수여야 합니다.');
    return errs;
  }

  function countParams(c) {
    const Dv = c.vision_hidden, Lv = c.vision_layers, P = c.patch_size, C = 3;
    const N = numPatches(c), Iv = Math.round(Dv * c.vision_mlp_ratio);
    const vlayer = {
      layer_norm1: 2 * Dv,
      q_proj: Dv * Dv + Dv, k_proj: Dv * Dv + Dv, v_proj: Dv * Dv + Dv, out_proj: Dv * Dv + Dv,
      layer_norm2: 2 * Dv,
      fc1: Dv * Iv + Iv, fc2: Iv * Dv + Dv,
    };
    const vPerLayer = Object.values(vlayer).reduce((a, b) => a + b, 0);
    const vision = {
      patch_embedding: C * P * P * Dv,
      class_embedding: Dv,
      position_embedding: (N + 1) * Dv,
      pre_layernorm: 2 * Dv,
      layers: vPerLayer * Lv,
      post_layernorm: 2 * Dv,
    };
    const visionTotal = Object.values(vision).reduce((a, b) => a + b, 0);
    const d = c.hidden_dim;
    const projector = c.projector_type === 'linear'
      ? { 'linear.weight': Dv * d, 'linear.bias': d }
      : { 'mm.0.weight': Dv * d, 'mm.0.bias': d, 'mm.2.weight': d * d, 'mm.2.bias': d };
    const projTotal = Object.values(projector).reduce((a, b) => a + b, 0);
    const llm = M.countParams(c);
    return {
      vision, vlayer, vPerLayer, visionTotal,
      projector, projTotal,
      llm, llmTotal: llm.total,
      total: visionTotal + projTotal + llm.total,
      groups: { vision: visionTotal, projector: projTotal, llm: llm.total },
    };
  }

  function initLlava(c, seed) {
    const randn = (function () {
      // model.js 와 다른 스트림을 쓰기 위해 seed 를 섞는다.
      let s = (seed ^ 0x9e3779b9) >>> 0;
      const rand = () => {
        s = (s + 0x6D2B79F5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
      let spare = null;
      return () => {
        if (spare !== null) { const v = spare; spare = null; return v; }
        let u = 0;
        while (u === 0) u = rand();
        const v = rand(), m = Math.sqrt(-2 * Math.log(u));
        spare = m * Math.sin(2 * Math.PI * v);
        return m * Math.cos(2 * Math.PI * v);
      };
    })();
    const W = (r, k, std) => {
      const t = new Tensor(r, k);
      for (let i = 0; i < t.data.length; i++) t.data[i] = randn() * (std === undefined ? c.init_std : std);
      return t;
    };
    const zeros = (n) => new Tensor(1, n);
    const ones = (n) => { const t = new Tensor(1, n); t.data.fill(1); return t; };
    const ln = (n) => ({ g: ones(n), b: zeros(n) });

    const Dv = c.vision_hidden, N = numPatches(c), Iv = Math.round(Dv * c.vision_mlp_ratio), d = c.hidden_dim;
    const vision = {
      patch_embed: W(Dv, 3 * c.patch_size * c.patch_size),
      cls: W(1, Dv),
      pos: W(N + 1, Dv),
      ln_pre: ln(Dv),
      layers: Array.from({ length: c.vision_layers }, () => ({
        ln1: ln(Dv),
        q: { W: W(Dv, Dv), b: zeros(Dv) },
        k: { W: W(Dv, Dv), b: zeros(Dv) },
        v: { W: W(Dv, Dv), b: zeros(Dv) },
        out: { W: W(Dv, Dv), b: zeros(Dv) },
        ln2: ln(Dv),
        fc1: { W: W(Iv, Dv), b: zeros(Iv) },
        fc2: { W: W(Dv, Iv), b: zeros(Dv) },
      })),
      ln_post: ln(Dv),
    };
    const projector = c.projector_type === 'linear'
      ? { fc1: { W: W(d, Dv), b: zeros(d) } }
      : { fc1: { W: W(d, Dv), b: zeros(d) }, fc2: { W: W(d, d), b: zeros(d) } };
    const llm = M.initParams(c, seed | 0);
    return { vision, projector, llm, cfg: { projector_type: c.projector_type } };
  }

  function paramList(p) {
    const out = [];
    const v = p.vision;
    out.push(['vision_tower.…patch_embedding.weight', v.patch_embed],
      ['vision_tower.…class_embedding', v.cls],
      ['vision_tower.…position_embedding.weight', v.pos],
      ['vision_tower.…pre_layrnorm.weight', v.ln_pre.g],
      ['vision_tower.…pre_layrnorm.bias', v.ln_pre.b]);
    v.layers.forEach((L, i) => {
      const pre = `vision_tower.…layers.${i}.`;
      out.push([pre + 'layer_norm1.weight', L.ln1.g], [pre + 'layer_norm1.bias', L.ln1.b]);
      for (const [n, m] of [['q_proj', L.q], ['k_proj', L.k], ['v_proj', L.v], ['out_proj', L.out]]) {
        out.push([pre + 'self_attn.' + n + '.weight', m.W], [pre + 'self_attn.' + n + '.bias', m.b]);
      }
      out.push([pre + 'layer_norm2.weight', L.ln2.g], [pre + 'layer_norm2.bias', L.ln2.b],
        [pre + 'mlp.fc1.weight', L.fc1.W], [pre + 'mlp.fc1.bias', L.fc1.b],
        [pre + 'mlp.fc2.weight', L.fc2.W], [pre + 'mlp.fc2.bias', L.fc2.b]);
    });
    out.push(['vision_tower.…post_layernorm.weight', v.ln_post.g], ['vision_tower.…post_layernorm.bias', v.ln_post.b]);
    out.push(['multi_modal_projector.0.weight', p.projector.fc1.W], ['multi_modal_projector.0.bias', p.projector.fc1.b]);
    if (p.projector.fc2) out.push(['multi_modal_projector.2.weight', p.projector.fc2.W], ['multi_modal_projector.2.bias', p.projector.fc2.b]);
    for (const [n, t] of M.paramList(p.llm)) out.push(['language_model.' + n, t]);
    return out;
  }

  // ---------------------------------------------------------------- forward
  /**
   * image: Tensor [S, S*3] (정규화된 픽셀), text: { pre: ids[], post: ids[], answerStart, targets }
   */
  function forward(c, p, image, text) {
    const tape = [];
    const acts = {};
    const visionAttn = [];
    const llmAttn = [];
    const Dv = c.vision_hidden, Hv = c.vision_heads, hdv = Dv / Hv;
    const N = numPatches(c);

    // ---- vision tower (CLIP ViT) ----
    acts.image = image;
    const patches = patchify(tape, image, c.image_size, c.patch_size, 3); acts.patchify = patches;
    const pe = ops.linear(tape, patches, p.vision.patch_embed); acts.patch_embed = pe;
    const withCls = concatRows(tape, [p.vision.cls, pe]); acts.cls_cat = withCls;
    const posed = ops.add(tape, withCls, p.vision.pos); acts.pos_add = posed;
    let h = layernorm(tape, posed, p.vision.ln_pre.g, p.vision.ln_pre.b, c.ln_eps); acts.ln_pre = h;

    const hidden = [h]; // hidden_states[0] = embedding 출력
    for (let l = 0; l < c.vision_layers; l++) {
      const w = p.vision.layers[l], pre = `V${l}.`;
      const n1 = layernorm(tape, h, w.ln1.g, w.ln1.b, c.ln_eps); acts[pre + 'ln1'] = n1;
      const q = linearB(tape, n1, w.q.W, w.q.b); acts[pre + 'q_proj'] = q;
      const k = linearB(tape, n1, w.k.W, w.k.b); acts[pre + 'k_proj'] = k;
      const v = linearB(tape, n1, w.v.W, w.v.b); acts[pre + 'v_proj'] = v;
      const at = attentionFull(tape, q, k, v, Hv, hdv);
      acts[pre + 'attn'] = at.out;
      visionAttn.push(at.probs);
      const o = linearB(tape, at.out, w.out.W, w.out.b); acts[pre + 'out_proj'] = o;
      h = ops.add(tape, h, o); acts[pre + 'resid1'] = h;
      const n2 = layernorm(tape, h, w.ln2.g, w.ln2.b, c.ln_eps); acts[pre + 'ln2'] = n2;
      const f1 = linearB(tape, n2, w.fc1.W, w.fc1.b); acts[pre + 'fc1'] = f1;
      const ac = gelu(tape, f1, 'quick'); acts[pre + 'act'] = ac;
      const f2 = linearB(tape, ac, w.fc2.W, w.fc2.b); acts[pre + 'fc2'] = f2;
      h = ops.add(tape, h, f2); acts[pre + 'resid2'] = h;
      hidden.push(h);
    }
    acts.ln_post = layernorm(tape, h, p.vision.ln_post.g, p.vision.ln_post.b, c.ln_eps);

    // ---- feature select: hidden_states[vision_feature_layer], CLS 제거 ----
    const sel = hidden[hidden.length + (c.vision_feature_layer === -1 ? -1 : -2)];
    const feats = c.feature_strategy === 'cls_patch' ? sliceRows(tape, sel, 0, N + 1) : sliceRows(tape, sel, 1, N + 1);
    acts.feature_select = feats;

    // ---- projector ----
    let img = linearB(tape, feats, p.projector.fc1.W, p.projector.fc1.b); acts.proj_fc1 = img;
    if (p.projector.fc2) {
      img = gelu(tape, img, 'tanh'); acts.proj_act = img;
      img = linearB(tape, img, p.projector.fc2.W, p.projector.fc2.b); acts.proj_fc2 = img;
    }
    acts.image_features = img;

    // ---- 텍스트 임베딩 + 결합 ----
    const allIds = text.pre.concat(text.post);
    const temb = ops.embedding(tape, allIds, p.llm.embed); acts.text_embed = temb;
    const parts = [];
    if (text.pre.length) parts.push(sliceRows(tape, temb, 0, text.pre.length));
    parts.push(img);
    if (text.post.length) parts.push(sliceRows(tape, temb, text.pre.length, allIds.length));
    let x = parts.length > 1 ? concatRows(tape, parts) : parts[0];
    acts.assemble = x;

    // ---- language model (Llama decoder) ----
    for (let l = 0; l < c.num_layers; l++) {
      const w = p.llm.layers[l], pre = `L${l}.`;
      const h1 = ops.rmsnorm(tape, x, w.attn_norm, c.rms_eps); acts[pre + 'attn_norm'] = h1;
      const q = ops.linear(tape, h1, w.q_proj); acts[pre + 'q_proj'] = q;
      const k = ops.linear(tape, h1, w.k_proj); acts[pre + 'k_proj'] = k;
      const v = ops.linear(tape, h1, w.v_proj); acts[pre + 'v_proj'] = v;
      const qr = ops.rope(tape, q, c.num_heads, c.head_dim, c.rope_theta); acts[pre + 'rope_q'] = qr;
      const kr = ops.rope(tape, k, c.num_kv_heads, c.head_dim, c.rope_theta); acts[pre + 'rope_k'] = kr;
      const at = ops.attention(tape, qr, kr, v, c.num_heads, c.num_kv_heads, c.head_dim);
      acts[pre + 'attn'] = at.out;
      llmAttn.push(at.probs);
      const o = ops.linear(tape, at.out, w.o_proj); acts[pre + 'o_proj'] = o;
      x = ops.add(tape, x, o); acts[pre + 'resid1'] = x;
      const h2 = ops.rmsnorm(tape, x, w.mlp_norm, c.rms_eps); acts[pre + 'mlp_norm'] = h2;
      const g = ops.linear(tape, h2, w.gate_proj); acts[pre + 'gate_proj'] = g;
      const u = ops.linear(tape, h2, w.up_proj); acts[pre + 'up_proj'] = u;
      const a = ops.silu(tape, g); acts[pre + 'act'] = a;
      const m = ops.mul(tape, a, u); acts[pre + 'mul'] = m;
      const dn = ops.linear(tape, m, w.down_proj); acts[pre + 'down_proj'] = dn;
      x = ops.add(tape, x, dn); acts[pre + 'resid2'] = x;
    }
    const xf = ops.rmsnorm(tape, x, p.llm.final_norm, c.rms_eps); acts.final_norm = xf;
    const logits = ops.linear(tape, xf, p.llm.lm_head); acts.lm_head = logits;
    const ce = ops.crossEntropy(tape, logits, text.targets);
    acts.loss = ce.loss;

    return {
      tape, acts, visionAttn, llmAttn, probs: ce.probs,
      loss: ce.loss.data[0], lossTensor: ce.loss, count: ce.count, backwardDone: false,
    };
  }

  function backward(fw, p, opts) {
    if (fw.backwardDone) throw new Error('backward 는 forward 결과마다 한 번만 실행할 수 있습니다.');
    for (const [, t] of paramList(p)) t.grad = null;
    if (opts && opts.image) opts.image.grad = null;
    fw.lossTensor.ensureGrad()[0] = 1;
    for (let i = fw.tape.length - 1; i >= 0; i--) fw.tape[i]();
    fw.backwardDone = true;
  }

  // 학습 단계별로 어떤 부분을 얼리는지 (LLaVA 논문/1.5 레시피)
  const STAGES = {
    stage1: { label: 'Stage 1: 사전학습 (projector 만 학습)', train: { vision: false, projector: true, llm: false } },
    stage2: { label: 'Stage 2: 시각 지시 튜닝 (projector + LLM)', train: { vision: false, projector: true, llm: true } },
    full: { label: '전체 미세조정 (vision tower 까지)', train: { vision: true, projector: true, llm: true } },
  };

  function trainableList(p, stage) {
    const t = STAGES[stage].train;
    return paramList(p).filter(([n]) => {
      if (n.startsWith('vision_tower')) return t.vision;
      if (n.startsWith('multi_modal_projector')) return t.projector;
      return t.llm;
    });
  }

  const api = {
    ops: { patchify, linearB, concatRows, sliceRows, layernorm, gelu, attentionFull },
    numPatches, validateConfig, countParams, initLlava, paramList, forward, backward,
    STAGES, trainableList, Tensor,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else global.TVision = api;
})(typeof window !== 'undefined' ? window : globalThis);
