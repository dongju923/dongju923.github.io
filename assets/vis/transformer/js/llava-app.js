(function () {
  'use strict';

  const M = window.TModel, V = window.TVision;
  const { $, esc, fmt, vis, sleep, frame, short, num, bytes, heatmap } = window.TUI;
  const statLine = (arr) => window.TUI.statLine(arr, M.tensorStats);

  const COMPUTE_LIMIT = 1_500_000;   // 이 이상이면 shape / 파라미터 수만 표시
  const MAX_T = 200;                 // LLM 시퀀스 길이 한도
  const CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073];
  const CLIP_STD = [0.26862954, 0.26130258, 0.27577711];

  const PRESETS = {
    tiny: {
      label: 'Tiny (기본, 실시간 계산)',
      image_size: 32, patch_size: 8, vision_hidden: 48, vision_layers: 2, vision_heads: 4, vision_mlp_ratio: 4,
      projector_type: 'mlp2x_gelu', vision_feature_layer: -2, feature_strategy: 'patch',
      vocab_size: 96, hidden_dim: 64, num_layers: 2, num_heads: 4, num_kv_heads: 2, head_dim: 16,
      intermediate_size: 176, rope_theta: 10000, tie_word_embeddings: false,
    },
    small: {
      label: 'Small (~1M, 느리지만 계산 가능)',
      image_size: 48, patch_size: 8, vision_hidden: 96, vision_layers: 3, vision_heads: 6, vision_mlp_ratio: 4,
      projector_type: 'mlp2x_gelu', vision_feature_layer: -2, feature_strategy: 'patch',
      vocab_size: 128, hidden_dim: 128, num_layers: 3, num_heads: 8, num_kv_heads: 4, head_dim: 16,
      intermediate_size: 344, rope_theta: 10000, tie_word_embeddings: false,
    },
    llava15_7b: {
      label: 'LLaVA-1.5-7B (shape 전용)',
      image_size: 336, patch_size: 14, vision_hidden: 1024, vision_layers: 24, vision_heads: 16, vision_mlp_ratio: 4,
      projector_type: 'mlp2x_gelu', vision_feature_layer: -2, feature_strategy: 'patch',
      vocab_size: 32000, hidden_dim: 4096, num_layers: 32, num_heads: 32, num_kv_heads: 32, head_dim: 128,
      intermediate_size: 11008, rope_theta: 10000, tie_word_embeddings: false,
    },
    llava15_13b: {
      label: 'LLaVA-1.5-13B (shape 전용)',
      image_size: 336, patch_size: 14, vision_hidden: 1024, vision_layers: 24, vision_heads: 16, vision_mlp_ratio: 4,
      projector_type: 'mlp2x_gelu', vision_feature_layer: -2, feature_strategy: 'patch',
      vocab_size: 32000, hidden_dim: 5120, num_layers: 40, num_heads: 40, num_kv_heads: 40, head_dim: 128,
      intermediate_size: 13824, rope_theta: 10000, tie_word_embeddings: false,
    },
  };

  const FIELDS = [
    { key: 'image_size', group: 'image', value: 32 },
    { key: 'patch_size', group: 'image', value: 8 },
    { key: 'vision_hidden', group: 'vision', label: 'vision_hidden (Dv)', value: 48, primary: true },
    { key: 'vision_layers', group: 'vision', label: 'vision_layers (Lv)', value: 2 },
    { key: 'vision_heads', group: 'vision', label: 'vision_heads (Hv)', value: 4 },
    { key: 'vision_mlp_ratio', group: 'vision', label: 'mlp_ratio', value: 4, float: true },
    { key: 'ln_eps', group: 'vision', label: 'layer_norm_eps', value: 1e-5, float: true, step: 'any' },
    { key: 'hidden_dim', group: 'llm', label: 'hidden_dim (d)', value: 64, primary: true },
    { key: 'head_dim', group: 'llm', label: 'head_dim (hd)', value: 16, step: 2 },
    { key: 'num_heads', group: 'llm', label: 'num_heads (H)', value: 4 },
    { key: 'num_kv_heads', group: 'llm', label: 'num_kv_heads (KV)', value: 2 },
    { key: 'num_layers', group: 'llm', label: 'num_layers (L)', value: 2 },
    { key: 'intermediate_size', group: 'llm', label: 'intermediate (I)', value: 176 },
    { key: 'vocab_size', group: 'llm', label: 'vocab_size (V)', value: 96 },
    { key: 'rope_theta', group: 'llm', label: 'rope_theta', value: 10000, float: true },
    { key: 'rms_eps', group: 'llm', label: 'rms_norm_eps', value: 1e-6, float: true, step: 'any' },
    { key: 'init_std', group: 'llm', label: 'init std', value: 0.02, float: true, step: 'any' },
    { key: 'seed', group: 'llm', label: 'seed', value: 42 },
  ];
  const SELECTS = ['vision_feature_layer', 'feature_strategy', 'projector_type'];
  const ARCH_KEYS = ['image_size', 'patch_size', 'vision_hidden', 'vision_layers', 'vision_heads', 'vision_mlp_ratio',
    'projector_type', 'vocab_size', 'hidden_dim', 'num_layers', 'num_heads', 'num_kv_heads', 'head_dim',
    'intermediate_size', 'init_std', 'seed', 'tie_word_embeddings'];

  const VKEYS = ['ln1', 'q_proj', 'k_proj', 'v_proj', 'attn', 'out_proj', 'resid1', 'ln2', 'fc1', 'act', 'fc2', 'resid2'];
  const VGROUPS = [['ln1'], ['q_proj', 'k_proj', 'v_proj'], ['attn'], ['out_proj'], ['resid1'], ['ln2'], ['fc1'], ['act'], ['fc2'], ['resid2']];
  const LGROUPS = [['attn_norm'], ['q_proj', 'k_proj', 'v_proj'], ['rope_q', 'rope_k'], ['attn'], ['o_proj'], ['resid1'],
    ['mlp_norm'], ['gate_proj', 'up_proj'], ['act'], ['mul'], ['down_proj'], ['resid2']];

  const state = {
    cfg: null, specs: null, tok: null,
    imgSource: { type: 'preset', name: 'square' }, imgRGB: null, image: null,
    text: null, labels: [], segs: [],
    params: null, sig: null, adam: null, fw: null, phase: 'none',
    status: {}, active: new Set(), activeDir: null,
    selected: 'image', vlayer: 0, layer: 0, view: 'act', head: 0, vhead: 0, patchSel: 0, pos: null,
    history: [], computeOK: false, errors: [], warns: [], busy: false, skip: false,
  };

  // ================================================================ 샘플 이미지
  const IMG_PRESETS = {
    square: ['왼쪽 빨간 사각형', (x, y, S) => (x < S * 0.45 && y > S * 0.25 && y < S * 0.75 ? [0.85, 0.15, 0.15] : [0.93, 0.93, 0.9])],
    circle: ['가운데 파란 원', (x, y, S) => {
      const dx = x - S / 2 + 0.5, dy = y - S / 2 + 0.5;
      return Math.sqrt(dx * dx + dy * dy) < S * 0.3 ? [0.15, 0.35, 0.9] : [0.95, 0.95, 0.92];
    }],
    checker: ['체커보드', (x, y, S) => {
      const k = Math.max(2, Math.floor(S / 8));
      return (Math.floor(x / k) + Math.floor(y / k)) % 2 ? [0.12, 0.12, 0.14] : [0.95, 0.95, 0.95];
    }],
    gradient: ['RGB 그라디언트', (x, y, S) => [x / S, y / S, 1 - x / S]],
    face: ['도형 얼굴', (x, y, S) => {
      const n = (a, b, r) => Math.hypot(x - S * a + 0.5, y - S * b + 0.5) < S * r;
      if (n(0.35, 0.38, 0.07) || n(0.65, 0.38, 0.07)) return [0.1, 0.1, 0.12];
      if (y > S * 0.62 && y < S * 0.72 && x > S * 0.3 && x < S * 0.7) return [0.8, 0.2, 0.3];
      if (n(0.5, 0.5, 0.42)) return [0.98, 0.83, 0.6];
      return [0.4, 0.62, 0.85];
    }],
  };

  function renderImagePixels() {
    const S = state.cfg.image_size;
    const rgb = new Float64Array(S * S * 3);
    const src = state.imgSource;
    if (src.type === 'file' && src.canvas) {
      const c = document.createElement('canvas');
      c.width = S; c.height = S;
      const ctx = c.getContext('2d');
      ctx.drawImage(src.canvas, 0, 0, S, S);
      const d = ctx.getImageData(0, 0, S, S).data;
      for (let i = 0; i < S * S; i++) for (let k = 0; k < 3; k++) rgb[i * 3 + k] = d[i * 4 + k] / 255;
    } else {
      const fn = (IMG_PRESETS[src.name] || IMG_PRESETS.square)[1];
      for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
        const c = fn(x, y, S);
        for (let k = 0; k < 3; k++) rgb[(y * S + x) * 3 + k] = c[k];
      }
    }
    state.imgRGB = rgb;
    // CLIP 전처리: (픽셀 − mean) / std
    const img = new M.Tensor(S, S * 3);
    for (let i = 0; i < S * S; i++) for (let k = 0; k < 3; k++) {
      img.data[i * 3 + k] = (rgb[i * 3 + k] - CLIP_MEAN[k]) / CLIP_STD[k];
    }
    state.image = img;
  }

  function drawImageTo(canvas, opts) {
    const S = state.cfg.image_size;
    canvas.width = S; canvas.height = S;
    const ctx = canvas.getContext('2d');
    const im = ctx.createImageData(S, S);
    for (let i = 0; i < S * S; i++) {
      for (let k = 0; k < 3; k++) im.data[i * 4 + k] = Math.round(255 * Math.min(1, Math.max(0, state.imgRGB[i * 3 + k])));
      im.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    if (opts && opts.saliency && state.image.grad) {
      const g = state.image.grad;
      let max = 0;
      const mag = new Float64Array(S * S);
      for (let i = 0; i < S * S; i++) {
        let m = 0;
        for (let k = 0; k < 3; k++) m += Math.abs(g[i * 3 + k]);
        mag[i] = m;
        if (m > max) max = m;
      }
      const ov = ctx.getImageData(0, 0, S, S);
      for (let i = 0; i < S * S; i++) {
        const t = max ? mag[i] / max : 0;
        ov.data[i * 4] = ov.data[i * 4] * (1 - t) + 255 * t;
        ov.data[i * 4 + 1] = ov.data[i * 4 + 1] * (1 - t) + 40 * t;
        ov.data[i * 4 + 2] = ov.data[i * 4 + 2] * (1 - t) + 40 * t;
      }
      ctx.putImageData(ov, 0, 0);
    }
  }

  // attention 가중치를 패치 격자 위에 겹쳐 그린다
  function drawPatchOverlay(canvas, weights) {
    const S = state.cfg.image_size, P = state.cfg.patch_size, G = S / P;
    canvas.width = S; canvas.height = S;
    const ctx = canvas.getContext('2d');
    const im = ctx.createImageData(S, S);
    // 최소값을 투명으로 잡아야 (거의 균등한 attention 에서도) 차이가 보인다
    let max = -Infinity, min = Infinity;
    for (const w of weights) { if (w > max) max = w; if (w < min) min = w; }
    const span = max - min || 1;
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      const n = Math.floor(y / P) * G + Math.floor(x / P);
      const t = (weights[n] - min) / span;
      const o = (y * S + x) * 4;
      im.data[o] = 255; im.data[o + 1] = 190; im.data[o + 2] = 40;
      im.data[o + 3] = Math.round(30 + 185 * t);
    }
    ctx.putImageData(im, 0, 0);
  }

  // ================================================================ 모듈 명세
  function buildSpecs(c, dims) {
    const { S, P, N, Np, Dv, Iv, hdv, nImg, Ttext, T, d, V: Vsz } = dims;
    const B = ['B', 1];
    const IMG = [B, ['C', 3], ['H', S], ['W', S]];
    const PATCH = [B, ['N', N], ['C·P·P', 3 * P * P]];
    const VT = [B, ['N', N], ['Dv', Dv]];
    const VT1 = [B, ['N+1', Np], ['Dv', Dv]];
    const VTI = [B, ['N+1', Np], ['Iv', Iv]];
    const BTD = [B, ['T', T], ['d', d]];
    const IMGTOK = [B, ['N_img', nImg], ['d', d]];
    const vHead = [B, ['Hv', c.vision_heads], ['N+1', Np], ['hdv', hdv]];

    const lnSpec = (title, name, get, desc, extra) => ({
      kind: 'norm', title, sub: `LayerNorm(${Dv})`, ins: [['x', VT1]], out: VT1,
      params: [{ name: name + '.weight', shape: [Dv], get: (p, l) => get(p, l).g }, { name: name + '.bias', shape: [Dv], get: (p, l) => get(p, l).b }],
      desc, extra,
      fwd: 'μ = mean(x),  σ² = var(x)\ny = (x − μ)/√(σ² + ε) · γ + β',
      bwd: 'x̂ = (x − μ)/σ\n∂L/∂x = [ĝ − mean(ĝ) − x̂·mean(ĝ⊙x̂)] / σ,  ĝ = ∂L/∂y⊙γ\n∂L/∂γ = Σ ∂L/∂y⊙x̂,  ∂L/∂β = Σ ∂L/∂y',
    });
    const vlin = (title, name, get, inDims, outDims, wShape, desc) => ({
      kind: 'linear', title, sub: `Linear ${inDims[2][1]} → ${outDims[2][1]} (+bias)`, ins: [['x', inDims]], out: outDims,
      params: [{ name: name + '.weight', shape: wShape, get: (p, l) => get(p, l).W }, { name: name + '.bias', shape: [wShape[0]], get: (p, l) => get(p, l).b }],
      desc, fwd: `y = x·Wᵀ + b,   W ∈ ℝ^(${wShape[0]}×${wShape[1]})`,
      bwd: '∂L/∂x = ∂L/∂y·W\n∂L/∂W = (∂L/∂y)ᵀ·x\n∂L/∂b = Σ_t ∂L/∂y_t',
    });
    const llin = (title, name, key, inDims, outDims, desc) => ({
      kind: 'linear', title, sub: `Linear ${inDims[2][0]} → ${outDims[2][0]}`, ins: [['x', inDims]], out: outDims,
      params: [{ name, shape: [outDims[2][1], inDims[2][1]], get: (p, l) => p.llm.layers[l][key] }],
      desc, fwd: 'y = x·Wᵀ (bias 없음)', bwd: '∂L/∂x = ∂L/∂y·W,  ∂L/∂W = (∂L/∂y)ᵀ·x',
    });
    const rms = (title, name, get, desc) => ({
      kind: 'norm', title, sub: `RMSNorm(${d})`, ins: [['x', BTD]], out: BTD,
      params: [{ name, shape: [d], get }], desc,
      fwd: 'y = x/√(mean(x²)+ε) ⊙ γ', bwd: '∂L/∂x = (ĝ − x̂·mean(ĝ⊙x̂))/rms,  ĝ = ∂L/∂y⊙γ',
    });
    const BTq = [B, ['T', T], ['H·hd', c.num_heads * c.head_dim]];
    const BTk = [B, ['T', T], ['KV·hd', c.num_kv_heads * c.head_dim]];
    const BTI = [B, ['T', T], ['I', c.intermediate_size]];
    const qView = [B, ['H', c.num_heads], ['T', T], ['hd', c.head_dim]];
    const kView = [B, ['KV', c.num_kv_heads], ['T', T], ['hd', c.head_dim]];

    return {
      // ---------------- 입력 + vision tower ----------------
      g: {
        image: {
          kind: 'image', title: 'image', sub: `${S}×${S} RGB`, out: IMG,
          desc: `입력 이미지는 vision tower 해상도(${S}×${S})로 리사이즈된 뒤 CLIP 통계로 정규화됩니다: (픽셀 − mean)/std, mean=(0.481, 0.458, 0.408), std=(0.269, 0.261, 0.276).`,
          fwd: 'pixel_values = (resize(image)/255 − mean) / std', bwd: '∂L/∂image = saliency map (픽셀이 답에 준 영향)',
        },
        patchify: {
          kind: 'image', title: 'patchify', sub: `${P}×${P} 패치 ${N}개`, ins: [['pixel_values', IMG]], out: PATCH,
          desc: `이미지를 겹치지 않는 ${P}×${P} 패치 ${N}개로 자르고 각 패치를 ${3 * P * P} 차원 벡터로 폅니다. 실제 구현에서는 stride=kernel=${P} 인 Conv2d 한 번으로 처리되며, 이 단계는 그 conv 의 im2col 에 해당합니다.`,
          fwd: `N = (S/P)² = (${S}/${P})² = ${N}`, bwd: '패치 조각들을 원래 픽셀 위치로 되돌려 더한다 (scatter-add)',
        },
        patch_embed: {
          kind: 'vision', title: 'patch_embedding', sub: `Conv2d(3, ${Dv}, k=${P}, s=${P})`, ins: [['patches', PATCH]], out: VT,
          params: [{ name: 'vision_tower.…patch_embedding.weight', shape: [Dv, 3 * P * P], get: (p) => p.vision.patch_embed }],
          desc: '각 패치를 Dv 차원으로 투영합니다. CLIP 의 patch embedding conv 에는 bias 가 없습니다. 여기서부터 이미지는 "패치 토큰 시퀀스" 가 됩니다.',
          fwd: 'z_n = patch_n · Wᵀ', bwd: '∂L/∂W = (∂L/∂z)ᵀ·patches',
        },
        cls_cat: {
          kind: 'vision', title: '+ class_embedding', sub: 'CLS 토큰 prepend', ins: [['patch tokens', VT]], out: VT1,
          params: [{ name: 'vision_tower.…class_embedding', shape: [1, Dv], get: (p) => p.vision.cls }],
          desc: '학습되는 CLS 토큰 하나를 맨 앞에 붙입니다. CLIP 은 이 토큰으로 이미지 전체를 요약하지만, LLaVA 는 보통 CLS 를 버리고 패치 토큰만 LLM 에 넘깁니다.',
          fwd: 'x = [CLS; z₁ … z_N]', bwd: '∂L/∂CLS = ∂L/∂x[0]',
        },
        pos_add: {
          kind: 'vision', title: '+ position_embedding', sub: `학습형 위치 임베딩 (${Np}개)`, ins: [['x', VT1]], out: VT1,
          params: [{ name: 'vision_tower.…position_embedding.weight', shape: [Np, Dv], get: (p) => p.vision.pos }],
          desc: 'ViT 는 RoPE 대신 학습되는 절대 위치 임베딩을 더합니다. 위치 임베딩 개수가 패치 수 + 1 로 고정되므로, 해상도를 바꾸려면 위치 임베딩을 보간해야 합니다 (LLaVA-NeXT 가 하는 일).',
          fwd: 'x = x + E_pos', bwd: '∂L/∂E_pos = ∂L/∂x (위치별 합산)',
        },
        ln_pre: lnSpec('pre_layrnorm', 'vision_tower.…pre_layrnorm', (p) => p.vision.ln_pre,
          'CLIP ViT 는 encoder 에 들어가기 전에 LayerNorm 을 한 번 적용합니다.'),
        ln_post: lnSpec('post_layernorm', 'vision_tower.…post_layernorm', (p) => p.vision.ln_post,
          'CLIP 의 마지막 LayerNorm 입니다. LLaVA 가 hidden_states[−2] 를 쓰면 이 층과 마지막 encoder layer 는 사용되지 않아 기울기도 0 입니다.',
          c.vision_feature_layer === -2 ? 'LLaVA 기본 설정(-2)에서는 사용되지 않음' : null),
        feature_select: {
          kind: 'vision', title: 'feature select', sub: `hidden_states[${c.vision_feature_layer}]${c.feature_strategy === 'patch' ? ', CLS 제거' : ', CLS 포함'}`,
          ins: [['hidden_states', VT1]], out: [B, ['N_img', nImg], ['Dv', Dv]],
          desc: `vision tower 의 중간 출력 중 ${c.vision_feature_layer} 번째 layer 의 hidden state 를 고릅니다. 마지막 층은 CLIP 의 대조학습 목적에 과하게 특화되어 있어서, LLaVA 는 한 층 앞(−2)의 표현을 씁니다. ${c.feature_strategy === 'patch' ? 'CLS 토큰은 버리고 패치 토큰 ' + N + '개만 남깁니다.' : 'CLS 토큰까지 함께 넘깁니다.'}`,
          fwd: `features = hidden_states[${c.vision_feature_layer}][:, ${c.feature_strategy === 'patch' ? '1:' : ':'}]`,
          bwd: '선택되지 않은 층으로는 기울기가 전혀 흐르지 않는다',
        },
        proj_fc1: {
          kind: 'proj', title: c.projector_type === 'linear' ? 'projector (linear)' : 'mm_projector.0', sub: `Linear Dv → d (${Dv} → ${d})`,
          ins: [['features', [B, ['N_img', nImg], ['Dv', Dv]]]], out: IMGTOK,
          params: [{ name: 'multi_modal_projector.0.weight', shape: [d, Dv], get: (p) => p.projector.fc1.W },
            { name: 'multi_modal_projector.0.bias', shape: [d], get: (p) => p.projector.fc1.b }],
          desc: '비전 특징을 LLM 의 단어 임베딩과 같은 차원 d 로 옮깁니다. LLaVA 의 핵심이자 거의 유일한 새 파라미터로, Stage 1 에서는 이 부분만 학습합니다.',
          fwd: 'H_img = features · W₀ᵀ + b₀', bwd: '∂L/∂W₀ = (∂L/∂H)ᵀ·features',
        },
        proj_act: {
          kind: 'act', title: 'GELU', sub: 'projector 활성함수', ins: [['h', IMGTOK]], out: IMGTOK,
          desc: 'LLaVA-1.5 의 projector 는 Linear–GELU–Linear (mlp2x_gelu) 2층 MLP 입니다. LLaVA-1.0 은 단일 Linear 였습니다.',
          fwd: 'GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])', bwd: '∂GELU/∂x = 0.5(1+tanh u) + 0.5x(1−tanh²u)·du/dx',
        },
        proj_fc2: {
          kind: 'proj', title: 'mm_projector.2', sub: `Linear d → d (${d} → ${d})`, ins: [['h', IMGTOK]], out: IMGTOK,
          params: [{ name: 'multi_modal_projector.2.weight', shape: [d, d], get: (p) => p.projector.fc2 && p.projector.fc2.W },
            { name: 'multi_modal_projector.2.bias', shape: [d], get: (p) => p.projector.fc2 && p.projector.fc2.b }],
          desc: 'projector 의 두 번째 Linear 입니다. 출력은 "이미지 토큰" 이 되어 텍스트 토큰과 완전히 같은 자격으로 LLM 에 들어갑니다.',
          fwd: 'image_tokens = GELU(H)·W₂ᵀ + b₂', bwd: '∂L/∂W₂ = (∂L/∂out)ᵀ·GELU(H)',
        },
        text_embed: {
          kind: 'embed', title: 'embed_tokens (텍스트)', sub: `Embedding(V=${Vsz}, d=${d})`, ins: [['input_ids', [B, ['T_text', Ttext]]]], out: [B, ['T_text', Ttext], ['d', d]],
          params: [{ name: 'language_model.model.embed_tokens.weight', shape: [Vsz, d], get: (p) => p.llm.embed }],
          desc: '프롬프트와 정답 텍스트는 평소처럼 임베딩 테이블에서 가져옵니다. 이미지 토큰은 이 테이블을 거치지 않습니다.',
          fwd: 'x_t = E[id_t]', bwd: '∂L/∂E[id] += ∂L/∂x (scatter-add)',
        },
        assemble: {
          kind: 'embed', title: '시퀀스 결합', sub: `[텍스트 | 이미지 ${nImg} | 텍스트]`,
          ins: [['text embeds', [B, ['T_text', Ttext], ['d', d]]], ['image tokens', IMGTOK]], out: BTD,
          desc: `<image> 자리에 이미지 토큰 ${nImg} 개를 끼워 넣어 하나의 시퀀스를 만듭니다. 이 지점부터 LLM 은 이미지와 텍스트를 구분하지 않고 똑같은 토큰 열로 처리합니다. 그래서 길이가 T = ${Ttext} + ${nImg} = ${T} 로 늘어납니다.`,
          fwd: 'inputs_embeds = concat(text[:k], image_tokens, text[k:])', bwd: '기울기가 텍스트 임베딩과 projector 로 각각 나뉘어 흐른다',
        },
        final_norm: rms('model.norm', 'language_model.model.norm.weight', (p) => p.llm.final_norm, '마지막 decoder layer 뒤의 RMSNorm 입니다.'),
        lm_head: {
          kind: 'head', title: 'lm_head', sub: `Linear d → V (${d} → ${Vsz})`, ins: [['x', BTD]], out: [B, ['T', T], ['V', Vsz]],
          params: [{ name: 'language_model.lm_head.weight', shape: [Vsz, d], get: (p) => p.llm.lm_head, tied: !!c.tie_word_embeddings }],
          desc: '각 위치에서 다음 토큰의 logits 를 만듭니다. 이미지 토큰 위치의 예측은 학습에 쓰이지 않습니다(라벨이 −100).',
          fwd: 'logits = x·Wᵀ', bwd: '∂L/∂x = ∂L/∂logits·W',
        },
        loss: {
          kind: 'loss', title: 'CE loss (answer 만)', sub: '시각 지시 튜닝 목표', ins: [['logits', [B, ['T', T], ['V', Vsz]]], ['labels', [B, ['T', T]]]], out: [],
          desc: 'LLaVA 학습은 정답(ASSISTANT) 구간의 토큰만 지도합니다. 이미지 토큰과 프롬프트 위치의 라벨은 −100 으로 무시되므로, 모델은 "이미지를 보고 답을 이어 쓰는" 것만 배웁니다.',
          fwd: 'L = −(1/N_ans) Σ_{t∈answer} log p_t[target_t]', bwd: '∂L/∂logits_t = (p_t − onehot)/N_ans  (answer 밖은 0)',
        },
      },
      // ---------------- vision encoder layer ----------------
      v: {
        ln1: lnSpec('layer_norm1', 'vision_tower.…layer_norm1', (p, l) => p.vision.layers[l].ln1,
          'Pre-Norm 구조: attention 앞의 LayerNorm 입니다. Llama 의 RMSNorm 과 달리 평균을 빼고 bias 도 있습니다.'),
        q_proj: vlin('q_proj', 'vision_tower.…self_attn.q_proj', (p, l) => p.vision.layers[l].q, VT1, VT1, [Dv, Dv],
          `Query 투영. head ${c.vision_heads} 개로 나누면 head 당 ${hdv} 차원입니다. CLIP 의 attention Linear 들은 bias 를 씁니다.`),
        k_proj: vlin('k_proj', 'vision_tower.…self_attn.k_proj', (p, l) => p.vision.layers[l].k, VT1, VT1, [Dv, Dv], 'Key 투영입니다.'),
        v_proj: vlin('v_proj', 'vision_tower.…self_attn.v_proj', (p, l) => p.vision.layers[l].v, VT1, VT1, [Dv, Dv], 'Value 투영입니다.'),
        attn: {
          kind: 'attn', title: 'self-attention', sub: `양방향 · ${c.vision_heads} heads`, ins: [['Q', vHead], ['K', vHead], ['V', vHead]],
          scores: [B, ['Hv', c.vision_heads], ['N+1', Np], ['N+1', Np]], out: VT1,
          desc: '텍스트 decoder 와 달리 causal mask 가 없습니다. 모든 패치가 서로를 봅니다 — 이미지에는 "앞뒤" 순서가 없기 때문입니다. 또 GQA 없이 일반 MHA 를 씁니다.',
          fwd: 'A = softmax(Q·Kᵀ/√hdv)   (마스크 없음)\nO = A·V', bwd: 'dA = dO·Vᵀ,  dV = Aᵀ·dO\ndS = A⊙(dA − rowsum(dA⊙A))\ndQ = dS·K/√hdv,  dK = dSᵀ·Q/√hdv',
        },
        out_proj: vlin('out_proj', 'vision_tower.…self_attn.out_proj', (p, l) => p.vision.layers[l].out, VT1, VT1, [Dv, Dv], 'head 들을 합쳐 다시 Dv 로 투영합니다.'),
        resid1: {
          kind: 'add', title: '⊕ residual', sub: 'x + Attn(LN(x))', ins: [['x', VT1], ['attn', VT1]], out: VT1,
          desc: 'attention 블록의 잔차 연결입니다.', fwd: 'x ← x + out_proj(Attn(LN₁(x)))', bwd: '기울기가 두 경로로 그대로 복사된다',
        },
        ln2: lnSpec('layer_norm2', 'vision_tower.…layer_norm2', (p, l) => p.vision.layers[l].ln2, 'MLP 앞의 LayerNorm 입니다.'),
        fc1: vlin('mlp.fc1', 'vision_tower.…mlp.fc1', (p, l) => p.vision.layers[l].fc1, VT1, VTI, [Iv, Dv],
          `MLP 확장: Dv → Iv (${Dv} → ${Iv}, ratio ${c.vision_mlp_ratio}).`),
        act: {
          kind: 'act', title: 'QuickGELU', sub: 'OpenAI CLIP 활성함수', ins: [['h', VTI]], out: VTI,
          desc: 'OpenAI CLIP 은 표준 GELU 대신 QuickGELU 를 씁니다. SwiGLU 를 쓰는 Llama MLP 와 달리 gate 경로가 없는 평범한 2층 MLP 입니다.',
          fwd: 'QuickGELU(x) = x · σ(1.702x)', bwd: '∂y/∂x = σ(1.702x)·(1 + 1.702x(1 − σ(1.702x)))',
        },
        fc2: vlin('mlp.fc2', 'vision_tower.…mlp.fc2', (p, l) => p.vision.layers[l].fc2, VTI, VT1, [Dv, Iv], `MLP 축소: Iv → Dv (${Iv} → ${Dv}).`),
        resid2: {
          kind: 'add', title: '⊕ residual', sub: 'x + MLP(LN(x))', ins: [['x', VT1], ['mlp', VT1]], out: VT1,
          desc: 'MLP 블록의 잔차 연결입니다.', fwd: 'x ← x + fc2(QuickGELU(fc1(LN₂(x))))', bwd: '기울기가 두 경로로 그대로 복사된다',
        },
      },
      // ---------------- language model layer ----------------
      l: {
        attn_norm: rms('input_layernorm', 'language_model.model.layers.{l}.input_layernorm.weight', (p, l) => p.llm.layers[l].attn_norm,
          'Llama decoder 의 pre-norm 입니다 (RMSNorm, bias 없음).'),
        q_proj: llin('q_proj', 'language_model.…q_proj.weight', 'q_proj', BTD, BTq, 'Query 투영 (이미지 토큰 위치도 똑같이 처리됩니다).'),
        k_proj: llin('k_proj', 'language_model.…k_proj.weight', 'k_proj', BTD, BTk, 'Key 투영 (GQA).'),
        v_proj: llin('v_proj', 'language_model.…v_proj.weight', 'v_proj', BTD, BTk, 'Value 투영 (GQA).'),
        rope_q: { kind: 'rope', title: 'RoPE(Q)', sub: `θ=${c.rope_theta}`, ins: [['q', qView]], out: qView,
          desc: '이미지 토큰에도 텍스트와 같은 1D 위치 인덱스로 RoPE 가 적용됩니다. 즉 LLM 은 이미지의 2D 구조를 모르고, 패치 순서(래스터 스캔)만 봅니다.',
          fwd: '[x₁,x₂] → [x₁cos−x₂sin, x₂cos+x₁sin]', bwd: '역회전 R(−pθ) 적용' },
        rope_k: { kind: 'rope', title: 'RoPE(K)', sub: `θ=${c.rope_theta}`, ins: [['k', kView]], out: kView,
          desc: 'Key 에도 같은 회전을 적용합니다.', fwd: '동일', bwd: '역회전 R(−pθ) 적용' },
        attn: {
          kind: 'attn', title: 'causal attention', sub: c.num_heads === c.num_kv_heads ? 'MHA' : `GQA (${c.num_heads / c.num_kv_heads}:1)`,
          ins: [['Q', qView], ['K', kView], ['V', kView]], scores: [B, ['H', c.num_heads], ['T', T], ['T', T]], out: BTq,
          desc: '여기서 텍스트가 이미지 토큰을 "봅니다". 정답 토큰들은 앞쪽 이미지 토큰에 attention 을 주어 이미지 정보를 가져옵니다. mask 는 causal 이므로 이미지 토큰은 뒤의 텍스트를 볼 수 없습니다.',
          fwd: 'A = softmax(QKᵀ/√hd + causal mask),  O = A·V',
          bwd: 'dQ, dK, dV 로 나뉘어 흐르고, 이미지 토큰 위치의 기울기는 projector 를 거쳐 vision tower 로 간다',
        },
        o_proj: llin('o_proj', 'language_model.…o_proj.weight', 'o_proj', BTq, BTD, 'attention 출력을 d 로 되돌립니다.'),
        resid1: { kind: 'add', title: '⊕ residual', sub: 'x + Attn', ins: [['x', BTD]], out: BTD, desc: 'attention 블록 잔차 연결.', fwd: 'x ← x + o_proj(...)', bwd: '기울기 복사' },
        mlp_norm: rms('post_attention_layernorm', 'language_model.…post_attention_layernorm.weight', (p, l) => p.llm.layers[l].mlp_norm, 'MLP 앞 RMSNorm.'),
        gate_proj: llin('gate_proj', 'language_model.…gate_proj.weight', 'gate_proj', BTD, BTI, 'SwiGLU 의 gate 경로.'),
        up_proj: llin('up_proj', 'language_model.…up_proj.weight', 'up_proj', BTD, BTI, 'SwiGLU 의 up 경로.'),
        act: { kind: 'act', title: 'SiLU(gate)', sub: 'Swish', ins: [['gate', BTI]], out: BTI, desc: 'gate 경로 활성함수.', fwd: 'SiLU(g) = g·σ(g)', bwd: '∂SiLU/∂g = σ(g)(1+g(1−σ(g)))' },
        mul: { kind: 'act', title: '⊙ gate·up', sub: 'SwiGLU', ins: [['SiLU(gate)', BTI], ['up', BTI]], out: BTI, desc: 'gated linear unit.', fwd: 'h = SiLU(g)⊙u', bwd: '각 경로에 상대 항을 곱해 전달' },
        down_proj: llin('down_proj', 'language_model.…down_proj.weight', 'down_proj', BTI, BTD, 'I → d 축소.'),
        resid2: { kind: 'add', title: '⊕ residual', sub: 'x + MLP', ins: [['x', BTD]], out: BTD, desc: 'MLP 블록 잔차 연결.', fwd: 'x ← x + down(...)', bwd: '기울기 복사' },
      },
    };
  }

  // ================================================================ 보조
  function parseId(id) {
    let m = /^V(\d+)\.(.+)$/.exec(id);
    if (m) return { tower: 'v', l: +m[1], key: m[2] };
    m = /^L(\d+)\.(.+)$/.exec(id);
    if (m) return { tower: 'l', l: +m[1], key: m[2] };
    return { tower: 'g', l: null, key: id };
  }
  const specOf = (id) => { const r = parseId(id); return state.specs && state.specs[r.tower][r.key]; };
  const specParamCount = (sp) => (sp && sp.params || []).reduce((s, p) => s + (p.tied ? 0 : p.shape.reduce((a, b) => a * b, 1)), 0);
  const fmtShape = (dims) => (dims.length ? '[' + dims.map((x) => x[1]).join(', ') + ']' : 'scalar');
  const fmtShapeL = (dims) => (dims.length ? '[' + dims.map((x) => `${x[0]}=${x[1]}`).join(', ') + ']' : 'scalar []');

  function sequence() {
    const c = state.cfg;
    const seq = [['image'], ['patchify'], ['patch_embed'], ['cls_cat'], ['pos_add'], ['ln_pre']].map((ids) => ({ tower: 'g', ids }));
    for (let l = 0; l < c.vision_layers; l++) {
      for (const g of VGROUPS) seq.push({ tower: 'v', l, ids: g.map((k) => `V${l}.${k}`) });
    }
    seq.push({ tower: 'g', ids: ['ln_post'] }, { tower: 'g', ids: ['feature_select'] }, { tower: 'g', ids: ['proj_fc1'] });
    if (c.projector_type !== 'linear') seq.push({ tower: 'g', ids: ['proj_act'] }, { tower: 'g', ids: ['proj_fc2'] });
    seq.push({ tower: 'g', ids: ['text_embed'] }, { tower: 'g', ids: ['assemble'] });
    for (let l = 0; l < c.num_layers; l++) {
      for (const g of LGROUPS) seq.push({ tower: 'l', l, ids: g.map((k) => `L${l}.${k}`) });
    }
    seq.push({ tower: 'g', ids: ['final_norm'] }, { tower: 'g', ids: ['lm_head'] }, { tower: 'g', ids: ['loss'] });
    return seq;
  }

  function stageFrozen(paramName) {
    const t = V.STAGES[$('#stage').value].train;
    if (paramName.startsWith('vision_tower')) return !t.vision;
    if (paramName.startsWith('multi_modal_projector')) return !t.projector;
    return !t.llm;
  }

  // ================================================================ 설정 UI
  function buildConfigUI() {
    $('#preset').innerHTML = Object.entries(PRESETS).map(([k, p]) => `<option value="${k}">${esc(p.label)}</option>`).join('') +
      '<option value="custom">사용자 정의</option>';
    const fieldHTML = (f) => `<label class="field${f.primary ? ' primary' : ''}">
        <span>${esc(f.label || f.key)}</span>
        <input type="number" id="f_${f.key}" value="${f.value}" step="${f.step || (f.float ? 'any' : 1)}" min="0">
      </label>`;
    $('#visionFields').innerHTML = FIELDS.filter((f) => f.group === 'vision').map(fieldHTML).join('');
    $('#llmFields').innerHTML = FIELDS.filter((f) => f.group === 'llm').map(fieldHTML).join('');

    for (const f of FIELDS) {
      $('#f_' + f.key).addEventListener('input', () => { $('#preset').value = 'custom'; scheduleUpdate(); });
    }
    for (const s of SELECTS) $('#' + s).addEventListener('change', () => { $('#preset').value = 'custom'; update(); });
    $('#tie_word_embeddings').addEventListener('change', () => { $('#preset').value = 'custom'; update(); });
    $('#preset').addEventListener('change', () => { if ($('#preset').value !== 'custom') applyPreset($('#preset').value); });
    $('#promptText').addEventListener('input', scheduleUpdate);
    $('#answerText').addEventListener('input', scheduleUpdate);
    $('#showSaliency').addEventListener('change', renderImagePanel);
    $('#stage').addEventListener('change', () => { renderParams(); renderDiagram(); renderInspector(); });

    $('#imgPreset').innerHTML = Object.entries(IMG_PRESETS).map(([k, v]) => `<option value="${k}">${esc(v[0])}</option>`).join('');
    $('#imgPreset').addEventListener('change', () => {
      state.imgSource = { type: 'preset', name: $('#imgPreset').value };
      update();
    });
    $('#imgFile').addEventListener('change', (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const url = URL.createObjectURL(file);
      const im = new Image();
      im.onload = () => {
        const c = document.createElement('canvas');
        const side = Math.min(im.width, im.height);           // 가운데를 정사각형으로 자른다
        c.width = side; c.height = side;
        c.getContext('2d').drawImage(im, (im.width - side) / 2, (im.height - side) / 2, side, side, 0, 0, side, side);
        state.imgSource = { type: 'file', canvas: c, name: file.name };
        URL.revokeObjectURL(url);
        update();
      };
      im.src = url;
    });

    $('#trainSteps').addEventListener('input', () => {
      const n = $('#trainN');
      if (n) n.textContent = Math.max(1, Math.floor(+$('#trainSteps').value || 1));
    });
    $('#btnForward').addEventListener('click', onForward);
    $('#btnBackward').addEventListener('click', onBackward);
    $('#btnUpdate').addEventListener('click', onUpdate);
    $('#btnTrain').addEventListener('click', onTrain);
    $('#btnReset').addEventListener('click', () => { state.sig = null; update(); });
  }

  function applyPreset(k) {
    const p = PRESETS[k];
    for (const f of FIELDS) if (p[f.key] !== undefined) $('#f_' + f.key).value = p[f.key];
    for (const s of SELECTS) if (p[s] !== undefined) $('#' + s).value = p[s];
    $('#tie_word_embeddings').checked = !!p.tie_word_embeddings;
    $('#preset').value = k;
    update();
  }

  function readConfig() {
    const c = {};
    for (const f of FIELDS) {
      const el = $('#f_' + f.key);
      const v = el.value.trim() === '' ? NaN : Number(el.value);
      c[f.key] = v;
      el.classList.toggle('invalid', !Number.isFinite(v) || v <= 0 || (!f.float && !Number.isInteger(v)));
    }
    c.vision_feature_layer = +$('#vision_feature_layer').value;
    c.feature_strategy = $('#feature_strategy').value;
    c.projector_type = $('#projector_type').value;
    c.tie_word_embeddings = $('#tie_word_embeddings').checked;
    return c;
  }

  let timer = null;
  function scheduleUpdate() { clearTimeout(timer); timer = setTimeout(update, 250); }

  // ================================================================ 토큰 / 시퀀스
  function buildText(c, nImg) {
    const raw = $('#promptText').value;
    const marker = '<image>';
    const i = raw.indexOf(marker);
    const before = i >= 0 ? raw.slice(0, i) : raw;
    const afterPrompt = i >= 0 ? raw.slice(i + marker.length) : '';
    if (i < 0) state.warns.push('프롬프트에 <image> 가 없어 이미지 토큰을 맨 앞에 넣었습니다.');
    const answer = $('#answerText').value;

    const tok = state.tok;
    const pre = tok.encode(before);
    const postPrompt = tok.encode(afterPrompt);
    const ansIds = tok.encode(answer);
    const post = postPrompt.concat(ansIds);
    const T = pre.length + nImg + post.length;
    const ansStart = pre.length + nImg + postPrompt.length;

    const chars = Array.from(before).concat(new Array(nImg).fill(null), Array.from(afterPrompt), Array.from(answer));
    const idAt = (t) => (t < pre.length ? pre[t] : t < pre.length + nImg ? -1 : post[t - pre.length - nImg]);
    const targets = new Array(T).fill(-1);
    for (let t = Math.max(0, ansStart - 1); t < T - 1; t++) targets[t] = idAt(t + 1);

    const segs = new Array(T);
    for (let t = 0; t < T; t++) {
      segs[t] = t < pre.length ? 'pre' : t < pre.length + nImg ? 'image' : t < ansStart ? 'post' : 'answer';
    }
    return { pre, post, targets, T, ansStart, chars, segs, idAt, nAns: targets.filter((x) => x >= 0).length };
  }

  const tokenLabel = (t) => {
    const ch = state.text.chars[t];
    return ch === null ? `${t} ▣` : `${t} ${vis(ch)}`;
  };

  // ================================================================ 갱신
  function update() {
    if (state.busy) { scheduleUpdate(); return; }
    const c = readConfig();
    state.cfg = c;
    state.warns = [];
    state.errors = V.validateConfig(c);
    const valid = state.errors.length === 0;
    if (!valid) {
      state.specs = null; state.computeOK = false;
      renderParams(null); renderMessages(); renderDiagram(); renderInspector(); renderStatus();
      return;
    }

    renderImagePixels();
    if (!state.tok || state.tok.V !== c.vocab_size) state.tok = new M.CharTokenizer(c.vocab_size);

    const N = V.numPatches(c);
    const nImg = c.feature_strategy === 'cls_patch' ? N + 1 : N;
    state.text = buildText(c, nImg);

    const pc = V.countParams(c);
    renderParams(pc);

    const dims = {
      S: c.image_size, P: c.patch_size, N, Np: N + 1, Dv: c.vision_hidden,
      Iv: Math.round(c.vision_hidden * c.vision_mlp_ratio), hdv: c.vision_hidden / c.vision_heads,
      nImg, Ttext: state.text.T - nImg, T: state.text.T, d: c.hidden_dim, V: c.vocab_size,
    };
    state.specs = buildSpecs(c, dims);
    state.dims = dims;

    state.computeOK = true;
    if (state.text.nAns === 0) { state.warns.push('정답(answer) 텍스트가 비어 있어 loss 를 계산할 수 없습니다.'); state.computeOK = false; }
    if (state.text.T > MAX_T) { state.warns.push(`시퀀스가 너무 깁니다 (T=${state.text.T} > ${MAX_T}). shape 만 표시합니다.`); state.computeOK = false; }
    if (pc.total > COMPUTE_LIMIT) { state.warns.push(`파라미터가 ${short(pc.total)} 개라 브라우저 계산을 끄고 shape / 파라미터 수만 표시합니다 (한도 ${short(COMPUTE_LIMIT)}).`); state.computeOK = false; }
    if (state.tok.chars.length > c.vocab_size) state.warns.push(`고유 문자 수(${state.tok.chars.length})가 vocab_size 보다 많아 id 가 겹칩니다.`);

    const sig = ARCH_KEYS.map((k) => c[k]).join('|');
    if (state.computeOK) {
      if (sig !== state.sig || !state.params) {
        state.params = V.initLlava(c, c.seed | 0);
        state.adam = new M.Adam(+$('#lr').value);
        state.history = [];
        state.sig = sig;
      }
    } else {
      state.params = null; state.adam = null; state.sig = null; state.history = [];
    }

    if (state.vlayer >= c.vision_layers) state.vlayer = 0;
    if (state.layer >= c.num_layers) state.layer = 0;
    if (state.head >= c.num_heads) state.head = 0;
    if (state.vhead >= c.vision_heads) state.vhead = 0;
    if (state.patchSel >= N) state.patchSel = 0;
    if (!specOf(state.selected)) state.selected = 'image';

    state.fw = null; state.phase = 'none'; state.status = {};
    if (state.computeOK) runForward();

    renderImagePanel();
    renderSeqStrip();
    renderMessages();
    renderDiagram();
    renderInspector();
    renderStatus();
    drawLossChart();
  }

  function runForward() {
    state.fw = V.forward(state.cfg, state.params, state.image, state.text);
    state.phase = 'fwd';
    state.status = {};
    for (const g of sequence()) for (const id of g.ids) state.status[id] = 'fwd';
  }

  // ================================================================ 실행 버튼
  function setBusy(b) {
    state.busy = b;
    state.skip = false;
    $('#btnForward').textContent = b ? '⏭ 건너뛰기' : '▶ Forward';
    $('#btnBackward').textContent = b ? '⏭ 건너뛰기' : '◀ Backward';
    renderStatus();
  }

  async function animate(groups, dir) {
    state.activeDir = dir;
    const delay = () => 640 - +$('#speed').value;
    for (const g of groups) {
      if (state.skip) { for (const gg of groups) for (const id of gg.ids) state.status[id] = dir; break; }
      let moved = false;
      if (g.tower === 'v' && g.l !== state.vlayer) { state.vlayer = g.l; moved = true; }
      if (g.tower === 'l' && g.l !== state.layer) { state.layer = g.l; moved = true; }
      if (moved) renderDiagram();
      state.active = new Set(g.ids);
      if ($('#follow').checked) { state.selected = g.ids[0]; renderInspector(); }
      updateNodes();
      await sleep(delay());
      for (const id of g.ids) state.status[id] = dir;
    }
    state.active = new Set();
    state.activeDir = null;
    renderDiagram();
    renderInspector();
  }

  async function onForward() {
    if (state.busy) { state.skip = true; return; }
    if (!state.specs) return;
    setBusy(true);
    if (state.computeOK) runForward();
    state.status = {};
    renderDiagram();
    await animate(sequence(), 'fwd');
    setBusy(false);
  }

  async function onBackward() {
    if (state.busy) { state.skip = true; return; }
    if (!state.specs) return;
    setBusy(true);
    if (state.computeOK) {
      if (!state.fw || state.fw.backwardDone) runForward();
      V.backward(state.fw, state.params, { image: state.image });
      state.phase = 'bwd';
      if (state.view === 'act') state.view = 'grad';
      renderImagePanel();
    }
    await animate(sequence().filter((g) => g.ids[0] !== 'image').reverse(), 'bwd');
    setBusy(false);
  }

  function onUpdate() {
    if (state.busy || !state.computeOK || state.phase !== 'bwd') return;
    state.adam.lr = +$('#lr').value;
    state.history.push(state.fw.loss);
    state.adam.step(V.trainableList(state.params, $('#stage').value));
    runForward();
    if (state.view === 'grad' || state.view === 'wgrad') state.view = 'act';
    renderDiagram(); renderInspector(); renderStatus(); drawLossChart();
  }

  async function onTrain() {
    if (state.busy || !state.computeOK) return;
    const n = Math.max(1, Math.floor(+$('#trainSteps').value || 1));
    setBusy(true);
    const btn = $('#btnTrain');
    const list = V.trainableList(state.params, $('#stage').value);
    let last = performance.now();
    for (let i = 0; i < n; i++) {
      const fw = V.forward(state.cfg, state.params, state.image, state.text);
      V.backward(fw, state.params, { image: state.image });
      state.adam.lr = +$('#lr').value;
      state.adam.step(list);
      state.history.push(fw.loss);
      if (performance.now() - last > 60 || i === n - 1) {
        btn.textContent = `학습 중 ${i + 1}/${n}`;
        $('#lossVal').textContent = fw.loss.toFixed(4);
        drawLossChart();
        await frame();
        last = performance.now();
      }
    }
    btn.innerHTML = `학습 <span id="trainN">${n}</span> step`;
    runForward();
    if (state.view === 'grad' || state.view === 'wgrad') state.view = 'act';
    setBusy(false);
    renderDiagram(); renderInspector(); drawLossChart();
  }

  function renderStatus() {
    const ok = state.computeOK;
    $('#btnForward').disabled = !state.specs;
    $('#btnBackward').disabled = !state.specs;
    $('#btnUpdate').disabled = state.busy || !ok || state.phase !== 'bwd';
    $('#btnTrain').disabled = state.busy || !ok;
    $('#btnReset').disabled = state.busy || !ok;
    $('#lossVal').textContent = state.fw ? state.fw.loss.toFixed(4) : '—';
    $('#phaseVal').textContent = !ok ? 'shape 전용 모드'
      : state.phase === 'bwd' ? '기울기 계산됨'
        : state.text ? `answer ${state.text.nAns} 토큰 지도` : '';
  }

  // ================================================================ 사이드바
  function renderImagePanel() {
    if (!state.imgRGB) return;
    const c = state.cfg, S = c.image_size, P = c.patch_size, G = S / P, N = G * G;
    drawImageTo($('#imgCanvas'), { saliency: $('#showSaliency').checked && state.phase === 'bwd' });
    const grid = $('#patchGrid');
    grid.style.gridTemplateColumns = `repeat(${G}, 1fr)`;
    grid.classList.toggle('hide-labels', G > 12);
    grid.innerHTML = Array.from({ length: N }, (_, i) =>
      `<div class="patch-cell${i === state.patchSel ? ' on' : ''}" data-patch="${i}" title="패치 ${i} (row ${Math.floor(i / G)}, col ${i % G})">${i}</div>`).join('');
    grid.querySelectorAll('.patch-cell').forEach((el) => el.addEventListener('click', () => {
      state.patchSel = +el.dataset.patch;
      renderImagePanel();
      renderInspector();
    }));
    $('#imgInfo').innerHTML = `${S}×${S}×3 → ${P}×${P} 패치 <b>${N}</b>개 (${G}×${G})` +
      (state.imgSource.type === 'file' ? ` · ${esc(state.imgSource.name)}` : '') +
      ($('#showSaliency').checked && state.phase === 'bwd' ? ' · 빨강 = |∂L/∂pixel| 큼' : '');
  }

  function renderSeqStrip() {
    const t = state.text;
    const strip = $('#seqStrip');
    const max = 240;
    strip.innerHTML = t.segs.slice(0, max).map((seg, i) => {
      const ch = t.chars[i];
      const sup = t.targets[i] >= 0 ? ' sup' : '';
      const label = ch === null ? '▣' : esc(vis(ch));
      return `<span class="seq-cell ${seg}${sup}${state.pos === i ? ' on' : ''}" data-pos="${i}" title="t=${i} · ${seg}${t.targets[i] >= 0 ? ' · 다음 토큰 지도' : ''}">${label}</span>`;
    }).join('') + (t.T > max ? `<span class="muted">…+${t.T - max}</span>` : '');
    strip.querySelectorAll('.seq-cell').forEach((el) => el.addEventListener('click', () => {
      state.pos = +el.dataset.pos;
      renderSeqStrip();
      renderInspector();
    }));
    $('#seqLegend').innerHTML =
      `<span><i style="background:var(--k-input)"></i>프롬프트</span>
       <span><i style="background:var(--k-image)"></i>이미지 토큰 ${state.dims.nImg}</span>
       <span><i style="background:var(--k-text)"></i>프롬프트(뒤)</span>
       <span><i style="background:var(--accent)"></i>정답</span>
       <span><i style="background:var(--bwd)"></i>아래 밑줄 = loss 지도 위치 (${t.nAns})</span>
       <span class="mono">T = ${t.T}</span>`;
  }

  function renderMessages() {
    $('#messages').innerHTML =
      state.errors.map((e) => `<div class="msg err">${esc(e)}</div>`).join('') +
      state.warns.map((w) => `<div class="msg warn">${esc(w)}</div>`).join('');
  }

  function renderParams(pc) {
    // 인자 없이 부르면 (학습 단계 변경 등) 현재 설정으로 다시 계산한다
    if (!pc && state.cfg && !state.errors.length) pc = V.countParams(state.cfg);
    if (!pc) {
      $('#totalParams').textContent = '—';
      $('#totalShort').textContent = '';
      $('#paramBreakdown').innerHTML = '<div class="muted">설정값을 확인하세요.</div>';
      return;
    }
    const c = state.cfg;
    $('#totalParams').textContent = fmt(pc.total);
    $('#totalShort').textContent = `(${short(pc.total)})`;
    const stage = V.STAGES[$('#stage').value].train;
    const rows = [
      ['Vision tower', pc.visionTotal, 'var(--k-vision)', stage.vision],
      ['Projector', pc.projTotal, 'var(--k-proj)', stage.projector],
      ['Language model', pc.llmTotal, 'var(--k-linear)', stage.llm],
    ];
    const trainable = rows.filter((r) => r[3]).reduce((s, r) => s + r[1], 0);
    $('#paramBreakdown').innerHTML = rows.map(([name, v, col, tr]) => `
      <div class="pb-row">
        <span>${name}${tr ? '' : ' 🔒'}</span>
        <span class="pb-bar"><i style="width:${(100 * v / pc.total).toFixed(2)}%;background:${col};opacity:${tr ? 1 : 0.4}"></i></span>
        <span class="pb-val" title="${fmt(v)}">${short(v)}</span>
      </div>`).join('') +
      `<div class="pb-extra">이번 단계에서 학습되는 파라미터: <b>${short(trainable)}</b> (${(100 * trainable / pc.total).toFixed(1)}%) · 🔒 = 동결</div>
       <div class="pb-extra">이미지 토큰 ${fmt(state.dims ? state.dims.nImg : V.numPatches(c))}개 = 패치 ${fmt(V.numPatches(c))}개${c.feature_strategy === 'cls_patch' ? ' + CLS' : ''}</div>
       <div class="pb-extra">메모리: fp32 ${bytes(pc.total * 4)} · bf16 ${bytes(pc.total * 2)}</div>
       <div class="pb-formula">vision = 3P²·Dv + Dv + (N+1)·Dv + Lv·[4(Dv²+Dv) + 2Dv·Iv + Iv + Dv + 4Dv] + 4Dv
projector = Dv·d + d${c.projector_type === 'linear' ? '' : ' + d² + d'}
llm = V·d + L·[d·hd·(2H+2KV) + 3·d·I + 2d] + d${c.tie_word_embeddings ? '' : ' + V·d'}</div>`;
  }

  function drawLossChart() {
    const cv = $('#lossChart');
    const w = cv.clientWidth || 260, h = 110, dpr = window.devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr;
    const ctx = cv.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    const cs = getComputedStyle(document.documentElement);
    ctx.font = '11px ' + cs.getPropertyValue('--mono');
    ctx.fillStyle = cs.getPropertyValue('--muted');
    const hist = state.history;
    if (hist.length < 2) { ctx.fillText('학습 loss 곡선 (Adam step 후 표시)', 8, h / 2 + 4); return; }
    const pad = { l: 8, r: 8, t: 18, b: 8 };
    const max = Math.max(...hist), min = Math.min(...hist), span = max - min || 1;
    ctx.fillText(`loss ${hist[hist.length - 1].toFixed(4)}  (step ${hist.length})`, pad.l, 12);
    ctx.strokeStyle = cs.getPropertyValue('--accent');
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    hist.forEach((v, i) => {
      const x = pad.l + (i / (hist.length - 1)) * (w - pad.l - pad.r);
      const y = pad.t + (1 - (v - min) / span) * (h - pad.t - pad.b);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }

  // ================================================================ 다이어그램
  function renderDiagram() {
    const el = $('#diagram');
    if (!state.specs) { el.innerHTML = '<div class="placeholder">설정 오류를 수정하면 구조가 표시됩니다.</div>'; return; }
    const c = state.cfg, sp = state.specs, d = state.dims;

    const card = (id) => {
      const s = specOf(id);
      if (!s) return '';
      const pc = specParamCount(s);
      const view = s.scores ? `<span class="view">scores ${fmtShape(s.scores)}</span>` : '';
      return `<div class="node k-${s.kind}${s.kind === 'add' ? ' add-node' : ''}${s.extra ? ' dim-node' : ''}" data-id="${id}" title="${esc(s.title + ' — ' + fmtShapeL(s.out))}">
        <div class="node-title">${esc(s.title)}</div>
        ${s.kind === 'add' ? '' : `<div class="node-sub">${esc(s.sub)}</div>`}
        <div class="node-shape">${fmtShape(s.out)}${view}</div>
        <div class="node-meta"><span class="pc">${pc ? short(pc) + ' params' : ''}${s.extra ? ' · 미사용' : ''}</span><span class="gn"></span></div>
      </div>`;
    };
    const row = (...ids) => `<div class="frow">${ids.map(card).join('')}</div>`;
    const arrow = (dims, note) => `<div class="arrow"><span>${dims ? fmtShape(dims) : ''}${note ? ' ' + note : ''}</span></div>`;
    const tabs = (n, cur, kind) => Array.from({ length: n }, (_, i) =>
      `<button type="button" data-tab="${kind}" data-i="${i}" class="${i === cur ? 'on' : ''}">${i}</button>`).join('');

    const vl = state.vlayer, ll = state.layer;
    const VP = (k) => `V${vl}.${k}`, LP = (k) => `L${ll}.${k}`;
    const stage = V.STAGES[$('#stage').value].train;

    el.innerHTML = [
      row('image'), arrow(sp.g.image.out),
      row('patchify'), arrow(sp.g.patchify.out),
      `<div class="tower vision">
        <div class="tower-head"><b>Vision tower · CLIP ViT ${stage.vision ? '' : '🔒'}</b><span class="muted">${c.vision_layers} layers · Dv=${c.vision_hidden} · ${c.vision_heads} heads</span></div>
        ${row('patch_embed')}${arrow(sp.g.patch_embed.out)}
        ${row('cls_cat')}${arrow(sp.g.cls_cat.out)}
        ${row('pos_add')}${arrow(sp.g.pos_add.out)}
        ${row('ln_pre')}${arrow(sp.g.ln_pre.out)}
        <div class="layer" style="width:100%">
          <div class="layer-head"><b>Encoder Layer ${vl} <span class="muted">/ ${c.vision_layers}개 (× Lv)</span></b><div class="layer-tabs">${tabs(c.vision_layers, vl, 'v')}</div></div>
          <div class="sub"><span class="sub-label">residual</span><div class="sub-title">Self-Attention (마스크 없음)</div>
            ${row(VP('ln1'))}${arrow()}
            ${row(VP('q_proj'), VP('k_proj'), VP('v_proj'))}${arrow()}
            ${row(VP('attn'))}${arrow()}
            ${row(VP('out_proj'))}
          </div>
          ${arrow()}${row(VP('resid1'))}${arrow()}
          <div class="sub"><span class="sub-label">residual</span><div class="sub-title">MLP (QuickGELU)</div>
            ${row(VP('ln2'))}${arrow()}
            ${row(VP('fc1'))}${arrow()}
            ${row(VP('act'))}${arrow()}
            ${row(VP('fc2'))}
          </div>
          ${arrow()}${row(VP('resid2'))}
        </div>
        ${arrow(sp.g.ln_pre.out)}${row('ln_post')}
      </div>`,
      arrow(sp.g.feature_select.ins[0][1], '(hidden_states)'),
      row('feature_select'), arrow(sp.g.feature_select.out),
      `<div class="tower proj">
        <div class="tower-head"><b>Projector ${stage.projector ? '' : '🔒'}</b><span class="muted">${c.projector_type}</span></div>
        ${row('proj_fc1')}${c.projector_type === 'linear' ? '' : arrow() + row('proj_act') + arrow() + row('proj_fc2')}
      </div>`,
      arrow(sp.g.assemble.ins[1][1], '이미지 토큰'),
      row('text_embed', 'assemble'), arrow(sp.g.assemble.out),
      `<div class="tower llm">
        <div class="tower-head"><b>Language model · Llama decoder ${stage.llm ? '' : '🔒'}</b><span class="muted">${c.num_layers} layers · d=${c.hidden_dim} · T=${d.T}</span></div>
        <div class="layer" style="width:100%">
          <div class="layer-head"><b>Decoder Layer ${ll} <span class="muted">/ ${c.num_layers}개 (× L)</span></b><div class="layer-tabs">${tabs(c.num_layers, ll, 'l')}</div></div>
          <div class="sub"><span class="sub-label">residual</span><div class="sub-title">Causal Self-Attention (GQA + RoPE)</div>
            ${row(LP('attn_norm'))}${arrow()}
            ${row(LP('q_proj'), LP('k_proj'), LP('v_proj'))}${arrow()}
            ${row(LP('rope_q'), LP('rope_k'))}${arrow()}
            ${row(LP('attn'))}${arrow()}
            ${row(LP('o_proj'))}
          </div>
          ${arrow()}${row(LP('resid1'))}${arrow()}
          <div class="sub"><span class="sub-label">residual</span><div class="sub-title">MLP (SwiGLU)</div>
            ${row(LP('mlp_norm'))}${arrow()}
            ${row(LP('gate_proj'), LP('up_proj'))}${arrow()}
            ${row(LP('act'), LP('mul'))}${arrow()}
            ${row(LP('down_proj'))}
          </div>
          ${arrow()}${row(LP('resid2'))}
        </div>
      </div>`,
      arrow(sp.g.final_norm.out),
      row('final_norm'), arrow(sp.g.final_norm.out),
      row('lm_head'), arrow(sp.g.lm_head.out),
      row('loss'),
    ].join('');

    el.querySelectorAll('.node[data-id]').forEach((n) => n.addEventListener('click', () => {
      state.selected = n.dataset.id;
      updateNodes();
      renderInspector();
    }));
    el.querySelectorAll('.layer-tabs button').forEach((b) => b.addEventListener('click', () => {
      const i = +b.dataset.i;
      const r = parseId(state.selected);
      if (b.dataset.tab === 'v') { state.vlayer = i; if (r.tower === 'v') state.selected = `V${i}.${r.key}`; }
      else { state.layer = i; if (r.tower === 'l') state.selected = `L${i}.${r.key}`; }
      renderDiagram();
      renderInspector();
    }));
    updateNodes();
  }

  function updateNodes() {
    const fw = state.fw;
    document.querySelectorAll('#diagram .node[data-id]').forEach((n) => {
      const id = n.dataset.id, st = state.status[id], active = state.active.has(id);
      n.classList.toggle('selected', id === state.selected);
      n.classList.toggle('active-fwd', active && state.activeDir === 'fwd');
      n.classList.toggle('active-bwd', active && state.activeDir === 'bwd');
      n.classList.toggle('done-fwd', !active && st === 'fwd');
      n.classList.toggle('done-bwd', !active && st === 'bwd');
      const gn = n.querySelector('.gn');
      let txt = '';
      if (fw) {
        const t = fw.acts[id];
        if (id === 'loss' && st) txt = 'L = ' + fw.loss.toFixed(3);
        else if (st === 'bwd' && t && t.grad) txt = '‖∇‖ ' + num(M.tensorStats(t.grad).norm);
        else if (st === 'bwd' && t && !t.grad) txt = '∇ 없음';
      }
      gn.textContent = txt;
    });
  }

  // ================================================================ inspector
  function renderInspector() {
    const el = $('#inspector');
    if (!state.specs) { el.innerHTML = '<div class="placeholder">설정 오류를 수정하세요.</div>'; return; }
    const id = state.selected, r = parseId(id), s = specOf(id);
    if (!s) { el.innerHTML = '<div class="placeholder">모듈을 선택하세요.</div>'; return; }
    const title = r.tower === 'v' ? `vision.layers[${r.l}].${s.title}` : r.tower === 'l' ? `llm.layers[${r.l}].${s.title}` : s.title;
    const fw = state.fw;

    let html = `<div class="insp-head k-${s.kind}"><h2>${esc(title)}</h2><span class="badge">${esc(s.sub)}</span></div>`;
    if (s.extra) html += `<div class="msg warn" style="margin:8px 0">${esc(s.extra)}</div>`;
    html += `<p class="desc">${esc(s.desc)}</p>
      <h3>Forward</h3><div class="formula fwd-f">${esc(s.fwd)}</div>`;
    if (s.bwd) html += `<h3>Backward</h3><div class="formula bwd-f">${esc(s.bwd)}</div>`;

    html += '<h3>Shape</h3><table class="kv"><tr><th>텐서</th><th>shape</th></tr>';
    for (const [n, dd] of s.ins || []) html += `<tr><td>입력 ${esc(n)}</td><td class="mono">${fmtShapeL(dd)}</td></tr>`;
    if (s.scores) html += `<tr><td>attention scores</td><td class="mono">${fmtShapeL(s.scores)}</td></tr>`;
    html += `<tr><td><b>출력</b></td><td class="mono"><b>${fmtShapeL(s.out)}</b></td></tr></table>`;

    if (s.params && s.params.length) {
      html += '<h3>파라미터</h3><table class="kv"><tr><th>이름</th><th>shape</th><th class="r">개수</th><th class="r">‖W‖</th><th class="r">‖∇W‖</th></tr>';
      for (const p of s.params) {
        const t = state.params ? p.get(state.params, r.l) : null;
        const cnt = p.tied ? 0 : p.shape.reduce((a, b) => a * b, 1);
        const frozen = stageFrozen(p.name);
        html += `<tr><td class="mono">${esc(p.name.replace('{l}', r.l))}${frozen ? ' 🔒' : ''}</td>
          <td class="mono">[${p.shape.join(', ')}]</td><td class="mono r">${fmt(cnt)}</td>
          <td class="mono r">${t ? num(M.tensorStats(t.data).norm) : '—'}</td>
          <td class="mono r">${t && t.grad && state.phase === 'bwd' ? num(M.tensorStats(t.grad).norm) : '—'}</td></tr>`;
      }
      html += '</table>';
    }

    html += '<h3>텐서 보기</h3><div id="tv"></div>';
    if (s.kind === 'attn') html += `<h3>Attention</h3><div id="attnView"></div>`;
    if (r.key === 'image' || r.key === 'patchify') html += '<h3>이미지</h3><div id="imgView"></div>';
    if (r.key === 'lm_head' || r.key === 'loss') html += '<h3>다음 토큰 예측</h3><div id="predView"></div>';
    if (r.key === 'assemble') html += '<h3>시퀀스 구성</h3><div id="asmView"></div>';
    el.innerHTML = html;

    if (r.key === 'image' || r.key === 'patchify') renderImageView($('#imgView'));
    const tv = $('#tv');
    if (!state.computeOK || !fw) {
      tv.innerHTML = `<div class="placeholder">${state.computeOK ? 'Forward 를 실행하세요.' : 'shape 전용 모드입니다. 실제 값을 보려면 모델/이미지를 작게 줄이세요.'}</div>`;
      return;
    }
    renderTensorView(tv, id, r, s);
    if (s.kind === 'attn') (r.tower === 'v' ? renderVisionAttention : renderLlmAttention)($('#attnView'), r.l);
    if (r.key === 'lm_head' || r.key === 'loss') $('#predView').innerHTML = predTable();
    if (r.key === 'assemble') renderAssembleView($('#asmView'));
  }

  function renderImageView(el) {
    const wrap = document.createElement('div');
    wrap.className = 'two-col';
    const a = document.createElement('div');
    const cv = document.createElement('canvas');
    cv.className = 'img-view';
    a.appendChild(cv);
    a.insertAdjacentHTML('beforeend', `<div class="mini-note">입력 이미지 (${state.cfg.image_size}×${state.cfg.image_size})</div>`);
    wrap.appendChild(a);
    el.appendChild(wrap);
    drawImageTo(cv, {});
    if (state.phase === 'bwd' && state.image.grad) {
      const b = document.createElement('div');
      const cv2 = document.createElement('canvas');
      cv2.className = 'img-view';
      b.appendChild(cv2);
      b.insertAdjacentHTML('beforeend', '<div class="mini-note">∂L/∂image saliency (빨강 = 답에 큰 영향)</div>');
      wrap.appendChild(b);
      drawImageTo(cv2, { saliency: true });
      el.insertAdjacentHTML('beforeend', statLine(state.image.grad));
    }
  }

  function renderTensorView(tv, id, r, s) {
    const fw = state.fw, act = fw.acts[id];
    const hasParam = s.params && s.params.length && state.params && s.params[0].get(state.params, r.l);
    const views = [['act', '출력 활성값 y', !!act], ['grad', '∂L/∂y', !!(act && act.grad && state.phase === 'bwd')]];
    if (hasParam) views.push(['w', '가중치 W', true], ['wgrad', '∂L/∂W', state.phase === 'bwd' && !!s.params[0].get(state.params, r.l).grad]);
    if (!views.some((v) => v[0] === state.view)) state.view = 'act';

    const tabs = document.createElement('div');
    tabs.className = 'tabs';
    for (const [k, label, ok] of views) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = label;
      b.className = (state.view === k ? 'on' : '') + (k.includes('grad') ? ' g' : '');
      b.addEventListener('click', () => { state.view = k; renderInspector(); });
      if (!ok) b.title = 'Backward 를 실행하면 볼 수 있습니다';
      tabs.appendChild(b);
    }
    tv.appendChild(tabs);

    const cur = views.find((v) => v[0] === state.view);
    if (!cur[2]) {
      tv.insertAdjacentHTML('beforeend',
        `<div class="placeholder">${state.phase === 'bwd' ? '이 텐서에는 기울기가 흐르지 않습니다 (선택되지 않은 층이거나 동결된 부분).' : '◀ Backward 를 실행하면 기울기가 표시됩니다.'}</div>`);
      return;
    }

    if (r.key === 'loss' && (state.view === 'act' || state.view === 'grad')) {
      const v = state.view === 'act' ? act.data[0] : act.grad[0];
      tv.insertAdjacentHTML('beforeend', `<div class="formula" style="font-size:18px;text-align:center">${state.view === 'act' ? 'L' : '∂L/∂L'} = ${num(v)}</div>
        <div class="muted small-text">지도 위치 N = ${fw.count} (answer 토큰) · 균등분포 기준 ln V = ${Math.log(state.cfg.vocab_size).toFixed(4)}</div>`);
      return;
    }

    const c = state.cfg;
    let data, rows, cols, rl, cap, group = 0;
    if (state.view === 'act' || state.view === 'grad') {
      data = state.view === 'act' ? act.data : act.grad;
      rows = act.rows; cols = act.cols;
      // 행 라벨: vision tower 는 패치 번호, LLM 은 토큰
      if (r.tower === 'v' || ['ln_pre', 'ln_post', 'patch_embed', 'cls_cat', 'pos_add'].includes(r.key)) {
        rl = Array.from({ length: rows }, (_, i) => (rows === state.dims.Np ? (i === 0 ? 'CLS' : 'p' + (i - 1)) : 'p' + i));
        if (r.tower === 'v' && ['q_proj', 'k_proj', 'v_proj', 'attn'].includes(r.key)) group = state.dims.hdv;
      } else if (rows === state.dims.T) {
        rl = Array.from({ length: rows }, (_, i) => tokenLabel(i));
        if (['q_proj', 'k_proj', 'v_proj', 'rope_q', 'rope_k', 'attn'].includes(r.key)) group = c.head_dim;
      } else if (rows === state.dims.nImg) {
        rl = Array.from({ length: rows }, (_, i) => '이미지 ' + i);
      } else {
        rl = Array.from({ length: rows }, (_, i) => String(i));
      }
      cap = state.view === 'act' ? `y (행 = 토큰/패치, 열 = 채널${group ? ', 세로선 = head 경계' : ''})` : '∂L/∂y';
    } else {
      const p = s.params[0], t = p.get(state.params, r.l);
      data = state.view === 'w' ? t.data : t.grad;
      rows = t.rows; cols = t.cols;
      rl = rows === 1 ? ['γ/b'] : null;
      if (p.name.includes('position_embedding')) rl = Array.from({ length: rows }, (_, i) => (i === 0 ? 'CLS' : 'p' + (i - 1)));
      if (p.name.includes('embed_tokens') || p.name.includes('lm_head')) {
        rl = Array.from({ length: rows }, (_, i) => `${i} ${state.tok.decode(i).map(vis).join('/') || '·'}`);
      }
      cap = `${state.view === 'w' ? 'W' : '∂L/∂W'} [${rows} × ${cols}]`;
    }
    tv.appendChild(heatmap({ data, rows, cols, rowLabels: rl, colGroup: group, caption: cap }));
    tv.insertAdjacentHTML('beforeend', statLine(data));
  }

  function renderVisionAttention(el, l) {
    const c = state.cfg, Np = state.dims.Np, G = c.image_size / c.patch_size;
    const P = state.fw.visionAttn[l];
    const heads = document.createElement('div');
    heads.className = 'heads';
    for (let h = 0; h < c.vision_heads; h++) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = 'h' + h;
      b.className = h === state.vhead ? 'on' : '';
      b.addEventListener('click', () => { state.vhead = h; renderInspector(); });
      heads.appendChild(b);
    }
    el.appendChild(heads);

    const h = state.vhead;
    const labels = Array.from({ length: Np }, (_, i) => (i === 0 ? 'CLS' : 'p' + (i - 1)));
    const qRow = state.patchSel + 1; // 0 = CLS

    // 선택한 패치가 어디를 보는지 이미지 위에 겹쳐 그린다
    const wrap = document.createElement('div');
    wrap.innerHTML = `<div class="mini-note">패치 <b>${state.patchSel}</b> (query) 가 보는 곳 — 왼쪽 이미지에서 패치를 클릭해 바꿀 수 있습니다.</div>`;
    const ov = document.createElement('div');
    ov.className = 'overlay-wrap';
    const base = document.createElement('canvas'), over = document.createElement('canvas');
    over.className = 'ov';
    ov.appendChild(base); ov.appendChild(over);
    wrap.appendChild(ov);
    el.appendChild(wrap);
    drawImageTo(base, {});
    const row = P.subarray((h * Np + qRow) * Np, (h * Np + qRow + 1) * Np);
    drawPatchOverlay(over, Array.from(row.subarray(1)));      // CLS 제외하고 패치 격자에 매핑
    el.insertAdjacentHTML('beforeend',
      `<div class="mini-note">CLS 토큰에 준 attention: ${(row[0] * 100).toFixed(1)}% · 격자 ${G}×${G}</div>`);

    el.appendChild(heatmap({
      data: P.subarray(h * Np * Np, (h + 1) * Np * Np), rows: Np, cols: Np, mode: 'seq',
      rowLabels: labels, colLabels: labels,
      caption: `head ${h} attention (행 = query, 열 = key · causal mask 없음)`,
      onCell: (rr) => { if (rr > 0) { state.patchSel = rr - 1; renderImagePanel(); renderInspector(); } },
    }));
  }

  function renderLlmAttention(el, l) {
    const c = state.cfg, T = state.dims.T, grp = c.num_heads / c.num_kv_heads;
    const P = state.fw.llmAttn[l];
    const heads = document.createElement('div');
    heads.className = 'heads';
    for (let h = 0; h < c.num_heads; h++) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = 'h' + h;
      b.className = h === state.head ? 'on' : '';
      b.addEventListener('click', () => { state.head = h; renderInspector(); });
      heads.appendChild(b);
    }
    el.appendChild(heads);
    const h = state.head;
    const labels = Array.from({ length: T }, (_, i) => tokenLabel(i));

    // 선택한 위치가 이미지 토큰에 준 attention 비율
    const pos = state.pos !== null && state.pos < T ? state.pos : T - 1;
    let imgMass = 0, total = 0;
    for (let j = 0; j <= pos; j++) {
      const v = P[(h * T + pos) * T + j];
      total += v;
      if (state.text.segs[j] === 'image') imgMass += v;
    }
    el.insertAdjacentHTML('beforeend',
      `<div class="mini-note">위치 <b>t=${pos}</b> (${esc(state.text.chars[pos] === null ? '이미지 토큰' : vis(state.text.chars[pos]))}) 의 attention 중 <b>${(100 * imgMass / (total || 1)).toFixed(1)}%</b> 가 이미지 토큰으로 갑니다. 왼쪽 시퀀스 막대를 클릭해 위치를 바꿀 수 있습니다.</div>`);

    el.appendChild(heatmap({
      data: P.subarray(h * T * T, (h + 1) * T * T), rows: T, cols: T, mode: 'seq',
      rowLabels: labels, colLabels: labels, masked: (r, cc) => cc > r,
      caption: `head ${h} → KV head ${Math.floor(h / grp)} (▣ = 이미지 토큰, 빗금 = causal mask)`,
      onCell: (rr) => { state.pos = rr; renderSeqStrip(); renderInspector(); },
    }));
  }

  function renderAssembleView(el) {
    const t = state.text, d = state.dims;
    el.innerHTML = `<table class="kv">
      <tr><th>구간</th><th class="r">토큰 수</th><th>내용</th></tr>
      <tr><td>프롬프트 앞</td><td class="mono r">${t.pre.length}</td><td class="mono">${esc(Array.from(t.chars.slice(0, t.pre.length)).map(vis).join(''))}</td></tr>
      <tr><td>이미지 토큰</td><td class="mono r">${d.nImg}</td><td class="mono">projector 출력 (▣)</td></tr>
      <tr><td>프롬프트 뒤</td><td class="mono r">${t.ansStart - t.pre.length - d.nImg}</td><td class="mono">${esc(t.chars.slice(t.pre.length + d.nImg, t.ansStart).map(vis).join(''))}</td></tr>
      <tr><td>정답 (지도 구간)</td><td class="mono r">${t.T - t.ansStart}</td><td class="mono">${esc(t.chars.slice(t.ansStart).map(vis).join(''))}</td></tr>
      <tr><td><b>합계 T</b></td><td class="mono r"><b>${t.T}</b></td><td class="mono">loss 지도 위치 ${t.nAns}개</td></tr>
    </table>`;
  }

  function predTable() {
    const Vsz = state.cfg.vocab_size, P = state.fw.probs, t = state.text;
    let html = '<table class="kv"><tr><th>t</th><th>구간</th><th>정답</th><th class="r">p(정답)</th><th>top-3</th></tr>';
    for (let i = 0; i < t.T; i++) {
      if (t.targets[i] < 0 && t.segs[i] !== 'answer') continue;   // 지도되는 구간 위주로 표시
      const row = Array.from(P.subarray(i * Vsz, (i + 1) * Vsz)).map((p, k) => [k, p]).sort((a, b) => b[1] - a[1]).slice(0, 3);
      const tg = t.targets[i];
      const top = row.map(([k, p]) => `<span class="${tg >= 0 && k === tg ? 'pred-ok' : ''}">${esc(state.tok.decode(k).map(vis).join('/') || '·')}:${(p * 100).toFixed(1)}%</span>`).join(' ');
      const ok = tg >= 0 && row[0][0] === tg;
      html += `<tr><td class="mono">${i}</td><td class="mono">${t.segs[i]}</td>
        <td class="mono">${tg >= 0 ? esc(vis(t.chars[i + 1] === null ? '▣' : t.chars[i + 1])) : '—'}</td>
        <td class="mono r ${tg >= 0 ? (ok ? 'pred-ok' : 'pred-bad') : ''}">${tg >= 0 ? (P[i * Vsz + tg] * 100).toFixed(2) + '%' : '—'}</td>
        <td class="mono">${top}</td></tr>`;
    }
    return html + '</table>';
  }

  // ================================================================ 시작
  buildConfigUI();
  applyPreset('tiny');
  window.addEventListener('resize', drawLossChart);
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { renderInspector(); drawLossChart(); renderImagePanel(); });
})();
