(function () {
  'use strict';

  const M = window.TModel;
  const { $, esc, fmt, vis, sleep, frame, short, num, bytes, heatmap, hmColors, tooltipEl } = window.TUI;
  const statLine = (arr) => window.TUI.statLine(arr, M.tensorStats);

  // 브라우저에서 실제 연산을 돌릴 수 있는 한계. 넘으면 shape / 파라미터 수만 보여준다.
  const COMPUTE_LIMIT = 3_000_000;
  const MAX_T = 128;

  const PRESETS = {
    tiny: { label: 'Tiny (기본, 실시간 계산)', vocab_size: 64, hidden_dim: 32, num_layers: 2, num_heads: 4, num_kv_heads: 2, head_dim: 8, intermediate_size: 88, rope_theta: 10000, tie_word_embeddings: false },
    small: { label: 'Small (~0.76M, 계산 가능)', vocab_size: 128, hidden_dim: 128, num_layers: 4, num_heads: 8, num_kv_heads: 4, head_dim: 16, intermediate_size: 344, rope_theta: 10000, tie_word_embeddings: false },
    llama32_1b: { label: 'Llama-3.2-1B (shape 전용)', vocab_size: 128256, hidden_dim: 2048, num_layers: 16, num_heads: 32, num_kv_heads: 8, head_dim: 64, intermediate_size: 8192, rope_theta: 500000, tie_word_embeddings: true },
    mistral_7b: { label: 'Mistral-7B (shape 전용)', vocab_size: 32000, hidden_dim: 4096, num_layers: 32, num_heads: 32, num_kv_heads: 8, head_dim: 128, intermediate_size: 14336, rope_theta: 10000, tie_word_embeddings: false },
    llama3_8b: { label: 'Llama-3-8B (shape 전용)', vocab_size: 128256, hidden_dim: 4096, num_layers: 32, num_heads: 32, num_kv_heads: 8, head_dim: 128, intermediate_size: 14336, rope_theta: 500000, tie_word_embeddings: false },
  };

  const FIELDS = [
    { key: 'hidden_dim', label: 'hidden_dim (d)', value: 32, primary: true },
    { key: 'head_dim', label: 'head_dim (hd)', value: 8, step: 2, primary: true },
    { key: 'num_heads', label: 'num_heads (H)', value: 4 },
    { key: 'num_kv_heads', label: 'num_kv_heads (KV)', value: 2 },
    { key: 'num_layers', label: 'num_layers (L)', value: 2 },
    { key: 'intermediate_size', label: 'intermediate (I)', value: 88 },
    { key: 'vocab_size', label: 'vocab_size (V)', value: 64 },
    { key: 'rope_theta', label: 'rope_theta', value: 10000, float: true },
    { key: 'rms_eps', label: 'rms_norm_eps', value: 1e-6, float: true, step: 'any' },
    { key: 'init_std', label: 'init std', value: 0.02, float: true, step: 'any' },
    { key: 'seed', label: 'seed', value: 42 },
  ];
  // 이 값들이 바뀌면 가중치를 새로 만든다.
  const ARCH_KEYS = ['vocab_size', 'hidden_dim', 'num_layers', 'num_heads', 'num_kv_heads', 'head_dim', 'intermediate_size', 'init_std', 'seed', 'tie_word_embeddings'];

  const LAYER_GROUPS = [
    ['attn_norm'], ['q_proj', 'k_proj', 'v_proj'], ['rope_q', 'rope_k'], ['attn'], ['o_proj'], ['resid1'],
    ['mlp_norm'], ['gate_proj', 'up_proj'], ['act'], ['mul'], ['down_proj'], ['resid2'],
  ];

  const state = {
    cfg: null, specs: null, tok: null, chars: [], ids: [], targets: [], T: 0,
    params: null, sig: null, adam: null, fw: null, phase: 'none',
    status: {}, active: new Set(), activeDir: null,
    selected: 'embed', layer: 0, view: 'act', head: 0,
    history: [], computeOK: false, errors: [], warns: [], busy: false, skip: false,
  };

  // ================================================================ 모듈 명세 (shape, 파라미터, 수식)
  const LIN_BWD = '∂L/∂x = ∂L/∂y · W\n∂L/∂W = (∂L/∂y)ᵀ · x';
  const NORM_FWD = 'rms = √(mean(x²) + ε)\ny = (x / rms) ⊙ γ';
  const NORM_BWD = 'x̂ = x / rms,  ĝ = ∂L/∂y ⊙ γ\n∂L/∂x = (ĝ − x̂ · mean(ĝ ⊙ x̂)) / rms\n∂L/∂γ = Σ_t ∂L/∂y ⊙ x̂';
  const ROPE_FWD = 'θ_i = base^(−2i/hd),  i = 0 … hd/2−1\n[x₁, x₂] → [x₁·cos(pθ) − x₂·sin(pθ),\n            x₂·cos(pθ) + x₁·sin(pθ)]';
  const ROPE_BWD = '회전 행렬은 직교행렬 → 역전파는 역회전 R(−pθ)\n∂L/∂x₁ = d₁·cos + d₂·sin\n∂L/∂x₂ = −d₁·sin + d₂·cos';
  const ADD_BWD = '∂L/∂x = ∂L/∂y,  ∂L/∂f(x) = ∂L/∂y\n(기울기가 두 경로로 그대로 복사됨)';

  function buildSpecs(c, T) {
    const V = c.vocab_size, D = c.hidden_dim, H = c.num_heads, KV = c.num_kv_heads, hd = c.head_dim, I = c.intermediate_size;
    const B = ['B', 1], Tt = ['T', T];
    const BTD = [B, Tt, ['d', D]];
    const LP = 'model.layers.{l}.';
    const norm = (title, pname, ref, desc) => ({
      kind: 'norm', title, sub: `RMSNorm(d=${D})`, ins: [['x', BTD]], out: BTD,
      params: [{ name: pname, shape: [D], ...ref }], desc, fwd: NORM_FWD, bwd: NORM_BWD,
    });
    const lin = (title, inName, inDims, outDims, lp, pname, desc, view) => {
      const iN = inDims[2], oN = outDims[2];
      return {
        kind: 'linear', title, sub: `Linear ${iN[0]} → ${oN[0]}`, ins: [[inName, inDims]], out: outDims, view,
        params: [{ name: LP + pname, shape: [oN[1], iN[1]], lp }], desc,
        fwd: `y = x · Wᵀ\nW ∈ ℝ^(${oN[1]}×${iN[1]})   [${oN[0]} × ${iN[0]}]`, bwd: LIN_BWD,
      };
    };
    const BTq = [B, Tt, ['H·hd', H * hd]], BTk = [B, Tt, ['KV·hd', KV * hd]], BTI = [B, Tt, ['I', I]];
    const qView = [B, ['H', H], Tt, ['hd', hd]], kView = [B, ['KV', KV], Tt, ['hd', hd]];

    return {
      input: {
        kind: 'input', title: 'input_ids', sub: '문자 단위 토큰 id', out: [B, Tt],
        desc: '입력 문자열을 문자 단위로 토큰화한 정수 id 시퀀스입니다. 위치 t 의 정답(target)은 t+1 번째 토큰이며(next-token prediction), 마지막 위치는 loss 에서 제외됩니다.',
        fwd: 'text → [id₀, id₁, …, id_{T−1}]', bwd: null,
      },
      embed: {
        kind: 'embed', title: 'embed_tokens', sub: `Embedding(V=${V}, d=${D})`, ins: [['input_ids', [B, Tt]]], out: BTD,
        params: [{ name: 'model.embed_tokens.weight', shape: [V, D], p: 'embed' }],
        desc: '각 토큰 id 를 임베딩 테이블 E 의 행으로 바꿔 d 차원 벡터를 만듭니다. Llama 계열은 여기서 위치 임베딩을 더하지 않고, attention 내부에서 RoPE 로 위치 정보를 넣습니다.',
        fwd: 'x_t = E[id_t],   E ∈ ℝ^(V×d)', bwd: '∂L/∂E[id_t] += ∂L/∂x_t   (scatter-add)\n같은 토큰이 여러 번 나오면 기울기가 누적됨',
      },
      attn_norm: norm('input_layernorm', LP + 'input_layernorm.weight', { lp: 'attn_norm' },
        'Pre-Norm: attention 앞에 RMSNorm 을 둡니다. LayerNorm 과 달리 평균을 빼지 않고 RMS 로만 나누며 bias 가 없습니다. 학습 파라미터는 γ (d 개) 뿐입니다.'),
      q_proj: lin('q_proj', 'x', BTD, BTq, 'q_proj', 'self_attn.q_proj.weight',
        'Query 투영. 결과를 H 개 head 로 쪼개 [B, H, T, hd] 로 봅니다. hidden_dim 과 H·hd 가 같을 필요는 없습니다 (o_proj 가 다시 d 로 맞춤).', qView),
      k_proj: lin('k_proj', 'x', BTD, BTk, 'k_proj', 'self_attn.k_proj.weight',
        'Key 투영. GQA 에서는 KV head 수가 Q head 수보다 적어 파라미터와 KV cache 가 H/KV 배 줄어듭니다.', kView),
      v_proj: lin('v_proj', 'x', BTD, BTk, 'v_proj', 'self_attn.v_proj.weight',
        'Value 투영. Value 에는 RoPE 를 적용하지 않습니다.', kView),
      rope_q: {
        kind: 'rope', title: 'RoPE(Q)', sub: `rotary, θ=${c.rope_theta}`, ins: [['q', qView]], out: qView,
        desc: 'Rotary Position Embedding: head 벡터의 앞/뒤 절반을 짝지어 위치 p 에 비례한 각도로 회전합니다. 회전 후 q·k 내적은 상대 위치 (p − p′) 에만 의존합니다. 파라미터는 없습니다.',
        fwd: ROPE_FWD, bwd: ROPE_BWD,
      },
      rope_k: {
        kind: 'rope', title: 'RoPE(K)', sub: `rotary, θ=${c.rope_theta}`, ins: [['k', kView]], out: kView,
        desc: 'Key 에도 같은 위치 회전을 적용합니다. 파라미터는 없습니다.', fwd: ROPE_FWD, bwd: ROPE_BWD,
      },
      attn: {
        kind: 'attn', title: 'causal attention', sub: KV === H ? 'MHA' : KV === 1 ? 'MQA (KV=1)' : `GQA (${H / KV} Q heads / KV head)`,
        ins: [['Q', qView], ['K', kView], ['V', kView]], scores: [B, ['H', H], Tt, ['T', T]], out: BTq,
        desc: 'Causal scaled dot-product attention. 미래 토큰은 마스킹되고, Q head 들은 H/KV 개씩 묶여 같은 KV head 를 공유합니다 (GQA). 출력 head 들을 이어붙여 [B, T, H·hd] 로 되돌립니다.',
        fwd: 'S = Q·Kᵀ / √hd + M_causal     [B, H, T, T]\nA = softmax(S)\nO = A · V  →  concat heads',
        bwd: 'dA = dO · Vᵀ,   dV = Aᵀ · dO\ndS = A ⊙ (dA − rowsum(dA ⊙ A))\ndQ = dS · K / √hd,   dK = dSᵀ · Q / √hd\n(GQA: 공유 KV head 로 기울기 합산)',
      },
      o_proj: lin('o_proj', 'attn_out', BTq, BTD, 'o_proj', 'self_attn.o_proj.weight',
        'head 들을 이어붙인 attention 출력을 hidden 차원 d 로 다시 투영합니다.'),
      resid1: {
        kind: 'add', title: '⊕ residual', sub: 'x + Attn(Norm(x))', ins: [['x', BTD], ['attn', BTD]], out: BTD,
        desc: '잔차 연결. attention 블록 출력을 블록 입력에 더합니다. backward 에서는 기울기가 두 경로로 그대로 복사되어 깊은 모델에서도 기울기가 잘 흐릅니다.',
        fwd: 'x ← x + o_proj(Attn(RMSNorm(x)))', bwd: ADD_BWD,
      },
      mlp_norm: norm('post_attention_layernorm', LP + 'post_attention_layernorm.weight', { lp: 'mlp_norm' },
        'MLP 앞에 두는 RMSNorm 입니다.'),
      gate_proj: lin('gate_proj', 'x', BTD, BTI, 'gate_proj', 'mlp.gate_proj.weight', 'SwiGLU 의 gate 경로. d → I 로 확장합니다.'),
      up_proj: lin('up_proj', 'x', BTD, BTI, 'up_proj', 'mlp.up_proj.weight', 'SwiGLU 의 up 경로. d → I 로 확장합니다.'),
      act: {
        kind: 'act', title: 'SiLU(gate)', sub: 'Swish 활성함수', ins: [['gate', BTI]], out: BTI,
        desc: 'gate 경로에 SiLU(Swish) 활성함수를 적용합니다. 파라미터는 없습니다.',
        fwd: 'SiLU(g) = g · σ(g)', bwd: '∂SiLU/∂g = σ(g) · (1 + g · (1 − σ(g)))',
      },
      mul: {
        kind: 'act', title: '⊙ gate · up', sub: 'SwiGLU', ins: [['SiLU(gate)', BTI], ['up', BTI]], out: BTI,
        desc: 'SiLU(gate) 와 up 을 원소별로 곱합니다 (gated linear unit). gate 가 up 경로의 정보를 얼마나 통과시킬지 결정합니다.',
        fwd: 'h = SiLU(g) ⊙ u', bwd: '∂L/∂SiLU(g) = ∂L/∂h ⊙ u\n∂L/∂u = ∂L/∂h ⊙ SiLU(g)',
      },
      down_proj: lin('down_proj', 'h', BTI, BTD, 'down_proj', 'mlp.down_proj.weight', 'I → d 로 다시 축소합니다.'),
      resid2: {
        kind: 'add', title: '⊕ residual', sub: 'x + MLP(Norm(x))', ins: [['x', BTD], ['mlp', BTD]], out: BTD,
        desc: 'MLP 블록의 잔차 연결입니다.', fwd: 'x ← x + down(SiLU(gate(x̃)) ⊙ up(x̃)),  x̃ = RMSNorm(x)', bwd: ADD_BWD,
      },
      final_norm: norm('norm', 'model.norm.weight', { p: 'final_norm' }, '마지막 decoder layer 뒤에 적용하는 최종 RMSNorm 입니다.'),
      lm_head: {
        kind: 'head', title: 'lm_head', sub: c.tie_word_embeddings ? `Linear d → V (tied)` : 'Linear d → V',
        ins: [['x', BTD]], out: [B, Tt, ['V', V]],
        params: [{ name: c.tie_word_embeddings ? 'lm_head.weight (= embed_tokens, 공유)' : 'lm_head.weight', shape: [V, D], p: 'lm_head', tied: !!c.tie_word_embeddings }],
        desc: 'hidden 벡터를 vocab 크기의 logits 로 투영합니다. tie_word_embeddings 를 켜면 embedding 행렬을 그대로 재사용하므로 추가 파라미터가 0 개입니다.',
        fwd: 'logits = x · Wᵀ,   W ∈ ℝ^(V×d)', bwd: LIN_BWD + (c.tie_word_embeddings ? '\n(tied: ∂L/∂E 에 embedding 쪽 기울기와 합산)' : ''),
      },
      loss: {
        kind: 'loss', title: 'softmax + CE loss', sub: 'next-token prediction', ins: [['logits', [B, Tt, ['V', V]]], ['targets', [B, Tt]]], out: [],
        desc: '각 위치의 logits 에 softmax 를 취하고 다음 토큰의 음의 로그확률을 평균냅니다. Backward 는 여기서 ∂L/∂L = 1 로 시작합니다.',
        fwd: 'p_t = softmax(logits_t)\nL = −(1/N) Σ_t log p_t[target_t]', bwd: '∂L/∂logits_t = (p_t − onehot(target_t)) / N',
      },
    };
  }

  const specParamCount = (sp) => (sp.params || []).reduce((s, p) => s + (p.tied ? 0 : p.shape.reduce((a, b) => a * b, 1)), 0);
  const fmtShape = (dims) => (dims.length ? '[' + dims.map((d) => d[1]).join(', ') + ']' : 'scalar');
  const fmtShapeL = (dims) => (dims.length ? '[' + dims.map((d) => `${d[0]}=${d[1]}`).join(', ') + ']' : 'scalar []');
  const baseKey = (id) => id.replace(/^L\d+\./, '');
  const layerOf = (id) => { const m = /^L(\d+)\./.exec(id); return m ? +m[1] : null; };
  const idFor = (key, l) => (LAYER_GROUPS.some((g) => g.includes(key)) ? `L${l}.${key}` : key);

  function paramTensor(ref, id) {
    if (!state.params) return null;
    if (ref.lp) return state.params.layers[layerOf(id)][ref.lp];
    return state.params[ref.p];
  }

  function sequence() {
    const seq = [{ layer: null, ids: ['input'] }, { layer: null, ids: ['embed'] }];
    for (let l = 0; l < state.cfg.num_layers; l++) {
      for (const g of LAYER_GROUPS) seq.push({ layer: l, ids: g.map((k) => `L${l}.${k}`) });
    }
    seq.push({ layer: null, ids: ['final_norm'] }, { layer: null, ids: ['lm_head'] }, { layer: null, ids: ['loss'] });
    return seq;
  }

  // ================================================================ 설정 UI
  function buildConfigUI() {
    const sel = $('#preset');
    sel.innerHTML = Object.entries(PRESETS).map(([k, p]) => `<option value="${k}">${esc(p.label)}</option>`).join('') +
      '<option value="custom">사용자 정의</option>';
    $('#fields').innerHTML = FIELDS.map((f) => `
      <label class="field${f.primary ? ' primary' : ''}">
        <span>${esc(f.label)}</span>
        <input type="number" id="f_${f.key}" value="${f.value}" step="${f.step || (f.float ? 'any' : 1)}" min="0">
      </label>`).join('');

    sel.addEventListener('change', () => { if (sel.value !== 'custom') applyPreset(sel.value); });
    for (const f of FIELDS) {
      $('#f_' + f.key).addEventListener('input', () => { $('#preset').value = 'custom'; scheduleUpdate(); });
    }
    $('#tie_word_embeddings').addEventListener('change', () => { $('#preset').value = 'custom'; scheduleUpdate(); });
    $('#inputText').addEventListener('input', scheduleUpdate);
    $('#hdAuto').addEventListener('click', () => {
      const d = +$('#f_hidden_dim').value, h = +$('#f_num_heads').value;
      if (d > 0 && h > 0) { let hd = Math.max(2, Math.floor(d / h)); if (hd % 2) hd -= 1; $('#f_head_dim').value = hd; $('#preset').value = 'custom'; update(); }
    });
    $('#iAuto').addEventListener('click', () => {
      const d = +$('#f_hidden_dim').value;
      if (d > 0) { $('#f_intermediate_size').value = Math.max(1, Math.round((8 / 3) * d / 8) * 8); $('#preset').value = 'custom'; update(); }
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
      el.classList.toggle('invalid', !Number.isFinite(v) || v < 0 || (!f.float && !Number.isInteger(v)));
    }
    c.tie_word_embeddings = $('#tie_word_embeddings').checked;
    return c;
  }

  let timer = null;
  function scheduleUpdate() { clearTimeout(timer); timer = setTimeout(update, 200); }

  // ================================================================ 메인 갱신
  function update() {
    if (state.busy) { scheduleUpdate(); return; }
    const c = readConfig();
    state.cfg = c;
    state.errors = M.validateConfig(c);
    state.warns = [];
    const valid = state.errors.length === 0;

    // 토큰화
    const text = $('#inputText').value;
    if (!state.tok || state.tok.V !== c.vocab_size) state.tok = valid ? new M.CharTokenizer(c.vocab_size) : null;
    state.chars = Array.from(text);
    state.ids = state.tok ? state.tok.encode(text) : state.chars.map(() => 0);
    state.targets = state.ids.slice(1).concat([-1]);
    state.T = state.ids.length;

    const pc = valid ? M.countParams(c) : null;
    renderParams(pc);

    state.computeOK = false;
    if (valid) {
      if (state.T < 2) state.warns.push('forward/backward 를 계산하려면 입력이 2글자 이상이어야 합니다.');
      else if (state.T > MAX_T) state.warns.push(`입력이 너무 깁니다 (${state.T} > ${MAX_T}). shape 만 표시합니다.`);
      else if (pc.total > COMPUTE_LIMIT) state.warns.push(`파라미터가 ${short(pc.total)} 개라 브라우저 계산은 끄고 shape / 파라미터 수만 표시합니다 (한도 ${short(COMPUTE_LIMIT)}).`);
      else state.computeOK = true;
      if (state.tok && state.tok.chars.length > c.vocab_size) {
        state.warns.push(`지금까지 본 고유 문자 수(${state.tok.chars.length})가 vocab_size(${c.vocab_size})보다 많아 일부 id 가 겹칩니다.`);
      }
    }

    // 가중치 (구조가 바뀐 경우에만 새로 초기화)
    const sig = ARCH_KEYS.map((k) => c[k]).join('|');
    if (state.computeOK) {
      if (sig !== state.sig || !state.params) {
        state.params = M.initParams(c, c.seed | 0);
        state.adam = new M.Adam(+$('#lr').value);
        state.history = [];
        state.sig = sig;
      }
    } else {
      state.params = null; state.adam = null; state.sig = null; state.history = [];
    }

    state.specs = valid ? buildSpecs(c, state.T) : null;
    if (valid) {
      if (state.layer >= c.num_layers) state.layer = 0;
      const sl = layerOf(state.selected);
      if (sl !== null && sl >= c.num_layers) state.selected = 'embed';
      if (state.head >= c.num_heads) state.head = 0;
    }

    state.fw = null; state.phase = 'none'; state.status = {};
    if (state.computeOK) runForward();

    renderTokens();
    renderMessages();
    renderDiagram();
    renderInspector();
    renderStatus();
    drawLossChart();
  }

  function runForward() {
    state.fw = M.forward(state.cfg, state.params, state.ids, state.targets);
    state.phase = 'fwd';
    state.status = {};
    for (const g of sequence()) for (const id of g.ids) state.status[id] = 'fwd';
  }

  // ================================================================ 버튼 동작
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
      if (g.layer !== null && g.layer !== state.layer) { state.layer = g.layer; renderDiagram(); }
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
      M.backward(state.fw, state.params);
      state.phase = 'bwd';
      if (state.view === 'act') state.view = 'grad';
    }
    const seq = sequence().filter((g) => g.ids[0] !== 'input').reverse();
    await animate(seq, 'bwd');
    setBusy(false);
  }

  function onUpdate() {
    if (state.busy || !state.computeOK || state.phase !== 'bwd') return;
    state.adam.lr = +$('#lr').value;
    state.history.push(state.fw.loss);
    state.adam.step(M.paramList(state.params));
    runForward();
    if (state.view === 'grad' || state.view === 'wgrad') state.view = 'act';
    renderDiagram(); renderInspector(); renderStatus(); drawLossChart();
  }

  async function onTrain() {
    if (state.busy || !state.computeOK) return;
    const n = Math.max(1, Math.floor(+$('#trainSteps').value || 1));
    setBusy(true);
    const btn = $('#btnTrain');
    const list = M.paramList(state.params);
    let last = performance.now();
    for (let i = 0; i < n; i++) {
      const fw = M.forward(state.cfg, state.params, state.ids, state.targets);
      M.backward(fw, state.params);
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
    $('#phaseVal').textContent = !ok ? 'shape 전용 모드' :
      state.phase === 'bwd' ? '기울기 계산됨' : state.phase === 'fwd' ? `ln V = ${Math.log(state.cfg.vocab_size).toFixed(3)}` : '';
  }

  // ================================================================ 사이드바 렌더링
  function renderTokens() {
    const max = 160;
    $('#tokenPreview').innerHTML = state.chars.slice(0, max).map((ch, t) =>
      `<span class="chip" title="t=${t}">${esc(vis(ch))}<small>${state.ids[t]}</small></span>`).join('') +
      (state.chars.length > max ? `<span class="muted"> … +${state.chars.length - max}</span>` : '') +
      `<span class="muted" style="align-self:center;margin-left:4px">T = ${state.T}</span>`;
  }

  function renderMessages() {
    $('#messages').innerHTML =
      state.errors.map((e) => `<div class="msg err">${esc(e)}</div>`).join('') +
      state.warns.map((w) => `<div class="msg warn">${esc(w)}</div>`).join('');
  }

  function renderParams(pc) {
    if (!pc) {
      $('#totalParams').textContent = '—';
      $('#totalShort').textContent = '';
      $('#paramBreakdown').innerHTML = '<div class="muted">설정값을 확인하세요.</div>';
      return;
    }
    const c = state.cfg;
    $('#totalParams').textContent = fmt(pc.total);
    $('#totalShort').textContent = `(${short(pc.total)})`;
    const g = pc.groups;
    const rows = [
      ['Embedding', g.embed, 'var(--k-embed)'],
      ['Attention', g.attn, 'var(--k-attn)'],
      ['MLP', g.mlp, 'var(--k-act)'],
      ['RMSNorm', g.norm, 'var(--k-norm)'],
      ['LM head', g.lm_head, 'var(--k-head)'],
    ];
    const H = c.num_heads, KV = c.num_kv_heads;
    $('#paramBreakdown').innerHTML = rows.map(([name, v, col]) => `
      <div class="pb-row">
        <span>${name}</span>
        <span class="pb-bar"><i style="width:${(100 * v / pc.total).toFixed(2)}%;background:${col}"></i></span>
        <span class="pb-val" title="${fmt(v)}">${short(v)}</span>
      </div>`).join('') +
      `<div class="pb-extra">레이어 1개 = ${fmt(pc.perLayer)} × L(${c.num_layers})</div>
       <div class="pb-extra">메모리: fp32 ${bytes(pc.total * 4)} · bf16 ${bytes(pc.total * 2)}</div>
       <div class="pb-formula">V·d + L·[d·hd·(2H + 2KV) + 3·d·I + 2d] + d${c.tie_word_embeddings ? '' : ' + V·d'}
= ${fmt(c.vocab_size)}·${c.hidden_dim} + ${c.num_layers}·[${c.hidden_dim}·${c.head_dim}·(${2 * H}+${2 * KV}) + 3·${c.hidden_dim}·${c.intermediate_size} + ${2 * c.hidden_dim}] + ${c.hidden_dim}${c.tie_word_embeddings ? '' : ` + ${fmt(c.vocab_size)}·${c.hidden_dim}`}</div>`;
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
    if (hist.length < 2) {
      ctx.fillText('학습 loss 곡선 (Adam step 후 표시)', 8, h / 2 + 4);
      return;
    }
    const pad = { l: 8, r: 8, t: 18, b: 8 };
    const max = Math.max(...hist), min = Math.min(...hist);
    const span = max - min || 1;
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
    const sp = state.specs;
    if (!sp) { el.innerHTML = '<div class="placeholder">설정 오류를 수정하면 구조가 표시됩니다.</div>'; return; }
    const c = state.cfg, l = state.layer;
    const card = (key) => {
      const s = sp[key], id = idFor(key, l), pc = specParamCount(s);
      const view = s.view ? `<span class="view">view → ${fmtShape(s.view)}</span>` :
        s.scores ? `<span class="view">scores ${fmtShape(s.scores)}</span>` : '';
      return `<div class="node k-${s.kind}${s.kind === 'add' ? ' add-node' : ''}" data-id="${id}" title="${esc(s.title + ' — ' + fmtShapeL(s.out))}">
        <div class="node-title">${esc(s.title)}</div>
        ${s.kind === 'add' ? '' : `<div class="node-sub">${esc(s.sub)}</div>`}
        <div class="node-shape">${fmtShape(s.out)}${view}</div>
        <div class="node-meta"><span class="pc">${pc ? short(pc) + ' params' : ''}</span><span class="gn"></span></div>
      </div>`;
    };
    const ghost = (t) => `<div class="node ghost"><div class="node-title">${esc(t)}</div></div>`;
    const row = (...items) => `<div class="frow">${items.join('')}</div>`;
    const arrow = (dims) => `<div class="arrow"><span>${dims ? fmtShape(dims) : ''}</span></div>`;
    const BTD = sp.embed.out;

    const tabs = Array.from({ length: c.num_layers }, (_, i) =>
      `<button type="button" data-layer="${i}" class="${i === l ? 'on' : ''}">${i}</button>`).join('');

    el.innerHTML = [
      row(card('input')), arrow(sp.input.out),
      row(card('embed')), arrow(BTD),
      `<div class="layer">
        <div class="layer-head"><b>Decoder Layer ${l} <span class="muted">/ ${c.num_layers} 개 (× L)</span></b><div class="layer-tabs">${tabs}</div></div>
        <div class="sub"><span class="sub-label">residual</span><div class="sub-title">Self-Attention</div>
          ${row(card('attn_norm'))}${arrow(BTD)}
          ${row(card('q_proj'), card('k_proj'), card('v_proj'))}${arrow()}
          ${row(card('rope_q'), card('rope_k'), ghost('V (RoPE 없음)'))}${arrow()}
          ${row(card('attn'))}${arrow(sp.attn.out)}
          ${row(card('o_proj'))}
        </div>
        ${arrow(BTD)}${row(card('resid1'))}${arrow(BTD)}
        <div class="sub"><span class="sub-label">residual</span><div class="sub-title">MLP (SwiGLU)</div>
          ${row(card('mlp_norm'))}${arrow(BTD)}
          ${row(card('gate_proj'), card('up_proj'))}${arrow()}
          ${row(card('act'), ghost('up'))}${arrow()}
          ${row(card('mul'))}${arrow(sp.mul.out)}
          ${row(card('down_proj'))}
        </div>
        ${arrow(BTD)}${row(card('resid2'))}
      </div>`,
      arrow(BTD),
      row(card('final_norm')), arrow(BTD),
      row(card('lm_head')), arrow(sp.lm_head.out),
      row(card('loss')),
    ].join('');

    el.querySelectorAll('.node[data-id]').forEach((n) => n.addEventListener('click', () => {
      state.selected = n.dataset.id;
      updateNodes();
      renderInspector();
    }));
    el.querySelectorAll('.layer-tabs button').forEach((b) => b.addEventListener('click', () => {
      state.layer = +b.dataset.layer;
      const k = baseKey(state.selected);
      if (layerOf(state.selected) !== null) state.selected = `L${state.layer}.${k}`;
      renderDiagram();
      renderInspector();
    }));
    updateNodes();
  }

  function updateNodes() {
    const fw = state.fw;
    document.querySelectorAll('#diagram .node[data-id]').forEach((n) => {
      const id = n.dataset.id, st = state.status[id];
      const active = state.active.has(id);
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
      }
      gn.textContent = txt;
    });
  }


  // ================================================================ inspector
  function renderInspector() {
    const el = $('#inspector');
    const sp = state.specs;
    if (!sp) { el.innerHTML = '<div class="placeholder">설정 오류를 수정하세요.</div>'; return; }
    const id = state.selected, key = baseKey(id), l = layerOf(id), s = sp[key];
    const title = l !== null ? `layers[${l}].${s.title}` : s.title;
    const fw = state.fw;

    let html = `<div class="insp-head k-${s.kind}"><h2>${esc(title)}</h2><span class="badge">${esc(s.sub)}</span></div>
      <p class="desc">${esc(s.desc)}</p>
      <h3>Forward</h3><div class="formula fwd-f">${esc(s.fwd)}</div>`;
    if (s.bwd) html += `<h3>Backward</h3><div class="formula bwd-f">${esc(s.bwd)}</div>`;

    html += '<h3>Shape</h3><table class="kv"><tr><th>텐서</th><th>shape</th></tr>';
    for (const [n, d] of s.ins || []) html += `<tr><td>입력 ${esc(n)}</td><td class="mono">${fmtShapeL(d)}</td></tr>`;
    if (s.scores) html += `<tr><td>attention scores</td><td class="mono">${fmtShapeL(s.scores)}</td></tr>`;
    html += `<tr><td><b>출력</b></td><td class="mono"><b>${fmtShapeL(s.out)}</b></td></tr>`;
    if (s.view) html += `<tr><td>view (head 분리)</td><td class="mono">${fmtShapeL(s.view)}</td></tr>`;
    html += '</table>';

    if (s.params && s.params.length) {
      html += '<h3>파라미터</h3><table class="kv"><tr><th>이름</th><th>shape</th><th class="r">개수</th><th class="r">‖W‖</th><th class="r">‖∇W‖</th></tr>';
      for (const p of s.params) {
        const t = paramTensor(p, id);
        const cnt = p.tied ? 0 : p.shape.reduce((a, b) => a * b, 1);
        html += `<tr><td class="mono">${esc(p.name.replace('{l}', l))}</td><td class="mono">[${p.shape.join(', ')}]</td>
          <td class="mono r">${fmt(cnt)}</td>
          <td class="mono r">${t ? num(M.tensorStats(t.data).norm) : '—'}</td>
          <td class="mono r">${t && t.grad && state.phase === 'bwd' ? num(M.tensorStats(t.grad).norm) : '—'}</td></tr>`;
      }
      html += '</table>';
    }

    html += '<h3>텐서 보기</h3><div id="tv"></div>';
    if (key === 'attn') html += '<h3>Attention 확률 A = softmax(QKᵀ/√hd)</h3><div id="attnView"></div>';
    if (key === 'lm_head' || key === 'loss') html += '<h3>다음 토큰 예측</h3><div id="predView"></div>';
    el.innerHTML = html;

    const tv = $('#tv');
    if (!state.computeOK || !fw) {
      tv.innerHTML = `<div class="placeholder">${state.computeOK ? 'Forward 를 실행하세요.' : '이 설정은 shape 전용 모드입니다. 실제 값을 보려면 모델을 작게 줄이세요.'}</div>`;
      if (key === 'input') tv.innerHTML = tokenTable();
      return;
    }
    if (key === 'input') { tv.innerHTML = tokenTable(); return; }
    renderTensorView(tv, id, key, s);
    if (key === 'attn') renderAttention($('#attnView'), l);
    if (key === 'lm_head' || key === 'loss') $('#predView').innerHTML = predTable();
  }

  function rowLabels() { return state.chars.map((ch, t) => `${t} ${vis(ch)}`); }
  function vocabLabel(id) { const hits = state.tok.decode(id); return `${id} ${hits.length ? hits.map(vis).join('/') : '·'}`; }

  function renderTensorView(tv, id, key, s) {
    const fw = state.fw, act = fw.acts[id];
    const hasParam = s.params && s.params.length;
    const views = [
      ['act', '출력 활성값 y', !!act],
      ['grad', '∂L/∂y', !!(act && act.grad && state.phase === 'bwd')],
    ];
    if (hasParam) views.push(['w', '가중치 W', true], ['wgrad', '∂L/∂W', state.phase === 'bwd']);
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
      tv.insertAdjacentHTML('beforeend', '<div class="placeholder">◀ Backward 를 실행하면 기울기가 표시됩니다.</div>');
      return;
    }

    const c = state.cfg;
    const headCols = ['q_proj', 'k_proj', 'v_proj', 'rope_q', 'rope_k', 'attn'].includes(key) ? c.head_dim : 0;

    if (key === 'loss' && (state.view === 'act' || state.view === 'grad')) {
      const v = state.view === 'act' ? act.data[0] : act.grad[0];
      tv.insertAdjacentHTML('beforeend', `<div class="formula" style="font-size:18px;text-align:center">${state.view === 'act' ? 'L' : '∂L/∂L'} = ${num(v)}</div>
        <div class="muted small-text">평균을 낸 위치 수 N = ${fw.count} · 균등 분포 기준 ln V = ${Math.log(c.vocab_size).toFixed(4)}</div>`);
      return;
    }

    let data, rows, cols, rl, cap, group = 0;
    if (state.view === 'act' || state.view === 'grad') {
      data = state.view === 'act' ? act.data : act.grad;
      rows = act.rows; cols = act.cols; rl = rowLabels(); group = headCols;
      cap = state.view === 'act' ? `y  (행 = 토큰 위치 t, 열 = 채널${group ? ', 세로선 = head 경계' : ''})` : '∂L/∂y  (이 모듈 출력에 대한 기울기)';
    } else {
      const p = s.params[0];
      const t = paramTensor(p, id);
      data = state.view === 'w' ? t.data : t.grad;
      rows = t.rows; cols = t.cols;
      const vocabRows = p.p === 'embed' || p.p === 'lm_head';
      rl = vocabRows ? Array.from({ length: rows }, (_, i) => vocabLabel(i)) : rows === 1 ? ['γ'] : null;
      cap = `${state.view === 'w' ? 'W' : '∂L/∂W'}  [${rows} × ${cols}]${vocabRows ? '  (행 = vocab id)' : ''}`;
      if (!data) { tv.insertAdjacentHTML('beforeend', '<div class="placeholder">기울기가 없습니다.</div>'); return; }
    }
    tv.appendChild(heatmap({ data, rows, cols, rowLabels: rl, colGroup: group, caption: cap }));
    tv.insertAdjacentHTML('beforeend', statLine(data));
  }

  function renderAttention(el, l) {
    const c = state.cfg, T = state.T, H = c.num_heads, grp = H / c.num_kv_heads;
    const P = state.fw.attnProbs[l];
    const heads = document.createElement('div');
    heads.className = 'heads';
    for (let h = 0; h < H; h++) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = 'h' + h;
      b.className = h === state.head ? 'on' : '';
      b.addEventListener('click', () => { state.head = h; renderInspector(); });
      heads.appendChild(b);
    }
    el.appendChild(heads);
    const h = state.head;
    const labels = rowLabels();
    el.appendChild(heatmap({
      data: P.subarray(h * T * T, (h + 1) * T * T), rows: T, cols: T, mode: 'seq',
      rowLabels: labels, colLabels: labels, masked: (r, col) => col > r,
      caption: `Q head ${h} → KV head ${Math.floor(h / grp)}   (행 = query 위치, 열 = key 위치)`,
    }));
  }

  function tokenTable() {
    let html = '<table class="kv"><tr><th>t</th><th>문자</th><th class="r">id</th><th>target (t+1)</th></tr>';
    state.chars.forEach((ch, t) => {
      const tg = state.targets[t];
      html += `<tr><td class="mono">${t}</td><td class="mono">${esc(vis(ch))}</td><td class="mono r">${state.ids[t]}</td>
        <td class="mono">${tg >= 0 ? `${tg} (${esc(vis(state.chars[t + 1]))})` : '— (loss 제외)'}</td></tr>`;
    });
    return html + '</table>';
  }

  function predTable() {
    const V = state.cfg.vocab_size, P = state.fw.probs;
    let html = '<table class="kv"><tr><th>t</th><th>입력</th><th>정답</th><th class="r">p(정답)</th><th>top-3 예측</th></tr>';
    state.chars.forEach((ch, t) => {
      const row = Array.from(P.subarray(t * V, (t + 1) * V)).map((p, i) => [i, p]).sort((a, b) => b[1] - a[1]).slice(0, 3);
      const tg = state.targets[t];
      const top = row.map(([i, p]) => {
        const lab = state.tok.decode(i).map(vis).join('/') || '·';
        const cls = tg >= 0 ? (i === tg ? 'pred-ok' : '') : '';
        return `<span class="${cls}">${esc(lab)}:${(p * 100).toFixed(1)}%</span>`;
      }).join(' ');
      const ok = tg >= 0 && row[0][0] === tg;
      html += `<tr><td class="mono">${t}</td><td class="mono">${esc(vis(ch))}</td>
        <td class="mono">${tg >= 0 ? esc(vis(state.chars[t + 1])) : '—'}</td>
        <td class="mono r ${tg >= 0 ? (ok ? 'pred-ok' : 'pred-bad') : ''}">${tg >= 0 ? (P[t * V + tg] * 100).toFixed(2) + '%' : '—'}</td>
        <td class="mono">${top}</td></tr>`;
    });
    return html + '</table>';
  }

  // ================================================================ 시작
  buildConfigUI();
  applyPreset('tiny');
  window.addEventListener('resize', drawLossChart);
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { renderInspector(); drawLossChart(); });
})();
