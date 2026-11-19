---
title: "Transformer Visualizer - Llama / LLaVA 구조를 브라우저에서 forward·backward 해보기"
categories: Project
toc: true
toc_sticky: true
---
# Transformer Visualizer

논문이나 코드로 Transformer를 공부하다 보면 "이 텐서가 지금 shape이 뭐지?", "backward 때 기울기는 어디로 흘러가지?" 같은 질문이 계속 생깁니다.
그래서 **Llama 스타일 decoder**와 **LLaVA(비전-언어 모델)** 를 순수 HTML/CSS/JS로 구현하고, 각 연산을 카드로 그려서 forward와 backward를 눈으로 따라갈 수 있는 시각화 도구를 만들었습니다.

- 텍스트 LLM (Llama): [전체 화면으로 열기](/assets/vis/transformer/index.html){:target="_blank"}
- LLaVA (비전): [전체 화면으로 열기](/assets/vis/transformer/llava.html){:target="_blank"}

> 화면이 3단(설정 / 구조도 / inspector)으로 구성되어 있어서 **PC에서 전체 화면으로 여는 것**을 추천합니다.

라이브러리 없이 행렬 연산, autograd(수동 backward), Adam optimizer까지 전부 JS로 직접 작성했기 때문에 서버 없이 브라우저에서 바로 돌아갑니다.

# 1. 텍스트 LLM (Llama)

<iframe src="/assets/vis/transformer/index.html" style="width:100%;height:820px;border:1px solid #ddd;border-radius:6px;" loading="lazy"></iframe>

## 구조
Llama 계열 decoder의 구성 요소를 그대로 따라갑니다.

- **Embedding**: 문자 단위 toy tokenizer로 입력 문장을 토큰 id로 바꾸고 `embed_tokens`로 임베딩
- **RMSNorm**: 각 블록의 attention / MLP 앞에 적용
- **Q / K / V Linear + RoPE**: Q, K에 회전 위치 임베딩(RoPE)을 적용 (`rope_theta` 조절 가능)
- **Attention (GQA)**: `num_heads`와 `num_kv_heads`를 따로 설정할 수 있어서 MHA / GQA / MQA를 모두 확인 가능
- **SwiGLU MLP**: `down(silu(gate(x)) * up(x))`
- **Residual**: attention과 MLP 출력을 잔차로 더함
- **LM head + Cross Entropy**: 다음 토큰 예측(next-token)으로 loss 계산. `tie_word_embeddings` 옵션으로 lm_head와 embedding 가중치 공유 가능

## 기능
- **▶ Forward / ◀ Backward**: 연산 카드를 순서대로 하이라이트하면서 forward와 backward를 애니메이션으로 보여줍니다
- **Inspector**: 카드를 클릭하면 오른쪽에서 해당 연산의 출력 텐서(`y`), 출력 기울기(`∂L/∂y`), 가중치(`W`), 가중치 기울기(`∂L/∂W`)를 히트맵으로 볼 수 있습니다
- **파라미터 구성**: 설정값에 따라 Embedding / Attention / MLP / RMSNorm / LM head 별 파라미터 수를 계산해서 보여줍니다
- **학습**: Adam 1 step 또는 N step 학습을 돌리면서 loss 곡선을 확인할 수 있습니다
- **프리셋**
  - `Tiny`, `Small`: 실제로 forward/backward를 계산
  - `Llama-3.2-1B`, `Mistral-7B`, `Llama-3-8B`: 실제 모델 설정값으로 **shape과 파라미터 수만** 추적 (예: Llama-3-8B → 8,030,261,248개)

`head_dim = d / H`, `I ≈ 8/3 · d` 버튼으로 Llama에서 흔히 쓰는 관계식을 바로 적용해볼 수도 있습니다.

# 2. LLaVA (비전)

<iframe src="/assets/vis/transformer/llava.html" style="width:100%;height:820px;border:1px solid #ddd;border-radius:6px;" loading="lazy"></iframe>

## 구조
LLaVA는 **CLIP ViT vision tower → projector → Llama decoder** 로 이어지는 구조입니다.

- **Vision tower (CLIP ViT)**
  - 이미지를 `patch_size` 단위로 잘라 patch embedding, CLS 토큰, position embedding을 더함
  - `vision_feature_layer`: LLaVA 기본값인 `-2`(뒤에서 두 번째 layer) 또는 `-1` 선택
  - feature 선택: `patch`(CLS 제거) / `cls_patch`(CLS 포함)
- **Projector**
  - `mlp2x_gelu` (LLaVA-1.5): Linear → GELU → Linear
  - `linear` (LLaVA-1.0): Linear 하나
- **Language model (Llama)**: 위의 텍스트 LLM과 같은 decoder

프롬프트의 `<image>` 자리에 이미지 토큰 N개가 들어가고, loss는 **정답(ASSISTANT 응답) 구간에서만** 계산됩니다. 왼쪽 패널의 시퀀스 스트립에서 어떤 토큰이 이미지 / 프롬프트 / 정답인지 색으로 구분해서 볼 수 있습니다.

## 기능
- **이미지 입력**: 샘플 이미지(빨간 사각형, 파란 원, 체커보드, RGB 그라디언트, 도형 얼굴) 또는 직접 업로드한 이미지를 사용하고, patch 격자를 이미지 위에 표시
- **Saliency map**: backward 후 `∂L/∂image`를 이미지 위에 겹쳐서, 어떤 픽셀이 답에 큰 영향을 줬는지 확인
- **학습 단계**: 실제 LLaVA 학습 방식처럼 학습할 파라미터 범위를 선택
  - `Stage 1`: projector만 학습 (feature alignment)
  - `Stage 2`: projector + LLM 학습 (instruction tuning)
  - `전체`: vision tower까지 학습
- **프리셋**: `Tiny`, `Small`은 실제 계산, `LLaVA-1.5-7B`, `LLaVA-1.5-13B`는 shape 전용
  - 예: LLaVA-1.5-7B → vision 303,507,456 + projector 20,979,712 + LLM 6,738,415,616 = **7,062,902,784개**

# 3. 구현 메모
- 모든 연산(embedding, linear, RMSNorm, RoPE, attention, SiLU, 곱, 덧셈, cross entropy, vision tower, projector)에 대해 **backward를 직접 작성**했습니다
- 직접 짠 backward가 맞는지는 **수치 미분(중앙 차분)** 과 비교하는 gradcheck로 검증했습니다. 모든 설정에서 상대 오차가 `1e-7` 수준으로 일치합니다

```
MQA, H·hd ≠ d: OK  loss=3.6755  params=4668  worst rel err=3.82e-8
Llama-3-8B param count: 8,030,261,248 OK
LLaVA-1.5 스타일 (mlp2x_gelu, feature -2): OK  loss=4.2963  N=9  T=45  params=10716  worst rel err=1.14e-7
LLaVA-1.0 스타일 (linear, feature -1, CLS 포함): OK  loss=4.2880  N=9  T=46  params=9932  worst rel err=1.64e-7
LLaVA-1.5-7B: vision 303,507,456 + projector 20,979,712 + LLM 6,738,415,616 = 7,062,902,784 OK
```

- 시스템 다크 모드를 따라 색상이 자동으로 바뀝니다
