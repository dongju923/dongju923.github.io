---
title: "TanoGAN 논문리뷰"
toc: true
categories:
  - Paper
---

# TAnoGAN 논문 리뷰
> 원문 링크: [TAnoGAN: Time Series Anomaly Detection with Generative Adversarial Networks](https://arxiv.org/pdf/2008.09567.pdf)

# Abstract
시계열 데이터의 이상탐지는 제조, 의료 영상 및 사이버 보안 같은 많은 응용분야에서 직면한 중요한 문제이다. 본 논문에서는 적은 수의 데이터 포인트를 사용할 수 있을 때 시계열에서 이상을 탐지하기 위해 TAnoGAN이라는 새로운 GAN기반 비지도 학습방법을 제안한다. 광범위한 실험 결과에 따르면 TAnoGAN은 기존 신경망 모델보다 성능이 우수한 것을 확인했다. 

# Introduction
시계열 데이터의 이상탐지는 일반적으로 레이블 정보의 부족으로 비지도 학습으로 접근한다. 대부분 비지도 이상탐지는 대부분의 시스템이 본질적으로 매우 동적이고 정상적인 측정 범위를 정의하기 어렵기 때문에 실패할 확률이 높다. 따라서 본 논문에서는 TAnoGAN을 사용하여 시계열에서 이상을 탐지하기 위해 적대적 훈련 프로세스를 사용하여 시계열 데이터의 정상을 모델링 한 다음, 데이터 포인트가 정상에서 얼마나 벗어났는지 나타내는 anomaly score를 사용하여 이상을 탐지한다. anomaly score를 학습하기 위해 시계열 데이터 공간을 잠재 공간에 매핑한 다음 잠재 공간에서 데이터를 재구성한다. anomaly score는 실제 데이터와 재구성된 가짜 데이터 간의 손실로 구한다. AnoGAN은 CNN을 사용하여 이미지의 이상을 효과적으로 식별하지만 시계열을 처리하는 메커니즘이 포함되어 있지 않기 때문에 TAnoGAN은 시계열 데이터를 처리하기 위해 Generator와 Discriminator 모델로 LSTM을 사용하였다. 여러 도메인의 각기 다른 46개의 시계열 데이터에 대해 실험한 결과 기존 방법론보다 좋은 성능을 보인다.
> 기존의 AnoGAN과 개념이 비슷하나 시계열 개념이 추가된 느낌인 것 같다

![tanogan_fig1](/assets/images/tanogan/tanogan_fig1.png)
> (a)에서 GAN을 기반으로 하는 정상 데이터의 분포를 학습한다. (b)에서는 정상 데이터를 잠재 공간에 매핑하고 시계열을 재구성한다. 학습 방법은 AnoGAN과 같다.
>> [AnoGAN논문리뷰](https://dongju923.github.io/paper/anogan_review/)

# Model
![tanogan_train](/assets/images/tanogan/tanogan_train.jpg)
> 본 논문에서 G는 32, 64, 128인 staked layer LSTM을 사용했고, D는 hidden units수가 100인 single layer LSTM을 사용하였다.

![tanogan_loss](/assets/images/tanogan/tanogan_loss.jpg)
> loss를 구하는 전체적인 흐름도 이다. 각 loss들은 AnoGAN loss와 같다.
>> [AnoGAN논문리뷰](https://dongju923.github.io/paper/anogan_review/)


# Empirical Evaluation
1. DATA  
Numenta Anomaly Benchmark(NAB)데이터 셋 활용
2. Evaluation
- Accuracy, Precision, Recall, F1-score, Cohen Kappa, Area under ROC curve(AUROC)를 사용
- 총 8개의 모델과 비교
3. Results
![tanogan_results1](/assets/images/tanogan/tanogan_results1.jpg)
거의 모든 평가지표에서 TanoGAN이 높은 것을 확인 할 수 있다.
![tanogan_results2](/assets/images/tanogan/tanogan_results2.jpg)
TAnoGAN과 8개 모델의 pairwise 비교 한 것을 보면 거의 모든 dataset에서 TAnoGAN이 더 좋은 성능을 보이는 것을 알 수 있다,

# 오늘의 정리
1. TAnoGAN은 작은 시계열 데이터를 위한 모델임(큰 사이즈는 MADGAN이 성능이 더 우수한걸로 알고있다.)
2. AnoGAN과 마찬가지로 정상 데이터에 대해 잠재공간으로의 inverse mapping을 통해 Generator로부터 정상 데이터를 재구성한다.

# 참고자료
- [고려대학교 산업경영공학부 DSBA연구실 세미나 영상](https://www.youtube.com/watch?v=WkK52d0RWk8)
