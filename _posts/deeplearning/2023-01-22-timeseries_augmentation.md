---
title: "tsaug를 사용하여 시계열 데이터를 증강해보자!"
categories: DL
toc: true
toc_sticky: true
---

# tsaug 라이브러리
* 시계열 확장을 위한 Python패키지
* 시계열에 대한 augmentation 방법을 제공
* 여러개의 augmentation 방법을 파이프라인에 연결하는 API도 제공
* 시계열 뿐 아니라 2채널 오디오도 augmentation가능

# 환경
* numpy==1.23.5  
* tsaug==0.2.1

> * 본 포스트에서는 오디오에 대한 augmentation은 다루지 않았어요. 자세한 것은 [tsaug 도큐먼트](https://tsaug.readthedocs.io/en/stable/index.html)를 참고해주세요!  
> * 실험에 사용한 npy파일은 [여기](https://github.com/odnura/tsaug/tree/8d539c8246e26c9f365705fac396e4cdcc4b8d8e/docs/notebook)에서 다운받으실 수 있습니다.

### 데이터 로드


```python
import numpy as np
import tsaug
from tsaug.visualization import plot
```


```python
X = np.load("./X.npy")
Y = np.load("./Y.npy")

plot(X, Y);
```


    
![png](/assets/images/timeseries_augmentation/post_5_0.png)
    


### AddNoise
* 시계열에 임의의 노이즈를 추가
* 시계열의 모든 시점에 추가되는 노이즈는 독립적이며 동일하게 분포


```python
X_aug, Y_aug = tsaug.AddNoise(scale=0.02).augment(X, Y)

plot(X_aug, Y_aug);
```


    
![png](/assets/images/timeseries_augmentation/post_7_0.png)
    


### Convolve
* kernel window를 사용하여 시계열을 convolution함


```python
X_aug, Y_aug = tsaug.Convolve(window="flattop", size=11).augment(X, Y)

plot(X_aug, Y_aug);
```


    
![png](/assets/images/timeseries_augmentation/post_9_0.png)
    


### Crop
* 시계열에서 임의의 하위 시퀀스를 자름


```python
X_aug, Y_aug = tsaug.Crop(size=300).augment(X, Y)

plot(X_aug, Y_aug);
```


    
![png](/assets/images/timeseries_augmentation/post_11_0.png)
    


### Drift
* 시계열 값을 원래 값에서 무작위로 매끄럽게 Drift
* Drift의 정도는 최대 Drift와 Drift 지점의 수에 의해 제어됨


```python
X_aug, Y_aug = tsaug.Drift(max_drift=0.7, n_drift_points=5).augment(X, Y)

plot(X_aug, Y_aug);
```


    
![png](/assets/images/timeseries_augmentation/post_13_0.png)
    


### Dropout
* 시계열의 일부 임의 시점을 Dropout


```python
X_aug, Y_aug = tsaug.Dropout(p=0.2, size=(1, 5), fill=float("nan"), per_channel=True).augment(X, Y)

plot(X_aug, Y_aug);
```


    
![png](/assets/images/timeseries_augmentation/post_15_0.png)
    


### Pool
* 길이를 변경하지 않고 해상도를 줄임


```python
X_aug, Y_aug = tsaug.Pool(kind="ave", size=2).augment(X, Y)

plot(X_aug, Y_aug);
```


    
![png](/assets/images/timeseries_augmentation/post_17_0.png)
    


### Quantize
* 시계열을 레벨 세트로 양자화함
* 시계열의 값은 레벨 세트에서 가장 가까운 레벨로 반올림됨


```python
X_aug, Y_aug = tsaug.Quantize(n_levels=20).augment(X, Y)

plot(X_aug, Y_aug);
```


    
![png](/assets/images/timeseries_augmentation/post_19_0.png)
    


### Resize
* 시계열의 시간 해상도를 변경
* 크기가 조정된 시계열은 원래 시계열의 linear interpolation을 사용


```python
X_aug, Y_aug = tsaug.Resize(size=1000).augment(X, Y)

plot(X_aug, Y_aug);
```


    
![png](/assets/images/timeseries_augmentation/post_21_0.png)
    


### Reverse
* 시계열의 타임라인을 뒤집음


```python
X_aug, Y_aug = tsaug.Reverse().augment(X, Y)

plot(X_aug, Y_aug);
```


    
![png](/assets/images/timeseries_augmentation/post_23_0.png)
    


### TimeWarp
* 임의의 시간을 왜곡
* 속도 변경 횟수와, 속도의 최대 비율에 의해 제어
* 속도 비율이 크면 크게 왜곡됨


```python
X_aug, Y_aug = tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(X, Y)

plot(X_aug, Y_aug);
```


    
![png](/assets/images/timeseries_augmentation/post_25_0.png)
    


### 파이프라인 구축


```python
my_augmenter = (
tsaug.TimeWarp() * 2    # 무작위 시간왜곡 2회 병렬
+ tsaug.Crop(size=300)  # 길이가 300인 무작위 크롭
+ tsaug.Quantize(n_levels=[10, 20, 30]) # 10,20,30 레벨 세트로 무작위 양자화
+ tsaug.Drift(max_drift=(0.1, 0.5)) @ 0.8   # 80% 확률로 신호를 10% ~ 50% 까지 무작위 드리프트
+ tsaug.Reverse() @ 0.5 # 50% 확률로 순서를 반대로함
)
```


```python
X_aug, Y_aug = my_augmenter.augment(X, Y)
plot(X_aug, Y_aug);
```


    
![png](/assets/images/timeseries_augmentation/post_28_0.png)
    


### 오늘의 정리
이미지의 증강 기법은 많이 봐왔지만 시계열의 증강 기법은 처음이라 많이 신기했다. 시계열의 증강은 데이터와 라벨이 같이 증강이 되야하는데, resize나 crop 부분에서 라벨까지 같이 처리를 하는 것을 보고 신기했다. 앞으로 시계열 연구를 할 때 한번 참고를 해보는 것도 나쁘지 않을 것 같다!

### 참고자료
* tsaug 도큐먼트 <https://tsaug.readthedocs.io/en/stable/index.html>
