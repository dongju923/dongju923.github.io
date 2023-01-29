---
title: "텐서플로우 모델 compile 및 fit"
toc: true
published: true
categories: Tensorflow
---

모델이 생성되었으면 텐서플로우에서 제공하는 `compile()` 메서드와 `fit()` 메서드로 학습이 가능하다. 

# `compile()`
모델을 생성하고 학습시키기 이전에는 `compile()` 메서드를 통해서 학습방식에 대한 설정을 해야한다. 
1. 옵티마이저(optimizer)
- 최적화 알고리즘을 설정
- Adadelta, Adagrad, Adam 등이 있음
2. 손실함수(loss function)
- 모델 최적화에 사용되는 손실함수
- mse, categorical_crossentropy, binary_crossentropy 등이 있음
3. 평가지표(metric)
- 훈련과정을 모니터링하기 위한 지표
- 사용자가 메트릭을 직접 커스텀해서 사용이 가능
- accuracy, loss, mse 등이 있음

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

```python
# 커스텀 metric 만들기
from sklearn.metrics import mean_squared_error
def mse(y_test, y_pred):
    return mean_squared_error(y_test, y_pred)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy', mse])
```

# `fit()`
`compile()` 단계에서 지정한 방식으로 학습을 진행한다. `compile()` 단계에서 지정한 metrics 반환하여 기록을 살펴볼 수 있다.

```python
history = model.fit(
    x=None, # 입력데이터
    y=None, # 타켓데이터
    batch_size=None,    # 배치사이즈
    epochs=1,   # 훈련 반복 수
    verbose='auto', 
    callbacks=None, 
    validation_split=0.0,   # 검증데이터 비율
    validation_data=None,   # 검증데이터
    shuffle=True,   # 데이터 셔플
    class_weight=None,  # class에 따른 가중치 부여
    sample_weight=None, # sample에 따른 가중치 부여
    initial_epoch=0,    # 훈련을 시작할 수
    steps_per_epoch=None, # 단계별 배치 수
    validation_steps=None,  
    validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False
)
```
> 훈련이 끝나면 history에는 metrics의 값이 들어있다.


# 참고자료
1. https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
