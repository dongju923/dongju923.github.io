---
title: "텐서플로우 Gradient Tape로 자동 미분하기"
toc: true
toc_sticky: true
categories: Tensorflow
use_math: true
---


딥러닝에서 모델을 만든다는 것은 곧 계산그래프를 만든다는 것이다.  
일반적인 계산그래프와는 다르게 <span style="color:violet">딥러닝에서는 예측과 손실값을 계산하는 순전파와 편미분, 체인룰을 이용한 역전파가 존재한다.</span>  
텐서플로우의 Gradient Tape를 이용하면 즉시실행모드(Eager Excution)에서 쉽고 빠르게 미분 연산이 가능하다.
> 즉시실행모드란 그동안 1.x 버전에서 계산 그래프를 선언하고, 세션을 통해 계산을 하는 구조였던것을 텐서플로우 2.x 버전에서 세션과 그래프를 선언하지 않고 즉시 계산할 수 있게 만든 모드이다.

### 미분 방법

<span style="color:violet">Gradient Tape는 중간 연산과정(함수, 연산)을 테이프에 기록한다.</span>  
`with tf.GradientTape() as tape:`로 저장할 tape을 지정하면, `with`문맥 안에서의 관련 연산들은 tape에 기록이 된다. 이때 역전파를 위한 값들이 저장되고, tape에 저장된 연산 과정을 가져다가 `gradient()`메서드를 통해서 후진 방식 자동 미분(Reverse mode automatic differentiation)으로 미분을 계산한다.  
이렇게 계산한 값을 갱신하는 작업을 반복함으로써 원하는 답을 찾아가는 학습을 진행한다.

### 미분

그럼 간단하게 자동 미분을 수행해보자. Gradient Tape를 사용하기 위해서는 변수를 `tf.Tensor()` 형태로 저장해야 한다.  
임의로 $x^3$ 이라는 식을 세우고 $x$에 대해 미분해보자.  
고등학교때 배운 지식이라면 $3x^2$이 나오는 것을 알것이다.


```python
import tensorflow as tf

x = tf.Variable(4.0)

with tf.GradientTape() as tape:
  y = x**3

dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())
```

    48.0


대충 이런 느낌으로 미분을 진행한다.  
이건 그냥 단순한 미분일 뿐이다. 실전에서는 예측과 손실값 계산으로 인한 최적의 가중치와 편향을 찾는 과정이 필요하다. 

### 연습예제

이제 연습예제를 살펴보자.  
$y = a * x$ 라는 방정식에서 $a$와 $y$가 상수이고 $x$가 변수 일때, $x$를 구하려면 $|a*x-y|$의 값이 최소가 되는 값을 찾으면 된다.  
아래 예제에서 $8.0 = 2.0 * x$ 의 방정식에서 손실함수 값을 최소로 하는 $x$를 구하려면 손실함수에 대한 $x$의 미분값을 $x$에서 빼서 갱신해야한다.  
식으로 대충 나타내자면 $x = x - loss(x)$ 이렇게 되겠다.  
변수 $x$의 값은 4인데 어떻게 $x$가 4가 나오게 동작하는지 예제를 보자.


```python
a = tf.constant(2.0)
y = tf.constant(8.0)
x = tf.Variable(10.0)   # 랜덤 값

with tf.GradientTape() as tape:
    loss = tf.math.abs(a * x - y)

dx = tape.gradient(loss, x)
print(f"x값 : {x.numpy()}, 손실함수에 대해서 미분한 x값: {dx}")
x.assign(x-dx)
print(f"x-loss(x)의 값: {x.numpy()}")
```

    x값 : 10.0, 손실함수에 대해서 미분한 x값: 2.0
    x-loss(x)의 값: 8.0


위의 코드는 1번 실행했을 때의 결과이다. $x$가 10일때 손실함수를 미분하면 결국 a값(2.0)이 나온다. 그럼 그 다음번의 $x$값은 기존 $x$값에서 미분한 값(2.0)을 빼서 8.0이 된것이다.   
그럼 2번 실행했을 때의 결과는 어떨까? $x$에 8이 할당되고 미분값(2.0)을 빼면 그 다음의 $x$값은 6이 할당될것이다.  
이렇게 반복해가면서 $x$값을 찾는 것이다.  
밑의 코드를 한번 보자.


```python
a = tf.constant(2.0)
y = tf.constant(8.0)
x = tf.Variable(10.0)   # 랜덤 값

for i in range(4):
    with tf.GradientTape() as tape:
        loss = tf.math.abs(a * x - y)

    dx = tape.gradient(loss, x)
    print(f"{i}-x값 : {x.numpy()}, 손실함수에 대해서 미분한 x값: {dx}")
    x.assign(x-dx)
    print(f"x-loss(x)의 값: {x.numpy()}")
```

    0-x값 : 10.0, 손실함수에 대해서 미분한 x값: 2.0
    x-loss(x)의 값: 8.0
    1-x값 : 8.0, 손실함수에 대해서 미분한 x값: 2.0
    x-loss(x)의 값: 6.0
    2-x값 : 6.0, 손실함수에 대해서 미분한 x값: 2.0
    x-loss(x)의 값: 4.0
    3-x값 : 4.0, 손실함수에 대해서 미분한 x값: 0.0
    x-loss(x)의 값: 4.0


4번 반복했을 때의 결과이다. 최종 x의 값은 4가 나온다.  
모델을 만드는 방법은 정말 다양하다. Gradient Tape을 사용하여 학습을 할 수도 있고, Sequential모델을 만드는 방법도 있고, functional 모델을 만들 수도 있다. Gradient Tape은 그중 한가지일 뿐이다.  

### 실전예제

마지막으로 Gradient Tape를 사용하여 학습과 예측을 진행하는 예제를 살펴도록 하자.  
예제는 선형회귀를 구현하는 것이며, 제일 간단하기에 회귀를 구현하는 것을 선택했다.  
설명은 따로 하지않고 주석으로 달도록 하겠다!!


```python
import numpy as np

# 학습될 가중치와 편향을 선언
w = tf.Variable(4.0)
b = tf.Variable(1.0)

# 단순선형회귀의 계산 함수
def hypothesis(x):
  return w*x + b

# MSE loss 함수
def mse_loss(y_pred, y):
  return tf.reduce_mean(tf.square(y_pred - y))

# 데이터 선언
x = np.arange(1, 11)
y = x * 10 + 13
print(f"x: {x}")
print(f"y: {y}")

# SGD옵티마이저 사용. 학습률은 0.01
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# 500번 반복학습
for i in range(501):
    with tf.GradientTape() as tape:
        # 현재 w, b에 기반한 입력 x에 대한 예측값
        y_pred = hypothesis(x)

        # loss 계산
        loss = mse_loss(y_pred, y)

    # 손실 함수에 대한 w, b의 미분값 계산
    gradients = tape.gradient(loss, [w, b])

    # w, b 업데이트
    optimizer.apply_gradients(zip(gradients, [w, b]))

    if i % 50 == 0:
        print("epoch : {:3} | w의 값 : {:5.4f} | b의 값 : {:5.4} | loss : {:5.6f}".format(i, w.numpy(), b.numpy(), loss))
    
```

    x: [ 1  2  3  4  5  6  7  8  9 10]
    y: [ 23  33  43  53  63  73  83  93 103 113]
    epoch :   0 | w의 값 : 9.9400 | b의 값 :   1.9 | loss : 2322.000000
    epoch :  50 | w의 값 : 11.2648 | b의 값 : 4.195 | loss : 16.755640
    epoch : 100 | w의 값 : 11.0248 | b의 값 : 5.866 | loss : 11.000027
    epoch : 150 | w의 값 : 10.8303 | b의 값 :  7.22 | loss : 7.221475
    epoch : 200 | w의 값 : 10.6728 | b의 값 : 8.316 | loss : 4.740870
    epoch : 250 | w의 값 : 10.5451 | b의 값 : 9.205 | loss : 3.112357
    epoch : 300 | w의 값 : 10.4417 | b의 값 : 9.925 | loss : 2.043251
    epoch : 350 | w의 값 : 10.3579 | b의 값 : 10.51 | loss : 1.341386
    epoch : 400 | w의 값 : 10.2899 | b의 값 : 10.98 | loss : 0.880615
    epoch : 450 | w의 값 : 10.2349 | b의 값 : 11.36 | loss : 0.578120
    epoch : 500 | w의 값 : 10.1903 | b의 값 : 11.67 | loss : 0.379532



```python
x_test = np.array([3.5, 8.5, 11, 13])
y_true = x_test * 10 + 13
print(f"예측값: {hypothesis(x_test).numpy()}")
print(f"실제값: {y_true}")
```

    예측값: [ 47.341038  98.29279  123.76866  144.14937 ]
    실제값: [ 48.  98. 123. 143.]


> Gradient Tape의 사용은 회귀분석 말고 다른 모델에도 사용이 가능하다. 예를 들어 모델을 하나 만들고 `fit()`대신 Gradient Tape을 사용하여 학습할 수도 있다.
