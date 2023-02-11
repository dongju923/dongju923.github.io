---
title: "단순 로지스틱 회귀분석을 해보자!"
toc: true
toc_sticky: true
categories: ML
---


이번에는 단순로지스틱 회귀를 해보자. 로지스틱회귀를 모르는 사람은 [이글](https://dongju923.github.io/ml/Logistic_Regression/)을 참고하자.  
먼저 간단한 데이터로 먼저 감만 익힌다고 생각하자.  
다중 로지스틱회귀는 단순 로지스틱 회귀에서 독립변수만 여러개로 설정하면 된다.  

### 단순 로지스틱회귀(케라스)
케라스를 이용해서 로지스틱 회귀분석을 해보겠다. 학습과정과 예측 부분은 생략하였다.  
코드에서 볼 수 있듯이, 단순선형함수에 `activation`함수만 sigmoid으로 바꾸었고, loss는 binary_crossentropy를 사용하였다.  


```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

X = np.arange(5, 100, 5)
# x: [ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
y = np.array(list(map(lambda x: 0 if x<50 else 1, X)))
# y: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

model = Sequential()
model.add(Dense(1, input_dim=1, activation='sigmoid'))

sgd = optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])

model.fit(X, y, epochs=200)
```

### 단순 로지스틱 회귀(사이킷런)
이번에는 사이킷런을 이용하여 회귀분석을 해보았다.  
X에는 5~95까지 5단위로된 배열이 있으며, y값은 50이 넘으면 1, 아니면 0으로 만들었다.  
`LinearRegression` 과 마찬가지로 x에는 2차원 데이터를 넣어줘야 하기때문에 2차원으로 변환한 다음 넣어주었다. 


```python
from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.arange(5, 100, 5).reshape(-1, 1)
y = np.array(list(map(lambda x: 0 if x<50 else 1, X)))

model = LogisticRegression()
model.fit(X, y)
```

이제 예측을 해보도록 하자.  
테스트 데이터가 없으니 그냥 X값을 넣어보겠다.


```python
print(f"예측값: {model.predict(X)}")
print(f"실제값: {y}")
```

    예측값: [0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1]
    실제값: [0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1]


정확하게 분류한 것을 볼 수 있다. 로지스틱 회귀의 출력은 확률값이라고 했는데, 그냥 `predict()`메서드를 사용하면 0, 1로만 값이 나온다.  
각 항목에 대한 확률값을 보고싶으면, `predict_proba()` 메서드를 사용하면 된다.


```python
np.set_printoptions(precision=3, suppress=True)
model.predict_proba(X)
```

    array([[1.   , 0.   ],
           [1.   , 0.   ],
           [1.   , 0.   ],
           [1.   , 0.   ],
           [1.   , 0.   ],
           [1.   , 0.   ],
           [1.   , 0.   ],
           [0.996, 0.004],
           [0.864, 0.136],
           [0.136, 0.864],
           [0.004, 0.996],
           [0.   , 1.   ],
           [0.   , 1.   ],
           [0.   , 1.   ],
           [0.   , 1.   ],
           [0.   , 1.   ],
           [0.   , 1.   ],
           [0.   , 1.   ],
           [0.   , 1.   ]])



넘파이의 `set_printoptions`는 숫자가 지수표현식으로 나오는 것을 실수 표현으로 바꿔준다. 그냥 보기 편하라고 바꾼것이다.  
결과를 쉽게 이해해보면, X가 시험 점수라고 하고 y가 통과 여부라고 할 때,  
5점부터 45점까지는 불합격이고, 50점부터 95점까지는 통과라고 생각하면 되겠다.  
값을 하나씩만 넣어서 한번 보자


```python
print(model.predict_proba([[45]]))
print(model.predict_proba([[49]]))
```

    [[0.864 0.136]]
    [[0.248 0.752]]


첫번째로 45점을 넣었을 때, 시험에 불합격할 확률이 86.4%이고, 합격할 확률이 13.6%라는 뜻이다. 한마디로 0이 나올확률은 86.4%이고 1이 나올 확률이 13.6%라는 것이다.  

두번째로 49점을 넣었을 때, 시험에 불합격할 확률이 24.8%이고, 합격할 확률이 75.2%라는 뜻이다. 원래 데이터대로 라면 50점부터가 합격인데, 약간 잘못 예측한 것이다.  


다음 포스팅에서 다중 로지스틱 회귀분석을 해보자. 실제 데이터는 위의 데이터처럼 간단하지 않다. 독립변수도 여러개이고, 심지어 종속변수도 여러개인 경우도 있다. 

[다중 로지스틱 회귀분석](https://dongju923.github.io/ml/Multiple_Logistic_Regression/)
