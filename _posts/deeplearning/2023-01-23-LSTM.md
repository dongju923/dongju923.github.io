---
title: "케라스로 간단한 LSTM 구현하기!"
toc: true
toc_sticky: true
categories: DL
---

### LSTM(Long Short-Term Memory)이란?
* RNN(Recurrent Neural Networks)의 한 종류
* RNN의 장기 의존성 문제 때문에 고완됨

### 코드 구현 1


```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from keras.callbacks import EarlyStopping
```



```python
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
y = np.array([2, 5, 8, 11]) # x 배열의 각 1번째 인덱스의 값
print(x.shape)
print(y.shape)
```

    (4, 3)
    (4,)



```python
# reshape를 하는 이유: LSTM은 3차원으로 입력 받음 -> (None, _, _)
x = x.reshape((x.shape[0], x.shape[1], 1))
print(x.shape)
```

    (4, 3, 1)



```python
model = Sequential()
model.add(Input(shape=(3, 1)))   # (None, 3, 1) -> 3열 1개씩 묶음
model.add(LSTM(10, activation = 'relu'))
model.add(Dense(5))
model.add(Dense(1))
model.summary()
# LSTM 파라미터 수 구하기
# LSTM_Layer(4)*((input_shape + bias(1)) * output_shape + output_shape^2)
# 4*((1+1)*10 + 10^2) = 480
```

    Model: "sequential_4"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     lstm_4 (LSTM)               (None, 10)                480       
                                                                     
     dense_8 (Dense)             (None, 5)                 55        
                                                                     
     dense_9 (Dense)             (None, 1)                 6         
                                                                     
    =================================================================
    Total params: 541
    Trainable params: 541
    Non-trainable params: 0
    _________________________________________________________________



```python
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100, batch_size=1, verbose=0)
```

    <keras.callbacks.History at 0x7f0f601abe80>




```python
# 추론을 위한 데이터 (추론 결과는 14가 나와야함)
x_input = np.array([13, 14, 15])
# 데이터 reshape
x_input = x_input.reshape((1, x_input.shape[0], 1))
print(x_input.shape)
```

    (1, 3, 1)



```python
# 추론
y_pred = model.predict(x_input)
print(y_pred)
```

    1/1 [==============================] - 0s 17ms/step
    [[14.426681]]


### 코드 구현 2


```python
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15],
            [16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30],
            [50, 51, 52], [60, 61, 62], [63, 64, 65]])
y = np.array([2, 5, 8, 1, 14, 17, 20, 23, 26, 29, 51, 61, 64])
print(x.shape)
print(y.shape)
```

    (13, 3)
    (13,)



```python
x = x.reshape((x.shape[0], x.shape[1], 1))
print(x.shape) # (13,3,1)
```

    (13, 3, 1)



```python
model = Sequential()
model.add(LSTM(20, activation = 'relu', input_shape=(3,1), return_sequences=True))
model.add(LSTM(10, activation = 'relu')) # LSTM의 마지막은 return_sequence=False
model.add(Dense(1))
model.summary()
# return_sequences -> 이전의 결과들이 다음 레이어로 들어가게 하기 위함. timestep(4)만큼 추가
```

    Model: "sequential_28"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     lstm_51 (LSTM)              (None, 3, 20)             1760      
                                                                     
     lstm_52 (LSTM)              (None, 10)                1240      
                                                                     
     dense_53 (Dense)            (None, 1)                 11        
                                                                     
    =================================================================
    Total params: 3,011
    Trainable params: 3,011
    Non-trainable params: 0
    _________________________________________________________________



```python
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=200, batch_size=1, verbose=0, callbacks=[early_stopping])
```

    <keras.callbacks.History at 0x7f0eb30232b0>




```python
x_input1 = np.array([41, 42, 43])   # 추론 결과가 42가 나와야함
x_input1 = x_input1.reshape((1,x_input1.shape[0],1))
```


```python
y_pred1 = model.predict(x_input1)
print(y_pred1)
```

    1/1 [==============================] - 0s 223ms/step
    [[42.042114]]


### 코드 구현 3


```python
x = np.array([[1, 3, 5, 7], [9, 11, 13, 15], [17, 19, 21, 23], [25, 27, 29, 31], [33, 35, 37, 39],
            [49, 51, 53, 55], [65, 67, 69, 71], [97, 99, 101, 103], [43, 45, 47, 49]])
y = np.array([[3, 5], [11, 13], [19, 21], [27, 29], [35, 37], [51, 53], [67, 69], [99, 101], [45, 47]])
print(x.shape)
print(y.shape)
```

    (9, 4)
    (9, 2)



```python
x = x.reshape((x.shape[0], x.shape[1], 1))
print(x.shape)
```

    (9, 4, 1)



```python
model = Sequential()
model.add(LSTM(20, activation = 'relu', input_shape=(4,1), return_sequences=True))
model.add(LSTM(10, activation = 'relu'))
model.add(Dense(2))
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     lstm_5 (LSTM)               (None, 4, 20)             1760      
                                                                     
     lstm_6 (LSTM)               (None, 10)                1240      
                                                                     
     dense_1 (Dense)             (None, 2)                 22        
                                                                     
    =================================================================
    Total params: 3,022
    Trainable params: 3,022
    Non-trainable params: 0
    _________________________________________________________________



```python
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=200, batch_size=1, verbose=0, callbacks=[early_stopping])
```

    <keras.callbacks.History at 0x7faec0f8ec10>




```python
x_input1 = np.array([101, 103, 105, 107])
x_input1 = x_input1.reshape((1,x_input1.shape[0],1))
```


```python
y_pred1 = model.predict(x_input1)
print(y_pred1)
```

    1/1 [==============================] - 0s 188ms/step
    [[102.790436 104.9237  ]]


### 오늘의 정리
간단한 데이터로 다음 데이터를 예측하는 것을 해보았는데 레이어가 간단해도 꽤 정확한 성능이 나오는 것을 알 수 있었다. 다음 포스팅은 SimpleRNN과 LSTM을 비교해 보아야겠다.
