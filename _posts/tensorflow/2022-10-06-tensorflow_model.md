---
title: "텐서플로우 모델 만들기"
toc: true
categories: Tensorflow
---

텐서플로우 모델을 만드는 방법은 크게 세 가지가 있다.
1. Sequential로 만드는 모델
2. Funcional로 만드는 모델
3. Class로 만드는 모델
> 이번 예제에서는 세 가지 방법으로 Basic Autoencoder를 만들어 보는 것이다.

# Sequential 모델
Sequential 방식은 직관적이고 편리하다는 장점이 있지만 단순히 층을 쌓는 것만으로는 구현할 수 없는 복잡한 신경망을 구현할 수 없는것이 단점도 있다. 사용방법은 `Sequential()`모델을 정의하고 `add()` 메소드를 통해서 층을 추가하면 된다.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

autoencoder = Sequential()
# autoencoder.add(Input(shape=(784,)))
autoencoder.add(Dense(32, input_shape=(784,) ,activation = "relu"))
autoencoder.add(Dense(784, activation = "sigmoid"))

encoder = Sequential()
encoder.add(autoencoder.input)
encoder.add(autoencoder.layers[0])

decoder = Sequential()
decoder.add(Input(shape=(32,)))
decoder.add(autoencoder.layers[1])

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```

# Funcional 모델
Funcional 모델은 각 층을 일종의 함수로 정의하는 것이다. 그리고 각 함수를 조합하기 위한 연산자들을 지원하기 때문에 Sequential 모델보다 더 정교한 모델을 만들 수 있다. Funcional 모델은 입력의 크기를 명시한 입력층(Input layer)을 모델의 맨 앞단에 정의해야한다. 
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

input_img = Input(shape=(784,))
encoded = Dense(32, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
 
# autoencoder 모델
autoencoder = Model(input_img, decoded)
 
# encoder 모델
encoder = Model(input_img, encoded)
 
# decoder 모델
encoded_input = Input(shape=(32,))
decoded_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoded_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```

# Class형 모델
Class형 모델을 사용할 때, `__init__()` 메서드에 사용할 층을 정의해야 하고, `call()` 메서드를 통해서 모델의 정방향 전달을 구현해야 한다. 
```python
class Autoencoder(Model):
  def __init__(self):
    super(Autoencoder, self).__init__() 
    self.encoder = tf.keras.Sequential([
        layers.Dense(32, input_shape=(784,), activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
        layers.Dense(784, activation='sigmoid'),
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```
