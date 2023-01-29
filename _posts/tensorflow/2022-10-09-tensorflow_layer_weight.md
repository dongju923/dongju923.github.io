---
title: "텐서플로우 커스텀 레이어 생성 및 가중치 확인"
toc: true
published: true
categories: Tensorflow
---

텐서플로우에는 가중치를 초기화 및 확인이 가능하다. 일반적으로  layer의 `build()`함수가 작동할 때, 또는 모델에서 `call()`함수가 들어올 때 자동으로 초기화가 된다. 가중치와 편향은 각 `layer.get_weight()`로 확인이 가능한데, return값이 두개이므로(weight, bias) 변수를 두개 선언 해주어야 한다.
> 이번 예제에서는 대충 만든 CNN구조로 가중치들을 살펴보겠다.
 
# 텐서플로우 가중치 확인

```python
model = Sequential()
model.add(layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=(32,32,1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy')
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 32, 32, 32)        832       
                                                                     
     max_pooling2d (MaxPooling2D  (None, 16, 16, 32)       0         
     )                                                               
                                                                     
     conv2d_1 (Conv2D)           (None, 16, 16, 64)        8256      
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 8, 8, 64)         0         
     2D)                                                             
                                                                     
     dropout (Dropout)           (None, 8, 8, 64)          0         
                                                                     
     flatten (Flatten)           (None, 4096)              0         
                                                                     
     dense (Dense)               (None, 1000)              4097000   
                                                                     
     dropout_1 (Dropout)         (None, 1000)              0         
                                                                     
     dense_1 (Dense)             (None, 1)                 1001      
                                                                     
    =================================================================
    Total params: 4,107,089
    Trainable params: 4,107,089
    Non-trainable params: 0
    _________________________________________________________________


```python
w, b = model.layers[0].get_weights()
print(w.shape)
print(b.shape)
```
    (5, 5, 1, 32)
    (32,)
```python
model.layers[1].get_weights()
```

    []

> 모델 구조에서 파라미터의 수가 0인 layer는 가중치가 없다!

# 텐서플로우 커스텀 레이어 만들기
텐서플로우에는 커스텀 레이어를 만들어서 레이어의 가중치를 원하는 값으로 설정이 가능하다. 
```python
class SimpleDense(tf.keras.layers.Layer):
    
  def __init__(self, units=32):
      super(SimpleDense, self).__init__()
      self.units = units

  def build(self, input_shape):  # layer의 상태 선언
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(
        initial_value=w_init(shape=(input_shape[-1], self.units),
                             dtype='float32'),
        trainable=True)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(
        initial_value=b_init(shape=(self.units,), dtype='float32'),
        trainable=True)

  def call(self, inputs):  # 모델의 input과 output 계산방식 선언
      return tf.matmul(inputs, self.w) + self.b

linear_layer = SimpleDense(4)
# 가중치 초기화
value = linear_layer(tf.ones((3, 3)))
```
```python
linear_layer.weights
```
```
    [<tf.Variable 'simple_dense_8/Variable:0' shape=(3, 4) dtype=float32, numpy=
    array([[ 0.01979813, -0.04338467, -0.07237624, -0.06622894],
            [-0.04669333, -0.00611689, -0.03860446, -0.04537704],
            [-0.01581768, -0.01695884, -0.03870974,  0.00631301]],
        dtype=float32)>,
    <tf.Variable 'simple_dense_8/Variable:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>]
```
> 이렇게 만든 레이어는 `model.add(SimpleDense())` 형태로 사용이 가능하다.


# 참고자료
1. https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
> 텐서플로우 layer에 관한 공식 문서이다. 이밖에도 많은 방법들이 자세히 적혀있다!
