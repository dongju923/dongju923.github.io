---
title: "LSTM의 return_state와 return_sequences"
toc: true
toc_sticky: true
categories: DL
---


### `return_state=False, return_sequences=False`

`return_sequences`와 `return_state`를 False로 할 경우 LSTM레이어는 하나의 결과를 반환한다. 여기서 결과값은 마지막 은닉상태의 값이다.

```python
input_layer = Input(shape=(30,))
output = LSTM(64)(input_layer)
```
여기서 output의 형태는 (None, 64)가 출력된다.

### `return_state=True, return_sequences=False`

`return_state=True, return_sequences=False`로 할 경우에는 결과값 3개를 반환한다. 첫번째는 마지막 은닉상태 값이고, 두번째도 마지막 은닉상태 값이고, 세번째는 셀 상태 값이다.  
`return_sequences=False`이기 때문에 첫번째와 두번째의 결과값은 LSTM의 마지막 은닉상태 값이다.

```python
input_layer = Input(shape=(30,))
output, state_h, state_c = LSTM(64, return_state=True)(input_layer)
```
여기서 output, state_h, state_c의 형태는 (None, 64)이다. output, state_h는 값이 같지만 state_c는 값이 다르다.

### `return_state=False, return_sequences=True`

`return_state=False, return_sequences=True`로 할 경우에는 결과값을 한개만 반환한다. 여기서 결과값은 LSTM의 모든 시점의 은닉상태 값이다.

```python
input_layer = Input(shape=(30,))
output = LSTM(64, return_sequences=True)(input_layer)
```
여기서의 output의 형태는 (None, 30, 64)가 출력된다. 이는 30만큼의 은닉상태 값이 모두 포함된 상태이다.

### `return_state=True, return_sequences=True`

`return_state=True, return_sequences=True`로 할 경우에는 결과값을 3개 반환한다. 첫번째는 모든 시점의 은닉상태 값이고, 두번째는 마지막 은닉상태 값이고, 세번째는 셀 상태 값이다.

```python
input_layer = Input(shape=(30,))
output, state_h, state_c = LSTM(64, return_sequences=True)(input_layer)
```
여기서의 output의 형태는 (None, 30, 64)가 출력되고, state_h, state_c의 형태는 (None, 64)이다. output[:, -1, :]의 값은 state_h의 값과 같다. 이유는 마지막 시점의 은닉상태 값이기 때문이다.
