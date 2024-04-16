---
title: "파이토치-커스텀 데이터셋 만들기"
toc: true
toc_sticky: true
categories: Torch
---

파이토치에서는 데이터를 좀 더 쉽게 다룰 수 있도록 하는 Dataset 클래스가 있다.  
최근에 프로젝트를 진행할 때 100GB나 되는 데이터를 한번에 전처리 해서 데이터를 gpu에 올려놨었는데, 이렇게 하면
전처리 과정에서 시스템 메모리를 엄청나게 먹고 데이터도 gpu공간을 너무 많이 차지하는 단점이 있었다.  
그래서 최근에 토치 데이터셋을 공부하면서 느꼈던 지식을 공유하고자 한다!

### Custom Dataset
`torch.utils.data.Dataset`클래스를 상속받아서 나만의 dataset을 만들면 된다.  
```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass
    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass
```
데이터셋을 구성하는데 필요한 define은 3개이다.  
1. `__init__()`  
```text
필요한 변수들을 선언하는 단계이다. 
나중에 `CustomDataset`클래스를 호출할 때 받은 인자를 담을 변수를 선언한다.
```
2. `__len__()`
```text
데이터의 길이를 지정하는 단계이다. 
나중에 idx별로 데이터를 가져올 때, idx의 총 길이를 지정해야 한다.
```
3. `__getitem__()`
```text
데이터에서 특정 idx의 샘플을 가져오는 함수이다. 
여기서의 return 값이 최종 output이 된다.
```

### Data Loader
데이터셋을 만들었으면 학습에 사용할 미니 배치 데이터를 만들기 위해 `torch.utils.data.DataLoader`를 사용한다.  
dataloader를 통해 dataset의 전체 데이터가 batch_size로 슬라이싱 된다.
```python
train_loader = data.DataLoader(dataset=custom_dataset, batch_size=128, shuffle=True)
```
만들어 놓았던 custom_dataset을 넣어주면 배치만큼 데이터를 묶고, 셔플까지 해서 배치를 만들어 준다.


### 예제 1
```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    

data = torch.rand(50)
label = torch.randint(0, 10, (50,))
custom_datasets = CustomDataset(data, label)
custom_loaders = DataLoader(custom_datasets, batch_size=25, shuffle=True)
```
간단히 임시 데이터로 구성해 보았다.  
여기서 `__init__()`에 `self.x`와 `self.y`의 값은 data와 label값이 들어갈 것이다. `__len()`의 return은 `self.x`의 길이 이므로  
500이라는 값이 return된다. 이 50이라는 값이 `__getitem__()`에서 사용될 idx값이다. 최종 return은 `x[idx]`, `y[idx]`가 된다.  

```python
for x, y in custom_loaders:
    print(f"x값: {x}")
    print(f"y값: {y}")
```
이 코드를 실행시켜보면 50개의 데이터가 배치만큼 묶여서 나오는걸 알 수 있다. 

```text
x값: tensor([0.1772, 0.6119, 0.8013, 0.0062, 0.2600, 0.1827, 0.2496, 0.0108, 0.2483,
        0.5210, 0.8130, 0.6939, 0.1215, 0.7873, 0.1470, 0.9115, 0.2817, 0.4671,
        0.5618, 0.0946, 0.1604, 0.3909, 0.3788, 0.9520, 0.7144])
y값: tensor([1, 1, 3, 7, 0, 8, 4, 1, 8, 1, 9, 7, 5, 9, 7, 0, 5, 5, 3, 6, 0, 1, 4, 6,
        9])
x값: tensor([0.7404, 0.8087, 0.3474, 0.7505, 0.6372, 0.8600, 0.1880, 0.6901, 0.4890,
        0.7598, 0.1416, 0.8061, 0.7002, 0.2195, 0.4503, 0.9567, 0.1318, 0.6409,
        0.3760, 0.5300, 0.7330, 0.7779, 0.8534, 0.7478, 0.9667])
y값: tensor([4, 0, 7, 1, 3, 0, 0, 3, 5, 2, 3, 5, 3, 4, 6, 0, 7, 4, 0, 2, 1, 1, 8, 2,
        2])
```


### 예제 2
데이터가 특정 디렉토리에 있다고 가정하고 만들어 보겠다.

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, audio_dir, label_dir):
        self.audio_file_list = sorted([os.path.join(root, file) for root, _, files in os.walk(audio_dir) for file in files])
        self.label_file_list = sorted([os.path.join(root, file) for root, _, files in os.walk(label_dir) for file in files])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio = self.audio_file_list[idx]
        label = self.label_file_list[idx]
        # 전처리
        # 전처리 
        audio = torch.Tensor(audio)
        label = torch.Tensor(label)
        return audio, label

train_dataset = CustomDataset("./data/train/audio/", "./data/train/labels")
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
```
데이터셋을 호출할 때, 경로를 매개변수로 입력하면 `__init__()`부분에서 경로 내 모든 파일을 읽어와서 `self.audio_file_list`에 저장한다.  
그 다음 `__getitem()`에서 idx만큼 읽어서 전처리 후 return하는 코드이다.  

### 정리 및 요약
> 예제 2는 실제로 내가 음성모델을 학습할 때 사용한 코드이다. 데이터는 100만개 정도였기 때문에, 한번에 오디오를 전처리해서 보관하기에는
한계가 있었을 뿐더러 100만개가 되는 데이터를 전부 gpu로 올릴 수도 없었다. Torch Dataset은 배치만큼 데이터를 전처리하고, 배치만큼 gpu로 
올릴수 있기 때문에 정상적으로 학습할 수 있었다.
