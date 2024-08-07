---
title: "[그래프 AI]GCN 구현하기 with PyTorch"
categories: 
  - Graph
  
toc: true
toc_sticky: true

date: 2024-08-05
last_modified_at: 2024-08-05
---

실행 코드는 `.ipynb`파일로 제공한다. (주소: [Github](https://github.com/meaningful96/CodeAttic/blob/main/5.%20GNN/readme.md))

# GCN 구현하기
## 1. 패키지 설치
```python
# Install required packages.
!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
```

## 2. Visualization을 위한 함수 정의
이 코드는 그래프와 임베딩을 시각화하는 두 가지 함수를 정의한 것이다. 각 함수의 목적은 다음과 같다:

1. **`visualizeGraph(G, color)` 함수**:
   - 그래프 `G`와 노드의 색상을 지정하는 `color` 매개변수를 입력으로 받는다.
   - `nx.spring_layout`을 사용하여 그래프의 레이아웃을 계산하고, `nx.draw_networkx`를 사용하여 그래프를 시각화한다.
   - 시각화된 그래프는 노드의 레이블을 표시하지 않으며, 노드의 색상은 `color` 매개변수를 기반으로 지정된 컬러맵(`Set2`)을 사용한다.

2. **`visualizeEmbedding(h, color, epoch=None, loss=None)` 함수**:
   - 임베딩 텐서 `h`, 색상을 지정하는 `color` 매개변수, 선택적인 `epoch` 및 `loss` 값을 입력으로 받는다.
   - 임베딩 텐서 `h`를 NumPy 배열로 변환하고, `plt.scatter`를 사용하여 2D 평면에 점으로 시각화한다.
   - `epoch`와 `loss` 값이 제공되면, 플롯의 x축 레이블에 해당 정보를 표시한다.
   - 시각화된 임베딩은 지정된 `Set2` 컬러맵을 사용하여 노드의 색상을 표시한다.

이 코드는 주로 그래프 데이터를 시각화하고, 신경망의 임베딩 결과를 시각화하는 데 사용된다. 시각화를 통해 그래프의 구조와 임베딩의 분포를 쉽게 이해할 수 있다.

```python
import torch
import networkx as nx
import matplotlib.pyplot as plt

def visualizeGraph(G, color):
  plt.figure(figsize=(7,7))
  plt.xticks([])
  plt.yticks([])
  nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                   node_color=color, cmap="Set2")
  plt.show()

def visualizeEmbedding(h, color, epoch=None, loss=None):
  plt.figure(figsize=(7, 7))
  plt.xticks([])
  plt.yticks([])
  h = h.detach().cpu().numpy()
  plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
  if epoch is not None and loss is not None:
    plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
  plt.show()

```

## 3. Load Data
PyTorch Geometric을 사용하여 Karate Club 데이터셋을 로드하고 데이터셋 및 그래프에 대한 다양한 통계를 출력한다.  

1. **데이터셋 로드**:
   - `KarateClub` 데이터셋을 로드하여 `dataset` 변수에 할당한다.
   - 데이터셋에 대한 기본 정보(그래프 수, 특징 수, 클래스 수)를 출력한다.

2. **데이터셋 통계 출력**:
   - 데이터셋 내 그래프의 수.
   - 데이터셋의 노드당 특징 수.
   - 데이터셋 내 클래스 수.

3. **그래프 데이터 출력**:
   - 데이터셋의 첫 번째 그래프 데이터를 출력한다.

4. **그래프 통계 수집 및 출력**:
   - 그래프의 노드 수.
   - 그래프의 엣지 수.
   - 평균 노드 차수(엣지 수를 노드 수로 나눈 값).
   - 학습 노드 수.
   - 학습 노드 라벨 비율(전체 노드 수 대비 학습 노드의 비율).
   - 그래프에 고립된 노드가 있는지 여부.
   - 그래프에 자기 루프가 있는지 여부.
   - 그래프가 무방향인지 여부.

```python
from torch_geometric.datasets import KarateClub
dataset = KarateClub()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')


data = dataset[0]
print(data)
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
```

```bash
Dataset: KarateClub():
======================
Number of graphs: 1
Number of features: 34
Number of classes: 4
Data(x=[34, 34], edge_index=[2, 156], y=[34], train_mask=[34])
==============================================================
Number of nodes: 34
Number of edges: 156
Average node degree: 4.59
Number of training nodes: 4
Training node label rate: 0.12
Has isolated nodes: False
Has self-loops: False
Is undirected: True
```

`NetworkX` 라이브러리는 그래프 분석을 위한 여러가지 함수를 제공한다. 그 중 그래프를 시각화할 수 있는 함수 또한 존재한다.
```python
from torch_geometric.utils import to_networkx

G = to_networkx(data, to_undirected=True)
visualizeGraph(G, color=data.y)
```

<p align="center">
<img width="400" alt="1" src="https://github.com/user-attachments/assets/edb7cbc8-aeba-476d-a9a7-58c2765dfe73">
</p>

## 5. GCN 정의하기
1. **임포트**:
   - PyTorch의 주요 라이브러리인 `torch`.
   - 선형 계층을 위한 `torch.nn`의 `Linear`.
   - 그래프 컨볼루션 계층을 위한 `torch_geometric.nn`의 `GCNConv`.

2. **GCN 클래스 정의**:
   - `GCN` 클래스는 `torch.nn.Module`을 상속한다.
   - `__init__` 메서드는 모델을 초기화한다:
     - 재현성을 위해 시드를 수동으로 설정한다.
     - 세 개의 그래프 컨볼루션 계층(`conv1`, `conv2`, `conv3`)을 정의하고, 입력 및 출력 차원을 지정한다:
       - `conv1`: 데이터셋의 특징 수에서 4로 변환.
       - `conv2`: 4에서 4로 변환.
       - `conv3`: 4에서 2로 변환.
     - 2차원 입력을 데이터셋의 클래스 수로 매핑하는 선형 분류기 계층(`classifier`)을 정의한다.
   - `forward` 메서드는 모델의 순전파를 정의한다:
     - 첫 번째 그래프 컨볼루션 계층을 적용하고 `tanh` 활성화 함수를 적용한다.
     - 두 번째 그래프 컨볼루션 계층을 적용하고 `tanh` 활성화 함수를 적용한다.
     - 세 번째 그래프 컨볼루션 계층을 적용하고 `tanh` 활성화 함수를 적용한다.
     - 세 번째 그래프 컨볼루션 계층의 출력에 선형 분류기를 적용한다.

3. **모델 초기화**:
   - `GCN` 클래스의 인스턴스를 생성하고 출력한다.

```python
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    torch.manual_seed(1234)
    self.conv1 = GCNConv(dataset.num_features, 4)
    self.conv2 = GCNConv(4, 4)
    self.conv3 = GCNConv(4, 2)
    self.classifier = Linear(2, dataset.num_classes)

  def forward(self, x, edge_index):
    h = self.conv1(x, edge_index)
    h = h.tanh()
    h = self.conv2(h, edge_index)
    h = h.tanh()
    h = self.conv3(h, edge_index)
    h = h.tanh()

    out = self.classifier(h)
    return out, h

model = GCN()
print(model)

_, h = model(data.x, data.edge_index)
print(f'Embedding shape: {list(h.shape)}')
visualizeEmbedding(h, color=data.y)
```

```bash
GCN(
  (conv1): GCNConv(34, 4)
  (conv2): GCNConv(4, 4)
  (conv3): GCNConv(4, 2)
  (classifier): Linear(in_features=2, out_features=4, bias=True)
)

Embedding shape: [34, 2]
```

<p align="center">
<img width="400" alt="1" src="https://github.com/user-attachments/assets/63749506-be60-4d97-9e2f-f1eaf30abc6a">
</p>

## 6. 모델 학습
```python
import time
from IPython.display import Javascript
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 430})'''))
model = GCN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(data):
  optimizer.zero_grad()
  out, h = model(data.x, data.edge_index)
  loss = criterion(out[data.train_mask], data.y[data.train_mask])
  loss.backward()
  optimizer.step()
  return loss, h

for epoch in range(401):
  loss, h = train(data)
  if epoch % 100 == 0:
    visualizeEmbedding(h, color=data.y, epoch=epoch, loss=loss)
    time.sleep(0.3)
```

<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/0ca8c5bc-4296-4024-86af-35b31a88b365">
</p>

<br/>
<br/>

# MLP vs GCN
## 1. t-SNE 분석을 위한 함수 정의
**t-SNE**:
   - t-SNE (t-distributed Stochastic Neighbor Embedding)은 차원 축소를 위한 기계 학습 알고리즘이다. 특히 고차원 데이터셋을 2차원 또는 3차원으로 축소하여 데이터 포인트 간의 구조와 관계를 최대한 유지하면서 시각화하는 데 적합하다.

**시각화 함수**:
   - `visualize(h, color)` 함수:
     - t-SNE를 사용하여 고차원 텐서 `h`의 차원을 2D로 축소한다.
     - Matplotlib을 사용하여 2D 데이터를 플로팅한다.
     - 그림 크기를 10x10 인치로 설정한다.
     - 플롯에서 x와 y 틱을 제거한다.
     - "Set2" 컬러맵을 사용하여 `color` 매개변수에 따라 점의 색상을 지정하여 축소된 데이터를 산점도로 플로팅한다.

 ```python
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize(h, color):
  z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
  plt.figure(figsize=(10,10))
  plt.xticks=([])
  plt.yticks=([])
  plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
  plt.show()
 ```

## 2. Load Data

```python
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

print()
print(f'Dataset: {dataset}')
print('='*20)
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]
print()
print(data)
print('='*40)

print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loop: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
print(f'len(x): {len(data.x)}, len(x[0]): {len(data.x[0])}')
```

```bash
Dataset: Cora()
====================
Number of graphs: 1
Number of features: 1433
Number of classes: 7

Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
========================================
Number of nodes: 2708
Number of edges: 10556
Average node degree: 3.90
Number of training nodes: 140
Training node label rate: 0.05
Has isolated nodes: False
Has self-loop: False
Is undirected: True
len(x): 2708, len(x[0]): 1433
Processing...
Done!
```

## 3. MLP 
### 3-1. 클래스 정의의
**MLP 클래스 정의**:
   - `MLP` 클래스는 `torch.nn.Module`을 상속한다.
   - `__init__` 메서드는 모델을 초기화한다:
     - 재현성을 위해 시드를 수동으로 설정한다.
     - 두 개의 선형 계층(`lin1`, `lin2`)을 정의한다:
       - `lin1`: 데이터셋의 특징 수에서 지정된 숨겨진 채널 수로 변환.
       - `lin2`: 숨겨진 채널 수에서 데이터셋의 클래스 수로 변환.
   - `forward` 메서드는 모델의 순전파를 정의한다:
     - 첫 번째 선형 계층과 ReLU 활성화 함수를 적용한다.
     - 훈련 중일 때 0.5의 확률로 드롭아웃을 적용한다.
     - 두 번째 선형 계층을 적용하여 출력을 생성한다.

**모델 초기화**:
   - 16개의 숨겨진 채널로 `MLP` 클래스의 인스턴스를 생성하고 출력한다.

```python
import torch
from torch.nn import Linear
import torch.nn.functional as F

class MLP(torch.nn.Module):
  def __init__(self, hidden_channels):
    super().__init__()
    torch.manual_seed(12345)
    self.lin1 = Linear(dataset.num_features, hidden_channels)
    self.lin2 = Linear(hidden_channels, dataset.num_classes)

  def forward(self, x):
    x = self.lin1(x)
    x = x.relu()
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.lin2(x)
    return x

model = MLP(hidden_channels=16)
print(model)
```

```bash
MLP(
  (lin1): Linear(in_features=1433, out_features=16, bias=True)
  (lin2): Linear(in_features=16, out_features=7, bias=True)
)
```

### 3-2. Model Train
```python
from IPython.display import Javascript
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

model = MLP(hidden_channels=16)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
  model.train()
  optimizer.zero_grad()
  out = model(data.x)
  loss = criterion(out[data.train_mask], data.y[data.train_mask])
  loss.backward()
  optimizer.step()
  return loss

def test():
  model.eval()
  out = model(data.x)
  pred = out.argmax(dim=1)
  testCorrect = pred[data.test_mask] == data.y[data.test_mask]
  testAcc = int(testCorrect.sum()) / int(data.test_mask.sum())
  return testAcc

for epoch in range(1, 401):
  loss = train()
  print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
```

```bash
....
Epoch: 387, Loss: 0.3372
Epoch: 388, Loss: 0.3821
Epoch: 389, Loss: 0.2899
Epoch: 390, Loss: 0.2618
Epoch: 391, Loss: 0.3027
Epoch: 392, Loss: 0.2968
Epoch: 393, Loss: 0.3082
Epoch: 394, Loss: 0.3408
Epoch: 395, Loss: 0.2697
Epoch: 396, Loss: 0.2699
Epoch: 397, Loss: 0.2998
Epoch: 398, Loss: 0.3067
Epoch: 399, Loss: 0.3163
Epoch: 400, Loss: 0.3159
```

### 3-3. Evalutation
```python
testAcc = test()
print(f'Test Accuracy: {testAcc:.4f}')
```
```bash
Test Accuracy: 0.5920
```

## 4. 2-layer GCN
### 4-1. 클래스 정의
```python
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
  def __init__(self, hidden_channels=16):
    super().__init__()
    torch.manual_seed(1234567)
    self.conv1 = GCNConv(dataset.num_features, hidden_channels)
    self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

  def forward(self, x, edge_index):
    x = self.conv1(x, edge_index)
    x = x.relu()
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.conv2(x, edge_index)
    return x

model = GCN()
print(model)
```
```bash
GCN(
  (conv1): GCNConv(1433, 16)
  (conv2): GCNConv(16, 7)
)
```

### 4-2. Train
```python
model = GCN()
model.eval()
out = model(data.x, data.edge_index)
print(out.shape)
visualize(out, color=data.y)
```

<p align="center">
<img width="800" alt="1" src="https://github.com/user-attachments/assets/d7c80b9d-6a0e-4abd-881e-44fc86f52d10">
</p>

```python
from IPython.display import Javascript
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
  model.train()
  optimizer.zero_grad()
  out = model(data.x, data.edge_index)
  loss = criterion(out[data.train_mask], data.y[data.train_mask])
  loss.backward()
  optimizer.step()
  return loss

def test():
  model.eval()
  out = model(data.x, data.edge_index)
  pred = out.argmax(dim=1)
  testCorrect = data.y[data.test_mask] == pred[data.test_mask]
  testAcc = int(testCorrect.sum()) / int(data.test_mask.sum())
  return testAcc

for epoch in range(1, 401):
  loss = train()
  print(f'Epoch: {epoch:3d}, Loss: {loss:.4f}')

```
```bash
....
Epoch: 385, Loss: 0.2146
Epoch: 386, Loss: 0.2294
Epoch: 387, Loss: 0.2336
Epoch: 388, Loss: 0.2365
Epoch: 389, Loss: 0.2612
Epoch: 390, Loss: 0.2742
Epoch: 391, Loss: 0.2523
Epoch: 392, Loss: 0.2296
Epoch: 393, Loss: 0.2360
Epoch: 394, Loss: 0.2283
Epoch: 395, Loss: 0.2498
Epoch: 396, Loss: 0.2070
Epoch: 397, Loss: 0.2391
Epoch: 398, Loss: 0.2255
Epoch: 399, Loss: 0.2216
Epoch: 400, Loss: 0.2379
```

### 4-3. Evaluation
```python
testAcc = test()
print(f'Test Accuracy: {testAcc:.4f}')
```
```bash
Test Accuracy: 0.8060
```

```python
model.eval()
out = model(data.x, data.edge_index)
visualize(out, color=data.y)
```

<p align="center">
<img width="800" alt="1" src="https://github.com/user-attachments/assets/0c4da4fa-89af-4d0e-b527-9dc1d54956a1">
</p>

# Reference
[1] [Graph Convolutional Networks: Introduction to GNNs](https://mlabonne.github.io/blog/posts/2022-02-20-Graph_Convolution_Network.html)    
[2] [Github](https://github.com/kimiyoung/planetoid/tree/master)
