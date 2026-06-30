---
title: "[그래프 AI]GraphSAGE(Inductive Representation Learning on Large Graphs)"
categories: 
  - Graph
  
toc: true
toc_sticky: true

date: 2024-08-24
last_modified_at: 2024-08-24
---
# GraphSAGE
## 1. GraphSAGE 배경
GCN이나 GAT는 반지도학습(Semi-Supervised Learning) 방식이다. 반면 비지도 학습(Unsupervised Learning) 방식은 채택한 모델은 GraphSAGE이다. Labeling이 되지 않은 데이터를 이용하거나, 큰 사이즈의 graph를 학습해야 하는 모델들은 일반적으로 비지도 학습에 속한다. 

GraphSAGE 이전의 연구들(e.g., GCN, GAT)은 주로 전통적인 그래프 임베딩 기법과 전통적인 신경망 아키텍처를 사용하여 노드와 그래프를 표현했다. 그러나 이러한 접근 방식들은 **대규모 그래프 데이터에서 학습 효율성이 떨어지고**, 계산 자원이 많이 소모된다는 한계가 있었다. 또한, 새로운 노드가 추가(Evolvoing Graph)되거나 그래프 구조가 변할 때마다 **전체 모델을 다시 학습**해야 하는 문제도 존재했다. 

GraphSAGE는 <span style="color:red">**고정된 크기의 그래프에 대한 노드 임베딩을 학습하는 Transductive learning 방식의 한계점을 지적하고, 새롭게 추가되는 노드들에 대해서도 임베딩을 생성할 수 있는 Inductive learning 방식을 제안**</span>한다.

# GraphSAGE Architecture
## 1. Embedding Generation

<p align="center">
<img width="800" alt="1" src="https://github.com/user-attachments/assets/26c4d210-2c9e-478f-8284-21b508689621">
</p>

그래프에 존재하는 노드들은 여러 가지 정보를 포함한다. 예를 들어, 어떤 노드가 '사람'일 경우 국적, 성별, 나이 등등이 이 노드를 표현하는 추가적인 정보이다. 추가적인 정보들은 **특성(feature)**이라 한다. GNN 계열의 모델들이 학습과정에서 이 특성 정보를 활용한다. 그래프에서 여러 노드들의 연결 관계를 나타낸 행렬을 **인접 행렬(adjacency matrix)**라 한다. 그리고 각 노드들의 특성을 나타낸 행렬을 **특성 행렬(feature matrix)**라고 한다. 노드를 표현하는 노드 임베딩 행렬(Node representation matrix)은 보통 인접 행렬과 특성 행렬의 곱으로 이루어진다.

참고로, **링크 예측(Link prediction)**은 그래프 위의 그래프에서 '다 빈치'와 '루브르 박물관' 같이 두 노드가 연결되어 있을 확률을 예측하는 것이다. 그래프의 노드 각각에 대한 임베딩을 직접 학습하게 되면, 새로운 노드가 추가되었을 때 그 새로운 노드에 대한 임베딩을 추론할 수 없습니다. 따라서 GraphSage는, 노드 임베딩이 아닌 **Aggregation Function을 학습하는 방법을 제안**합니다.

### 1) 알고리즘
<p align="center">
<img width="800" alt="1" src="https://github.com/user-attachments/assets/0fe7206d-4578-4403-980a-52f97011dd70">
</p>

위의 그림에서 빨간색 노드를 새롭게 추가된 노드라고 가정하자. 따라서 이 추가된 노드의 임베딩을 구해야 한다. GraphSAGE에서는 다음과 같은 과정을 통해 추가된 노드의 임베딩을 구하며, 알고리즘은 다음과 같다.

1. 거리($$k$$)를 기준으로 일정 개수의 이웃 노드(neighborhood node)를 샘플링한다.
2. GraphSAGE를 통해 학습된 aggregation function을 통해, 주변 노드의 특성으로부터 빨간 노드의 임베딩을 계산한다.
3. 이 임베딩을 기반으로 링크 예측 등 여러 downstream task에 이용한다.

<p align="center">
<img width="700" alt="1" src="https://github.com/user-attachments/assets/775fb98d-df6d-479f-b80e-72e0ef650a3a">
</p>

특정 노드의 임베딩을 계산할 때, 거리가 $$K$$만큼 떨어져 있는 노드에서부터 순차적으로 **특성 집계(feature aggregation)**을 적용한다. 하지만, 이를 위해서는 추가적으로 **배치(batch)를 샘플링**하는 방법과 **이웃 노드에 대한 정의**가 필요하다.

### 2) 배치 샘플링
<p align="center">
<img width="700" alt="1" src="https://github.com/user-attachments/assets/120e8dda-83d3-42b9-9e92-31ddf86c7aa5">
</p>

GraphSAGE에서 **배치 샘플링(batch sampling)**은 대규모 그래프에서 효율적으로 학습을 수행하기 위한 중요한 기법이다. 배치 샘플링을 통해 모델은 전체 그래프를 한 번에 처리하는 대신, 그래프의 일부를 샘플링하여 미니배치(mini-batch) 단위로 학습한다. 이를 통해 메모리 사용량을 줄이고, 계산 속도를 향상시키며, 대규모 그래프에서도 효과적으로 학습할 수 있게 된다.

GraphSAGE의 배치 샘플링은 다음과 같은 단계로 구성된다:

1. **루트 노드 선택 (Root Node Selection)**
  - 학습할 배치를 구성하기 위해 그래프에서 임의의 노드들을 샘플링하여 루트 노드로 선택한다. 이러한 루트 노드들은 해당 미니배치의 중심이 되며, 모델은 이 루트 노드들의 임베딩을 학습하게 된다.

2. **이웃 노드 샘플링 (Neighbor Node Sampling)**
  - 각 루트 노드에 대해, 일정한 수의 이웃 노드를 샘플링한다. GraphSAGE는 모든 이웃을 고려하는 대신, 각 노드의 k-hop 이웃 중 일부만 샘플링함으로써 계산 비용을 줄인다. 예를 들어, 2-계층 GraphSAGE에서 샘플링된 노드 집합은 1-계층 샘플링된 이웃들로부터, 그리고 이 1-계층 이웃들의 이웃으로부터 구성된다.
  - 이때, GraphSAGE는 각 레이어마다 정해진 수의 이웃을 샘플링하도록 하여, 샘플링된 이웃의 수가 기하급수적으로 늘어나지 않도록 제어한다. 예를 들어, 1-계층에서 이웃 노드를 10개 샘플링하고, 2-계층에서 각각의 이웃에 대해 10개의 이웃을 샘플링하면, 최종적으로는 각 루트 노드에 대해 최대 100개의 이웃 정보만 사용하게 된다.

3. **미니배치 학습 (Mini-batch Learning)**
  - 샘플링된 노드들과 이들의 이웃들을 기반으로 미니배치를 구성하여 모델을 학습한다. 이 과정에서 GraphSAGE는 샘플링된 이웃 노드들의 정보를 사용하여 루트 노드의 임베딩을 계산하고, 이 임베딩을 업데이트한다. 학습은 미니배치 단위로 반복되며, 각 배치가 그래프의 서로 다른 부분을 커버하도록 함으로써 모델은 전체 그래프의 구조적 정보를 효과적으로 학습할 수 있게 된다.

### 3) Neighbor Sampling
<p align="center">
<img width="600" alt="1" src="https://github.com/user-attachments/assets/c441a533-cf10-4daa-848e-a1fbbbdb710d">
</p>

GraphSAGE에서는 각 노드의 임베딩을 효율적으로 학습하기 위해 이웃 노드를 샘플링하는 방법을 사용한다. 노드 $$u$$의 이웃 $$\mathcal{N}(u)$$는 그래프 구조에 따라 정의되며, 모든 이웃을 사용하는 대신 계산 복잡도를 줄이기 위해 **유니폼 랜덤 샘플링 방식으로 일부 이웃을 선택**한다. 학습 과정의 각 반복(iteration)마다 노드 $$u$$의 이웃 $$\mathcal{N}(u)$$ 중에서 고정된 개수의 이웃을 무작위로 샘플링하여 모델에 사용하며, 이를 통해 모델은 대규모 그래프에서도 일정한 계산 비용으로 학습이 가능해진다. 샘플링할 이웃 노드의 수는 모델의 하이퍼파라미터로 설정되며, 각 레이어마다 샘플링이 반복되어 점진적으로 더 넓은 범위의 정보를 수집하게 된다. 이러한 방식을 **Uniform Random Draw**방식 이라고 하며, GraphSAGE는 <span style="color:red">**매 iteration마다 정해진 개수의 최근접 노드를 이웃으로 샘플링**</span>하는 것이다.  

## 2. Aggregation
GraphSAGE는 다시 한 번 강조하면 <span style="color:red">**Aggregation 함수**</span>를 학습하여 inductive learning이 가능하게 한다. 집계 함수(aggregator function)는 **이웃 노드들로부터의 정보를 취합하는 역할**을 합니다. 하지만 그래프 데이터의 특성 상, 노드의 **이웃들 간에는 어떤 순서가 없다**. 이러한 이유로 집계 함수는 symmetric하고 높은 수용력(high representational capacity)을 지녀야하며 학습이 가능해야 한다. 논문에서는 세 가지 Aggregator를 제안한다. Aggregator로 어떤 걸 사용 하는가에 따라 타겟 노드 자신의 정보를 포함할 수도, 포함하지 않을 수도 있다. 

**Mean Aggregator**  
<p align="center">
<img width="600" alt="1" src="https://github.com/user-attachments/assets/f39edf55-cd27-491a-9a03-4aa9fc42b6a1">
</p>

- Mean Aggregator는 주변 노드의 임베딩과 **자기 자신(ego node)의 임베딩**을 단순 평균한 후, 선형 변화와 ReLU를 적용해 줌으로써, 임베딩을 업데이트 한다.

**LSTM Aggregator**  
<p align="center">
<img width="600" alt="1" src="https://github.com/user-attachments/assets/1d0d7e94-86bd-4448-a245-4356c8d7a3ab">
</p>

- LSTM aggregator는 높은 수용력을 가진다는 장점이 있다. 하지만 LSTM 자체는 symmetric한 함수가 아니라는 문제가 있다. 따라서 본 연구에서는, 인풋 노드들의 순서를 랜덤하게 조합하는 방식을 취한다.
- LSTM Aggregator의 경우는 사실 Graph 구조에서는 부적합하다. 그 이유는, LSTM이 Sequence를 입력 받기 때문이고, 이 Sequence를 또 Sequential (순차적)하게 처리하기 때문이다. 이렇게 하면, 이웃 노드들의 정보에 순서가 매겨지게 되는데, **Graph에서 이웃 노드들의 순서는 무의미**하다.

**Pooling Aggregator**  
<p align="center">
<img width="600" alt="1" src="https://github.com/user-attachments/assets/2bdebf01-13c8-4206-adf9-c8ddb3922b42">
</p>

- 각 노드의 임베딩에 대해 **선형 변환(linear transformation)**을 수행한 뒤, **Element-Wise Max Pooling**을 통해 이웃 노드들의 정보를 aggregate하는 방식이다.

## 3. Training GraphSAGE
<p align="center">
<img width="800" alt="1" src="https://github.com/user-attachments/assets/e5ebacfb-6a56-4d10-a47d-196a9d2c61c4">
</p>

GraphSAGE는 대조 학습(Contrastive learning)을 통해 파라미터를 업데이트한다. 대조 학습을 위해서 Negative sampling을 해야하며, Negative sample은 $$k$$번의 **random walk를 통해 도달할 수 있는 엔티티**들로 만들어지며, 이 손실 함수의 목적은 Positive Sample(Ground Truth)에 대해서는 더 가깝게, Negative Sample에 대해서는 멀어지게 학습하는 것이다.

<center>$$J_G(z_u) = - \log \left( \sigma (z_u^\top z_v) \right) - Q \cdot \mathbb{E}_{v_n \sim P_n(v)} \log \left( \sigma (-z_u^\top z_{v_n}) \right)
$$</center>

다시 말해, $$z_v$$와 $$z_u$$는 random walk를 기반으로 이웃으로 설정된 Ground Truth 노드쌍이고, $$z_{v_n}$$은 $$z_u$$에 대한 negative 노드이다. 이웃 노드끼리는 유사도가 높은 임베딩을 갖도록하고, 이웃이 아닌 노드쌍에 대해서는 유사도가 낮아지도록 임베딩을 학습한다.

# Reference
\[1\] William L. Hamilton, Rex Ying, and Jure Leskovec.2018. Inductive representation learning on large graphs.  
\[2\] [\[논문리뷰\]GraphSage : Inductive Representation Learning on Large Graphs(2017)](https://velog.io/@dongdori/GraphSage-Inductive-Representation-Learning-on-Large-Graphs)
