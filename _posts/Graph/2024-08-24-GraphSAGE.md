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

그래프에 존재하는 노드들은 여러 가지 정보를 포함한다. 예를 들어, 어떤 노드가 '사람'일 경우 국적, 성별, 나이 등등이 이 노드를 표현하는 추가적인 정보이다. 추가적인 정보들은 **특성(feature)**이라고 한다. GNN 계열의 모델들이 학습과정에서 이 특성 정보를 활용한다. 그래프에서 여러 노드들의 연결 관계를 나타낸 행렬을 **인접 행렬(adjacency matrix)**라고 한다. 그리고 각 노드들의 특성을 나타낸 행렬을 **특성 행렬(feature matrix)**라고 한다. 노드를 표현하는 노드 임베딩 행렬(Node representation matrix)은 보통 인접 행렬과 특성 행렬의 곱으로 이루어진다.

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

