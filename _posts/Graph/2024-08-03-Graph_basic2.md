---
title: "[그래프 이론]그래프 머신러닝(딥러닝)"
categories: 
  - Graph
  
toc: true
toc_sticky: true

date: 2024-08-03
last_modified_at: 2024-08-03
---


<p align="center">
<img width="600" alt="1" src="https://github.com/user-attachments/assets/35cf9e2d-8ea9-4b26-a094-656a7ee9ea34">
</p>

Deep learning 기법을 이용해 그래프의 정보를 학습하는 방법은 크게 세 가지로 구분된다. **전통적인 방식(Traditional Method)**은 그래프의 구조적 패턴을 직접 분석하고 비교하여 그래프 간의 유사성을 측정한다. **노드 임베딩(Node Embedding)**은 노드(= 정점, 엔티티)를 벡터 공간에 임베딩하여 그래프의 구조적 정보를 저차원 표현으로 변환한다. 마지막으로, **그래프 신경망(GNN)**은 그래프 데이터에서 학습을 통해 노드와 그래프의 표현을 최적화하여 다양한 그래프 관련 작업을 수행한다.

# 노드 임베딩(Node Embedding)
<p align="center">
<img width="600" alt="1" src="https://github.com/user-attachments/assets/fc71d824-338d-452e-88f7-316a8494c726">
</p>

노드 임베딩(Node Embedding)은 정점을 <span style="color:red">**저차원의 벡터 공간**</span>에 임베딩하여 그래프의 구조적 정보를 저차원 표현으로 변환하는 방법이다. 이는 **정점 간의 관계**와 그래프의 **구조적 정보를 보존**하면서 벡터 공간에서 표현하는 방법이다. 정점 임베딩의 목적은 그래프에서 수행되는 다양한 머신러닝 작업(예: 정점 분류(Node Classification), 링크 예측(Link Prediction), 그래프 클러스터링(Graph Clustering) 등)을 효과적으로 처리하기 위함이다.
