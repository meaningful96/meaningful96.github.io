---
title: "[그래프 이론]그래프 머신러닝(딥러닝)"
categories: 
  - Graph
  
toc: true
toc_sticky: true

date: 2024-08-03
last_modified_at: 2024-08-03
---
<figure style="text-align: center; margin: auto;">
  <img width="1000" alt="1" src="https://github.com/user-attachments/assets/d32f9fb9-ad71-404e-a7bd-12726049a08b" style="display: block; margin: auto;">
  <figcaption style="font-size:70%; text-align: center; width: 100%; margin-top: 0px;">
    <em>ref) Stanford University <a href="https://web.stanford.edu/class/cs224w/">CS224W</a></em>
  </figcaption>
</figure>

그래프는 실생활에서 다양하게 사용되고 있다. 인스타그램, 메타, 링크드인과 같은 거대한 소셜 네트워크 안에서 팔로워 관계를 나타내는 소셜 네트워크(Social Network), 단백질간의 상호작용을 나타내는 그래프, 질병과 약물간의 상관 관계를 나타내는 그래프, 전자상거래 시스템에서 사용자와 구매 아이템 간의 상호 관계를 그려 놓은 사용자-아이템 그래프 등이 대표적이다.  

<p align="center">
<img width="600" alt="1" src="https://github.com/user-attachments/assets/35cf9e2d-8ea9-4b26-a094-656a7ee9ea34">
</p>

Deep learning 기법을 이용해 그래프의 정보를 학습하는 방법은 크게 세 가지로 구분된다. **전통적인 방식(Traditional Method)**은 그래프의 구조적 패턴을 직접 분석하고 비교하여 그래프 간의 유사성을 측정한다. **노드 임베딩(Node Embedding)**은 노드(= 정점, 엔티티)를 벡터 공간에 임베딩하여 그래프의 구조적 정보를 저차원 표현으로 변환한다. 마지막으로, **그래프 신경망(GNN)**은 그래프 데이터에서 학습을 통해 노드와 그래프의 표현을 최적화하여 다양한 그래프 관련 작업을 수행한다.

# 노드 임베딩(Node Embedding)
<p align="center">
<img width="600" alt="1" src="https://github.com/user-attachments/assets/fc71d824-338d-452e-88f7-316a8494c726">
</p>

노드 임베딩(Node Embedding)은 정점을 <span style="color:red">**저차원의 벡터 공간**</span>에 임베딩하여 그래프의 구조적 정보를 저차원 표현으로 변환하는 방법이다. 이는 **정점 간의 관계**와 그래프의 **구조적 정보를 보존**하면서 벡터 공간에서 표현하는 방법이다. 정점 임베딩의 목적은 그래프에서 수행되는 다양한 머신러닝 작업(예: 정점 분류(Node Classification), 링크 예측(Link Prediction), 그래프 클러스터링(Graph Clustering) 등)을 효과적으로 처리하기 위함이다.
