---
title: "[논문리뷰]DynaSemble: Dynamic Ensembling of Textual and Structure-Based Models for Knowledge Graph Completion"

categories: 
  - GR
  
toc: true
toc_sticky: true

date: 2024-07-24
last_modified_at: 2024-07-24
---

*Nandi, A., Kaur, N., Singla, P., & Mausam*. (2023, November 7). **DynASemble: dynamic ensembling of textual and Structure-Based models for knowledge graph completion**. arXiv.org. [https://arxiv.org/abs/2311.03780](https://arxiv.org/abs/2311.03780)

# Problem Statement

<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/090acad1-a9f8-484d-b22f-a9cf8ef3970f">
</p>

<span style="font-size:110%">**Knowledge Graph Completion**</span>  
이 논문에서 풀고자 하는 문제는 Knowledge Graph Completion (KGC)이다. Knowledge Graph는 불완전하다. 즉, 누락된 지식도 많고, 인간이 찾지 못하는 패턴 또한 존재한다. KGC는 모델을 통해 이런 불완전한 KG를 complete하게 완성시키는 것을 목표로 한다.

- **Input**: $$(h, r, ?)$$ or $$(?, r, t)$$.
  - ($$h, r, t$$): 각각 head entity, relation, tail entity

- **Output**: 타겟 엔티티 $$?$$

<br/>

<span style="font-size:110%">**Previous Study**</span>
- **Embedding-based model(임베딩 기반 모델)**
  - 구조 기반 모델이라고도 불리며, NBFNet과 RGHAT와 같은 모델들이 이에 해당한다. 그래프 신경망(GNN)을 사용하여 KG의 연결 구조를 활용한다.
  - KG의 구조 정보를 활용할 수 있지만, 자연어 정보를 사용하지 못한다.
- **Text-based model(텍스트 기반 모델)**
  - SimKGC와 HittER와 같은 모델들이 있다. KG에 존재하는 많은 자연어 정보를 통해 KG의 지식을 학습한다.
  - 풍부한 자연어 정보를 활용할 수 있지만, 구조적인 정보를 활용하지 못한다. 

<br/>
<br/>

# Method

<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/d7464b86-3d03-4ac5-9754-25e4d16ba8d5">
</p>

모델은 매우 간단하다. 여러 가지 모델을 단순하게 앙상블(Ensemble) 하는 것이기 때문에 model-agnostic하다. 모델은 크게 다섯 단계를 거쳐 학습을 하게 된다.

<span style="font-size:110%">**1. Min-max Normalization**</span>
먼저 각 모델의 출력 임베딩 $$M_i$$의 벡터 크기를 맞춰주기 위해서 normalization을 한다. 본 논문에서는 Max-min normalization을 통해 $$0 \sim 1$$ 범위로 통일시킨다.

<span style="font-size:110%">**2. 모델 $$M_i$$의 Score distribution)**</span>
다음으로 입력 쿼리($$q$$)에 대한 normalized된 출력 임베딩 $$M_i(h, r, t)$$의 평균과 분산을 구한 후 concat한 것이다. 

<span style="font-size:110%">**3. Query-Dependent Weights**</span>
각 모델의 score distribution이 구해지면, 모든 score distribution을 concat하여 **2-layer MLP**에 입력시킨다. 이렇게 함으로써 각 모델에 대한 가중치가 결정된다. 이 때, 모델이 총 $$k$$개가 있으면 MLP의 입력은 $$k$$개의 평균과 분산을 입력받으로 $$2k$$가 되고, 각 모델의 가중치를 출력해야하므로 출력 벡터(가중치)의 크기는 $$k$$이다.

<span style="font-size:110%">**4. Ensemble**</span>
positive의 가중치와 negative의 가중치를 각각구하고, 이를 모델의 normalized된 출력 벡터(=특징 벡터)에 곱해준다. 이를 각각 양성 앙상블 스코어(Positive Ensemble score)와 음성 앙상블 스코어(Negative Ensemble Score)

<span style="font-size:110%">**5. Training Loss**</span>
구해진 양성 및 음성 앙상블 스코어를 가지고 **Margin Ranking Loss**를 손실함수로 하여 모델을 학습한다.

<br/>
<br/>

# Experiments
<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/e6dd707e-9584-4b32-8cfd-7f50f23c4e05">
</p>

실험 결과 여러 데이터셋에서 SOTA를 달성.

<br/>
<br/>

# Limitations and Contributions
- **Contribution**
  - 높은 성능 달성
  - Embedding based model과 Text based model을 적절하게 결합함.
 
- **Limitation**
  - Novelty가 없음. 기존의 모델들의 출력을 그대로 활용하기 때문에 2-layer MLP이외에 추가되는게 없음.
  - Reproduce가 제대로 안됨.
  - 모델에서 결과가 가장 좋은 SimKGC, NBFNet모두 매우 크기가 큰 모델이라 병렬적으로 학습이 거의 불가능함.
  - 따라서 학습 시간이 매우 오래 걸릴 것으로 추정됨.  

