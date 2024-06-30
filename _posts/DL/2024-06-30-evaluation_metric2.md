---
title: "[딥러닝]Evaluation Metric(평가 지표) - (2) 순위 성능 지표"

categories: 
  - DeepLearning

toc: true
toc_sticky: true

date: 2024-06-30
last_modified_at: 2024-06-30
---

# Evaluation Metric

이전 포스터 [\[딥러닝\]Evaluation Metric(평가 지표)](https://meaningful96.github.io/deeplearning/evaluation_metric/#site-nav)에 이어서 순위 기반의 모델(Ranking Based Model)들의 성능을 측정하는 순위 성능 지표에 대해 알아보겠다.

## 2. 순위 기반 모델의 평가 지표

순위 기반 모델(Ranking based Model)들의 평가지표이다. Mean Rank(MR), Mean Reciprocal Rank(MRR), Hits@k, NDCG등이 있다.

### 1) Mean Rank(MR)

**MR**은 매우 간단한 개념이다. 모델이 예측한 샘플들의 순위의 평균을 의미한다. 수식은 다음과 같다. $$N$$은 테스트한 샘플의 수이고, $$rank_i$$는 $$i$$번째 샘플의 순위이다.

<center><span style="font-size:110%">$$\text{MR} \; = \; \frac{1}{N} \sum_{i=1}^N rank_i$$</span></center> 

예를 들어, 한 학생이 5번의 대회에 참가해 각각 1,3,3,5,2 등을 차지했다고 가정해보자. 이 때의 MR은 (1+3+5+5+2)/5 = 3.2가 된다. 즉, 평균적으로 이 학생은 3.2등을 한 것이다.

### 2) Mean Reciprocal Rank (MRR)

**MRR**은 실제 정답의 순위의 역수를 평균 낸 것이다. 추천 시스템, Knowledge Graph Completion, 정보 검색 등 여러 분야에서 자주 사용된다. <span style="color:gold">**MRR이 1에 가까울수록 모델의 성능이 좋은 것**</span>이다.

<p align="center">
<img width="700" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/3b390cdc-d0e7-4886-bde1-caf543e15f49">
</p>

