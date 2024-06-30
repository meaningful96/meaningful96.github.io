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
