---
title: "[딥러닝]Evaluation Metric(평가 지표)"

categories: 
  - DeepLearning

toc: true
toc_sticky: true

date: 2024-06-28
last_modified_at: 2024-06-28
---

# Evaluation Metric

## 1. Classification (분류 모델)

주로 Knowledge Graph Completion (KGC), Recommandar System, Machine Learning 에서 많이 사용된다.

### 1) Accuracy (정확도)
<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/a6d237e8-7fec-41bd-b9f6-38ccff3babd5">
</p>

Accuracy (정확도)는 **실제 정답(Ground truth) 중 모델이 정답이라 예측한 비율**이다. 즉 모델이 예측한 전체 샘플들 중에 TP는 TP로, TN은 TN으로 맞춘 비율인 것이다. 이 평가지표는 직관적으로 정확하게 예측한 비율을 확인 할 수 있다. 하지만, False Negative와 False Positive의 비율이 동일한 symmetric한 데이터셋에서만 사용할 수 있다. 예를 들어, Ground Truth에서 정답과 오답인 비율이 극단적으로 1:9라고 가정했을 때, 만약 모델이 모든 예측을 False로 해버리면 이 모델의 정확도는 90%가 되는 것이다. 따라서 데이터셋의 분포가 불균형할 경우 다른 평가지표를 사용해야한다.

<br/>

### 1) Precision (정밀도)
<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/2e52b60f-ab58-4d65-8e08-b10cc6718b65">
</p>

Precision (정밀도)는 **모델이 정답이라 예측한 것 중 실제 정답의 비율**이다. 
<br/>


### 2) Recall (재현율)
<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/c2613c66-e54f-41e5-9e33-7c80c4937a41">
</p>


<br/>
