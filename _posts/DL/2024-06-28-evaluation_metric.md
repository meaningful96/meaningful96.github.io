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

Precision (정밀도)는 **모델이 정답이라 예측한 것 중 실제 정답의 비율**이다. 이 평가지표는 낮은 False Positive(FP)의 비율이 중요할 때 사용할 수 있는 좋은 측정법이다. 하지만, False Negative는 전혀 측정하지 못한다는 단점이 있다.
<br/>


### 2) Recall (재현율)
<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/c2613c66-e54f-41e5-9e33-7c80c4937a41">
</p>

Recall (재현율)은 Precision과는 달리 **실제 정답 중 모델이 정답으로 예측한 것에 대한 비율**이다. Recall은 True Positive Rate(TPR) 혹은 통계학에서는 Sensitivity(민감도)라고도 한다. Recall은 낮은 False Negative(FN)의 비율이 실험에서 중요할 경우 좋은 측정법이다. 하지만, FP를 전혀 반영하지 못한다는 단점이 있다.

### 3) F1 Score

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/801b3a1d-6fd8-4c25-9256-f15623b365a6">
</p>

F1 Score는 **Precision과 Recall의 조화평균**이다. F1-Score는 $$0 ~ 1$$사이의 값을 가지며, 1에 가까울수록 모델의 성능이 좋은 것이다. 조화평균(harmonic mean)은 산술평균(arithmetic mean)과 달리 데이터 값들의 역수를 더한 후 그 역수의 산술평균을 구하는 방식으로 계산된다. 이는 특히 데이터 값들이 서로 상호 의존적인 경우나 비율을 나타내는 경우에 유용하다. 

<center><span style="font-size:110%">$$2 \times \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$</span></center>

F1-Score는 수식으로 표현하면 위와 같이 표기한다.




<br/>
