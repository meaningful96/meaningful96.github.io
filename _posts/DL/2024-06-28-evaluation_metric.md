---
title: "[딥러닝]Evaluation Metric(평가 지표) - (1) 분류 성능 지표"

categories: 
  - DeepLearning

toc: true
toc_sticky: true

date: 2024-06-28
last_modified_at: 2024-06-28
---

# Evaluation Metric

## 1. 분류 기반 모델의 평가 지표

분류 기반 모델(Classification based Model)들의 평가지표이다.

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

<center><span style="font-size:110%">$$\text{Precision} \; = \; \frac{\text{TP}}{\text{TP} + \text{FP}}$$</span></center>

위의 식은 Precision을 나타낸다.

<br/>

### 2) Recall (재현율)
<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/c2613c66-e54f-41e5-9e33-7c80c4937a41">
</p>

Recall (재현율)은 Precision과는 달리 **실제 정답 중 모델이 정답으로 예측한 것에 대한 비율**이다. Recall은 True Positive Rate(TPR) 혹은 통계학에서는 Sensitivity(민감도)라고도 한다. Recall은 낮은 False Negative(FN)의 비율이 실험에서 중요할 경우 좋은 측정법이다. 하지만, FP를 전혀 반영하지 못한다는 단점이 있다.

<center><span style="font-size:110%">$$\text{Recall} \; = \; \frac{\text{TP}}{\text{TP} + \text{FN}}$$</span></center>

위의 식은 Recall을 나타낸다.

<br/>

### 3) F1-Score

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/801b3a1d-6fd8-4c25-9256-f15623b365a6">
</p>

F1 Score는 **Precision과 Recall의 조화평균**이다. F1-Score는 $$0 \sim 1$$사이의 값을 가지며, 1에 가까울수록 모델의 성능이 좋은 것이다. 조화평균(harmonic mean)은 산술평균(arithmetic mean)과 달리 데이터 값들의 역수를 더한 후 그 역수의 산술평균을 구하는 방식으로 계산된다. 이는 특히 데이터 값들이 서로 상호 의존적인 경우나 비율을 나타내는 경우에 유용하다. 

<center><span style="font-size:110%">$$\text{F1-Score} \; = \; 2 \times \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$</span></center>

위의 식은 F1-Score를 나타낸다.

<br/>

### Ex) Precision, Recall, F1-Score in Recommender system

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/956016e2-dfa5-4fe9-a380-1a5700a7e7ce">
</p>

### 4) Area Under the ROC Cureve (AUC-ROC Curve)

Receiver Operating Characteristic Curve (ROC Curve)는 "수신자 판단 곡선"이라는 뜻을 가진다. ROC Curve는 이진 분류 모델의 성능을 시각적으로 평가하는 지표로, 임계값을 변화시켰을 때의 Sensitivity나타낸다. **Sensitivity**란 위에서 언급하였듯이, Recall과 식이 동일하며, 의미를 이해하기 쉽도록 이를 <span style="color:red">**True Positive Rate(TPR)**</span>이라고 한다. 그러면 **1 - Specificity**는 자연스럽게 <span style="color:red">**False Positive Rate(FPR)**</span>가 된다. 

여기서 주의해야할 점은, TPR은 Sensitivity(민감도)를 나타내고, FPR은 1에서 Specificity(특이도)를 뺀 값이다. 
- Sensitivity(민감도): True Positive Rate (TPR)로, 실제 Positive 샘플 중에서 모델이 정확히 Positive로 예측한 비율
- Specificity(특이도): True Negative Rate (TNR)로, 실제 Negative 샘플 중에서 모델이 정확히 Negative로 예측한 비율

TPR은 다시 말해 모델이 실제 Positive 샘플을 얼마나 잘 감지하는지를 나타낸다. 그리고 FPR은 모델이 실제 Negative 샘플을 잘못해서 Positive로 예측하는 비율을 나타낸다. 이를 수식으로 표현하면 다음과 같다. (TPR과 반대 개념) 이를 수식으로 표현하면 다음과 같다.

<center><span style="font-size:110%">$$\text{True Positive Rate(TPR)} \; = \; \frac{\text{TP}}{\text{TP} + \text{FN}}$$</span></center>  
<center><span style="font-size:110%">$$\text{False Positive Rate(FPR)} \; = \; \frac{\text{FP}}{\text{FP} + \text{TN}}$$</span></center>

이 두 개념을 통해서 $$x$$축에서는 FPR을, $$y$$축에는 TPR을 표시해 plot을 만들 수 있으며, Skew와 면적을 관찰할 수 있다. 이 때, 면적이 Area Under the ROC Curve (AUC)가 되는 것이다. AUC는 다음과 같은 특성을 가진다.

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/dd4bcfef-81b7-4007-b0ce-2ced518f3c36">
</p>

- AUC 값이 1에 가까울수록 모델의 성능이 좋다. 이 경우, 모든 임계값에서 TPR이 높고 FPR이 낮은 모델이다.
- AUC 값이 0.5에 가까우면, 모델의 분류 성능이 랜덤 수준에 가깝다는 의미이다. 즉, 모델이 클래스를 제대로 분류하지 못하고 있는 것을 나타낸다.
- AUC 값이 0보다 작으면, 모델이 랜덤한 예측보다 더 나쁜 성능을 보이는 경우이다. 이는 분류 모델이 잘못된 예측을 하고 있다는 것을 나타낸다.

<center><span style="font-size:110%">$$\text{AUCC} \; = \; \frac{1}{2} \sum_{i=1}^{m-1}(x_{i+1} - x_i)(y_i + y_{i+1})$$</span></center>

따라서 AUC는 ROC 커브를 요약하는 한 지표로, <span style="color:red">**AUC 값이 높을수록(즉, Positive와 Negative를 잘 구분하는)**</span> 성능이 좋은 모델이다. AUC값에 따른 case를 시각화하면 다음과 같다.

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/9e14633c-d94b-409e-83c7-4e8406ad1da6">
</p>

이처럼, AUC가 1일경우 모델은 모든 positive를 positive로, negative를 negative로 정확하게 분류함을 알 수 있다. 반면, AUC가 0.5일 경우 모델은 랜덤하게 분류하는 것과 동일하다. 즉, positive와 negative 샘플에 대한 discrimination capacity가 없다. 마지막으로 AUC가 0일 경우, 모델은 완전히 반대로 예측하게 된다.

물론, AUC는 이진 분류(binary classification)뿐만 아니라 다중 분류(multi-class)에서도 사용 가능하다. Multi-class 문제를 해결하기 위해 "One vs All" 방법론을 사용한다. 이는 각 클래스를 나머지 모든 클래스와 비교하여 N개의 ROC 곡선을 그리는 방식이다. 예를 들어 $$X, Y, Z$$ 세 개의 클래스가 있다고 가정할 때, 세 가지의 ROC 곡선을 그리면 된다.
- $$X$$를 $$Y$$와 $$Z$$에 대해 분류하는 ROC 곡선
- $$Y$$를 $$Z$$와 $$X$$에 대해 분류하는 ROC 곡선
- $$Z$$를 $$X$$와 $$Y$$에 대해 분류하는 ROC 곡선

결론적으로 AUC를 사용하였을 경우 다음과 같은 장단점이 있다.
- Pros
  - **Scale Variance**: AUC는 절대적인 값이 아닌 예측의 순위를 측정하므로, 스케일에 무관하다. 즉, 모델의 예측이 얼마나 잘 랭킹되어 있는지를 측정한다.
  - **Classification Threshold Invariant**: AUC는 분류 임계값의 선택과 무관하게 모델의 예측 품질을 측정합니다. 이는 임계값이 변하더라도 모델의 성능을 평가하는 데 유리하다.
 
- Cons
  - 스케일 불변성(Scale variance)은 잘 조정된 확률 출력이 필요할 때 바람직하지 않다.
  - 분류 임계값 불변성(Classification Threshold Invariant)은 비용의 큰 차이가 있는 경우 바람직하지 않다.

<br/>
