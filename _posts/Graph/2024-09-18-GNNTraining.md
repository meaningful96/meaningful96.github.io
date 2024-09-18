---
title: "[그래프 AI]GNN 학습과 데이터 분할(Split)"
categories: 
  - Graph
  
toc: true
toc_sticky: true

date: 2024-09-18
last_modified_at: 2024-09-18
---

# Training GNN

## Supervised Learning vs Unsupervised Learining
<p align="center">
<img width="700" alt="1" src="https://github.com/user-attachments/assets/77f424e2-3c49-4e56-8f92-369cefa7bca4">
</p>

**Supervised Learning**은 정답 레이블이 있는 데이터를 이용해 학습하는 방식이며, **Unsupervised Learning**은 레이블 없이 데이터의 패턴이나 구조를 학습하는 방식이다.

Supervised Learning에서는 Ground Truth가 레이블로부터 오고, Unsupervised Learning에서는 Unsupervised Signal로부터 온다. **Supervised Learning**에서는 <span style="color:red">**학습을 위해 제공되는 정답 레이블(Label)이 존재**</span>하며, 이 레이블은 외부 데이터를 통해 얻는다. 모델은 예측 값과 실제 레이블 간의 차이를 기반으로 손실함수를 계산하여 학습을 진행하며, 이때 데이터셋에 존재하는 정답 레이블이 Ground Truth가 된다.

반면, Unsupervised Learning에서는 명확한 레이블이 주어지지 않고, 대신 외부에서 제공된 Unsupervised Signal을 기반으로 학습이 이루어진다. 이 신호는 데이터 간의 패턴이나 구조를 발견하는 데 사용되며, 학습의 목적은 데이터의 관계를 파악하는 것이다. 따라서, **Unsupervised Learning**에서 Ground Truth는 레이블이 아닌 <span style="color:navy">**패턴이나 구조적 정보**</span>로부터 나온다.

**GNN(Graph Neural Network)**에서도 예측을 수행하려면 Ground Truth가 필요하다. Supervised GNN의 경우 Ground Truth는 노드나 엣지의 레이블로부터 오고, Unsupervised GNN의 경우 Ground Truth는 그래프 내에서 발견되는 구조적 패턴이나 노드 간 유사성과 같은 Unsupervised Signal에서 비롯된다. 이처럼 Supervised와 Unsupervised 두 방식 모두 모델 학습에 중요한 정보를 제공한다.

## 손실 함수(Loss)
손실 함수에 대표적으로 분류 문제를 위한 Cross-Entropy Loss와 회귀 문제를 위한 MSE가 있다.

<p align="center">
<img width="700" alt="1" src="https://github.com/user-attachments/assets/29249568-98bb-4818-8bd2-6f8366f3937a">
</p>

**Cross-Entropy Loss**는 모델이 출력한 확률 분포와 원-핫 인코딩된 정답 레이블 간의 차이를 측정하는 손실 함수이다. 이 손실 함수는 **정답 레이블에 해당하는 클래스의 확률값을 최대화하고, 나머지 클래스의 확률값을 최소화**하는 것을 목표로 한다. 이를 통해 최종적으로 손실 함수를 최소화(minimizing)하며, 모델이 더 정확한 예측을 하도록 학습을 유도한다.


<p align="center">
<img width="700" alt="1" src="https://github.com/user-attachments/assets/d733464f-52d5-4c16-9495-53314d7f3a68">
</p>

**MSE(Mean Squared Error)**는 예측 값과 Ground Truth(실제 값) 간의 차이를 제곱하여 평균을 구하는 손실 함수이다. 모델은 예측 값과 Ground Truth 간의 오차를 최소화하는 방향으로 학습을 진행하며, 이때 오차는 두 값의 차이를 제곱한 값으로 계산된다. 따라서, MSE는 **예측 값과 Ground Truth 사이의 차이를 제곱한 후 평균을 구해 그 값을 최소화**하려는 것이다. 주로 회귀 문제에서 사용되며, 예측 값이 실제 값과 얼마나 가까운지를 측정하는 데 유용하다.

## Evaluation Metric
<p align="center">
<img width="700" alt="1" src="https://github.com/user-attachments/assets/15dd12b4-30fd-43a1-a0cb-6771cd4c29bb">
</p>

**Regression (회귀)**    
a. Root Mean Square Error (RMSE)    
예측 값과 실제 값 간의 오차의 제곱 평균에 루트를 씌운 값이다. 값이 클수록 예측 성능이 나쁘며, 큰 오차에 더 민감한 평가 지표다.  

b. Mean Absolute Error (MAE)    
예측 값과 실제 값 간의 절대 차이의 평균이다. 오차의 크기를 그대로 반영하며, 값이 클수록 예측 성능이 떨어진다. MAE는 RMSE에 비해 큰 오차에 덜 민감하다.  

**Classification (분류)**     
a. Multi-Class Classification: Accuracy (Acc)    
전체 예측 중에서 올바르게 예측된 비율이다. 다중 클래스 분류에서 많이 사용되며, 전체 예측이 얼마나 정확한지 평가한다.  

b. Binary Classification: Accuracy, Precision, Recall  
- Accuracy: 이진 분류에서의 정확도. 전체 예측 중에서 맞춘 비율.
- Precision: 참이라고 예측한 것 중에서 실제로 참인 비율.
- Recall: 실제로 참인 것 중에서 모델이 참이라고 예측한 비율.

c. Metric agnostic to classification threshold: ROC-AOC  
ROC(Receiver Operating Characteristic) 곡선과 AUC(Area Under the Curve)는 분류 기준(Threshold)에 상관없이 모델의 성능을 평가하는 지표다. 곡선 아래 면적이 클수록 성능이 좋다.  

Accuracy, Precision, Recall, ROC-AOC Curve에 관한 자세한 설명은 [Evaluation Metric(평가 지표) - (1) 분류 성능 지표](https://meaningful96.github.io/deeplearning/evaluation_metric/)에 있다.
