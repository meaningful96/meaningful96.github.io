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

# Data Split
## Inductive Setting vs Transductive Setting

**Split Graph: Transductive Setting**    
<p align="center">
<img width="400" alt="1" src="https://github.com/user-attachments/assets/57a98570-2ce6-4b33-bba4-9d1b3cfadcfe">
</p>

**Transductive Setting**에서는 <span style="color:red">**Train/Valid/Test set이 모두 동일한 그래프에 존재**</span>한다. 즉, 하나의 큰 그래프를 여러 부분으로 나누어 Split하는 방식이다. 이 설정에서는 전체 그래프의 구조를 모두 알고 있는 상태에서, 노드 혹은 엣지를 예측하는 Node/Edge prediction task에 사용된다.

- Train/Valid/Test set이 **동일한 그래프를 공유**하며, 각 셋은 특정 노드와 엣지들에 대한 예측을 담당한다.
- 따라서 단순하게 **각 데이터셋에 서로 다른 노드가 포함**되도록 split한다.
- 이 방식은 Node classification이나 Edge prediction 같은 작업에 적합하다.
- 학습 시에는 그래프 전체를 사용해 Node embedding을 계산하지만, 학습에 사용되는 레이블블은 특정 노드에 한정된다.
- 따라서 새로운(Unseen) 노드나 엣지가 추가되었을때 예측을 하지 못한다.

**Splitting Graph: Inductive Setting**   
<p align="center">
<img width="400" alt="1" src="https://github.com/user-attachments/assets/7f770c23-689a-46bf-84c7-78825eb7561d">
</p>

**Inductive Setting**에서는 <span style="color:red">**Train/Valid/Test set이 각각 다른 그래프로 구성**</span>되어 있다. 즉, 여러 개의 독립된 그래프를 가지고 Split하며, 각 Split이 고유한 그래프를 관찰할 수 있다. 이 설정에서는 모델이 **Unseen 그래프에 대해 일반화**할 수 있는 능력을 요구한다.

- Train/Valid/Test set이 서로 다른 그래프를 포함하며, 하나의 Split은 다른 Split에서 관찰된 그래프를 볼 수 없다.
- 전체 Dataset은 여러 개의 그래프로 구성되어 있으며, 학습한 모델은 이전에 보지 못한 그래프에 대해서도 일반화할 수 있어야 한다.
- 이 방식은 Node prediction, Edge prediction, Graph prediction 작업 모두에 사용 가능하다.
- 예를 들어, 학습 시에는 1번과 2번 노드로 구성된 그래프를 사용해 학습한 뒤, validation 시에는 3번과 4번 노드로 구성된 그래프를 사용해 평가를 진행한다. 이때 학습된 모델이 새로운 그래프에서도 성능을 유지할 수 있어야 한다.

### 1) Node Classification을 위한 Split
<p align="center">
<img width="700" alt="1" src="https://github.com/user-attachments/assets/0372b4f4-0645-4cc4-a289-c9e0b9ec11c2">
</p>

**Transductive Node Classification**  
Transductive setting에서 Train/Validation/Test 모든 Split은 **동일한 그래프 구조**를 관찰할 수 있다. 즉, 그래프 전체를 알 수 있는 상태에서, 각 Split은 자신에게 할당된 노드의 레이블만을 가지고 학습과 평가를 진행한다.

- Training set은 자신에게 할당된 노드 B, C, F를 학습한다.
- Validation set은 노드 A, D로 validation을 수행한다.
- Test set은 노드 E를 통해 evaluation을 진행한다.
  
이 방식에서는 전체 그래프의 구조는 알고 있지만, 레이블은 각 Split에 할당된 노드에 대해서만 사용된다. 따라서 unseen 노드에 대한 예측은 불가능하다.

**Inductive Node Classification**  
Inductive setting에서는 Train/Validation/Test 각각의 Split이 **서로 독립적인 그래프로 구성**되어 있다. 이 방식은 각 Split이 서로 다른 그래프를 학습 및 평가하며, 모델이 새로운 그래프에 대해 일반화할 수 있는 능력을 평가한다.

- Training set은 첫 번째 그래프로 노드의 표현을 학습한다.
- Validation set은 두 번째 그래프로 validation을 수행한다.
- Test set은 세 번째 그래프로 evaluation을 진행한다.

이 방식에서는 모델이 unseen 그래프에서 일반화할 수 있어야 하며, 다양한 그래프 구조에 대한 Node classification이 가능하다.

<br/>

### 2) Graph Classification을 위한 Split
<p align="center">
<img width="700" alt="1" src="https://github.com/user-attachments/assets/db1f62cd-6ade-4ace-a229-baeb145cf011">
</p>

Graph Classification에서는 **unseen graph에 대한 예측을 테스트**해야 하므로, **Inductive setting만이 적합**하다. Inductive setting은 각 Split에 있는 Train/Validation/Test 데이터셋이 모두 **독립적인 그래프**로 구성되는 것을 의미한다.

Inductive setting에서 Graph Classification은 다음과 같은 방식으로 진행된다.
- Training set에는 여러 개의 독립적인 그래프가 포함되어 있으며, 모델은 이 그래프들을 사용해 학습을 진행한다.
- Validation set 역시 독립적인 그래프로 구성되어 있으며, 학습된 모델이 이 Validation 그래프에서 얼마나 잘 예측하는지를 평가한다.
- Test set은 학습 중에 보지 못한 새로운 그래프를 포함하며, 모델이 이 unseen graph에서 얼마나 잘 일반화하는지를 평가한다.

이 방식은 Graph Classification 작업에서 필수적이며, Transductive setting은 동일한 그래프 내에서만 학습과 예측이 가능하기 때문에 적합하지 않다. <span style="color:red">**Inductive setting에서는 새로운 그래프에서의 일반화 능력을 확인할 수 있다.**</span>

<br/>

### 3) Link Prediction을 위한 Split
<p align="center">
<img width="700" alt="1" src="https://github.com/user-attachments/assets/c5adecd6-e3f3-4cb4-9a92-3499001be0e0">
</p>

**Link prediction**은 missing edges를 예측하는 것이 목표인 unsupervised 또는 self-supervised 작업이다. 이 작업을 설정하는 과정은 다소 복잡한데, 그 이유는 레이블과 데이터셋의 분할을 우리가 직접 만들어야 하기 때문이다.

**Setting up Link Prediction**  
1. **Message Edges**:
   - 이 엣지들은 GNN의 **message passing 과정에서** 사용되며, 학습 과정에서 노드 간 정보를 교환하는 데 사용된다. 즉, GNN이 그래프의 구조와 패턴을 학습하는 데 중요한 역할을 한다.
   - GNN이 이 엣지들을 통해 그래프의 임베딩을 생성하고, 각 노드가 이웃 노드들로부터 정보를 받아 갱신된다.
   
2. **Supervision Edges**:
   - 이 엣지들은 모델이 **예측을 수행할 때**만 사용되며, 학습 과정에서는 직접적으로 사용되지 않는다.
   - Supervision Edges는 주로 평가 목적 또는 예측 정확도(정답 비교)를 확인하는 데 사용된다. GNN이 학습한 후, 노드 임베딩을 기반으로 **새로운 엣지**가 형성될 가능성을 예측할 때, 그 예측을 **실제 존재하는 엣지와 비교**하여 모델 성능을 평가하게 된다.
   - 또한 Supervision Edge는 학습 목표(Objective) 계산을 위해 사용되므로, 결국 모델이 올바르게 학습되었는지 확인하는 역할을 한다.

Link prediction이 까다로운 이유는, 레이블이 없는 상태에서 수행되는 unsupervised 또는 semi-supervised 학습 방식이라는 점이다. 따라서 Edge 자체에 대한 supervision이 필요하다. 이를 위해 Message Edges와 Supervision Edges를 나누어 설정하며, 학습 과정에서는 Message Edges만을 사용하고, Supervision Edges는 예측 시에만 고려된다.

### 3-1) Inductive Link Prediction
<p align="center">
<img width="700" alt="1" src="https://github.com/user-attachments/assets/177e5aea-70f3-4430-b7fe-62a3c0e1d041">
</p>

**Inductive Setting**은 **독립적인 3개의 그래프**를 각각 Training set, Validation set, Test set으로 나누고, 각각의 set에서 edge들을 학습에 사용되는 message edge와 예측에 사용되는 supervision edge로 나눈다.

- **Message edge (검은 선)**:
  - GNN 모델이 학습하는 데 사용하는 엣지이다.
  - 노드 간의 정보를 전달하는 경로(**Message Passing**)로, 그래프 구조를 학습할 때 중요한 역할을 한다.

- **Supervision edge (점선)**:
  - 예측을 평가하거나 학습할 때 사용하는 엣지이다. 모델의 출력과 비교되는 정답 레이블의 역할을 한다.
  - 그러나 supervision edges는 GNN의 입력으로 사용되지 않는다.
  - 즉 모델이 학습하는 동안에는 고려되지 않으며, **예측 결과와 비교할 때만 사용**된다.

<br/>

### 3-2) Transductive Link Prediction
<p align="center">
<img width="700" alt="1" src="https://github.com/user-attachments/assets/251a89c8-7a73-4bc6-9e08-d2bd64e19cae">
</p>

**Transductive Setting**은 Link Prediction에서 많이 사용되는 방식으로, <span style="color:red">**하나의 큰 그래프를 여러 그래프로 나누어**</span> 학습 및 예측을 수행한다. 이 방식은 Training, Validation, Test 세 단계에서 각기 다른 엣지를 사용하는 방법을 따른다. 

- **Edges**
  - **Training message edges**: 학습에 사용되는 엣지.
  - **Training supervision edges**: 학습된 메시지 엣지를 바탕으로 예측에 사용되는 엣지.
  - **Validation edges**: 검증 단계에서 예측해야 하는 엣지.
  - **Test edges**: 테스트 단계에서 최종적으로 예측해야 하는 엣지. 

**(1) Training (학습 단계)**  
Message edges와 Supervision edges를 사용하여 학습을 진행한다. Message edge는 GNN이 학습할 때 사용하는 엣지로, 그래프에서 노드 간의 정보를 전달하며, 모델이 관계를 학습하는 데 중요한 역할을 한다. Supervision edge는 학습된 메시지 엣지를 기반으로 예측을 수행하며, 주로 손실함수를 계산하는 데 사용된다.

**(2) Validation (검증 단계)**  
학습한 Training message edges와 Training supervision edges를 입력으로 사용하여 Validation edges를 예측한다. Validation 단계에서는 학습된 모델의 성능을 평가하기 위해 기존 학습 데이터에서 얻은 정보뿐만 아니라, 검증을 위한 새로운 엣지를 예측하여 성능을 측정한다.

**(3) Test (테스트 단계)**  
학습과 검증에서 사용된 모든 엣지를 입력으로 하여 Test edges를 예측한다. Test 단계는 학습 및 검증에서 사용한 모든 데이터를 활용하여 최종적으로 예측 성능을 평가하는 과정이다. 여기서, 학습된 엣지들과 검증된 엣지들을 모두 사용하여 테스트 엣지를 맞추는 데 활용한다.

<p align="center">
<img width="700" alt="1" src="https://github.com/user-attachments/assets/19340d65-1378-4582-9460-ea64bbfe3a59">
</p>

<br/>
<br/>

# Reference
\[1\] [CS224W 강의](https://web.stanford.edu/class/cs224w/)

