---
title: Chapter 0.Cross Validation

categories:
  - ML
tags:
  - [ML,DL,Machine learning, Deep Learning]

toc: true
toc_sticky: true

date: 2022-11-21
last_modified_at: 2022-11-21 
---

## Cross Validation
### 1. Dataset에 관하여
Machine Learning과 Deep Learning에 있어서 가지고 있는 데이터를 전처리하고 가고하는 과정은 매우 중요하다. 전처리 과정을 거친 데이터들은 training할 준비가 된 데이터들이다.
여기서 가지고 있는 Dataset을 모두 training에 활용하지 않는다. Dataset을 왜 모두 tarining에 쓰지 않을까? 가장 근본적이고, 표면적인 이유는 결국 모든 Data들은 '돈'과 직결
되기 때문이다.

1) 새로운 data는 결국 돈!! money !!
2) 따라서 가지고 있는 Dataset을 쪼개서 활용하는 것이 바람직하다.

### 2. Training set, Test set
<img width="208" alt="1" src="https://user-images.githubusercontent.com/111734605/202988535-4672985f-2840-4bbf-96cf-6885d64ee8b6.png">
위의 그림처럼 하나의 Dataset을 두 개로 나눈 경우 Dataset의 90% 정도는 Training을 하는데 사용하고, 나머지 10%로는 Prediction을 하는데 사용한다. 하지만 여기서 문제가 
발생한다. 흔히 말하는 변수들, Parameter에는 크게 두 가지 종류가 존재한다. 

- Model Parameter
- HpyerParameter

이때, Model Parameter라는 것이 결국 Training을 통해서 찾게되는 prarameter이다. 그럼 Hpyerparameter는 무엇인가? 바로, 우리가 프로그래밍시 직접 정해줘야하는 환경
변수들이다. Dataset을 두 개로 나누면 결국 Hyperparameter를 적합하는데 Test set이 이용이된다. Test set은 반드시 Prediction을해 성능 평가하는데만 사용되어야 한다.
따라서, 두 개로 나누는 것은 부적합하다. 이를위해, 아주 적합한 Hpyerparameter를 찾는데 새로운 분할 방식이 필요하다.

### 3. Validation Set
궁극적으로 하나의 머신러닝, 딥러닝 알고리즘을 학습시키기 위해서 Dataset을 세 개의 part로 쪼개야 한다. **1)Training Set**, **2) Test Set**, **3) Validation Set**

<img width="224" alt="2" src="https://user-images.githubusercontent.com/111734605/202989971-d2d6e812-a466-4641-81f5-efe2e01a3fb2.png">


### 4. k-fold Cross Validation
<img width="289" alt="3" src="https://user-images.githubusercontent.com/111734605/202990137-8d15a64c-7d84-4762-923c-f5fb947c95ca.png">

k-fold cross validation은 좀 더 좋은 Hpyerparameter를 찾기 위해서 위의 그림과 같이 Training set과 validation을 교차시켜 바꿔가면서 Training하는 것이다. 이를 통해
좀 더 좋은 Hpyerparameter를 적합할 수 있다.

### 5. Summary
<img width="349" alt="4" src="https://user-images.githubusercontent.com/111734605/202990430-c0dc5400-6a74-45ce-b7da-6ef23f883e81.png">
