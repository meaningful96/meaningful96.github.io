---
title: Transductive Learning vs Inductive Learning

categories:
  - DeepLearning
tags:
  - [DL,Graph]

toc: true
toc_sticky: true

date: 2022-12-30
last_modified_at: 2022-12-30 
---

```
Trasnduction is reasoning from observed, specific training cases to specific test cases.

In contrast, induction is reasoning from observed training cases to general rules, wihch
are then applied to the test cases.
```

## 1. Inductive Learning
- Supervised learning과 유사
- 이미 레이블링된 훈련 데이터셋을 활용하여 모델 구축
- **레이블링된 데이터로 구축된 모델이 본 적이 없는(Unseen) Test Dataset의 레이블을 예측**
- 레이블링된 Training dataset 과 레이블링되지 않은 Test dataset이 **철.저.하.게** 분리됨
- 새로운 데이터가 들어와도 알고리즘 재시작할 필요 없음
- 이로 인해 계산 효율 높음(Less Computational Cost)

```
Inductive learning is the same as what we commonly know as traditional supervised learning. 
We build and train a machine learning model based on a labelled training dataset we already 
have. Then we use this trained model to predict the labels of a testing dataset which we 
have never encountered before.
```

Inductive Learning은 기존의 Supervised learning이라 말해도 무방하다. 즉, <span style = "color:gold">Test-set과 Training-set이 분리되어있고, 모델 학습에 오직 Training Data-set만 사용되고,
예측에는 Test-Set만 사용</span>된다.

즉, Inductive Learning이 추구하는 것은 **general한 모델**을 만드는 것이다. 어떤 새로운 데이터가 들어오더라도 그것에 대해 보다 더 정확하게 예측을 하는 것을
목표로 하는 학습 방식을 말한다.


## 2. Transductive Learning
- Semi-supervised Learning과 유사
- 사전에 Train data set과 Test data set을 모두 관찰
- 테스트 데이터의 레이블링이 안되어 있지만, **그 안에서 유의미한 패턴을 추출**할 수 있다.
- 모델을 구축하지 않아서 새로운 데이터가 입력으로 들어오면 알고리즘을 처음부터 재시작해야 한다
- 이로 인해 **낮은 계산 효율(High Computational Cost)**을 보인다.
 
```
In contrast to inductive learning, transductive learning techniques have observed all the data 
beforehand, both the training and testing datasets. We learn from the already observed training 
dataset and then predict the labels of the testing dataset. Even though we do not know the labels 
of the testing datasets, we can make use of the patterns and additional information present 
in this data during the learning process.
```
- Transductive learning의 가장 대표적인 예는 ***Trasnductive SVM(TSVM)*** 과 ***graph-based label propagation algorithm(LPA)*** 이다.

Transductive Learning을 다시 말하면, 하나의 Dataset으로 Training과 Test를 모두 진행한다고 할 수 있다. 다시 말하면, <span style = "color:aqua">모델을 훈련시키는 와중에 이미 Test Set과 Training Set이 무엇인지 알고 있는 것</span>이다.

Transductive Learning은 예측 모델을 만드는 것이 아니다. 만약, 새로운 데이터가 유입되면, Model training을 처음부터 다시 돌려야한다.  
이를 통해 결국은 가지고 있는 데이터를 training과 test에 모두 사용한다는 것을 알 수 있다.

**Trnasductive learning**은 <span stlye = "color:gold">Training에서 데이터의 label을 쓰고, 그 데이터들이 가지고 있는 다른 attribute로 Test를 진행하는 것이다. 다시 말해, **Training과 Test에서 데이터가 분리되어 있지 않다.**</span>

## 3. Example & Summary 

### Example of Inductive Learning

<p align="center">
<img width="100%" alt="1" src="https://user-images.githubusercontent.com/111734605/213160745-878dd5ed-6ad2-471c-841f-c7de960958d3.png">
</p>

- 레이블링된 데이터: A,B,C,D 4개
- 육안으로도 언뜻, 두 개의 Cluster로 나눌 수 있음 
- By ***Nearest Neighbour***, 두 개의 Cluster로 나눠짐
- 하지만, 레이블링된 데이터가 적어 실제로는 Inductive learning으로 학습하기 어려움

### Example of Transductive Learning

<p align="center">
<img width="100%" alt="1" src="https://user-images.githubusercontent.com/111734605/213161891-5fb865b4-1153-46fb-8001-68daffa001a1.png">
</p>

Inductive Learning에서와는 다르게, 추가적인 정보인 Edge에 대한 정보를 가지고 있다고 해보자. 이 때, 모델 학습과정에서 이 Edge에 대한 정보를 이용할 수 있다.
위의 그림과 같은 Transductive Learning 모델을 Semi-supervised graph base ***label propagation algorithm*** 이라고 한다. 이를 통해, 위의 Inductive learning과는
다른 결과를 얻게 된다.

### Summary

<p align="center">
<img width="100%" alt="1" src="https://user-images.githubusercontent.com/111734605/213159615-242e7e88-c16d-4e82-9b2c-fcf16e8ac354.png">
</p>

## Reference
[Inductive vs. Transductive Learning](https://towardsdatascience.com/inductive-vs-transductive-learning-e608e786f7d)    
[Transduction (machine learning) Wikipedia](https://en.wikipedia.org/wiki/Transduction_(machine_learning))    
[What is the difference between inductive and transductive learning](https://www.quora.com/What-is-the-difference-between-inductive-and-transductive-learning)  



