---
title: "[딥러닝]Contrastive Learning(대조 학습)이란?"

categories: 
  - DeepLearning

toc: true
toc_sticky: true

date: 2023-03-09
last_modified_at: 2023-03-09
---
# Contrastive Learning이란?
Contrastive Learning이란 입력 샘플 간의 **비교**를 통해 학습을 하는 것으로 볼 수 있다. <b>Self-Supervised Learning(자기지도학습)</b>에 사용되는 접근법 중 하나로 사전에 정답 데이터를 구축하지 않는 판별 모델(Discriminative Model)이라 할 수 있다. 따라서, 데이터 구축 비용이 들지 않음과 동시에 학습 과정에 있어서 보다 용이하다는 장점을 가져가게 된다. 
<u>데이터 구축 비용이 들지 않음</u>과 동시에 <u>Label도 없기</u> 때문에 **1)보다 더 일반적인 feature representation이 가능**하고 **2)새로운 class가 들어와도 대응이 가능**하다. 이후 Classification등 다양한 downstream task에 대해서 네트워크를 fine-tuning시키는 방향으로 활용하고 한다. 

<span style = "font-size:110%"><center><b><i>"A Contrast is a great difference between two or more things which is clear when you compare them."</i></b></center></span>

**Contrastive learning**이란 다시 말해 <span style = "color:green">**대상들의 차이를 좀 더 명확하게 보여줄 수 있도록 학습**</span>하는 방법을 말한다. 차이라는 것은 어떤 **기준**과 비교하여 만들어지는 것이다. 예를 들어, 어떤 이미지들이 유사한지 판단하게 하기 위해서 **Similar**에 대한 기준을 명확하게 해줘야 한다.

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/231803381-c9c0d8e0-da54-45cc-982e-a1d4347909b4.png">
</p>


## 1. Similarity learning
이러한 **유사도**를 이용해 학습하는 방식을 Similarity learning이라고 한다. Similarity learning의 사전적 정의는 다음과 같고, 크게 3가지 종류가 있다.

<span style = "font-size:110%"><center><b><i>"Similarity learning is closely realted to regresison and classification, but the goal is to learn a similarity funciton that measures <span style = "color:green">how similar or related two objects are</span>.</i></b></center></span>

결국, Constrastive learning과 similarity learnig모두 다 <span style = "color:green">**어떤 객체들에 대한 유사도**</span>와 관련이 있다는 걸 알 수 있다.

<br/>

### 1.1 Regression Similarity Learning

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/231949535-f5aa3f86-50ff-4bcc-bdb8-c8be9f671226.png">
</p>

**Regression Similarity Learning**은 두 객체 간의 유사도를 알고 있다는 전제하에 **Supervised Learning 학습**을 시키는 방법이다. 유사도는 Pre-defined된 어떤 기준에 의해 설정되고, 기준에 따라 모델이 학습된다. 위의 <b>$$y$$</b>가 **유사도**를 나타낸다. 유사도가 높으면 $$y$$값이 높게 설정된다. 앞의 설정한 유사도에 따라 모델이 학습되면, <span style = "color:green">**학습된 모델에 test 데이터인 두 객체가 입력 될 때 pre-defined 기준에 따라 유사도를 결정**</span>된다.

예를 들어, 강아지 이미지 데이터들 끼리는 모두 강한 유사도를 주고, 강아지 이미지 데이터와 고양이 이미지 데이터들의 유사도는 매우 낮은 값으로 설정해주어 학습 시키면, 학습 한 모델은 강아지 이미지들끼리에 대해서 높은 유사도 값을 regression할 것이다.

하지만, 이러한 유사도($$y$$)를 어떻게 설정할지는 매우 난해한 문제이다.

<br/>

### 1.2 Classification Similarity Learning

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/231978599-9e80ecb1-8145-480b-b23b-81d14c703740.png">
</p>

Regression Similarity Learning과 식은 유사하다. 다만 다른 점은 이름에서 알 수 있다. Regression은 연속된 데이터를 분류하는 것을 말한다(ex) Linear Regression). 반면 Classification은 이산적인 분류를 하는 것을 말한다(ex) Binary Classification). 따라서, **Classification Similarity Learning**은 두 객체가 유사한지 아닌지만 알 수 있다.

- Regression: $$y \in R$$
  - 유사도값의 범위는 실수 $$\rightarrow$$ 유사도의 정도를 파악하기 어려움
  - R값의 범위 설정 + 어떤 y값을 해줘야 하는지 어려움
- Classification: $$y \in {0,1}$$
  - 두 객체가 <span style = "color:green">**유사한지 아닌지만 알려줌**</span>(마치 NSP: Next Sentence Prediction과 비슷). 어느 정도로 유사한지는 알 수 없음.

<br/>

### 1.3 Ranking Similarity Learning

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/231981078-7a301fc1-ce63-4e48-95ec-787544ff2e6d.png">
</p>

앞선 두 가지 방식과 다른 점은 **Input이 3개(Triplet of objects)라는 점이다.** 일반적인 데이터 $$x$$와 유사한 데이터 $$x^{+}$$, 유사하지 않은 데이터 $$x^{-}$$가 입력으로 들어간다. 이런식으로 <span style = "color:green">**유사한 데이터들 간의 유사도와 유사하지 않은 데이터들 간의 유사도 차이를 설정하여 학습**</span>하게 된다.

데이터들끼리 유사도가 높다는 것을 <span style = "color:green">**거리(Distance)**</span>관점에서 해석할 수 있다. **유사한 데이터는 (상대적인) 거리가 가깝다**는 식으로 해석 가능하다. 이와 비슷한 것이 이전 Graph Embedding 모델들이다. head와 relation의 상대적인 위치를 통해 벡터 임베딩 형태로 표현하고, tail과의 상대적인 거리가 좁아질수록 정답인 True triple이 되는 것이다. 

거리를 이용한 해석 방식은 **Distance Metric Learning**이라고 한다. 

## 2 Distance Metric Learning

유사도를 판단하는 방법 중 거리의 관점에서 해석하는 방식이다. 보통 거리라는 개념을 단순히 점과 점 사이의 최단 거리로만 이해하는 경우가 있지만, 거리를 측정하는 다양한 방법이 존재한다. 또한 진짜 물리적인 거리만이 아니라, 여러 가지 트릭을 이용해 **상대적인**거리로 표현할 수도 있다. **Metric Learning**은 <span style = "color:green">**객체(데이터)들 간의 거리(or 유사도)를 수량화**</span>하는 방법이다. Metric Learning에서는 다음과 같은 4가지 속성을 반영한다.

- Non-negativity: $$f(x,y) \geq 0$$ = 두 데이터 간의 거리는 음수가 될 수 없다.
- Identity of Discernile: $$f(x,y) = 0 \Leftrightarrow x = y$$ = 두 데이터 간의 거리가 0이면 x와 y는 동일하다.
- Symmetry: $$f(x,y) = f(y,x)$$ = <x,y>간의 거리나 <y,x>
- Triangle Inequality: $$f(x,z) \leq f(x,y) + f(y,z)$$ = <x,z>간의 거리는 <x,y>간의 거리와 <y,z>간의 거리를 합한 것보다 클 수 없다.

### 2.1 Metric of Two Types

거리를 측정하는 metric 방식에도 크게 두 가지 종류가 있다.
- Pre-Defined Metrics
  - 단순히 데이터들을 정의 된 metric 공식에 입력하고 거리값을 도출하여 유사도를 비교
- Leraned Metrics
  - 데이터들로부터 추정할 수 있는 다른 지표들을 metric 공식에 적용하여 거리값을 도출

### 2.2 Deep Metric Learning(Feat. Contrastive Loss)

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/232093884-14893cff-7629-427f-b9e3-210e699056e7.png">
</p>


데이터들의 차원수가 높을 경우, 서로 간의 유사도를 Euclidean Distance를 통해 구하는 것은 매우 힘들다. 그 이유는, **'Curse of Dimension'**으로 인해 의미 있는 **manifold를 찾미 못하기 때문**이다. 즉, 실제 Euclidean Distance는 manifold 상에서 구해야 하기 때문에, **manifold를 잘 찾는 것이 두 데이터간 유의미한 Similarity를 구하는데 결정적인 역할**을 할 수 있다.

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/232100220-f5602e72-9217-4563-9afb-a6624a3f2623.png">
</p>

유의미한 manifold를 찾기 위해서는 <span style = "color:green">**demension reduction**</span>(차원축소 e.g. PCA, SVD, LDA) 방식이 필요하고, 그 방식이 바로 <span style = "color:green">**Deep Neural Network**</span>이다. 결국 특정 metric을 기준으로 한 유사도를 찾기 위해 딥러닝 모델들의 파라미터가 학습된다면, 이는 해당 metric을 찾기 위한 <span style = "color:green">**manifold를 찾는 과정이며, 이 과정 자체가 Estimate from data를 의미**</span>한다. 


## 3 Contrastive Learning 

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/232206814-1bb5b197-b7f1-421c-8c11-1d4c7ba59353.png">
</p>

위의 그림을 먼저 보면, 유사한 이미지를 한 쌍으로 한 **Positive pair끼리는 Euclidian Loss가 최소화** 되도록 학습시키면, <span style = "color:green">**DNN(Deep Neural Network)은 고차원의 Positive pair 데이터의 거리가 가깝도록 low dimension으로 차원 축소(Dimension reduction or embedding)**</span>한다. 반면 **Negative pair끼리는 Euclidian Loss값이 커지도록 설정**해줄 수 있다. 이로 인해 <b>margin(m)</b>이라는 개념이 도입되고, <span style = "color:green">**margin은 negative pair간의 최소한의 거리**</span>를 의미한다. 

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/232207762-a89577d7-d885-4753-97be-2a0e2972995e.png">
</p>

예를 들어, loss값이 최소가 되기를 바라는데, negative pair($$x_n, x_q$$)의 거리가 m보다 작다면 계속해서 loss값을 생성해낼 것이다. 하지만, 학습을 통해 negative pair간의 거리가 m보다 크게 되면 loss값을 0으로 수렴시킬 수 있다. 이 두식이 결합되어 하나의 loss, <span style = "color:red">**Contrastive loss**</span>라 한다. 

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/232207966-9a8e1b6e-c485-45f6-9015-1da2c974cf16.png">
</p>

## 4. Knowledge Graph관점에서 Contrastive Learning

Contrastive Loss는 Negative pair loss와 Positive pair loss의 합으로 구성된다. Knowledge Graph에서 종종 학습의 성능을 높여주기 위해 <span style = "color:red">**Negative Sampling**</span>을 자주 사용한다. 여기서 Negative sampling은 KG에서 실제로 정의가 된 Triple이 아닌, <u>실제로는 연결관계가 없는 Triple</u>을 말한다. 즉, 어떤 head와 tail이 실제로는 어떤 릴레이션으로 연결이 안되어 있는 Triple을 말한다. 

따라서 학습이 Positive pair에게는 점수를 더 주며, Negative pair에게는 점수를 덜 주는 방향으로 Loss가 구성되어 학습이 진행된다. 대표적인 모델로 SimKGC가 있다.

[SimKGC]("https://arxiv.org/pdf/2203.02167.pdf")


<br/>
<br/>

# Reference
[Contrastive Learning이란? (Feat. Contrastive Loss)]("https://89douner.tistory.com/334")    
[Contrastive Learning이란]("https://daebaq27.tistory.com/97")    
[Contrastive Learning이란? (Feat. Contrastive Loss, Self-Superviesd Learning)]("https://iambeginnerdeveloper.tistory.com/198")  

