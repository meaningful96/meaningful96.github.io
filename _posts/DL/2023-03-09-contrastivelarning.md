---
title: Contrastive Learning(대조 학습)이란?

categories: 
  - DeepLearning
tags:
  - [DL,ANN,Neural Network]

toc: true
toc_sticky: true

date: 2023-03-09
last_modified_at: 2023-03-09
---
# Contrastive Learning이란?
Contrastive Learning이란 입력 샘플 간의 **비교**를 통해 학습을 하는 것으로 볼 수 있다. <b>Self-Supervised Learning(자기지도학습)</b>에 사용되는 접근법 중 하나로 사전에 정답 데이터를 구축하지 않는 판별 모델(Discriminative Model)이라 할 수 있다. 따라서, 데이터 구축 비용이 들지 않음과 동시에 학습 과정에 있어서 보다 용이하다는 장점을 가져가게 된다. 
<u>데이터 구축 비용이 들지 않음</u>과 동시에 <u>Label도 없기</u> 때문에 **1)보다 더 일반적인 feature representation이 가능**하고 **2)새로운 class가 들어와도 대응이 가능**하다. 이후 Classification등 다양한 downstream task에 대해서 네트워크를 fine-tuning시키는 방향으로 활용하고 한다. 

<span style = "font-size:110%"><center><b><i>"A Contrast is a great difference between two or more things which is clear when you compare them."</i></b></center></span>

**Contrastive learning**이란 다시 말해 <span style = "color:aqua">**대상들의 차이를 좀 더 명확하게 보여줄 수 있도록 학습**</span>하는 방법을 말한다. 차이라는 것은 어떤 **기준**과 비교하여 만들어지는 것이다. 예를 들어, 어떤 이미지들이 유사한지 판단하게 하기 위해서 **Similar**에 대한 기준을 명확하게 해줘야 한다.

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/231803381-c9c0d8e0-da54-45cc-982e-a1d4347909b4.png">
</p>

<br/>
<br/>

## 1. Similarity learning
이러한 **유사도**를 이용해 학습하는 방식을 Similarity learning이라고 한다. Similarity learning의 사전적 정의는 다음과 같고, 크게 3가지 종류가 있다.

<span style = "font-size:110%"><center><b><i>"Similarity learning is closely realted to regresison and classification, but the goal is to learn a similarity funciton that measures <span style = "color:aqua">how similar or related two objects are</span>.</i></b></center></span>

결국, Constrastive learning과 similarity learnig모두 다 <span style = "color:aqua">**어떤 객체들에 대한 유사도**</span>와 관련이 있다는 걸 알 수 있다.

### 1.1 Regression Similarity Learning

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/231949535-f5aa3f86-50ff-4bcc-bdb8-c8be9f671226.png">
</p>

**Regression Similarity Learning**은 두 객체 간의 유사도를 알고 있다는 전제하에 **Supervised Learning 학습**을 시키는 방법이다. 유사도는 Pre-defined된 어떤 기준에 의해 설정되고, 기준에 따라 모델이 학습된다. 위의 <b>$$y$$</b>가 **유사도**를 나타낸다. 유사도가 높으면 y값이 높게 설정된다. 앞의 설정한 유사도에 따라 모델이 학습되면, <span style = "color:aqua">**학습된 모델에 test 데이터인 두 객체가 입력 될 때 pre-defined 기준에 따라 유사도를 결정**</span>된다.

예를 들어, 강아지 이미지 데이터들 끼리는 모두 강한 유사도를 주고, 강아지 이미지 데이터와 고양이 이미지 데이터들의 유사도는 매우 낮은 값으로 설정해주어 학습 시키면, 학습 한 모델은 강아지 이미지들끼리에 대해서 높은 유사도 값을 regression할 것이다.

하지만, 이러한 유사도($$y$$)를 어떻게 설정할지는 매우 난해한 문제이다.

<br/>

### 1.2 Classification Similarity Learning

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/231951745-0f01afda-a228-4411-a64d-c8f4288d4a15.png">
</p>

Regression Similarity Learning과 식은 유사하다. 다만 다른 점은 이름에서 알 수 있다. Regression은 연속된 데이터를 분류하는 것을 말한다(ex) Linear Regression). 반면 Classification은 이산적인 분류를 하는 것을 말한다(ex) Binary Classification). 따라서, **Classification Similarity Learning**은 두 객체가 유사한지 아닌지만 알 수 있다.

- Regression: $$y \in R$$
  - 유사도값의 범위는 실수 $$\rightarrow$$ 유사도의 정도를 파악하기 어려움
  - R값의 범위 설정 + 어떤 y값을 해줘야 하는지 어려움
- Classification: $$y \in {0,1}$$
  - 두 객체가 <span style = "color:aqua">**유사한지 아닌지만 알려줌**</span>(마치 NSP: Next Sentence Prediction과 비슷). 어느 정도로 유사한지는 알 수 없음.

### 1.3 Ranking Similarity Learning


<br/>
<br/>

# Reference
[Contrastive Learning이란? (Feat. Contrastive Loss)]("https://89douner.tistory.com/334")    
[Contrastive Learning이란]("https://daebaq27.tistory.com/97")    
[Contrastive Learning이란? (Feat. Contrastive Loss, Self-Superviesd Learning)]("https://iambeginnerdeveloper.tistory.com/198")  


