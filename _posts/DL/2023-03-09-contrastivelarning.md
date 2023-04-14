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

## 1. Similarity learning
이러한 **유사도**를 이용해 학습하는 방식을 Similarity learning이라고 한다. Similarity learning의 사전적 정의는 다음과 같다.

<span style = "font-size:110%"><center><b><i>"Similarity learning is closely realted to regresison and classification, but the goal is to learn a similarity funciton that measures how similar or related two objects are.</i></b></center></span>


