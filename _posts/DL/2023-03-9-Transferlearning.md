---
title: "[딥러닝]Transfer Learning(전이학습)이란?"

categories: 
  - DeepLearning

toc: true
toc_sticky: true

date: 2023-03-09
last_modified_at: 2023-03-09
---

# Transfer Learning, 전이 학습
Transfer Learning은 Pre-trained(사전 훈련)된 모델을 새로운 작업에 대한 몰델의 시작점으로 재사용하는 학습 방식이다. 쉽게 말해서, <span style = "color:green">**Transfer Learning은 어떤 목적을 이루기 위해 학습된 모델을 다른 작업에 이용**</span>하는 것이다. 

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/231578258-24dc8726-1ea5-4382-ae24-a497e33ae876.png">
</p>

위 그림은 Transfer Learning의 간단한 예이다. Transfer Learning을 수행하기 위해서는 Pre-trained된 모델이 필요하다. 이 예시에서는 1000개의 클래스로 구별하기 위해 큰 데이터셋으로 학습된 모델을 사용한다. 그리고 개와 고양이를 분류하기 위해 새롭게 작은 데이터셋을 준비해 학습할 때, 학습된 모델을 모두 가져오고, 제일 **마지막 레이어만 새로 학습**을 한다.

# Transfer Learning을 사용하는 이유
1. 범용적으로 학습된 **Pre-Traied Model**을 이용하기 때문에 학습을 빠르게 수행할 수 있다. 이미 입력된 데이터의 feature를 효율적으로 추출하기 때문에, 학습할 데이터에 대해 feature를 추출하기 위한 학습을 별도로 진행하지 않아도 되기 때문이다.

2. 작은 데이터셋에 대해 학습을 진행할 때 Overfitting을 예방할 수 있다. 적은 데이터로 특징을 추출하기 위한 학습을 하게 되면, 데이터 수에 비해 모델의 가중치 수가 많을 수 있어 미세한 특징까지 모두 학습할 수 있다. Transfer Learning을 이용해 <u>마지막 레이어만 학습</u>하게 한다면, 학습할 가중치 수가 줄어 과한 학습이 이루어지지 않게 할 수 있습니다.

# Transfer Learning VS Fine-Tuning
Transfer Learning이 더 큰 개념으로, 특정 Task를 풀기 위해 pre-trained된 모델의 파라미터를 미세 조정하는 fine-tuning방식은 transfer learning의 기법 중 하나이다.

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/231588195-7b29c037-10a5-4762-b60e-75cfc21ddc26.png">
</p>
