---
title: Chapter 1. Weight Initialization

categories:
  - ML
tags:
  - [ML,DL,Machine learning, Deep Learning]

toc: true
toc_sticky: true

date: 2022-11-21
last_modified_at: 2022-11-21 
---

## 1. 초기 가중치 설정(Weight Initialization)의 필요성
Machine learning과 Deep learning의 알고리즘은 Cost fucntion을 감소시키는 방향으로 iterative하게 동작한다. 다시 말해서 Neural Network 등의 model들의 Loss(Cost)를
최소화하는 방법은 파라미터를 최적화(Parameter Optimization)하는 것이다.

(1) Gradient Descent

<img width="380" alt="1" src="https://user-images.githubusercontent.com/111734605/202996617-b816808f-5db0-4921-a878-cc97bbeb7e9e.png">

Gradient Descentsms Gradient(기울기)의 반대 방향으로 Update를 진행한다. 하지만, 기본적으로 gradient descent는 초기값에 매우 민감한 알고리즘이다. 결국, Minimum cost value가 
Weight의 Initial value가 어디냐에 따라 그 값이 달라지게 된다.

(2) **Initial Weight Value(초기 가중치 값)**에 따라 모델의 성능에 상당한 영향을 준다.

- 초기값이 Extremely High or Extremely Low일 경우
