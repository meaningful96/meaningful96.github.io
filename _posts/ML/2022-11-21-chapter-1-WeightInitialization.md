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

### (1) Gradient Descent

<p align="center">
<img width="500" alt="1" src="https://user-images.githubusercontent.com/111734605/202996617-b816808f-5db0-4921-a878-cc97bbeb7e9e.png">
</p>

Gradient Descentsms Gradient(기울기)의 반대 방향으로 Update를 진행한다. 하지만, 기본적으로 gradient descent는 초기값에 매우 민감한 알고리즘이다. 결국, Minimum cost value가 
Weight의 Initial value가 어디냐에 따라 그 값이 달라지게 된다.

- **Initial Weight Value**(초기 가중치 값)에 따라 모델의 성능에 상당한 영향을 준다.

### (2) 초기값이 Extremely High or Extremely Low일 경우
초기값이 극단적일 경우 여러가지 문제점이 발생할 수 있다.

- Vanishing Gradient
- Training에 많은 시간 소모

Vanishing gradient는 특히 딥러닝 알고리즘에서 치명적이다. 딥러닝을 할 때 Vanishing gradient 현상을 줄이는 많은 방법들이 존재하고, 그 중 하나가 Weight initialization을 적절
히 이용하는 것이다.

## 2. Zero Initialization
사실 Zero initialization은 그다지 좋은 방법은 아니다. 모든 초기값을 0으로 두고 시작하는 방법인데, 이는 연산 결과가 0으로 나오게 만들 가능성이 크기에 좋지 않다.

- Iltimately, 0으로 초기화 하는것 : Bad
- 학습이 제대로 안됨.
- Neuron이 Training 중에 feature를 학습하는데, Foward propagation시 input의 모든 가중치가 0이면 Next layer로 모두 같은 값이 전달 됨.
- Backpropagation시 모든 weight의 값이 똑같이 바뀌게 됨.

<p align="center">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/111734605/203003878-2f8764a7-d30e-47bd-8fe1-3806a8353b0e.png">
</p>

