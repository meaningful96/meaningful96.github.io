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

Ex) 2개의 Hidden Layer(은닉층)가 있는 MLP

> bias = 0, Weight = a 로 초기화했다고 가정(zero는 아니고 보다 더 직관적으로 보여주기 위해 상수로 설정)
> Activation fucntion = ReLU, ReLU = maximum(0,x)
> f(x1,x2) = (ReLU(ax1), ReLU(ax2)) 

이때 ax1과 ax2의 편미분값은 a로 동일하다. 이는 다시 말해 Loss에 동일한 영향을 미치는 것이고 동일한 영향을 미친다는 것은 기울기가 같다는 것이다.(대칭적 가중치) 이렇게 했을 경우
두 가지 문제점이 발생한다.

- 서로 다른 것을 학습하지 못함
- Weight가 여러 개인 것이 무의미
- 따라서 Weight의 초깃값은 **무작위**로 설정해야 함을 시사해줌

## 3. Random Initialization
Parameter를 모두 다르게 초기화할 수 있는 방법으로 가장 쉽게 생각해 볼 수 있는 방법은 확률분포를 이용하는 것이다. Gaussian Distribution(정규분포)을 이용하여 각 weight에 배정하여
Initial value를 설정할 수 있다. 이해를 위해 표준편차를 각각 다르게 설정하면서 가중치를 정규분포로 초기화한 신경망(Neural Net)의 활성화 함수(Activation fucntion) 출력 값을 살펴
보았다.

  (1) 평균 0.5, 표준편차가 1인 케이스, Activation function = Sigmoid(Logistic) function
  
  <p align="center">
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/111734605/203316223-d8c17028-4fdb-444a-b49a-f1c7a8a68ab7.png">
  </p>
  
  먼저 Sigmoid 함수의 가장 큰 특징은, Input 값이 0 주위에서만 미분값이 유의미한 값을 가진다. 0에서 작아질수록 sigmoid 값의 출력은 0에 가까워지고, 0에서 커질수록 sigmoid 출력은
  1에 수렴한다. 하지만, input값이 0에서 멀면 sigmoid 함수는 saturation이 되기 때문에 미분값(gradient)값이 0이 되고, 결국 **Vanishing Gradient**현상이 일어나게 된다. 위의 그
  림을 보면 sigmoid의 출력값이 0과 1에 가까울때만 출력되는 것을 확인할 수 있다. 그리고 앞서 말했듯, 이 경우 미분값은 0이 된다. 
  
  > (즉, 표준편차가 1이면 sigmoid 기준으로 input이 양 극단에 치우친 것과 마찬가지이다.)
  
  (2) 평균 0.5, 표준편차가 0.01인 케이스, Activation function = Sigmoid(Logistic) function
  
  <p align="center">
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/111734605/203321902-f02439cc-eb77-48f7-822e-96e727b170bf.png">
  </p>
  
  이 경우에는 Input이 0.5 주변에 모여있으므로 활성함수인 sigmoid를 취하게되면 유의미한 값을 가지게되며, 미분값이 0이 아니다. 하지만, 대부분의 출력값이 0.5 주변에 모여있기 때문에
  Zero initialization에서 봤던 예시 처럼 노드별로 gradient값의 출력값이 비슷해 결국은 Multi-Layer를 구성하는 의미가 사라지게 된다.
  
## 4. LeCun Initialization
LeCun은 CNN 모델을 사용한 Architecture인 LeNet의 창시자이다. CNN을 도입함으로서 인공지능 분야의 큰 획을 그은 분이다. LeCun은 효과적인 Backpropagation을 위한 논문으로서 초기화
방법을 제시했다. Gaussian Distribution과 Uniform Distribution을 따르는 두 가지 방법에 대해서 소개했다.
(논문 링크: [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)


  
