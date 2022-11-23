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

### (1) 표준편차가 1인 케이스, Activation function = Sigmoid(Logistic) function
 
```python
import numpy as np
import matplotlib.pyplot as plt

# 모델링 및 변수 초기화
def sigmoid(x):
  return 1/(1 + np.exp(-x))

x = np.linspace(-5,5, 500)
y = sigmoid(x)
plt.title("Sigdmoid function")
plt.plot(x,y,'b')
```
  
  <p align="center">
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/111734605/203447819-8f020237-553d-4a61-b0c7-5f69418dda6d.png">
  </p>
  
  먼저 Sigmoid 함수의 가장 큰 특징은, Input 값이 0 주위에서만 미분값이 유의미한 값을 가진다. 0에서 작아질수록 sigmoid 값의 출력은 0에 가까워지고, 0에서 커질수록 sigmoid 출력은
  1에 수렴한다. 하지만, input값이 0에서 멀면 sigmoid 함수는 saturation이 되기 때문에 미분값(gradient)값이 0이 되고, 결국 **Vanishing Gradient**현상이 일어나게 된다. 야래 그
  림을 보면 sigmoid의 출력값이 0과 1에 가까울때만 출력되는 것을 확인할 수 있다. 그리고 앞서 말했듯, 이 경우 미분값은 0이 된다. 
  
  > (즉, 표준편차가 1이면 sigmoid 기준으로 input이 양 극단에 치우친 것과 마찬가지이다.)
  
  ```python
  import numpy as np
  import matplotlib.pyplot as plt

  # 모델링 및 변수 초기화
  def sigmoid(x):
      return 1/(1 + np.exp(-x))

  x = np.random.randn(1000, 100) # mini batch : 1000, input : 100
  node_num = 100                 # 각 은닉층의 노드(뉴런) 수
  hidden_layer_size = 5          # 은닉층이 5개
  activations = {}               # 이곳에 활성화 결과(활성화값)를 저장

  plt.hist(x)

  for i in range(hidden_layer_size):
      if i != 0:
          x = activations[i - 1]
          
      w = np.random.randn(node_num, node_num) * 1
      a = np.dot(x, w)
      z = sigmoid(a)
      activations[i] = z

  # 히스토그램 그리기
  plt.figure(figsize=(20,5))

  for i, a in activations.items():
      plt.subplot(1, len(activations), i + 1)
      plt.title(str(i+1) + "-layer")
      plt.hist(a.flatten(), 30, range = (0,1))

  plt.show()
  ```
  
  - 데이터 분포를 보기위한 Histogram
  
  <p align="center">
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/111734605/203448439-a42bcab1-d216-4482-ad51-41c81e800ecd.png">
  </p>
  
  - Activation value in Each Layer
  
  <p align="center">
  <img width="800" alt="image" src="https://user-images.githubusercontent.com/111734605/203448548-d3fe5ff2-61f2-45da-bdaa-141fc08f3493.png">
  </p>
  
  각 Layer에서 나온 활성화 값을 보면,
  
  **<span style = "background-color:#fff5b1;">결국 데이터들이 각 레이어에서 0과 1에 집중되어있고, 다음 Layer로 Sigmoid를 취해서 넘어갈 경우 결국 미분값은 0됨을 알 수 있다.</span>**
  
  
### (2) 표준편차가 0.01인 케이스, Activation function = Sigmoid(Logistic) function
  
```python
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")
# 모델링 및 변수 초기화
def sigmoid(x):
  return 1/(1 + np.exp(-x))

x = np.random.randn(1000, 100) # mini batch : 1000, input : 100
node_num = 100                 # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5          # 은닉층이 5개
activations = {}               # 이곳에 활성화 결과(활성화값)를 저장

plt.hist(x)

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]
        
    w = np.random.randn(node_num, node_num) * 0.01 # 표준편차가 0.01로 바뀜
    a = np.dot(x, w)
    z = sigmoid(a)
    activations[i] = z

# 히스토그램 그리기
plt.figure(figsize=(20,5))

for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30, range = (0,1))

plt.show()
```

  - Activation value in Each Layer
  
  <p align="center">
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/111734605/203449805-258361ff-3280-400c-82c9-ede44ae26b4e.png">
  </p>
  
  이 경우에는 Input이 0.5 주변에 모여있으므로 활성함수인 sigmoid를 취하게되면 유의미한 값을 가지게되며, 미분값이 0이 아니다. 하지만, 대부분의 출력값이 0.5 주변에 모여있기 때문에
  Zero initialization에서 봤던 예시 처럼 노드별로 gradient값의 출력값이 비슷해 결국은 Multi-Layer를 구성하는 의미가 사라지게 된다.
  
  Zero initialization이나, 지금의 경우를 두고 'model **표현력**이 떨어진다'라고 한다.


**<span style = "background-color:#fff5b1;">Important</span>**  
```
각 층의 활성화값은 적당히 고루 분포되어야 한다.층과 층 사이에 적당하게    
다양한 데이터가 흐르게 해야 신경망 학습이 효율적으로 이뤄지기 때문이다.      
반대로 치우친 데이터가 흐르면 기울기 소실이나 표현력 제한 문제에 빠져서   
학습이 잘 이뤄지지 않는 경우가 생긴다.
```

## 4. LeCun Initialization

LeCun은 CNN 모델을 사용한 Architecture인 LeNet의 창시자이다. CNN을 도입함으로서 인공지능 분야의 큰 획을 그은 분이다. LeCun은 효과적인 Backpropagation을 위한 논문으로서 초기화
방법을 제시했다. Gaussian Distribution과 Uniform Distribution을 따르는 두 가지 방법에 대해서 소개했다.
(논문 링크: [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf))

  <p align="center">
  <img width="500" alt="KakaoTalk_20221122_223809519" src="https://user-images.githubusercontent.com/111734605/203328282-214d9daf-8573-4fb2-91b5-2eb081742cba.png">
  </p>
  
## 5. Xavier Initialization
Xavier 초기화 기법은 딥러닝 분야에서 자주 사용되는 방법 중 하나이다. Xavier는 위의 Zero initialization이나, Random initialization에서 발생한땐 문제들을 해결하기 위해 고안된  
방법이다. Xavier initialization에서는 고정된 표준편차를 사용하지 않는다. 이전 Hidden layer의 노드 수에 맞추어 변화시키는 것이 특징이다.
[Xavier Initializaion이 고안된 논문](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
이 말하는 것을 결국 각 층의 활성화값들을 광범위하게 분포시킬 목적으로 가중치의 적절한 분포를 찾자는 것이다. 앞 계층의 노드가 n개라면 표준편차가
**$$\frac{1}{\sqrt{n}}$$ 인 분포를 사용**하면 된다는 것이 결론이다.

**Xavier Intialization 의 특징**
- Neuron의 개수가 많을수록 초깃값으로 설정하는 weight이 더 좁게 퍼짐
- Layer가 깊어질수록 앞의 본 방식보다 더 넓게 분포됨을 알 수 있음.
- 따라서 sigmoid를 쓰더라도 표현력이 어느정도 보장됨
- Neuron의 개수에 따라 초기화되므로 좀 더 robust함

<p align="center">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/111734605/203454262-e5a0fd44-78fe-4564-bfbf-1a10115c3628.png">
</p>
  

### (1) Xavier Intialization 실험

```python
# 모델링 및 변수 초기화
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.random.randn(1000,100) # mini batch 1000, input 100
node_num = 100                # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5         # 은닉층의 5개
activations = {}              # 이곳에 활성화 결과(활성화값)를 저장

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]
        
    w = np.random.randn(node_num, node_num)/np.sqrt(node_num) # node의 개수에 루트씌우고 나눠줌
    a = np.dot(x,w)
    z = sigmoid(a)
    activations[i] = z

# 히스토그램 그리기
plt.figure(figsize = (20,5))
plt.suptitle("Weight Initialization = Xavier", fontsize = 16)
for i,a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i + 1) + 'layer')
    plt.hist(a.flatten(), range = (0,1))   
```

<p align="center">
<img width="800" alt="image" src="https://user-images.githubusercontent.com/111734605/203458530-14d5855a-cf95-492f-a564-62e21c795398.png">
</p>

결과적으로 층이 깊어질수록 일그러지는 경향성이 있지만, 앞에서 본 zero & random initialization에 비하면 확실히 넓게 분포되어 있음을 알 수 있다. 각 층에 데이터가 골구로 분포되어
있으므로, Sigmoid 함수의 표현력도 제한받지 않고 효율적인 학습을 이끌어 낼 수 있다.

**<span style = "background-color:#fff5b1;">How to reduce distortion</span>** 
```
위의 그림에서 오른쪽으로 갈수록 층이 깊어지고, 갈수록 분포가 왜곡된다(일그러진다).  
이는 Sigmoid함수에의해 발생한다. Activation function을 Sigmoid가 아닌 쌍곡함수  
중 tahn 힘수를 이용하면 개선된다. tanh를 사용하면 종 모양으로 분포가 된다. 이와  
같은 현상이 발생하는 이유는, sigmoid는 원점대칭이 아니기 때문이다. (0,0.5) 대칭
반면 tanh는 완벽하게 원점대칭이다.
```
```python
import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

x = np.linspace(-10,10,500)
y = tanh(x)
plt.plot(x,y,'b')
```

딥러닝을 위한 Framework로 나는 Pytorch를 주력으로 사용한다. Pytorch에서 Xavier Initialization을 사용하면 다음과 같다.

```
# PyTorch
torch.nn.init.xavier_normal_()
```

### (2) Xavier initialization with ReLU
앞서서 지금까지 Sigmoid 함수를 활성화 함수로 사용했다. 하지만, Sigmoid 함수는 Vanishing gradient issue에 치명적인 원인을 제공한다. 그 이유인 즉슨, 0 부근에서만 미분값이 유의미한
결과를 가지고, 그 이외의 지점에서 미분값이 0이기 때문이다. 이는 결국 Deep-Layer Model에서 Hidden layer를 거치면 거칠수록 vanishing gradient issue를 심화시킨다. 따라서, 다른 활
성화 함수를 사용할 필요가 있다. 그것의 대안으로 나온 함수가 바로 ReLU이다. 

#### - ReLU
```python
import numpy as np
import matplotlib.pyplot as plt

def ReLU(x):
    return np.maximum(0, x)

x = np.linspace(-5,5,500)
y = ReLU(x)
plt.plot(x,y,'b')
```

<p align="center">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/111734605/203460964-84d02eea-c509-4abe-88cc-7dc792d2c5eb.png">
</p>

논문에서 말해주듯이, Xavier Intialization은 활성화 함수가 선형인 것을 전제로 결과를 도출한다. Sigmoid 함수와 tanh는 점 대칭 함수이고, 0 부근에서 Linear한 특성을 보인다.
따라서 이 두 함수는 Xavier initialization을 사용하는데 적합하다는 것을 알 수 있다. 하지만, ReLU는 0이하인 지점에서는 그 값이 0이고, 양의 구간에서만 선형인 것을 알 수 있다.

### (3) ReLU 함수를 이용해 Xavier Initialization 실험

- 표준편차가 0.01인 정규분포, activation function = ReLU

```python
# 모델링 및 변수 초기화
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.random.randn(1000, 100) # mini batch : 1000, input : 100
node_num = 100                 # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5          # 은닉층이 5개
activations = {}               # 이곳에 활성화 결과(활성화값)를 저장

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]
        
    w = np.random.randn(node_num, node_num) * 0.01
    a = np.dot(x, w)
    z = relu(a)
    activations[i] = z

# 히스토그램 그리기
plt.figure(figsize=(20,5))
plt.suptitle("Normal Distribution, Standard deviation = 0.01 ", fontsize=16)
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i+1) + "-layer")
    plt.ylim(0, 7000) # y축 최대 7000으로 제한, y축 범위
    plt.hist(a.flatten(), 30, range = (0,1))
plt.show()
```

<p align="center">
<img width="800" alt="image" src="https://user-images.githubusercontent.com/111734605/203464868-50a8ed5a-d518-45e4-862e-9580c8fe0f63.png">
</p>

- Xavier Initialization, activation function = ReLU

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.random.randn(1000, 100) # mini batch : 1000, input : 100
node_num = 100                 # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5          # 은닉층이 5개
activations = {}               # 이곳에 활성화 결과(활성화값)를 저장

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]
    # Xavier 초깃값 적용          
    w = np.random.randn(node_num, node_num)/ np.sqrt(node_num)
    a = np.dot(x, w)
    z = relu(a)
    activations[i] = z

plt.figure(figsize=(20,5))
plt.suptitle("Xavier Initialization with ReLU", fontsize=16)
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i+1) + "-layer")
    plt.ylim(0, 7000)    
    plt.hist(a.flatten(), 30, range = (0,1))


plt.show()
```

<p align="center">
<img width="800" alt="image" src="https://user-images.githubusercontent.com/111734605/203465528-2d3a7691-af55-47bb-96e1-3e0e82cf106b.png">
</p>

**<span style = "background-color:#fff5b1;">Important</span>**  
```
- 학습이 잘되기 위해서는 Layer마다 분포들의 표준편차는 같거나 비슷해야함.
```

**결과**  
1. 표준편차가 0.01인 정규 분포를 사용한 결과: <span style = "color: aqua">학습이 잘 이루어지지 않을 것으로 예상</span>
2. Xavier Initialization을 사용한 결과    : <span style = "color: aqua">층이 깊어지면서 치우침이 조금씩 커진다. 즉, 층이 깊어질수록 분포들이 0 쪽으로 쏠리는 경향성을 보이고 ReLU는 0에서 미분값이 0이므로 **Vanishing Graident issue**가 발생하게 된다</span> 

**고찰**
층이 깊어질수록 치우침이 없어지게 하도록하는 방법이 필요하다. 근데, ReLU는 양의 구간에서만 미분값이 유의미하다. 즉, 절반의 구간에서만 유의미하므로, 그 효과를 증폭시키기 위해 **두 배의 계수**가 필요할꺼 같다. 이를 만족시키기위해 나온 초기화 기법이 He Initialization이다.

## 6. He Initialization
He Initialization의 기본적인 틀은 Xavier와 동일하다. 다만, 그 효과를 두 배로 만들기 위해서 표준편차에 x2 를 해준 **$$\frac{2}{\sqrt{n}}$$** 인 정규 분포를 사용하는 것이다.
역시 마찬가지로 Layer수에 Dependent하다.(n = Layer 수)

### (1) He initialization 실험

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.random.randn(1000, 100) # mini batch : 1000, input : 100
node_num = 100                 # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5          # 은닉층이 5개
activations = {}               # 이곳에 활성화 결과(활성화값)를 저장

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]
    # Xavier 초깃값 적용          
    w = np.random.randn(node_num, node_num)*np.sqrt(2/node_num)
    a = np.dot(x, w)
    z = relu(a)
    activations[i] = z

plt.figure(figsize=(20,5))
plt.suptitle("Xavier Initialization with ReLU", fontsize=16)
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i+1) + "-layer")
    plt.ylim(0, 7000)    
    plt.hist(a.flatten(), 30, range = (0,1))
plt.show()
```

<p align="center">
<img width="800" alt="image" src="https://user-images.githubusercontent.com/111734605/203467948-2b04d971-bc3f-4f43-89eb-03dd8c4babe6.png">
</p>

### (2) He Initialization 결과
1. 모든 층에서 균일하게 분포, 층이 깊어져도 분포가 균일하게 유지되기에 Backpropagation을 진행할 때도 적절한 값이 나올 것으로 예상된다.
2. 따라서 ReLU에 적합한 weight initialization은 He initialization이다.

## 7. Weight Initialization 실험 결과

**<span style="color:aqua">1) ReLU를 활성화 함수로 사용할 때는 He을 사용!!</span>**  
**<span style="color:aqua">2) Sigmoid, tanh 처럼 zero centered fucntion을 사용할 경우 Xavier를 사용!!</span>**

