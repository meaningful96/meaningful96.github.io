---
title: "[논문리뷰]Transformer: Attention Is All You Need"

categories: 
  - PaperReview
  
tags:
  - [KG Completion, NLP]
  
toc: true
toc_sticky: true

date: 2023-03-01
last_modified_at: 2023-03-01
---

# 1. Problem Statement
<span style = "font-size:110%">1. Sequential 모델들의 <span style = "color:aqua">Computational Complexity</span>가 너무 높음</span>
  - Recurrent model(RNN)을 기반으로 한 여러가지 Architecture들이 존재: RNN, LSTM, Seq2Seq
  - 최근 연구에서 factorization trick이나 conditional computation을 통해 계산 효율성을 많이 개선
  - 특히 Conditional Computation은 모델 성능도 개선
  - 하지만, 여전히 계산 복잡도 문제 존재

<center><span style = "font-size:80%">LSTM의 문제점: Input Sequence를 하나의 Context Vector로 압축➜병목현상</span></center>

2. <span style = "font-size:110%">Attention Mechanism이 다양한 분야의 Sequence Modeling에 사용되지만, 그럼에도 <span style = "color:aqua">RNN을 사용</span>.</span>
  - Attention Mechainsm은 Input과 Output간 Sequence의 길이를 신경쓰지 않아도 됨.

3. 기존의 RNN모델들은 Parallelization이 불가능 ➜ Training에 많은 시간 소요

4. Sequ2Seq의 문제점
  - 하나의 Context Vector에 모든 정보를 압축해야 하므로 정보의 타당성이 떨어져 성능 저하 발생

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/227275202-0c2ce492-7f17-4db3-bf7a-88cac2c23521.png">
</p>  

<span style = "font-size:120%">➜ '<span style = "color:gold">매번 소스 문장에서의 출력 전부를 입력으로</span> 받으면 어떨까?'라는 질문에서 시작</span> 
  - 최근 GPU가 많은 메모리와 빠른 병렬 처리를 지원  

<span style = "font-size:120%">➜ Transformer는 <span style = "color:gold">input과 output간 global dependency를 뽑아내기 위해 Recurrence를 사용하지 않고, Attention mechanism만을 사용</span>함.</span> 

# 2. Relation Work
1. RNN, LSTM, Seq2Seq

2. Sequential Computation을 줄이는 것은 Extended Neural GPU, ByteNet등에서도 다뤄진다.
  - CNN을 기반으로 한 모델들임
  - input output 거리에서 dependency를 학습하기 어려움
  - <span style = "color:aqua">Transformer에서는 Multi-Head Attention으로 상수 시간으로 줄어든다.</span>

3. Self-Attention

4. End-to-End Memory Network
  - sequence-aligned recurrence 보다 recurrent attention mechanism에 기반한다.
  - simple-language question answering 과 language modeling task에서 좋은 성능을 보인다.


# 3. Method
## 1) Overview

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/227331054-832086ea-5c2f-4e58-abc3-290db460a0aa.png">
</p>

- Encoder
  - **2개의 Sub-layer**로 구성되어 있으며, 총 **6개의 층**으로 구성되어 있다.(N=6)
  - 두 개의 Sub-layer는 **Multi-head attention**과 **position-wise fully connected feed-forward network**이다.
  - Residual Connection 존재, Encoder의 최종 Output은 차원이 512이다.($$d_{model}$$ = 512)

- Decoder
  - **세 개의 Sub layer**로 구성되어 있으며, 총 **6개의 층**이 stack되어 있다.(N=6)
  - 세 개의 Sub layer는 **Masked Multi-head attention**, **Multi-head attention**, **position-wise fully connected feed-forward network**이다.
  - Residual Connection 존재
 
## 2) Positional Encoding
Transformer는 RNN이나 CNN을 전혀 사용하지 않는다. 대신 <span style = "color:aqua">**Postional Encoding**</span>을 많이 사용한다. 트랜스포머 이전의 전통적인 임베딩에 Positional Encoding을 더해준 형태가 트랜스포머의 Input이 된다.

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/227280781-2baf7e42-38af-46ea-82a8-067671475685.png">
</p>

Postional Encoding은 주기 함수를 활용한 공식을 사용하며 <span style = "color:aqua">각 단어의 상대적인 위치 정보를 네트워크에게 입력</span>한다. 수식은 다음과 같다.  

<p align="center">
<img width="400" alt="1" src="https://user-images.githubusercontent.com/111734605/227283915-f70f70a0-da08-4f5d-8097-75de375a9779.png">
</p>

$$pos$$는 position이고, $$i$$는 차원이다. 중요한 것은 Postional Encoding은 임베딩으로서의 $$d_{model}$$과 차원수가 동일하다는 것이다.

## 3) Multi-head Attention

먼저 Attention Mechanism에 대해 살펴보면 다음과 같다. Attention mechanism의 목적은 한 토큰이 다른 토큰과 얼마나 연관성이 있는지를 나타내기 위한 수학적 방법으로서, Query와 Key, Value를 통해 연산을 한다.
- Query: 비교의 주체 대상. 입력 시퀀스에서 초점을 둔 토큰으로 트랜스포머에서는 Decoder 또는 Encoder Layer의 숨겨진 상태(Hidden state)를 변환하여 형성된다.
- Key: 비교의 객체. 입력 시퀀스에 있는 모든 토큰이다. Query와 입력 시퀀스의 각 항목 간의 관련성을 결정하는데 사용된다.
- Value: 입력 시퀀스의 각 토큰과 관련된 실제 정보를 수치로 나타낸 실제 값이다. 각 요소의 중요도를 결정했을 때 모델에 필요한 정보를 제공하는 데 사용된다.

트랜스포머에서 Self-Attention은 <span style = "color:aqua">Scaled-Dot Product</span>로 이름에서도 알 수 있듯, 행렬곱과 스케일링으로 이루어진 연산이다. 그림으로 나타내면 다음과 같다.

<p align="center">
<img width="300" alt="1" src="https://user-images.githubusercontent.com/111734605/227312790-3e8d658a-737a-41c3-be42-6d8b4a71ea40.png">
</p>

**Scaled-Dot Product Self-Attention**
- Attention <span style = "color:aqua">**Energy**</span> = **Dot-Product** of (Query & Key) = <span style = "font-size:110%">$$QK^T$$</span> = $$e_{ij}$$
- Attention <span style = "color:aqua">**Score**</span> = **Scailing** of Key's Dimension = <span style = "font-size:120%">$$\frac{QK^T}{\sqrt{d_k}}$$</span> 
- Attention <span style = "color:aqua">**Weight**</span> = **Softmax**(Attention Score) = <span style = "font-size:120%">$$softmax(\frac{QK^T}{\sqrt{d_k}})$$</span> = $$a_{ij}$$ 

<span style = "color:gold"><span style = "font-size:120%">➜ Attention(Query, Key, Value) = $$softmax(\frac{QK^T}{\sqrt{d_k}})V$$ </span></span>이다.

트랜스포머에서는 인코더와 디코더 모두에서 <span style = "color:aqua">**Multi-head Attention**</span>을 사용한다. 병렬로 Head의 개수만큼 한 번에 어텐션을 진행하는 것으로, 동시에 여러 개의 Attention value값을 추출해 낼 수 있다. Multi-Head Attention을 사용하는 이유는 여러가지이다.

**Multi-head Attention**
- Improved representation learning: 모델이 입력 시퀀스의 다양한 측면에 어텐션할 수 있으므로 데이터를 더 포괄적이고, 미묘한 차이까지도 이해할 수 있다.
- Increased model capacity: 모델이 Key와 Query의 더 복잡하고 다양한 interaction을 학습할 수 있다. 이로써 더 복잡한 관계를 포착해낸다.
- Efficient Parallelization: 병렬화를 통해 빠른 학습과 추론이 가능하다.

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/227325573-f5ca67b9-3b5a-4e51-bab8-dbeefffa36e8.png">
</p>

논문에서는 head의 수는 8개이고, $$d_k = d_v = d_{model}/h$$ = 64이다. 각 head의 차원수가 감소했기 때문에 Total Computational Cost가 full dimensionality일 때의 single-head attention가 같다. 다시 말해서, Multi-head attention에서 d의 차원이 줄어든 것의 결과는 Single-head attention에서 d의 차원을 늘렸을때랑 계산 결과가 수렴한다. 또한 Multi-head Attention에서 중요한 것은 <span style = "gold">**어텐션을 수행한 뒤에도 입력과 차원이 동일하게 유지**</span>된다는 것이다.

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/227327896-7e526443-5d28-44b4-9fc1-d06cffe1f440.png">
</p>

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/227328122-057ad1b2-71d9-4152-bb8f-7ce55413c9c6.png">
</p>

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/227328526-af54acf1-accd-44de-bc44-91e2e60b1873.png">
</p>

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/227328797-5656e5ad-92ec-4562-a811-651b6957a960.png">
</p>

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/227329001-6d267496-3759-4cd8-a2fb-6d1c522af08b.png">
</p>

트랜스포머에는 총 세 가지 종류의 어텐션(Attention) 레이어가 사용된다. 

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/227329824-4e53276c-35f4-4775-93a8-87359d436d72.png">
</p>

## 4) Encoder
트랜스포머의 인코더는 두 가지의 Sub layer로 구성된다. Sub-layer를 보기 전 앞서 말했듯, Input Embedding Matrix에 Positional Encoding이 더해진 값이 첫 번째 Sub layer인 Multi-head Attention에 Input으로 들어간다. 그 이후, Attention의 결과와 Multi-head Attention의 Input이 더해지고 정규화를 거친다. 이 때, Multi-head Attention의 Input이
'Add + Norm'의 인풋으로 가는 것을 <span style = "color:aqua">**Residual Connection**</span>이라고 하고, 이러한 학습 방식을 '**잔여학습(Residual Learning)**' 이라고 한다.
첫 번째 Sub layer는 결국 'Multi-head Attention Layer'와 'Add + Norm Layer'로 구성된다.

두 번째 Sub layer는 'Fully Connected Feedforward Layer'와 'Add + Norm Layer'로 구성된다. 또한 여기서도 마찬가지로 Residual Connection을 한다. Residual Connection을 하는 이유는 어떤 Layer를 거쳤을 때 변환되서 나온 값에 실제 Data의 Input을 더해줘서 Input을 좀 더 반영하게 해주는 것이다. 이렇게하면 결론적으로 성능이 향상된다.

인코더는 총 6개의 Layer로 구성된다. 다시 말해서 2개의 Sub Layer가 포함된 하나의 Layer가 6개(N = 6)인 것이고 같은 Operation이 총 6번이라는 것이다. 그리고 이는 Input Sequence가 인코더에서 결론적으로 총 12개의 Sub layer를 거치는 것이다. <u>6개의 Layer는 서로 다른 파라미터를 가진다.</u>

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/227334714-702e866b-a05f-482a-a7b9-bf6daa115f10.png">
</p>

## 5) Decoder
디코더의 입력은 총 두개이며 디코더는 총 세 개의 Sub layer로 구성된다. 먼저 트랜스포머가 학습을 할 때, Epoch마다 나오는 Output Embedding Matrix가 출력 단어의 상대적인 위치를 나타내는 Positional Encoding과 더해져서 디코더의 입력으로 들어간다. 즉, 출력 단어에 대한 정보와 상대적인 위치정보를 더해서 입력으로 들어가고 이 값은 **Maksed Multi-Head Attention**을 거치게 된다.  

Masked Self-Attention을 하는 이유는, 학습과 추론과정에 정보가 새는(Information Leakage)것을 방지하기 위함이다. 트랜스포머에서 마스킹된 Self Attention은 모델이 <u>한 번에 하나씩 출력 토큰을 생성할 수 있도록 하면서 모델이 미래의 토큰이 아닌 이전에 생성된 토큰에만 주의를 기울이도록 하기 위함</u>이다. 이를 더 자세히 말하자면, Encoder-Decoder로 이루어진 모델들의 경우 입력을 순차적으로 전달받기 때문에 t + 1 시점의 예측을 위해 사용할 수 있는 데이터가 t 시점까지로 한정된다. 하지만 트랜스포머의 현재 시점의 출력값을 만들어 내는데 미래 시점의 입력값까지 사용할 수 있게되는 문제가 발생하기 때문이다.

이 이유는 트랜스포머의 초기값, 1 Epoch을 생각해보면 이해하기 쉽다. 처음에 입력으로 들어가 인코더를 거친 값이 디코더로 들어가는데, 디코더로 들어가는 또 다른 입력은 이전 Epoch에서의 출력 임베딩값이다. 하지만 1 Epoch에서는 과거의 값은 존재하지 않아 초기에 설정해준 값이 들어간다. 즉, 1 Epoch에서 이미 출력값을 입력으로 요구하기 때문에 시점이 미래라 할 수 있는 것이고, 결국은 현재의 출력 값을 예측하는데 미래의 값을 이용한다고 말할 수 있다. 이러한 문제를 방지하기 위해 **Look-Ahead Mask** 기법이 나왔다. 

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/227343660-9676f01e-c7d1-4973-b005-6db96d06753a.png">
</p>

트랜스포머에서는 기존의 연산을 유지하며 Attentio Value를 계산할 때 $$i<j$$인 요소들은 고려하지 않는다. Attention(i,j)에서 여기서 i는 Query의 값이고, j는 Value의 값이다. 이를 그림으로 표현하면 위와 같다. 디테일하게 Atttention Score를 계산한 행렬의 대각선 윗부분을 -∞로 만들어 softmax를 취했을 때 그 값이 0이되게 만든다. 즉, Masking된 값의 Attnetion Weight는 0이된다. 이렇게 함으로서 Attention Value를 계산할 때 미래 시점의 값을 고려하지 않게된다. 

이렇게 Maksed Multi-head Attention을 거친후 Residual Connection과 함께 <u>'Add + Norm' Layer를 거치면 그 출력값이 인코더의 출력값과 함께 두 번째 Sub Layer인 <b>Multi-Head Attention</b>의 입력</u>으로 들어간다. 이는 의미적으로 Seq2Seq에서의 인코더-디코더 어텐션과 동일하다. <span style = "color:aqua">출력 단어가 입력 단어와 얼마나 연관이 있는지를 나타내기 위함</span>이다.

예를 들어서, 입력 단어가 'I am a teacher'이고 출력 단어가 '나는 선생님 입니다' 일 때, '선생님'이라는 단어를 번역한다고 했을때 I, am, a, teacher들 중 어느 단어와 가장 큰 관계가 있는지 구하는 것이 바로 두 번째 Sub layer의 역할이며 이를 **Encoder-Decoder Attention**이라고 하는 것이다. 따라서 인코더의 출력이 Key가되고 디코더의 첫 번째 Sub layer의 출력이 Query가 된다.


<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/227349449-5f2879f4-4879-4d92-b14f-0c5cad134215.png">
</p>

마찬가지로 디코더의 경우 총 6개(N = 6)의 Layer로 구성되므로 총 18개의 Sub layer로 구성된 것이다. 또한 인코더의 출력의 경우 모든 디코더 레이어의 입력으로 들어가게 된다. 다시 말해, 인코더의 출력값은 총 6개의 Layer로 똑같이 들어가 Encoder-Decoder Attention을 수행하게 된다.

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/227349573-880ed1ee-f418-4add-946f-c41254dce991.png">
</p>

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/227351382-068b514e-1afd-46bb-a0b5-1b4e1412761b.png">
</p>

## 6) Why Self-Attention
Recurrent, Convolution layer와 Self-Attention의 시간 복잡도를 비교하였다.
1. Layer당 전체 계산 복잡도(Computational Complexity)
2. Sequential Paralize 할 수 있는 계산의 양
3. 네트워크에서 Long-range Dependency 사이의 path 길이
  - Sequence Transduction 문제에서 Long-range dependency를 학습하는 것이 major challenge이다.
  - 이러한 학습에 영향을 주는 한 가지 요인은 네트워크에서 forwad와 backward 시그널이 순회해야하는 Path length이다.
  - 입력과 출력 시퀀스의 위치조합사이 path가 짧을수록, Long-range dependency를 학습하기 쉬움
    - Input과 Output Position 사이의 최대 Path 길이를 비교 

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/227423400-4e4f8813-8541-4fbc-971a-a5bd0f764613.png">
</p>

- <u>Self-Attention layer는 모든 Postion을 상수 시간만에 연결</u> 가능하다. 반면 Recurrent layer의 경우 $$O(n)$$이 소요된다.
- <u>계산 복잡도 측면에서 n < d일때 Self-attention 층이 Recurrent 층보다 빠르다.</u>
  - n: Sequence Length
  - d: Representation Dimensionality
  - 기계 번역 대부분이 n < d인 경우에 속한다.

<br/>
- 아주 긴 Sequence의 경우 계산 성능을 개선하기 위해 Self-attention은 입력 시퀀스의 neighborhood size를 r로 제한할 수 있다. 
  - 이를 통해 Maximum path의 길이를 $$O(n/r)$$로 증가시킬 수 있다.

<br/>
- $$k<n$$인 kernel width 의 single convolutional layer는 input 과 output의 모든 쌍을 연결하지 않는다.
  - contiguous kernel의 경우  $$O(n/k)$$의 stack이 필요
  - dilated convolution의 경우 $$O(log_k(n))$$이 필요
           
<br/>
- Convolution layer는 일반적으로 recurrent layer보다 더 비용이 많이 든다.
  - Separable Convolution의 경우 복잡도를 $$O(knd + nd^2)$$ 까지 줄일 수 있다.
  - 그러나<span style = "color:gold"> $$k = n$$인 경우 트랜스포머와 마찬가지로 Self-attention layer와 Point-wise Feedforward layer의 조합과 복잡도가 같다.</span>
<br/>
결론적으로 Self-attention을 통해 더 Interpretable한 모델을 만들 수 있다. Attention head들은 다양한 task를 잘 수행해내고, 문장의 구조적, 의미적 구조를 잘 연관시키는 성질을 보이기도 한다. 

<br/>

# Experiment & Result
## 1) DataSet
1. WMT 2014 English-German
  - 4.5백만의 Sequence Pair
  - 문장들을 byte-pair 인코딩 되어있음
  
2. English-French
  - 36M 개의 문장과 32000개의 word-peice vocabulary로 쪼개진 토큰들
  
## 2) Experiment
- Optimizer
  - Adam
  - learning rate = $$lrate = d_{model}^{-0.5} /cdot min(stepnum^{-0.5}, stepnum /cdot warmupsteps^{-1.5})$$ 이다.
<br/>
- Regularization
  1. Residual Dropout
    - 각 Sub-layer의 output에 dropout 적용
    - 임베딩의 합과 positional 인코딩에 dropout 적용
  2. Label Smoothing
    - One-hot encoding은 출력 시퀀스에서 단 하나만 1이고 나머지는 0
    - 이렇게 할 경우 다른 출력값의 영향력을 완전히 무시하기 때문에 정보의 편협 발생
    - 따라서, <u>정답이 아닌 Label에 대해서도 조금의 확률</u>을 부여 ex) 0.9/0.025/0.025/0.025/0.025
    - ($\epsilon_{ls} = 0.1$)
## 3) Result

1. Machine Translation: SOTA 달성
<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/227427547-c3efea78-0595-4e5f-8d59-3e15ec25e2d4.png">
</p>   

2. Model Variation
<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/227427738-919ad9bc-4d28-4977-98d9-ea3f349a5d0e.png">
</p>

3. English Constituency Parsing
<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/227427796-67b959fa-4bf8-4237-96d8-516e92427057.png">
</p>

English Constituency Parsing에서도 잘 일반화해서 사용할 수 있는지 실험하였다. 구체적인 tuning 없이도 놀라운 성능을 보였다.

# Contribution
1. Recurrent Model을 사용하지않고 오직 <span style = "color:gold">Attention Mechanism만을 이용해서 새로운 모델을 제시</span>하였다.
2. Benchmark Dataset에 대하여 SOTA 달성

# Reference
[Paper]("https://arxiv.org/abs/1706.03762")  
[나동빈 Youtube]("https://www.youtube.com/watch?v=AA621UofTUA&t=2664s")  
[Github]("https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice")    
[Blog: Transformer 논문 리뷰]("https://wandukong.tistory.com/19")
