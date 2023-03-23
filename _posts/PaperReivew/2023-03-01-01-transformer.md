---
title: (0.1) Transformer - Attention Is All You Need

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
  - **3개의 Sub-layer**로 구성되어 있으며, 총 **6개의 층**이 stack되어 있다.(N=6)
  - 세 갸의 Sub-layer는 **Masked Multi-head attention**, **Multi-head attention**, **position-wise fully connected feed-forward network**이다.
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
- Improved representation learning: 모델이 입력 시퀀스의 다양한 측면에 어텐션할 수 있으므로 데이터를 더 포괄적으로, 미묘한 차이도 이해할 수 있다.
- Increased model capacity: 모델이 Key와 Query의 더 복잡하고 다양한 interaction을 학습할 수 있다. 이로써 더 복잡한 관계를 포착해낸다.
- Efficient Parallelization: 병렬화를 통해 빠른 학습과 추론이 가능하다.

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/227325573-f5ca67b9-3b5a-4e51-bab8-dbeefffa36e8.png">
</p>

논문에서는 head의 수는 8개이고, $$d_k = d_v = d_{model}/h$$ = 64이다. 각 head의 차원수가 감소했기 때문에 Total Computational Cost가 full dimensionality일 때의 single-head attention가 같다. 다시 말해서, Multi-head attention에서 d의 차원이 줄어든 것의 결과는 Single-head attention에서 d의 차원을 늘렸을때랑 계산 결과가 수렴한다.

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/227327896-7e526443-5d28-44b4-9fc1-d06cffe1f440.png">
</p>

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/227328122-057ad1b2-71d9-4152-bb8f-7ce55413c9c6.png">
</p>

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/227328526-af54acf1-accd-44de-bc44-91e2e60b1873.png">
</p>

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/227328797-5656e5ad-92ec-4562-a811-651b6957a960.png">
</p>

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/227329001-6d267496-3759-4cd8-a2fb-6d1c522af08b.png">
</p>

트랜스포머에는 총 세 가지 종류의 어텐션(Attention) 레이어가 사용된다. 

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/227329824-4e53276c-35f4-4775-93a8-87359d436d72.png">
</p>

## 4) Encoder
트랜스포머의 인코더는 두 가지의 Sub layer로 구성된다. Sub-layer를 보기 전 앞서 말했듯, Input Embedding Matrix에 Positional Encoding이 더해진 값이 첫 번째 Sub layer인 Multi-head Attention에 Input으로 들어간다. 그 이후, Attention의 결과와 Multi-head Attention의 Input이 더해지고 정규화를 거친다. 이 때, Multi-head Attention의 Input이
'Add + Norm'의 인풋으로 가는 것을 <span style = "color:aqua">**Residual Connection**</span>이라고 하고, 이러한 학습 방식을 '**잔여학습(Residual Learning)**' 이라고 한다.
첫 번째 Sub layer는 결국 'Multi-head Attention Layer'와 'Add + Norm Layer'로 구성된다.

두 번째 Sub layer는 'Fully Connected Feedforward Layer'와 'Add + Norm Layer'로 구성된다. 또한 여기서도 마찬가지로 Residual Connection을 한다. Residual Connection을 하는 이유는 어떤 Layer를 거쳤을 때 변환되서 나온 값에 실제 Data의 Input을 더해줘서 Input을 좀 더 반영하게 해주는 것이다. 이렇게하면 결론적으로 성능이 향상된다.

인코더는 총 6개의 Layer로 구성된다. 다시 말해서 2개의 Sub Layer가 포함된 하나의 Layer가 6개(N = 6)인 것이고 같은 Operation이 총 6번이라는 것이다. 그리고 이는 Input Sequence가 인코더에서 결론적으로 총 12개의 Sub layer를 거치는 것이다.

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/227334714-702e866b-a05f-482a-a7b9-bf6daa115f10.png">
</p>

## 5) Decoder

# Experiment & Result

# Contribution
