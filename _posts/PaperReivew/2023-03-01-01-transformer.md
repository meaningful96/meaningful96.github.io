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
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/227272230-fe6bdd47-1dc7-4389-be26-cda76efd7e37.png">
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

$$PE_{(pos, 2i)} = sin(pos/10000^{\frac{2i}{d_{model}}})$$

### 3) Multi-head Attention
먼저 Attention Mechanism에 대해 살펴보면 다음과 같다. Attention mechanism의 목적은 한 토큰


# Experiment & Result

# Contribution
