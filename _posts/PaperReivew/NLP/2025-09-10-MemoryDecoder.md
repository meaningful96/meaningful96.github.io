---
title: "[논문리뷰]Memory Decoder: Memory for Large Language Models"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2025-09-10
last_modified_at: 2025-09-10
---

*Jiaqi Cao, Jiarui Wang, Rubin Wei, Qipeng Guo, Kai Chen, Bowen Zhou, and Zhouhan Lin*. 2025. [Memory Decoder: A Pretrained, Plug-and-Play Memory for Large Language Models](https://arxiv.org/abs/2508.09874). arXiv:2508.09874 [cs.CL] https://arxiv.org/abs/2508.09874

# Problem Statement
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.10%5DMemoryDecoder/figure1.png?raw=true">
</p>

<span style="font-size:110%">**Domain Adaptive PreTraining (DAPT)의 근본적인 한계**</span>  
DAPT는 특정 도메인의 데이터로 LLM을 continuous pretraining하는 방법이다. 하지만, LLM의 모든 파라미터를 full fine-tuning해야 하므로 계산 비용이 매우 높고, 모델 크기가 커질수록 매우 비효율적이다. 또한, 여러 모델을 동일한 도메인에 적응시키려면 각 모델마다 별도로 학습해야 하는 자원 비효율성이 존재한다. 가장 중요한 한계점은, <span style="color:gold">**학습 과정에서 모델이 가진 기존의 일반화 능력을 잃어버리는 catastrophic forgetting 현상이 발생**</span>한다.

<span style="font-size:110%">**Retrieval-Augmented Generation (RAG)의 근본적인 한계**</span>  
기존의 RAG 모델은 파라미터를 유지하면서(freeze) 외부 저장소에서 관련 정보를 검색하여 모델의 출력을 보강한다. 그러나 추론 시마다 방대한 데이터베이스에서 최근접 이웃(kNN) 검색을 수행해야 하므로 상당한 **추론 지연(latency)**이 발생한다.

<br/>
<br/>

# Methodology
## Overview
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.10%5DMemoryDecoder/figure2.png?raw=true">
</p>

Memory Decoder는 기존의 LLM을 특정 도메인에 효율적으로 적응시키기 위해 고안된 새로운 방법이다. 이 방법론은 사전 학습 시  Memory Decoder와 **비매개변수 검색기(non-parametric retriever) 간의 출력 분포를 정렬(align)**하는 과정을 통해 방대한 도메인 지식을 소형 매개변수 모델에 압축하여 인코딩한다. 이러한 과정을 통해 학습된 Memory Decoder는 추론 시 검색 오버헤드를 없애고, 어떤 LLM과도 파라미터를 수정하지 않고 결합 가능한 plug-and-play 방식의 효율적인 추론 메커니즘을 제공한다. 구체적으로 앞서 언급한 DAPT와 RAG의 한계를 다음과 같은 방법으로 해결한다.

<span style="font-size:110%">**DAPT의 한계 해결**</span>    
DAPT의 **치명적인 망각(catastrophic forgetting) 현상**과 자원 비효율성은 모델의 모든 파라미터를 미세 조정하는 것에서 비롯된다. 이를 해결하기 위해 Memory Decoder는 다음과 같은 방법을 제안한다.

- **[소형 디코더 학습]**: Memory Decoder는 기본 LLM의 파라미터를 동결(freeze)하고 건드리지 않습니다. 대신, <span style="color:gold">**도메인 지식을 담당하는 별도의 소형 트랜스포머 디코더를 학습**</span>시킨다. 이 방식은 LLM의 기존 일반화 능력을 보존하면서 도메인별 지식을 추가하기 때문에 치명적인 망각 현상이 발생하지 않는다.
- **[토크나이저 공유]**: Memory Decoder는 단일한 소형 모델을 사전 학습하여 <span style="color:gold">**동일한 토크나이저를 사용하는 모든 LLM에 plug-and-play 방식으로 적용**</span>할 수 있다. DAPT처럼 모델 크기별로 또는 모델 종류별로 각각 학습할 필요가 없으므로, 자원 활용 측면에서 매우 효율적이다.

<span style="font-size:110%">**RAG의 한계 해결**</span>  

- **[검색 과정의 제거]**: Memory Decoder는 사전 학습 단계에서 비매개변수 검색기가 생성하는 분포를 모방하도록 학습하여 <span style="color:gold">**도메인 지식을 내제화**</span>한다. 즉, 추론 시에는 외부 데이터베이스에서 문서를 검색하는 대신, 이미 학습된 Memory Decoder 모델의 순방향 전달(forward pass) 연산만으로 필요한 지식을 즉각적으로 활용한다. 이 방식은 **kNN 검색에 소요되는 시간을 완전히 없애므로**, 기존 RAG에 비해 훨씬 낮은 추론 오버헤드만으로도 도메인 적응을 가능하게 한다.

## LLM과 Non-parametric Retriever의 Token generation의 차이
LLM은 파라미터 안에 내재된 확률 분포만 사용하여, 주어진 문맥(context)으로부터 다음 토큰의 확률을 예측한다. LLM의 입력으로 들어가는 토큰 시퀀스를 $$x = (x_1, x_2, \cdots, x_{t-1})$$, 생성해야할 다음 토큰을 $$y_t$$라고 할 때, LLM의 입력 텍스트에 대한 다음 토큰 예측 분포는 다음과 같다.

- **Next Token Prediction**
<center>$$p_{\text{LLM}}(y_t \vert x; \theta) = $$</center>

- **Autoregressive Sequence Factorization**
<center>$$p_{\text{LLM}}(y_{1:T}) = \prod_{t=1}^T p_{\text{LM}}(y_t \vert x) $$</center>

## Pretraining
사전 학습 단계의 목표는 **“입력 context에 대해,** <span style="color:gold">**non-parametric retrieval가 생성한 확률 분포가 Memory Decoder가 생성한 확률 분포와 동일**</span>**”해지도록 만드는 것**이다. 이 방식은 큰 datastore에 존재하는 key-value 쌍으로 된 **도메인 지식을 compact한 모델에 내제화** 시키는 과정이다. 입력 텍스트의 토큰 시퀀스를 $$x = (x_1, x_2, \cdots, x_{t-1})$$, 타겟 토큰을 $$y_t$$라고 할 때,


<br/>
<br/>

# Experiments



<br/>
<br/>

# Conclusion
