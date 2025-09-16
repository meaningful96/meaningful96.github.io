---
title: "[논문리뷰]Memory Decoder: Memory for Large Language Models(2025)"

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
<span style="font-size:110%">**Large Language Models (LLMs)**</span>  
**LLM**은 파라미터 안에 내재된 확률 분포만 사용하여, 주어진 문맥(context)으로부터 다음 토큰의 확률을 예측한다. LLM의 입력으로 들어가는 토큰 시퀀스를 $$x = (x_1, x_2, \cdots, x_{t-1})$$, 생성해야할 다음 토큰을 $$y_t$$라고 할 때, LLM의 입력 텍스트에 대한 다음 토큰 예측 분포는 다음과 같다.

- **Next Token Prediction**  
<center>$$p_{\text{LLM}}(y_t \vert y_{<t}; \theta)$$</center>

- **Autoregressive Sequence Factorization**  
<center>$$p_{\text{LLM}}(y_{1:T}) = \prod_{t=1}^T p_{\text{LM}}(y_t \vert y_{<t})$$</center>

LLM이 특정 도메인에 대한 Knowledge를 이용해서 추론을 진행해야 할 경우에는, 일반적으로 RAG를 활용해 외부 문서를 검색해 LLM에게 질문과 함께 입력시킨다. 하지만, 이럴 경우 주어진 쿼리에 대해 검색에 소요되는 시간과 검색되는 문서의 개수나 컨텍스트 길이에 따라 추론 시간이 매우 길어진다.

<span style="font-size:110%">**Non-parametric Retrieval**</span>    
**비매개변수 검색기(Non-parametric Retrieval)**는 도메인 지식을 활용하면서도, 전통적인 RAG 방식에서 발생하는 검색 지연과 컨텍스트 확장으로 인한 attention 연산량 증가 문제를 완화하기 위해 제안된 방법이다. 구체적으로, 도메인 코퍼스에 대해 가능한 <span style="color:gold">**모든 문맥에 대해 LLM 특정 레이어의 임베딩을 key, 해당 문맥 직후에 등장하는 토큰을 value**</span>로 하는 KV 데이터스토어를 사전에 구축한다. 문맥(context)은 예를 들어, 도메인 문서의 "Pancreatic cancer arises when cells in the pancreas, a glandular organ behind the stomach, begin to multiply out of control and form a mass."라는 문장이 있을때, "Pancreatic", "Pancreatic cancer", "Pancreatic cancer arises", ... 이런식으로 나올 수 있는 모든 경우의 수를 문맥이라고 한다.

<center>$$p_{\text{kNN}}(y_t \vert y_{<t}) \propto \displaystyle\sum_{(k_i, v_i) \in N(k_t, k)} \mathbf{1}_{y_t = v_i} \exp \left( - \frac{d(k_t, k_i)}{\tau} \right)$$</center>

이후 입력 쿼리로부터 LLM이 다음 토큰을 예측할 때, 동일한 레이어의 임베딩을 추출하여 데이터스토어에서 kNN 검색을 수행하고, 유사한 문맥들의 임베딩을 기반으로 다음 토큰 확률 분포를 계산한다. 

- **Next Token Prediction**
<center>$$p(y_t \vert y_{<t}) = \lambda p_{\text{kNN}}(y_t \vert y_{<t}) + (1 - \lambda)p_{\text{LLM}} (y_t \vert y_{<t})$$</center>

- **Autoregressive Sequence Factorization**  
<center>$$p(y_{1:T}) = \prod_{t=1}^T \big[ \lambda p_{\text{kNN}}(y_t \vert x) + (1 - \lambda)p_{\text{LLM}} (y_t \vert x) \big]$$</center>

이 확률 분포는 원래 LLM이 예측한 분포와 가중 합을 이루어 최종 분포를 결정한다. 다만 이 방식은 여전히 대규모 데이터스토어 구축으로 인한 메모리 비용과, **매 토큰 예측마다 반복되는 유사도 계산으로 인한 추론 지연(latency)**이라는 한계를 가진다.

## Pretraining
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.10%5DMemoryDecoder/figure8.png?raw=true">
</p>

사전 학습 단계의 목표는 **“입력 context에 대해,** <span style="color:gold">**non-parametric retrieval가 생성한 확률 분포가 Memory Decoder가 생성한 확률 분포와 동일**</span>**”해지도록 만드는 것**이다. 이 방식은 큰 datastore에 존재하는 key-value 쌍으로 된 **도메인 지식을 compact한 모델에 내제화** 시키는 과정이다. 입력 텍스트의 토큰 시퀀스를 $$x = (x_1, x_2, \cdots, x_{t-1})$$, 타겟 토큰을 $$y_t$$라고 할 때,

- **Input**: Domain Corpus내 문맥 $$x_i$$ + non-parametric retrieval가 생성한 확률 분포 $$p_{\text{kNN}}(\cdot \vert x_i)$$
- **Output**: Memory Decoder가 계산한 확률 분포 $$p_{\text{Mem}}(\cdot \vert x_i)$$, Next Token $$y_{i}$$

먼저 Memory Decoder를 학습하기 위해서는 non-parametric retrieval가 생성한 확률 분포를 supervision으로 사용해야 한다. 이를 위해 도메인별 코퍼스를 사용하여 문맥 임베딩-다음에 올 토큰 쌍을 Key-Value로 하는 KV 데이터스토어를 구축한다.

<center>$$(K, V) = \{ (\phi(x_i), y_i) \vert (x_i, y_i) \in \mathcal{D}_{\text{Domain}} \}$$</center>

이렇게 만들어진 데이터 스토어를 기반으로 입력된 도메인 코퍼스의 문맥에 대한 확률 분포 $$p_{\text{kNN}}(\cdot \vert x_i)$$를 계산하게 된다. 

## Objective Function
학습은 이 캐싱된 kNN 분포 $$p_{\text{kNN}}(\cdot \vert x_i)$$를 감독 신호로 사용하여 Memory Deocder의 출력 분포 $$p_{\text{Mem}}(\cdot \vert x_i)$$를 정렬(align) 시키는 방ㅂ식으로 진행된다. 학습에는 총 두 가지 목적 함수를 사용한다.

- **KL Divergence Loss**  
<center>$$\mathcal{L}_{\text{KL}}(x_i) = \text{KL}(p_{\text{kNN}}(\cdot \vert x_i) \vert \vert p_{\text{Mem}}(\cdot \vert x_i))$$</center>

먼저 정렬을 위해서 KL Divergence 기반의 loss를 정의한다.

- **Next Token Prediction Loss**  
<center>$$\mathcal{L}_{\text{LM}}(x_i) = -\log p_{\text{Mem}}(y_i \vert x_i)$$</center>  
<center>$$\mathcal{L} (x_i) = \beta \cdot \mathcal{L}_{\text{KL}}(x_i) + (1-\beta) \cdot \mathcal{L}_{\text{LM}}(x_i)$$</center>  

또한, corpus의 분포에서 과도한 편차를 방지하기 위해, 보완적으로 표준 언어 모델링을 위한 목적 함수를 사용한다. 즉, KL Divergence는 Memory Decoder의 출력 분포가 kNN 분포를 모방하도록 유도하고, 언어 모델링 손실은 모델이 코퍼스의 구조적 패턴을 유지하도록 만든다.

## Inference
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.10%5DMemoryDecoder/figure9.png?raw=true">
</p>

사전 학습이 완료되면, 메모리 디코더는 플러그 앤 플레이 기능을 통해 호환 가능한 토크나이저를 가진 어떤 언어 모델이든 단순한 보간(interpolation)만으로 대상 도메인에 적응할 수 있다. 추론시에는 LLM과 Memory Decoder는 동일한 입력 text를($$x$$) 병렬로 처리한다. 

<center>$$p_{\text{Mem-PLM}} (y_t \vert x) = \alpha \cdot p_{\text{Mem}}(y_t \vert x) + (1-\alpha) \cdot p_{\text{PLM}}(y_t \vert x) $$</center>

Memory Decoder는 단 한 번의 forward pass만으로 도메인 특화된 확률 분포를 생성한다. 그런 다음, 이 Memory Decoder의 출력 분포 $$p_{\text{Mem}}(\cdot \vert x_i)$$와 기본 LLM의 출력 분포 $$p_{\text{PLM}}(\cdot \vert x_i)$$를 interpolate하여 최종 예측을 생성한다. 이러한 방법은

- 별도의 문서를 검색하지 않고 도메인 지식을 활용할 수 있음 (RAG의 한계)
- 별도의 문서 검색이 없기 때문에 입력 텍스트가 길어지지 않음 (RAG의 한계)
- 별도의 kNN연산이 필요하지 않음 (non-parametric retrieval의 한계)
- LLM을 full fine-tuning하지 않고 크기가 작은 Memory Decoder만 학습 (DAPT의 한계) 

라는 장점들을 가지며, **추론 속도가 일반적인 RAG기법보다 빠르다**.

<br/>
<br/>

# Experiments
## Main Result: Perplexity Comparison
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.10%5DMemoryDecoder/figure4.png?raw=true">
</p>

GPT-2 계열 모델(GPT-2 small, medium, large, xl)에 대해 Wikitext-103에서의 perplexity를 비교한 결과를 제시한다. 결과적으로, Memory Decoder는 모든 크기의 모델에서 성능을 향상시켰으며, 특히 GPT-2 medium에 적용했을 때는 동일 크기의 DAPT보다 낮은 perplexity를 달성했다. 이는 <span style="color:gold">**원래 모델 파라미터를 수정하지 않고도 Memory Decoder가 도메인 지식을 효과적으로 반영하여 언어 모델링 성능을 개선**</span>할 수 있음을 보여준다

## Ablation Study
<p align="center">
<img width="600" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.10%5DMemoryDecoder/figure7.png?raw=true">
</p>

Table 9는 Memory Decoder를 학습할 때 사용하는 손실 함수 구성 요소의 효과를 분석한 결과이다. +KL Only(KL divergence만 사용)와 +CE Only(Cross-Entropy만 사용) 모두 단독으로는 충분한 성능을 내지 못하고, 특히 CE 단독일 때 가장 높은 perplexity를 보였다. 반면, 제안된 +MemDec(KL과 CE를 모두 사용)은 모든 모델 크기(Qwen2.5-3B, Qwen2.5-7B, Qwen2-7B)에서 가장 낮은 perplexity를 달성했다. 결과적으로 <span style="color:gold">**KL Deivergence를 사용하지 않으면(+CE) perplexity가 가장 커지므로, KL Divergence Loss의 성능 gain이 가장 크다**</span>.

즉, KL divergence를 통해 kNN 분포를 정밀하게 근사하면서도, Cross-Entropy를 통해 언어 모델링 분포와의 안정성을 유지하는 하이브리드 학습 방식이 가장 효과적임을 입증한다. 특히 Qwen2.5-7B에서 KL Only(3.84) 대비 MemDec(3.57)이 가장 큰 개선을 보이며, 이 결과는 두 손실이 상호 보완적이라는 점을 강조한다.

## Performance in Downstream NLP Tasks
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.10%5DMemoryDecoder/figure5.png?raw=true">
</p>

감정 분석(SST2, MR, CR, RT), 텍스트 분류(AGN, Yahoo), 자연어 추론(CB, RTE, HYP) 등 아홉 가지 다운스트림 NLP 과제에서의 성능을 보여준다. Memory Decoder는 **평균적으로 base 모델, kNN-LM, DAPT, LoRA를 모두 능가**했으며, 특히 자연어 추론 계열(CB, RTE)에서 뚜렷한 강점을 보였다. 또한 DAPT가 특정 과제에서 catastrophic forgetting을 일으키는 반면, Memory Decoder는 전반적으로 안정적인 성능 향상을 유지했다. 이는 Memory Decoder가 다양한 도메인과 태스크에서 일반성과 적응성을 동시에 달성할 수 있음을 입증한다.

## Inference Latency
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.10%5DMemoryDecoder/figure3.png?raw=true">
</p>

다양한 도메인 적응 방법들의 추론 지연(latency)을 비교한 결과를 보여준다. kNN-LM과 In-Context RAG는 매 토큰마다 최근접 이웃 검색 또는 긴 문맥 처리로 인해 추론 시간이 선형적으로 증가하는 반면, Memory Decoder는 작은 Transformer 디코더 한 번만 거치면 되므로 훨씬 효율적이다. 특히 대규모 데이터스토어(예: 5억 엔트리) 환경에서 kNN-LM의 검색 비용이 급격히 증가하는 상황에서도, <span style="color:gold">**Memory Decoder는 고정된 오버헤드(약 1.28배)에 그치며 최대 10배 이상의 속도 이점**</span>을 보였다

## Perplexity Comparison
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.10%5DMemoryDecoder/figure4.png?raw=true">
</p>

GPT-2 계열 모델(GPT-2 small, medium, large, xl)에 대해 Wikitext-103에서의 perplexity를 비교한 결과를 제시한다. 결과적으로, Memory Decoder는 모든 크기의 모델에서 성능을 향상시켰으며, 특히 GPT-2 medium에 적용했을 때는 동일 크기의 DAPT보다 낮은 perplexity를 달성했다. 이는 <span style="color:gold">**원래 모델 파라미터를 수정하지 않고도 Memory Decoder가 도메인 지식을 효과적으로 반영하여 언어 모델링 성능을 개선**</span>할 수 있음을 보여준다

## Cross-Model Adaptation & Cross-Vocabulary Adaptation
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.10%5DMemoryDecoder/figure6.png?raw=true">
</p>

**[Cross-Model Adaptation (Table 3)]** 하나의 MemDec을 훈련해 놓으면, 같은 tokenizer를 공유하는 다양한 크기의 모델들(0.5B~72B)에 그대로 붙여서 사용할 수 있는지 확인하기 위한 실험으로, 단일 MemDec(0.5B)로 Qwen2/2.5 계열의 0.5B~72B 모델 모두 개선, 심지어 Qwen2.5-0.5B + MemDec이 Qwen2.5-72B 기본 모델보다 성능 우위. 이는 140배 이상의 파라미터 효율성을 입증한다Memory Decoder.

**[Cross-Vocabulary Adaptation (Table 4)]** “tokenizer가 다른 모델에도 MemDec을 재사용할 수 있는가?”가에 대한 질문을 해결하기 위한 실험으로, 실험 결과  Qwen2.5에서 학습한 MemDec을 Llama 계열에 단 10% 추가 학습만으로 이식 가능하며, 특히 Llama3-8B에서 약 50% perplexity 감소 달성하였다. 전반적으로 LoRA보다 더 강력하고, cross-tokenizer 전이도 안정적으로 가능함을 입증했다.

<br/>
<br/>

# Conclusion
- **Contribution**
  - **[아키텍쳐 효율성]** 작은 Transformer 디코더가 **비매개변수 검색기(non-parametric retriever)**의 동작을 근사하도록 학습되어, 추론 시 외부 검색(latency, storage)을 제거함.
  - **[모델 호환성]** 동일한 토크나이저를 공유하는 어떤 LLM에도 쉽게 통합할 수 있는 plug-and-play 구조를 제공하여, 다양한 크기의 Qwen 및 LLaMA 모델에서 효과적인 도메인 적응을 달성함.
  - **[범용성]** 생의학, 금융, 법률 세 가지 전문 도메인에서 평균 6.17 perplexity 감소를 달성했으며, 다운스트림 태스크에서도 안정적이고 폭넓은 성능 개선을 보여줌.

<br/>

- Limitations
  - **[도메인 코퍼스 의존성]** 여전히 특정 도메인 코퍼스에 의존해야 하며, 학습 데이터 품질과 범위에 따라 성능이 제한될 수 있음.
  - **[평가 아티팩트 문제]** 일부 다운스트림 태스크(예: Yahoo, HYP)에서는 DAPT의 성능 저하가 본질적 한계가 아니라 평가 방법론(DCPMI) 아티팩트임이 드러났음.
