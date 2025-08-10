---
title: "[논문리뷰]Self-Retrieval: End-to-End Information Retrieval with One Large Language Model"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2025-08-10
last_modified_at: 2025-08-10
---

*Qiaoyu Tang, Jiawei Chen, Zhuoqun Li, et al*. **[Self-Retrieval: End-to-End Information Retrieval with One Large Language Model](https://arxiv.org/abs/2403.00801)**. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2024. arXiv:2403.00801.

# Problem Statement
**[구성 요소 분리로 인한 비효율성]** 기존 IR 시스템은 인덱싱 (indexing), 검색 (retrieval), 리랭킹(reranking) 모듈이 별도로 동작하며, LLM은 일부 구성 요소에만 제한적으로 적용된다. 이는 지식 공유와 모듈 간의 시너지 효과를 저해하며 구현 복잡성을 증가시킨다.

**[Dense/Generative Retrieval의 LLM 활용 제약]** Dense retrieval은 쿼리와 문서를 밀집 벡터로 매칭하여 LLM의 풍부한 언어 이해 능력을 충분히 활용하지 못한다. Generative retrieval은 문서 식별자(identifier) 생성에 의존하여 LLM의 자연스러운 텍스트 생성 능력과 지식을 온전히 활용하기 어렵다.

**[정확한 문서 매칭 문제]** 기존 generative retrieval은 LLM이 생성한 텍스트가 코퍼스의 실제 문서와 정확히 일치하지 않아, 후처리 과정에서 불일치 문제가 발생한다.

**[통합 RAG의 부재]** 대부분의 RAG(Retrieval-Augmented Generation) 시스템은 검색과 답변 생성을 분리하여, 컨텍스트 전달 과정에서 정보 손실과 비일관성이 발생한다.

<br/>
<br/>

# Methodology
## Overview
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.08.10%5DSelf-Retrieval/figure1.png?raw=true">
</p>

Self-Retrieval은 <span style="color:red">**하나의 LLM 안에 인덱싱, 검색, 리랭킹 과정을 통합한 End-to-end 정보 검색 아키텍처**</span>이다. 먼저 인덱싱 단계에서 self-supervised sentence-to-passage 학습을 통해 코퍼스 내용을 LLM 파라미터에 내재화한다. 이후 검색 단계에서는 쿼리를 입력받아 관련 문서 제목과 본문을 직접 생성하되, trie 기반 constrained decoding을 적용해 생성 결과가 실제 코퍼스의 문서와 정확히 일치하도록 보장한다. 마지막으로 리랭킹 단계에서는 LLM이 자체 평가(self-assessment)를 수행해 각 문서가 질의에 답변 가능한지 여부를 판단하고, 제목 생성 확률과 평가 점수를 결합해 최종 순위를 산출한다. 이 통합 구조를 통해 Self-Retrieval은 전통적인 모듈 분리형 IR 시스템 대비 높은 정확성과 효율성을 동시에 달성한다.

## Step 1. Indexing: Internalize the Corpus
<p align="center">
<img width="300" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.08.10%5DSelf-Retrieval/figure2_indexing.png?raw=true">
</p>

- **입력**
    - 단일 문장 $$s_i$$ (문서 $$p$$ 내의 각 문장),
    - 문서 $$p$$는 최대 $$L$$개의 문장으로 구성( $$p = \{s_1, s_2, \cdots, s_L\}$$)
- **출력**
    - 해당 문서 전체 $$p$$

인덱싱 과정에서는 Self-Supervised Sentence-to-Passage learning을 한다. 구체적으로, 문서 $$p = \{s_1, s_2, ..., s_L\}$$에서 한 문장 $$s_i$$를 입력으로 주고, 모델이 해당 문서 전체를 auto-regressive 방식으로 복원하도록 학습한다. 이를 통해 LLM은 코퍼스 내 문서들의 내용과 구조를 파라미터 $$\theta$$에 내재화하며, 인덱싱 과정 자체가 ‘부분 정보 → 전체 문서 복원’이라는 **retrieval-like task**로 변환한다.  이는 언어모델 사전학습(pretraining) 방식과 유사해, continuous pretraining 효과를 기대할 수 있다.인덱싱 과정에서 목적 함수 (objective function)는 $$P(p \vert s_i; \theta)$$이다. 

이러한 접근은 외부 인덱스 없이 LLM 내부 파라미터만으로 검색을 수행할 수 있게 한다. 복잡한 문서 식별자(identifier) 설계나 매핑 과정이 필요 없으며, 모델이 문서 내용을 직접 생성할 수 있다. 또한 인덱싱과 검색 능력을 동시에 학습하므로 retrieval 효율성을 높이고, 언어모델이 원문 재구성 능력을 갖추게 한다.

## Step 2. Retrieval: Generate Relevant Passage through Constrained Decoding
<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.08.10%5DSelf-Retrieval/figure3_retrieval.png?raw=true">
</p>

- **입력:** 쿼리 $$q$$
- **출력:** 제목 $$\hat{t}$$  + 관련 문서 본문 $$\hat{p}$$

검색 단계에서 Self-Retrieval은 쿼리에 대해 먼저 전역 정보를 제공하는 **문서 제목** $$\hat{t}$$을 생성하고, 이를 조건으로 **관련 문서 본문**  $$\hat{p}$$을 생성한다. 그러나 **LLM이 생성한 문장이 코퍼스의 실제 문서와 불일치할 가능성**이 있으므로, <span style="color:red">**trie 기반 constrained decoding**</span>을 사용한다. 코퍼스 전체를 prefix tree로 변환하고, 각 노드에는 다음 토큰 후보 집합을 저장한다. 생성 과정에서 모델은 이 후보 집합에 속한 토큰만 생성할 수 있으며, 특정 문서를 유일하게 식별할 수 있는 시점이 되면 나머지 부분은 코퍼스의 원문으로 자동 완성한다. 

- LLM이 쿼리 $$q$$에 대해 전역 정보 (global information)를 담은 **문서 제목 (title)**을 생성 → $$P(\hat{t} \vert q; \theta)$$
- 해당 제목을 조건으로 관련 문서 본문을 생성 → $$P(\hat{p} \vert q, \hat{t}; \theta)$$
- Ground Truth 문서와의 **정확한 코퍼스 매칭**을 위해 **trie 기반 constrained decoding**을 적용
    - 코퍼스 전체를 prefix tree $$T$$로 구축
    - 각 노드에 다음 토큰 후보 집합 저장
    - 생성 중에는 $$T$$에 허용된 토큰만 출력 가능
- 충분한 prefix가 생성되어 해당 문서가 유일하게 식별되면, 나머지는 코퍼스에서 자동완성

이로써 생성 결과와 코퍼스 문서가 정확히 일치하도록 보장한다. 이 방식은 dense retrieval처럼 임베딩 매칭에 의존하지 않고, LLM의 언어 생성 능력을 그대로 활용한다. 또한 generative retrieval에서 흔히 사용되는 식별자(identifier) 기반 접근 대신, 실제 문서 내용을 직접 생성함으로써 정보 손실을 최소화한다. trie 기반 제약 디코딩 덕분에 검색 결과가 코퍼스와 완벽히 매칭되며, 불필요한 후처리 절차가 필요 없다.

## Step 3. Reranking: Self-Assessment based Relevance Evaluation
<p align="center">
<img width="200" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.08.10%5DSelf-Retrieval/figure4_reranking.png?raw=true">
</p>

**입력(Input)**

- 후보 집합 $${(t_i, p_i)}$$ (생성된 제목, 본문 쌍)
- 쿼리 $$q$$

**출력(Output)**

- 최종 관련성 점수 $$S_i$$
- 리랭킹된 문서 목록

리랭킹 단계에서는 <span style="color:red">**LLM이 각 후보 문서에 대해 자체 평가를 수행**</span>한다. 모델은 주어진 쿼리와 문서 쌍에 대해 “can answer the query” 또는 “cannot answer the query”라는 응답을 생성하며, 이를 바탕으로 관련성을 판단한다. 평가 점수는 두 가지 요소로 구성된다. 첫째, 제목 생성 확률을 기반으로 한 제목 스코어 $$S_T$$이다. 둘째, self-assessment 결과의 거부 확률에 기반한 평가 스코어 $$S_P$$이다. 참고로 거부 확률이란 LLM이 해당 문장을 생성할 확률을 1에서 뺀 확률 값이다. 

1. LLM이 각 후보 문서에 대해 relevance 판단을 문장 형태로 생성
    - “can answer the query” (관련 있음) = Positive
    - “cannot answer the query” (관련 없음) = Rejection
2. 평가 스코어 계산
    - 제목 스코어: $$S_T^i = \text{SoftMax} \Big ( \frac{P(t_i \vert q; \theta}{\tau} \Big)$$
    - self-assessment 스코어: $$S_P^i = \text{SoftMax} \Big ( \frac{1 - P(\text{rejection} \vert q, t_i, p_i; \theta)}{\delta} \Big)$$
3. 최종 스코어: $$S_i = S_T^i \cdot S_P^i$$
4. 이 스코어로 모든 후보를 재정렬

이 두 점수는 곱셈으로 결합되어 최종 점수 $$S$$가 산출되며, 이를 기준으로 모든 후보 문서가 재정렬된다. 이 방법은 생성과 평가를 동일 LLM 내에서 처리하므로 외부 reranker가 불필요하다. 제목 확률과 본문 평가를 함께 고려하여 보다 정밀한 순위를 제공하며, 학습 시에는 positive/negative 예시를 활용해 모델이 정확한 판별 능력을 갖추도록 한다. 이를 통해 검색 결과의 품질이 향상되고, downstream RAG 단계에서도 더 정밀한 컨텍스트 제공이 가능해진다.

## Training & Inference

훈련 과정에서는 인덱싱, 검색, 리랭킹 과제를 모두 텍스트 생성 형태로 통합하여 auto-regressive cross-entropy loss로 학습한다. 인덱싱 단계에서는 문장-문서 복원 학습을 수행하고, 검색 단계에서는 문서 제목과 본문을 생성하며, 리랭킹 단계에서는 positive/negative 예시를 기반으로 관련성을 평가한다. 특히, RAG 형태의 응답 생성을 가능하게 하기 위해 golden answer를 self-assessment 응답 뒤에 이어 붙여 학습함으로써, 모델이 관련성 판단 직후 곧바로 답변을 생성할 수 있도록 한다. 

이를 통해 동일한 LLM이 retrieval과 answer generation을 end-to-end로 수행할 수 있는 구조를 갖추게 된다. 추론 시에는 먼저 beam search로 여러 개의 제목을 생성하고, 각 제목에 대해 여러 개의 본문을 생성한 뒤, self-assessment 점수로 리랭킹하여 최종 상위 문서를 선택하며, 필요할 경우 이 문서들과 질문을 함께 입력해 동일 모델이 최종 답변까지 생성한다.

- **Training**
    - **인덱싱 데이터**: self-supervised sentence-to-passage
    - **검색 데이터**: supervised query-passage pairs (제목+본문 생성)
    - **리랭킹 데이터**: positive/negative passage 샘플
    - Auto-regressive cross-entropy loss 사용
- **Inference**
    1. Beam search로 $$i$$개의 제목 생성
    2. 각 제목에 대해 $$j$$개의 본문 생성
    3. self-assessment 스코어 계산
    4. 최종 리랭킹 후 상위 $$k$$개 문서 반환 + 답변 생

이 통합 학습 접근은 전통적으로 분리된 IR 구성 요소를 하나의 LLM 안에 결합함으로써, 모듈 간 정보 손실을 줄이고 모델의 전반적인 효율성과 일관성을 높인다. 또한 RAG와 같은 downstream 태스크를 별도 설계 없이 직접 통합할 수 있어 범용성이 높다.

<br/>
<br/>

# Experiments
## Main Results
<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.08.10%5DSelf-Retrieval/table1.png?raw=true">
</p>

Self-Retrieval은 NQ와 TriviaQA에서 모든 sparse, dense, generative retrieval 기법을 초월했다. StableLM-3B 기반은 fine-tuned BGE 대비 NQ에서 MRR@5 +5.46, TriviaQA에서 +5.07 향상했고, Llama2-7B 기반은 NQ 70.00, TriviaQA 68.74의 MRR@5로 최고 성능을 기록했다. 기존 generative retrieval 모델인 DSI-XXL보다 최대 19포인트 이상 높았다.

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.08.10%5DSelf-Retrieval/table23.png?raw=true">
</p>

Self-Retrieval은 기존 SOTA인 GenRet 대비 R@1 +5.2, R@10 +3.8, MRR@100 +4.8 향상을 기록했다. 별도의 query generation 데이터 증강 없이도 높은 성능을 달성해, 문서 레벨 검색에서 강력한 일반화를 보였다.

Wikipedia 기반이 아닌 비정형 환경과 제목이 없는 상황에서도 GenRet과 유사한 SOTA 성능을 달성했다. 제목 부재 문제는 Llama2 기반 자동 제목 생성으로 해결했으며, R@1에서 47.8로 상위권을 유지했다.

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.08.10%5DSelf-Retrieval/table5.png?raw=true">
</p>

BGE-FT+Reader 파이프라인 대비 모든 실험 설정(10K/40K 문서, StableLM/Llama2 기반)에서 EM 점수가 크게 향상됐다. Llama2-7B 기반은 TriviaQA 40K에서 70.40의 EM을 기록하며 최고 성능을 보였다.

## Ablation Study
<p align="center">
<img width="600" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.08.10%5DSelf-Retrieval/table4.png?raw=true">
</p>

- 인덱싱 제거 → NQ MRR@5 -10.5 감소
- 제목 제거 → 전역 정보 손실로 NQ MRR@5 -16.6 감소
- Self-assessment 제거 → 관련성 평가 부정확, MRR@5 약 -6.7 감소
- 각 모듈이 성능 유지에 필수적임을 입증

세 구성 요소 중 **제목(title) 생성이 성능에 가장 큰 기여**를 했다. 제목 제거 시 NQ MRR@5가 –16.6, TriviaQA MRR@5가 –8.24 하락하며, 다른 요소 제거보다 감소 폭이 훨씬 컸다. 이는 제목이 전역적인 문서 식별 정보로 작용해, 본문 생성의 품질과 검색 정확도 모두에 중요한 역할을 한다는 것을 의미한다.

## Analysis 1. Scaling corpus size
<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.08.10%5DSelf-Retrieval/figure4.png?raw=true">
</p>

Figure 4에서 Self-Retrieval(3B)과 BGE-FT를 각각 NQ와 TriviaQA에서 10K200K 문서(약 290K3M passages) 규모로 확장해 성능 변화를 측정했다. 결과적으로 두 모델 모두 코퍼스가 커질수록 성능이 감소했지만, 감소율은 비슷했고 Self-Retrieval은 대규모 환경에서도 안정적으로 성능을 유지했다. 특히 기존 연구에서 generative retrieval(DGI, NCI 등)은 대규모로 갈수록 dense retrieval 대비 급격히 성능이 떨어지는 경향이 보고되었지만, Self-Retrieval은 이 한계를 완화했다. 이는 trie 기반 제약 디코딩과 내부 코퍼스 내재화 덕분에, 검색 공간이 커져도 노이즈나 불일치 문제를 효과적으로 억제할 수 있음을 시사한다. 따라서 Self-Retrieval은 수백만 단위의 대규모 문서 집합에서도 실용 가능성이 높다.

## Analysis 2. Efficiency
<p align="center">
<img width="500" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.08.10%5DSelf-Retrieval/table9.png?raw=true">
</p>

Self-Retrieval은 trie 기반 구조로 메모리 사용량을 30MB로 억제해, 같은 natural language decoding 기반인 SEAL(444MB)보다 14배 이상 효율적이다. Latency 측면에서 beam size 10에서 1.44초, beam size 100에서 6.06초로, SEAL과 비슷하거나 약간 높은 수준이지만 DSI-XL(0.23초)보다는 느리다. 그러나 beam size 10에서도 Hits@5가 76.17로 SEAL(61.91)이나 DSI-XL(60.21)보다 월등히 높아, 적은 연산량으로도 고품질 검색이 가능하다. Beam size를 100으로 늘리면 Hits@5가 81.49까지 올라가, 연산량과 성능 간 유연한 조절이 가능하다. 이러한 특성은 최적화 기법(quantization, attention acceleration)과 하드웨어 개선 시 효율성이 더 높아질 가능성을 보여준다.


