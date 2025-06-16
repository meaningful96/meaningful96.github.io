---
title: "[논문리뷰]Generate-then-Ground in Retrieval-Augmented Generation for Multi-hop Question Answering"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2025-06-15
last_modified_at: 2025-06-15
---

*Zhengliang Shi, Shuo Zhang, Weiwei Sun, Shen Gao, Pengjie Ren, Zhumin Chen, and Zhaochun Ren**. “[**Generate-then-Ground in Retrieval-Augmented Generation for Multi-hop Question Answering**](https://aclanthology.org/2024.acl-long.397/).” In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7339–7353, Bangkok, Thailand, August 2024.

# Problem Statement
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.06.15%5DGenGround/GenGround1.png?raw=true">
</p>

기존의 **retrieve-then-read 패러다임은 MHQA에서 비효율적이며, LLM의 내재적 지식을 제대로 활용하지 못하는** 한계가 존재한다. Multi-hop QA는 다양한 문서와 정보를 통합하여 논리적 추론을 요구하는 복잡한 작업이다. 대부분의 기존 연구는 retrieve-then-read 방식에 따라 외부 문서를 검색한 후 LLM이 이를 읽고 정답을 생성한다. 그러나 이 방식은 다음과 같은 두 가지 핵심 한계를 가진다.

- **Retrieval 의존성**: 정답 도출이 전적으로 검색된 문서에 의존하기 때문에, 중요한 정보를 포함하지 못한 문서가 검색되면 LLM은 정답을 유추하지 못한다.
- **Noise 및 환각(hallucination)**: 검색된 문서에는 노이즈가 많으며, 이로 인해 LLM이 사실이 아닌 응답을 생성할 수 있다.

이러한 한계를 극복하고자, 저자들은 LLM이 보유한 parametric knowledge와 외부 문서의 grounding evidence를 교대로 사용하는 **Generate-then-Ground (GenGround)** 프레임워크를 제안한다.

<br/>
<br/>

# Methodology
<p align="center">
<img width="900" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.06.15%5DGenGround/GenGround2.png?raw=true">
</p>

GenGround는 **Answer Deduction**과 **Instructional Knowledge Grounding**의 두 단계를 교대로 수행한다. 

## Answer Deduction
- **입력**: 질문 $$Q$$와 현재 iteration의 context $$H =  {(q_j, \tilde{a}*j)}*{j < i}$$
- **출력**: 단순화된 sub-question $$q_i$$와 그에 대한 LLM의 즉각적인 답변 $$a_i$$
- **핵심**: LLM의 내재된 world knowledge를 이용하여 **sub-question을 유도하고 답변을 생성**

**Answer Deduction** 모듈은 LLM의 매개변수 내 저장된 세계 지식(world knowledge)을 활용한다. 복잡한 다중 홉 질문을 단순한 단일 홉 질문들로 분해하고, 현재 컨텍스트와 입력된 질문을 바탕으로 하위 질문을 생성한 후, 해당 질문에 대한 즉각적인 답변을 직접 생성한다. 

## Instructional Knowledge Grounding
- **입력**: sub-question $$q_i$$, answer $$a_i$$
- **출력**: 외부 문서를 바탕으로 수정된 답변 $$\tilde{a}_i$$
- **단계**:
    1. $$q_i$$를 이용해 retriever를 통해 문서 집합 $$D_i$$를 수집
    2. $$a_i$$의 정당성을 문서 내에서 검증하고 필요시 수정
 
**Instructional Knowledge Grounding** 모듈은 LLM의 환각 문제를 해결하기 위해 외부 문서를 활용한다. <span style="color:red">**생성된 하위 질문을 사용하여 관련 문서를 검색**</span>한 후, <span style="color:red">**LLM이 질문-답변 쌍을 검색된 문서의 증거에 근거하여 검증하고 수정**</span>하도록 한다. 가장 관련성 높은 내용을 인용하고 답변을 수정하고 개선한다.

## Batch Knowledge Grounding
<p align="center">
<img width="500" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.06.15%5DGenGround/GenGround3.png?raw=true">
</p>

- **목적**: 긴 문서 리스트를 일괄로 검토하지 않고, 문서를 배치 단위로 나눠 점진적으로 evidence를 찾음으로써 효율성과 정밀도를 개선
- **전략**: 한 배치에 evidence가 있으면 grounding을 종료하고 다음 단계로 진행함.

**Batch Knowledge Ground 전략은** 검색된 문서들의 노이즈를 줄이기 위해 배치 단위로 문서를 처리한다. 배치 크기가 $$b$$일 때, 처음 ($$1, b$$)번째 문서들로 답변을 수정하고, 증거를 찾지 못하면 다음 배치 ($$b+1, 2b$$)로 진행한다.

## Instructional Grounding Distillation (IDG)
- **목적**: ChatGPT 기반 GenGround를 소형 모델(Mistral-7B 등)에 distill하기 위한 학습 데이터 생성
- **구성**: Natural Questions 데이터셋에서 50k 샘플을 활용하여 ChatGPT로 수정 trajectory를 생성하고, 이를 기반으로 student model을 지도 학습함.

**Instructional Grounding Distillation (IGD) 방법은** ChatGPT의 grounding 능력을 작은 모델에 전수한다. Natural Questions 데이터셋의 50k 질문을 사용하여 ChatGPT가 생성한 수정 궤적을 학생 모델에 instruction tuning으로 학습시킨다.

<br/>
<br/>
# Experiments
## Main Results
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.06.15%5DGenGround/GenGround4.png?raw=true">
</p>

4개의 MHQA 벤치마크에서 평가한 결과, GenGround는 모든 데이터셋에서 최고 성능을 달성했다. HotpotQA에서 Acc=47.27, F1=52.26, Acc†=55.73을 기록하여 기존 retrieve-then-read 방법들보다 4-6%p 향상된 정확도를 보였다. 특히 MuSiQue에서는 Acc=20.24로 기존 최고 성능 대비 상대적으로 13% 개선했고, 2WikiQA에서도 Acc=45.61로 우수한 성능을 달성했다.


## Ablation Study
<p align="center">
<img width="500" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.06.15%5DGenGround/GenGround5.png?raw=true">
</p>

- **Answer Deduction**의 성능 gain이 가장 크다.

HotpotQA와 StrategyQA에서 수행한 ablation study에서 각 구성요소의 중요성이 입증되었다. Answer deduction 단계를 제거했을 때 HotpotQA에서 Accuracy가 6%p, StrategyQA에서 10%p 하락했다. Grounding 단계를 제거했을 때는 HotpotQA에서 F1이 7%p, Accuracy가 4%p 하락했다. Batch grounding 전략을 제거했을 때도 HotpotQA에서 F1이 5%p, Accuracy가 2%p 하락하여 모든 구성요소가 성능에 중요한 역할을 함을 보여준다.

## Analysis 1. Knowlede Incorporation
<p align="center">
<img width="500" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.06.15%5DGenGround/GenGround6.png?raw=true">
</p>

100개의 HotpotQA 샘플을 분석한 결과, 전체 성공률은 53.2%였다. LLM이 직접 정답을 생성한 경우가 28.7%, 초기에 잘못된 답변을 생성했지만 외부 문서를 통해 수정한 경우가 24.5%였다. 오류율은 5.6%로 매우 낮아, LLM이 검색된 문서를 효과적으로 활용함을 보여준다. 이는 매개변수 지식과 외부 지식 모두를 통합하는 것의 중요성을 강조한다.

## Analysis 2. Efficiency
<p align="center">
<img width="500" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.06.15%5DGenGround/GenGround7.png?raw=true">
</p>

토큰 소비량 분석에서 GenGround는 평균 3541.6개의 토큰을 사용하여 GRG w/ decomposition(7806.4개)과 RetGen(8917.8개)보다 훨씬 효율적이었다. 이는 프레임워크가 LLM의 deduction 능력을 활용해 복잡한 질문을 단순한 하위 질문으로 분해하고 직접 답변을 생성하여 더 짧은 추론 경로를 만들기 때문이다.

<br/>
<br/>

# Conclusion
## Contributions
이 논문의 주요 기여는 다음과 같다. 첫째, Multi-hop Question Answering 태스크를 위한 새로운 Generate-then-Ground 프레임워크를 제안했다. 이는 기존의 retrieve-then-read 패러다임과 달리 LLM이 먼저 답변을 생성한 후 검색된 문서로 이를 검증하고 수정하는 방식이다. 둘째, LLM의 파라메트릭 지식과 외부 문서를 효과적으로 결합하는 방법을 제시했다. Answer deduction 단계에서는 LLM의 내재된 지식을 활용하고, grounding 단계에서는 외부 문서로 hallucination을 수정한다. 셋째, Instructional Grounding Distillation 방법을 통해 작은 모델에서도 이 프레임워크를 사용할 수 있도록 했다. 이를 통해 Mistral-7B가 ChatGPT 기반 방법들과 비슷하거나 더 좋은 성능을 달성했다.

## Limitations
이 연구의 한계점은 다음과 같다.
- 첫째, 프레임워크의 첫 단계인 초기 답변 생성이 태스크에 따라 달라질 수 있다. 다른 도메인이나 태스크에서는 의미 있는 초기 답변을 생성하기 어려울 수 있어 적용 범위가 제한된다.
- 둘째, 복잡한 질문을 단순한 질문들로 분해할 수 있다고 가정하지만, 질문 분해 자체가 어려운 문제이며 현재 프레임워크에서 충분히 탐구되지 않았다.
- 셋째, 외부 문서가 초기에 생성된 비사실적 진술을 수정할 수 있다고 가정하지만, 필요한 정보가 없거나 잘못된 정보가 포함된 경우 프레임워크의 효과성이 떨어질 수 있다.
