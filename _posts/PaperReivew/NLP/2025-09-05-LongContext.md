---
title: "[논문리뷰]Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2025-09-05
last_modified_at: 2025-09-05
---
*Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O. Arik*. 2025. [Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG](https://arxiv.org/abs/2410.05983). In Proceedings of the International Conference on Learning Representations (ICLR 2025). International Conference on Learning Representations.

# Problem Statement
<p align="center">
<img width="500" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.02%5DLongContext/figure2.png?raw=true">
</p>
<center><span style="font-size:80%"><em>Lost in the Middle: How Language Models Use Long Contexts</em></span></center>

<span style="font-size:110%">**Lost-in-the-middle**</span>    
**Lost-in-the-middle**는 LLM이 긴 입력을 처리할 때 시퀀스의 <span style="color:gold">**앞과 뒤 정보는 잘 활용하지만 중간에 위치한 정보는 무시**</span>하는 현상이다. 위의 그림은 20개의 검색 문서를 입력하고 정답 문서를 1~20번째 임의 위치에 배치했을 때의 성능 변화를 보여준다. 정답 문서가 7~16번째에 있을 경우 정답률이 크게 떨어지며, 심지어 검색 문서를 주지 않은 closed-book 설정보다도 낮아진다. 검색 문서 수가 많아질수록 핵심 문서가 중간에 위치할 가능성이 높기 때문에, 이 문제를 완화하는 것이 중요하다.

<span style="font-size:110%">**Double-edge Sword Effect of Retrieval**</span>  
이는 lost-in-the-middle 문제와 연결된다. 더 강력한 retrieval은 높은 recall과 precision을 제공하지만, 동시에 의미적으로 유사하나 실제로 정답을 포함하지 않거나 추론에 도움이 되지 않는 **distractor** 문서들을 더 많이 검색하게 된다. 즉, retrieval 성능이 향상될수록 정답 문서와 구분하기 어려운 <span style="color:gold">**hard negative**</span>가 증가하며, 이는 LLM의 추론을 방해한다.

<br/>
<br/>

# Chanllenges of Long Context LLMs in RAG
이 섹션에는 long-context상황에서 LLM에 발생할 수 있는 문제를 실험과 함께 좀 더 중점적으로 다룬다.

## The Effect of Retrieved Context Size on RAG Performance
<p align="center">
<img width="500" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.02%5DLongContext/figure3.png?raw=true">
</p>

**Research Question**  
- 더 많은 양의 Passage(or Document)를 검색해 LLM에게 입력시키면, 일관되게 성능이 향상되는가?

**Experimental Setup**  
- **Input**:  Question + Top-$$k$$ Passages
- **Output**: Answer

**Results**  
- Parameter 수가 200B에 달하는 Gemini-1.5-Pro는 문서 수에 robust하고, sLLM에서는 입력으로 passage가 10개만 넘어가도 성능 감소가 급격하게 발생함.
- 강력한 Retrieval(e5)일수록 passage수가 많아짐에 따라 성능 감소 폭이 커짐. 이는 강력한 retrieval가 hard negative를 더 많이 검색하게 됨을 의미함.

## The Interplay of Retrieval Quality and LLM Capabilites
<p align="center">
<img width="500" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.02%5DLongContext/figure4.png?raw=true">
</p>

**Research Question**  
- 성능의 병목 현상은 retrieval가 관련 정보를 식별하는 능력의 한계에서 비롯된 것인가? 아니면 long-context LLM이 retrieval된 문맥을 효과적으로 활용하는 능력의 한계에서 비롯된 것인가?

**Experimental Setup**  
- **Recall@$$k$$**: Top-$$k$$ Passage 안에 ground truth가 포함되어 있는 비율 (포함 여부)
- **Precision@$$k$$**: Top-$$k$$ Passage 중 실제로 ground truth인 문서의 비율 (개수)
- **Input**:  Question + Top-$$k$$ Passages
- **Output**: Answer

**Results**  
- 검색된 문서 수($$k$$)가 커질수록 Recall는 증가하고, Precision은 감소한다. 특히 <span style="color:green">**Recall**</span> 대비 <span style="color:blue">**QA Accuracy**</span>가 항상 더 낮음
- 특히 BM25대비 더 강력한 retriever인 e5에서는 더 높은 $$k$$에서 성능 하락 폭이 BM25대비 더 큼. 단, 성능 하락폭이 큰 것이지, 절대적인 성능은 e5가 더 좋음.
- 앞선 결과들은 검색된 문서에 정답 문서가 포함되어 있더라도  irrelevant passage(=hard negative)들로 인해 LLM이 mislead하게 됨을 의미함.

## The Importance of Hard Negative for Long-Context LLM Eval.
<p align="center">
<img width="600" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.02%5DLongContext/figure5.png?raw=true">
</p>

**Research Question**  
- 현재의 long-context LLM들은 이러한 Hard negative에 얼마나 강건한가?
- Hard negative의 영향은 사용된 retriever에 따라 달라지는가?

**Experimental Setup**  
- **Input**:  Question + Ground Truth Passage (Top-$$1$$에 위치) + Top-($$2~k$$) Passage
- **Output**: Answer
- Retriever로 검색된 문서들이 랜덤하게 뽑은 문서와 비교했을 때 Hard negative 일것이라고 가정함.

**Results**  
- 검색된 문서 수가 증가함에 따라 성능 감소가 급격하게 발생함.
- 반면 랜덤하게 선택한 문서를 추가할 땐 비교적 robust하며, QA Accuracy(=RAG Accuracy) 또한 더 높음.
- 이는 검색된 문서가 Hard negative임을 의미함.

<br/>
<br/>

# Methodology and Experiments Results
## Retrieval Reordering (Training-free)
<p align="center">
<img width="700" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.02%5DLongContext/figure6.png?raw=true">
</p>

**Retrieval Reordering**은 본 논문에서 지적하는 가장 핵심 적인 한계점인 **lost-in-the-middle 현상을 역이용**하는 방법이다. Retriever를 통해 $$k$$개의 문서를 뽑았을 때, 홀수 번째 문서들(1, 3, 5, ...)들은 앞쪽에 배치하고 짝수 번째 문서들은 역순으로(..., 8, 6, 4, 2) 그 뒤에 배치하는 것이다. LLM에 입력으로 들어가는 프롬프트의 Instruction을 $$I$$, $$i$$번째 문서를 $$d_i$$, 질문을 $$q$$라고 할 때 다음과 같은 순서로 프롬프트를 조정하는 것이다.

- Before: $$[I, d_1, d_2, d_3, \cdots, d_k, q]$$
- After : $$[I, d_1, d_3, d_5, \cdots, d_6, d_4, d_2, q]$$

실험 결과 reodering 방법이 검색된 문서수가 많아졌을 때 더 original order방식보다 항상 더 robust하다.

## Implicit Robustness Fine-tuning
<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.02%5DLongContext/figure7.png?raw=true">
</p>

Reordering은 hard negative의 영향을 줄이는 데 효과적이지만, LLM 자체의 강건성을 높이지는 못한다. 이를 보완하기 위해 논문에서는 RAG 전용 데이터로 LLM을 **Implicit Robustness Fine-tuning**하는 방법을 제안한다. 구체적으로는 LLM이 hard negative가 포함된 noisy retrieval 환경에서도 올바른 정답을 도출할 수 있도록 학습을 진행한다. 방식 자체는 일반적인 Instruction tuning과 유사하지만, 여러 RAG-QA 벤치마크(NQ, WoW, Fever, MMLU)와 Wikipedia Corpus를 혼합하여 학습 데이터로 사용했다는 점이 차별점이다.

- **Input:**  $$[I, d_1, d_2, d_3, \cdots, d_k, q]$$
- **Output:** $$a$$

이때 Direct FT는 검색 문서를 넣지 않고 $$[I,q]$$만을 입력으로 학습한 방식이고, RAG FT는 검색된 문서까지 포함해 학습하는 방식이다. 실험 결과 모든 데이터셋에서 RAG FT가 Direct FT보다 항상 높은 성능을 보였다. 또한 검색 문서 수가 늘어나 hard negative가 많아져도 RAG FT는 성능 저하가 완만하게 나타났으며, 성능 곡선도 상대적으로 평탄하게 유지되었다. 이는 RAG FT가 hard negative에 대해 더 강건한 특성을 가지도록 LLM을 학습시킨다는 것을 의미한다. 하지만, 사실 RAG FT가 다른 방법론보다 성능이 뛰어난 이유는, instruction tuning을 통해 기존에는 LLM에 내제되어 있지 않던 unseen document들을 학습을 통해 파라미터 내에 내제화했기 때문에 당연한 결과이다.

## Explicit Relevance Fine-tuning
<p align="center">
<img width="700" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.02%5DLongContext/figure8.png?raw=true">
</p>

**Explicit Relevance Fine-tuning**은 Implicit Robustness Fine-tuning이 단순히 노이즈 환경에서 강건성을 길러주는 것에 그치는 한계를 넘어, retrieved 문맥 내에서 relevant passage와 irrelevant passage를 명시적으로 구분하도록 학습하는 방법이다 이를 위해 학습 과정에 **중간 추론 단계(intermediate reasoning step)**를 추가한다. 즉, LLM이 최종 정답 $$a$$를 출력하기 전에 먼저 **reasoning paragraph** $$r$$를 생성하여 어떤 문서가 정답에 유효한 근거인지 식별하도록 하는 것이다. 

- **Input:**  $$[I, d_1, d_2, d_3, \cdots, d_k, q]$$
- **Output:** $$[r, a]$$

논문에서는 Gemini-1.5를 활용해 미리 생성된 reasoning paragraph를 ground truth로 사용하였다. 실험 결과 Direct FT < RAG FT < RAG FT w. Int의 일관된 성능 향상이 확인되었으며, 특히 RAG FT w. Int는 단순한 Instruction tuning이나 Implicit FT 대비 더 높은 성능을 보였다. 이는 LLM이 CoT 기반 reasoning을 통해 retrieval 문맥 내에서 중요한 문서를 식별하고, 그 근거를 토대로 정답을 도출하는 과정을 학습했기 때문이다. 따라서 이 방법은 lost-in-the-middle 문제와 hard negative 문제를 동시에 완화할 수 있다는 의의를 가진다.

## Analysis
<p align="center">
<img width="500" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.02%5DLongContext/figure9.png?raw=true">
</p>

**(a) Training Data Distribution 분석 결과**    
그림 (a)는 Gemma-2-9B-Base를 다양한 데이터 분포로 학습했을 때 HotpotQA에서의 성능을 비교한 것이다. 혼합 데이터(NQ, WoW, Fever, MMLU를 각각 12.5k 샘플씩 포함)로 학습한 모델이 단일 데이터(NQ only, WoW only, Fever only, MMLU only)로 학습한 모델보다 일관되게 더 높은 RAG 정확도를 달성했다. 이는 데이터 소스가 다양할수록 모델이 새로운 질의 유형과 retrieval 패턴에 잘 적응하며, 일반화 성능이 크게 향상됨을 보여준다. 즉, **데이터 다양성이 LLM의 적응력과 일반화 성능 향상에 핵심**적임을 입증한 결과이다.

**(b) Retriever Variation 분석 결과**  
그림 (b)는 동일한 NQ 데이터셋을 사용하되 retriever 종류를 달리하여 학습했을 때, 다양한 retriever 환경에서의 성능을 비교한 것이다. FT w. mix(BM25 + e5 혼합)로 학습한 모델이 모든 retriever 환경(BM25, e5뿐만 아니라 Contriever, BGE와 같은 unseen retriever 포함)에서 가장 높은 성능을 보였다. 또한 특정 retriever로 학습한 모델은 유사한 특성을 가진 retriever 환경에서 상대적으로 더 좋은 성능을 보였다. 예를 들어, BM25로 학습한 모델은 Contriever 환경에서, e5로 학습한 모델은 BGE 환경에서 강점을 보였다. 전반적으로 fine-tuning을 수행한 모든 모델은 No FT 대비 성능이 크게 향상되었으며, 이는 retriever 별 hard negative의 특성이 다르기 때문에 학습 시 **다양한 retriever 데이터를 혼합하는 것이 새로운 retrieval 환경에서도 더 높은 강건성과 일반화 성능을 제공**함을 시사한다.

<br/>
<br/>

# Conclusion
**Contribution**
- LLM에 RAG 기법을 적용했을 때, 어떤 한계가 존재하고 어떤 학습 방식이 각각을 해결할 수 있는지 구체적으로 보여줌.
- RAG + LLM 기반의 framework에서 병목이 ‘retriever 한계’가 아니라 ‘LLM의 활용 한계’에서 기인함을 분리해 보였습니다.
- LLM을 fine-tuning할 때는 혼합 데이터로 학습할 때가 단일 데이터로 학습하는 것보다 일반화가 더 잘됨을 실험적으로 보여줌.
