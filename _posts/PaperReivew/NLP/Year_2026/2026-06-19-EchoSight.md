---
title: "[논문리뷰]EchoSight-Advancing Visual-Language Models with Wiki Knowledge (EMNLP findings, 2024)"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2026-06-19
last_modified_at: 2026-06-19
---

*Yibin Yan and Weidi Xie*. 2024. [**EchoSight: Advancing Visual-Language Models with Wiki Knowledge**](https://aclanthology.org/2024.findings-emnlp.83/). In Findings of the Association for Computational Linguistics: EMNLP 2024, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Association for Computational Linguistics, Miami, Florida, USA, 1538–1551.

# 1. Problem Statements
이 논문은 **Knowledge-based Visual Question Answering(KVQA)**, 특히 이미지 속 세부 엔티티에 대해 Wikipedia 수준의 백과사전 지식을 검색하고 이를 이용해 답변을 생성하는 **multimodal RAG 기반 VQA task**를 해결한다. 표준 VQA는 이미지 안의 색, 개수, 행동처럼 시각 정보만으로 답할 수 있는 반면, 이 논문이 다루는 KVQA는 “이 산의 첫 등정 시기는 언제인가?”, “이 건물의 설계자는 누구인가?”처럼 이미지에 직접 드러나지 않는 fine-grained entity knowledge가 필요하다. 

**EchoSight**는 reference image와 textual question을 입력으로 받고, Wikipedia image-article knowledge base에서 관련 article/section을 검색·재정렬한 뒤 LLM이 최종 답변을 생성하도록 설계된 retrieval-augmented vision-language system이다.

<br/>
<br/>

# 2. Limitations of Existing Works
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.18%5DEchoSight/figure1.png?raw=true">
</p>

- **[VLM 내부 지식 인코딩의 한계]** 기존 대규모 generative VLM은 knowledge-based VQA에서 필요한 모든 백과사전 지식을 모델 파라미터 안에 충분히 저장하기 어렵다. 저자들은 그 원인을 제한된 모델 용량과 encyclopedic, long-tail information이 학습 데이터에 드물게 포함되는 문제로 설명한다. 따라서 특정 식물의 지역별 용도, 특정 건물의 설계자, 특정 산의 역사적 사건처럼 세밀한 entity-level fact가 필요한 질문에서는 VLM이 환각을 일으키거나 불완전한 답을 생성하기 쉽다.
- **[시각 단서와 엔티티 지식 연결의 한계]** knowledge-based VQA에서는 이미지 자체가 질문 해결에 충분한 단서를 제공하지 않는 경우가 많다. 예를 들어 교회 사진만으로는 그 건축 연도나 설계자 같은 속성을 알기 어렵고, 산이나 식물 사진만으로는 Wikipedia article 내의 정답 section을 직접 특정하기 어렵다. 즉 기존 접근은 visual attribute와 entity knowledge 사이의 의미적 연결을 안정적으로 형성하지 못하며, 단순 visual similarity retrieval만으로는 visually similar하지만 contextually wrong한 article을 상위에 둘 위험이 크다.

<br/>
<br/>

# 3. Methodology
## 3.1. Overview
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.18%5DEchoSight/figure2.png?raw=true">
</p>

EchoSight의 핵심 방법론은 **visual-only coarse retrieval → multimodal fine-grained reranking → LLM answer generation**으로 이어지는 3단계 multimodal RAG pipeline이다. Figure 2는 전체 구조를 보여주며, 

- **입력:** reference image $$I_{ref}$$, question $$Q$$, External KB
  - 외부 지식에 해당하는 Knowledge Base (KB)는 Wikipedia artical과 image pair로 구성됨.
  - KB는 $$B = \{ (a_1, I_1),\ldots, (a_n, I_n) \}$$으로 정의하면 $$a_i$$는 Wikipedia article, $$I_i$$는 각 article에 연결된 image임

### 3.1. Retriever
### 3.1.1. Corse-grained Retrieval
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.18%5DEchoSight/figure3.png?raw=true">
</p>

**1) Visual-only Search**는 reference image $$I_{ref}$$만을 사용해 <span style="color:red">**유사한 Wikipedia image를 먼저 찾는 coarse retrieval**</span> 단계이다. 먼저 사전 학습된 vision encoder $$\Phi_{vis}$$를 이용해 External KB에 존재하는 모든 이미지를 임베딩하여 vector DB를 구축한다. 

사용자의 question ($$q$$)과 reference image ($$I_{ref}$$)가 시스템에 입력이 되면 먼저 $$I_{ref}$$를 동일한 vision encoder로 임베딩하고 유사도 기반으로 top-$$k$$개의 image를 검색한다. 이를 통해 최종적으로 top-$$k$$개의 image-article 후보 집합 $$E_v = \{ (a_1, I_1),\ldots, (a_k, I_k) \}$$를 얻는다. 구현에서는 frozen `Eva-CLIP-8B` vision encoder의 마지막 레이어의 mean pooling된 임베딩과 `FAISS`를 사용해 대규모 image search를 수행한다.

### 3.1.2. Fine-grained Multimodal Reranking
Visual-only search를 통해 $$k$$개의 <span style="color:red">**image-article 후보 집합에 대해 질문과 실제로 관련이 있는지 image-question 기준으로 재정렬**</span>하는 fine-grained reranking을 진행하고, 이를 **Multimodal Reranking**이라고 정의한다. 먼저 Q-Former를 이용해 reference image와 textual question을 함께 받아 32개의 멀티모달 쿼리 토큰을 생성한다. 이를 수식으로 표현하면 다음과 같다. 

<center>$$z_m^i = \text{Q-Former} (I_{ref}, Q)$$</center>

여기서 $$z^i_m$$이 쿼리의 $$i$$번째 토큰 임베딩이다. 그리고 각 후보 article은 섹션 단위로 나누고 article title을 prefix로 붙여 $$a_i = \{sec_1^i, sec_2^i, \ldots, sec_p^i\}$$ 처럼 구성한 뒤 text encoer로 섹션 임베딩을 만든다. 섹션으로 나눈다는 것은, 쉽게 말해 하나의 long-context article을 여러 개의 passage로 나누는 것을 의미한다.

Reranking score는 멀티모달 쿼리 토큰과 Wikipedia 섹션의 [CLS] 토큰 임베딩 간 최대 pairwise 유사도로 정의된다.

<center>$$S_r^{sec} = max_{1\leq i \leq N_q} \Big ( \cos (z_m^i, z_s^{sec}) \Big ) $$</center>

이 방식은 질문에 중요한 특정 query token이 section text와 강하게 매칭되는 경우를 포착하는 late-interaction 성격의 scoring이다. 

최종적으로  Reranker는 vision-only search 단계에서는 각 image-article pair별 스코어와 멀티모달 reranking score를 결합해 스코어 $$sec_{vl}$$

<center>$$sec_{vl} = \arg\max_{sec \in a} \Big( \alpha \cdot S_v^{sec} + (1-\alpha)\cdot S_r^{sec} \Big)$$</center>

여기서 $$\alpha$$는 visual simiarlity와 reranking score 사이의 balanced parameter이다.

Reranker는 hard negative sampling을 활용한 대조 학습 방식으로 학습을 진행한다. 입력은 positive evidence 섹션과 negative 섹션들이며, negative는 무작위 article이 아니라 visual-only retrieval에서 image는 비슷하지만 정답 article이 아닌 후보들로 구성된다. 즉 모델은 **visually similar but contextually distinct**한 hard negatives 사이에서 질문과 실제로 관련 있는 section을 고르도록 학습된다. 목적함수는 in-batch기반 **InfoNCE**로 학습한다.

<br/>
<br/>

# 4. Experiments
## 4.1. Retrieval Results
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.18%5DEchoSight/figure4.png?raw=true">
</p>

Table 1은 **E-VQA에서 visual-only retrieval과 multimodal reranking의 차이를 비교한 retrieval 실험**이다. EchoSight는 reranking 없이 Recall@1 13.3을 기록하지만, reranking을 적용하면 Recall@1이 36.5로 크게 상승한다. 이는 visual-only search가 이미지가 유사한 Wikipedia article 후보를 찾는 데에는 효과적이지만, 질문과 실제로 관련 있는 article을 최상위에 배치하는 데에는 한계가 있음을 보여준다. 다만 Google Lens는 Recall@1 47.4로 더 높지만, 이는 인터넷 규모의 closed-source image index를 사용하는 강한 upperbound 성격의 비교 대상이라는 점에서 EchoSight와 동일 조건의 retrieval system은 아니다.

Table 2는 **InfoSeek에서 EchoSight의 retrieval 성능을 평가한 결과**이며, E-VQA에서 학습된 reranker가 다른 benchmark에서도 효과적으로 동작함을 보여준다. EchoSight는 reranking 없이도 Recall@1 45.6을 기록해 CLIP I-T의 Recall@1 32.0보다 높고, reranking을 적용하면 Recall@1이 53.2까지 상승한다. 특히 InfoSeek에서는 reranker를 별도로 학습하지 않고 zero-shot으로 사용했기 때문에, 제안한 multimodal reranking이 특정 데이터셋에만 맞춰진 것이 아니라 image-question 기반 **evidence ranking에 일반적으로 도움이 된다**는 점을 시사한다.

## 4.2. VQA Results
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.18%5DEchoSight/figure5.png?raw=true">
</p>

Table 3은 **최종 VQA accuracy를 기존 방법들과 비교한 main result**이다. EchoSight w. Reranking은 E-VQA에서 41.8, InfoSeek에서 31.3을 달성하며, Wiki-LLaVA와 DPR∗V+T 같은 기존 retrieval-augmented VQA 방법보다 높은 성능을 보인다. 특히 E-VQA에서 EchoSight w/o. Reranking은 19.4에 머무르지만, reranking을 적용하면 41.8까지 상승한다. 이는 최종 답변 성능의 향상이 단순히 LLM을 사용했기 때문이 아니라, 정답과 관련된 Wikipedia section을 정확히 상위에 올리는 retrieval-and-reranking 구조에서 비롯된다는 점을 보여준다.

## 4.2. Qualitative Results
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.18%5DEchoSight/figure6.png?raw=true">
</p>

Figure 3은 **EchoSight가 GPT-4V보다 fine-grained encyclopedic question에서 더 정확한 답변을 생성하는 사례와 실패 사례를 함께 보여준다**. 성공 사례에서 EchoSight는 건물이 workshops를 개최한다는 정보, museum designer가 J. Langer라는 정보, Osorno Volcano가 Chile에 있다는 정보, 특정 식물이 Bulgaria에서 wound healing에 사용되었다는 정보처럼 이미지에 직접 드러나지 않는 지식을 맞힌다. 이는 retrieved Wikipedia section을 context로 사용하는 방식이 내부 지식에만 의존하는 VLM보다 세부 factual question에 더 적합함을 보여준다. 반면 도시, breeding pairs 수, plant common name을 틀린 실패 사례도 제시되며, 이는 잘못된 section이 검색되거나 retrieved context에 정답 근거가 없을 경우 EchoSight도 오류를 낼 수 있음을 보여준다.

<br/>
<br/>

# 5. Concolusion
**Contribution**
- **[multimodal RAG 기반 KVQA framework 제안]** EchoSight는 fine-grained encyclopedic knowledge가 필요한 knowledge-based VQA를 해결하기 위한 retrieval-augmented vision-language system이다. 이미지와 질문만으로 답하기 어려운 경우, Wikipedia 기반 KB에서 관련 지식을 검색해 LLM의 답변 생성에 활용한다.
- **[Visual-only retrieval과 multimodal reranking 결합]** EchoSight는 먼저 reference image만으로 visually similar한 Wikipedia article 후보를 찾고, 이후 image-question query와 section text를 함께 고려해 후보를 reranking한다. 이를 통해 단순 시각 유사도만으로는 구분하기 어려운 visually similar but contextually wrong한 article을 효과적으로 걸러낸다.

**Limitations**
- **[Knowledge base 품질과 coverage 의존성]** EchoSight는 외부 KB에서 정답 근거를 검색하는 구조이므로, KB에 필요한 domain-specific knowledge가 없거나 article-image 매핑이 불완전하면 성능이 저하될 수 있다. 즉 모델의 성능은 underlying knowledge base의 품질과 포괄성에 크게 의존한다.
- **[Multimodal reranking의 계산 비용]** EchoSight는 visual-only retrieval 이후 candidate article의 section들을 다시 multimodal reranking하기 때문에 추가 계산 비용이 발생한다. 특히 reranking scope가 커질수록 성능은 좋아지지만 retrieval time도 증가하므로, real-time application에는 한계가 있다.
