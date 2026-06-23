---
title: "[논문리뷰]LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG (arXiv, 2026)"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2026-06-21
last_modified_at: 2026-06-21
---

*Yijia Zheng and Marcel Worring*. 2026. **LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG**. arXiv:2605.06285 [cs.CL]

# 1. Problem Statement
이 논문은 iterative reasoning과 retrieval의 장점은 유지하면서 명시적 thought/subquery 생성 비용을 제거하는 것이다. 이를 위해 LatentRAG는 reasoning과 subquery 생성을 LLM의 continuous hidden state인 latent token으로 수행하고, latent subquery를 dense retriever와 정렬하여 자연어 query 없이 직접 검색한다. 동시에 제한된 QA trajectory만으로 latent retrieval을 학습하고, 필요한 경우 latent token을 자연어로 복원하여 효율성과 투명성 사이의 선택을 가능하게 한다.

<br/>
<br/>

# 2. Limitations of Existing Works
기존 single-step RAG는 한 번의 검색만 수행하므로 복합 질문을 여러 단계로 분해해 필요한 정보를 순차적으로 수집하기 어렵다. Agentic RAG는 LLM이 매 단계에서 thought와 subquery를 생성하고 검색 결과를 반영하는 방식으로 이 문제를 완화하지만, 긴 중간 출력을 token-by-token으로 생성해야 하므로 naive RAG보다 약 16–22배 높은 지연을 유발한다.

- **[Single-step RAG의 복합 질문 처리 한계]** Traditional RAG는 원 질문으로 한 번만 검색한 뒤 답을 생성하는 구조를 사용한다. 따라서 질문이 여러 entity나 relation을 순차적으로 연결해야 하는 경우, 첫 검색 결과에 없는 후속 정보를 추가로 요청하거나 이전 검색 결과에 근거해 query를 수정하기 어렵다. 이 구조적 한계로 인해 HotpotQA나 2wiki와 같은 multi-hop QA에서 agentic RAG보다 낮은 성능을 보인다.
- **[명시적 thought 및 subquery 생성의 순차적 지연]** 기존 agentic RAG는 각 검색 단계에서 긴 natural-language thought와 subquery를 autoregressive decoding으로 생성한다. 각 출력 토큰이 이전 토큰에 의존하므로 여러 번의 순차적인 LLM forward pass가 필요하고 GPU 병렬화만으로 지연을 제거하기 어렵다. 논문의 측정에서는 Search-R1의 thought와 subquery 생성이 전체 지연의 약 90%를 차지하며, 전체 추론 시간은 naive RAG의 16–22배에 이른다.
- **[Discrete subquery interface]** 기존 방법은 LLM이 생성한 문자열 subquery를 외부 검색 시스템에 전달하는 구조를 사용한다. Text generation과 document retrieval 사이에 discrete sampling이 존재하므로 retrieval 결과에서 발생한 signal을 LLM의 subquery representation까지 직접 역전파하기 어렵다. 결과적으로 LLM과 retriever가 별도 구성요소로 학습되며, 최종 QA 목적에 맞게 두 구성요소의 representation을 공동 최적화하기 어렵다.

# 3. Methodology
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.21%5DLatentRAG/figure1.png?raw=true">
</p>
Figure 2의 핵심은 explicit agentic RAG의 반복 구조는 유지하되, 각 단계에서 생성하던 자연어 thought와 subquery를 고정된 수의 special token hidden state로 대체하는 것이다. Figure 2(a)의 기존 방식은 `Generation → Retrieval → Generation`을 반복하면서 매번 자연어 thought와 subquery를 순차 생성한다. Figure 2(b)의 LatentRAG는 같은 검색 순서를 따르지만 thought와 subquery 위치에 latent token만 배치하며, latent subquery hidden state로 검색을 수행한다.

Figure 2(c)의 전체 architecture는 세 구성요소로 나뉜다. **Generation**은 question, 이전 retrieved documents, latent thought/subquery token을 입력받아 다음 action과 latent representation을 생성한다. **Retrieval**은 latent subquery token을 projector와 dense retrieval model에 통과시켜 subquery embedding을 만들고 top-$k$ document를 반환한다. **Latent Decoding**은 학습 시 latent representation에 semantic supervision을 제공하고, 추론 시 선택적으로 이를 자연어 thought와 subquery로 복원한다.


<br/>
<br/>

# 4. Experiments


<br/>
<br/>

# 5. Conclusion
