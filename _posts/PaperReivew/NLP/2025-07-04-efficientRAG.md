---
title: "[논문리뷰]EfficientRAG: Efficient Retriever for Multi-Hop Question Answering"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2025-07-04
last_modified_at: 2025-07-04
---

*Ziyuan Zhuang, Zhiyang Zhang, Sitao Cheng, Fangkai Yang, Jia Liu, Shujian Huang, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, and Qi Zhang*. “[**EfficientRAG: Effective Retriever for Multi-Hop Question Answering**](https://aclanthology.org/2024.emnlp-main.199/).” In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 3392–3411, Miami, Florida, USA, November 2024

# Problem Statement
이 논문에서는 기존의 Standard RAG와 Single-step Retrieval의 한계점을 네 가지로 분류한다.

**[Single-step retrieval의 비효율성]** 기존 RAG는 주로 1회성 단순 retrieval에 의존하여 multi-hop reasoning이 필요한 복잡 질문에 비효율적이다. 전통적인 one-round RAG 방법들은 단순한 질문에는 효과적이지만, 여러 단계의 추론이 필요한 multi-hop 질문에서는 첫 번째 검색으로 얻은 정보만으로는 답변이 불가능하다.

**[Multi-step retrieval의 비용 문제]** 매 retrieval 라운드마다 LLM 호출이 필요하여 지연시간 (latency)과 비용 (cost)이 증가한다. 또한 복잡한 프롬프트 설계 및 few-shot 예시가 필요하여 도메인 확장성이 저하된다.

**[Noisy chunks로 인한 정확도 저하]** Multi-hop QA에서는 필요없는 정보 (noisy chunks)로 인해 정확도가 저하된다. 무관한 청크들의 존재가 LLM 생성기에게 지속적인 도전을 제기한다.

**[LLM의 noisy context 취약성]** LLM도 noisy context에 취약해, 중간 reasoning 단계에서 오류가 발생할 가능성이 있다. LLM 기반 시스템들이 노이즈가 있는 지식 입력으로 부분 답변을 생성해야 하는 중간 단계에서 실패하는 경우가 많다.

<br/>
<br/>

# Methodology



<br/>
<br/>

# Experiments



<br/>
<br/>

# Conclusion

