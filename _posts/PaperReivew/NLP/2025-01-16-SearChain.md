---
title: "[논문리뷰]Search-in-the-Chain: Interactively Enhancing Large Language Models with Search for Knowledge-intensive Tasks"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2025-01-16
last_modified_at: 2025-01-16
---

*Shicheng Xu, Liang Pang, Huawei Shen, Xueqi Cheng, and Tat-Seng Chua*. 2024. **[Search-in-the-Chain: Interactively Enhancing Large Language Models with Search for Knowledge-intensive Tasks](https://arxiv.org/abs/2304.14732)**. arXiv:2304.14732 

# Problem Statement
<span style="font-size:110%">Multi-hop Question Answering(MHQA)</span>
<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/4c0e9e29-b810-4c98-827b-0c4a61b87369">
</p>

**다중홉 추론(Multi-hop Question Answering, MHQA)**은 두개 이상의 문서(document) 혹은 패세지(passage)에 존재하는 정보를 종합해 주어진 질문에 정답을 추론할 수 있는 질의 응답 문제이다. 여러 개의 문서를 찾기 위해서는 **RAG(Retrieval-Augmented Generation)**를 사용해야 한다. 최근 많은 연구들은 RAG와 LLM을 결합하여, 질문에 가까운 문서를 검색해서 찾아 LLM에게 추론을 시키는 framework를 많이 사용한다. 

<span style="font-size:110%">Limitations of Existing Research</span>
<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/4ee7f85c-f4f2-4890-a9fd-5c7744dde45b">
</p>

하지만 기존 연구들에서는 LLM으로 검색된 문서를 추론하는데 여러 가지 문제가 존재한다. 논문에서는 크게 세 가지로 문제를 유형화한다.

**추론 체인의 단절(Breaking the reasoning chain)**  
- 기존 연구에서는 정보 검색을 LLM의 추론 과정에 직접 삽입하는 방식을 사용한다.
- 이 때 Self-Ask방식을 사용하는데, 기존 연구들 대부분은 검색된 문서와 질문을 LLM에게 입력시키고,
- LLM에게 여러 개의 서브 질문을 생성하게 하고 sub-QA pasage를 생성한다.
- 다시 말해, 검색된 문서로 Chain of Thought를 진행하는 방식을 의미한다.
- 만약에 LLM이 이미 정답을 알고 있더라도, 잘못 검색하는 경우가 발생한다.
- 이는 LLM은 각 step에서 하나의 sub-question만 해결하며, 전체적인 reasoning chain을 고려하지 못하게 됨을 의미한다.

**Retriever가 잘못된 정보를 제공하는 경우 LLM의 혼란 초래**
- LLM이 이미 정확히 기억하고 있는 지식이라도 IR(Information Retriever)이 검색한 정보가 잘못되었을 경우, LLM이 이를 신뢰하여 잘못된 응답을 생성하게 될 위험이 있다.

**추론 방향의 유연성 부족**
- LLM이 한 방향으로만 추론을 수행하도록 제한하여, 추론 과정 중 새로운 정보가 필요하거나 잘못된 방향으로 진행된 경우 이를 동적으로 수정하거나 새로운 방향으로 전환하는 기능이 부족하다.

<br/>
<br/>

# Method



<br/>
<br/>

# Experiments and Results



<br/>
<br/>

# Contribution


