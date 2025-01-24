---
title: "[논문리뷰]Search-in-the-Chain: Interactively Enhancing Large Language Models with Search for Knowledge-intensive Tasks"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2025-01-24
last_modified_at: 2025-01-24
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

SearChain은 이러한 세 가지 문제점을 해결하고자 하며, 

<br/>
<br/>

# Method
## Model Architecture
<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/30f4ecf2-9410-46ed-9325-2aff0a0e6acc">
</p>

SearChain은 크게 세 가지 파트로 나눠진다.
1. **Chain-of-Query (CoQ)**
2. **Verification& Completion**
3. **Tree-of-Reasoning (ToR)**

**CoQ**는 복잡한 질문을 <span style="color:red">**여러 개의 서브 질문과 답변으로 연결괸 글로벌 추론 체인으로 변환**</span>하는 것이다. 위의 그림에서  Round 1 (CoQ Gen.) 부분은 LLM이 질문을 분해해 초기 CoQ를 생성하는 과정이다. 이 단계에서 각 노드는 질문-답변 쌍으로 구성되며, 각 서브 질문의 정답이 노드가 된다. (A⮕B⮕C⮕D)

CoQ를 통해 여러 개의 서브 질문-정답 쌍으로 구성된 글로벌 체인이 성공적으로 구성되었으면, LLM은 순차적으로 서브 질문에 대해 정답을 추론한다. 이 과정에서 **Verification(검증)**과 **Completion(보완)**을 통해 LLM이 이미 알고 있는 지식과 부족한 지식을 외부로부터 검색해 적절하게 추론에 사용하게 된다. Verification은 <span style="color:red">**IR이 CoQ에서 각 서브 질문에 대한 LLM의 정답을 검증하고, 만약 잘못되었다 판단될 경우 수정**</span>하는 과정이다. Completion은 <span style="color:red">**LLM이 정답을 생성하지 못한 경우를 "Unsolved Query"로 표시하고, 이 표시된 노드들에 대해 IR을 통해 외부에서 추가적인 정보를 검색해 제공하는 것**</span>이다. 그림에서 Round 1(Verif.)와 Round 3(Verif. and Comp.)에서 IR이 노드(B, F 등)를 검증하고 필요한 경우 정보를 보완하여 새로운 CoQ를 생성한다.

이렇게 CoQ를 생성하고, 검증과 보완과정을 반복하다보면, 기존의 직선적인 추론 체인과 달리 Tree 형태의 추론 체인이 형성된다. 이를 **Tree-of-Reasoning(ToR)**이라고 한다. ToR은 <span style="color:red">**추론 방향을 동적으로 수정**</span>할 수 있음을 보여주며, 잘못된 노드가 발견되면 분기를 생성하고, 이를 포함한 전체 추론 트리를 관리하게 된다. 그림에서 Round 4 (CoQ Gen.)와 Tree-of-Reasoning (Trace.)에서 노드 수정 (J 추가) 및 새로운 분기 생성 과정을 통해 올바른 최종 답변에 도달하는 것을 볼 수 있다. 


<br/>
<br/>

# Experiments and Results



<br/>
<br/>

# Contribution


