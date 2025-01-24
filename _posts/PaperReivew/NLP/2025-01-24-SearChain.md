![image](https://github.com/user-attachments/assets/60af0aa5-7abb-4a59-97b1-928517e60886)---
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

## S-1. Chain-of-Query (CoQ)
<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/cd6b01ea-012a-4bae-850c-1d87089c744e">

Chain-of-Query는 하나의 복잡한 질문을 여러 개의 서브 질문과 정답 쌍으로 분리해 하나의 추론 체인을 형성하는 과정이다. 예를 들어, "Where do Greyhound buses that are in the birthplace of Spirit If...'s performer leave from?"라는 질문은 아래와 같이 세 개의 질문-정답 쌍으로 분리될 수 있다.

- 질문1 - "Who is the performer of Spirit If...?" / 답변 - "Kevin Drew“
- 질문2 - "Where was Kevin Drew born?" / 답변 - "Toronto“
- 질문3 - "Where do Greyhound buses in Toronto leave from?" / 답변 - "Toronto Coach Terminal"

이 경우 그림에서 각 노드는 답변인 "Kevin Drew", "Toronto", "Toronto Coach Terminal"에 대응된다.

## S-2. Verification & Completion

<span style="font-size:110%">**Verification (검증)**</span>  
<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/ed2a97a1-a83d-49b7-8529-0d30336066d9">

검증의 목적은 "잘못된 정보를 수정"하는 것이다. LLM이 생성한 답변이 정확한지 확인하고, IR에서 제공한 정보와 불일치할 경우 이를 수정하기 위한 피드백을 제공한다. 먼저 Retriever와 Reader는 문서를 찾고, 정답과 함께 **신뢰 점수(confidence score)**를 출력한다. 만약 LLM이 생성한 답변과 IR이 찾은 답변이 다르고, 그와 동시에 IR의 신뢰 점수가 임계값을 넘어가면 feedback 함수를 통해 아래 프롬프트를 생성해 LLM에게 입력시킨다.
LLM은 이를 통해 답변을 수정한다.

<br/>

<span style="font-size:110%">**Completion (보완)**</span>  
<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/0ffce3ed-8941-436e-ba97-d001a49e0334">

보완의 목적은 "부족한 정보를 검색 및 추가"하는 것이다. LLM이 답변을 생성하지 못한 경우, IR이 부족한 정보를 제공하여 LLM이 추론을 이어가도록 돕는다. 보완 과정을 크게 세 단계로 이루어진다.
1. **미해결 질문 탐지**: LLM이 특정 질문(Query)에 대해 답변을 생성하지 못할 경우, 해당 노드를 "Unsolved Query"로 표시한다.
2. **문서 검색 및 답변 추출**: LLM이 특정 질문(Query)에 대해 답변을 생성하지 못할 경우, 해당 노드를 "Unsolved Query"로 표시한다.
3. **피드백 생성**: LM이 새로운 답변을 생성하도록 다음과 같은 프롬프트를 생성한다.

## S-3. Tree-of-Reasoning (ToR)
<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/42a9e75e-6447-46b7-abab-5d3f7ad3453f">

첫 번째 단계인 CoQ는 복잡한 질문을 연속적인 서브 질문-정답 쌍으로 세분화하여, 추론 사슬의 구조를 **깊이 우선 탐색(DFS)** 기반 경로와 유사하게 구성한다. 이후 검증과 보완 작업을 통해 최종적으로 LLM의 추론 과정은 **트리 구조**의 형태를 띤다. 서브 질문에 대한 답변(sub-answer)이며, 각각의 브랜치(branch)는 잘못된 노드가 수정되거나 새로운 정보가 추가될 때 생성된다. 

기존의 CoT기반의 연구들은 선형 방식인 것과는 다르게, ToR은 <span style="color:red">**새로운 경로를 동적으로 추가하거나 잘못된 경로를 수정**</span>할 수 있다. ToR의 장점은 세 가지로 정리할 수 있다.
- **동적 추론 방향 수정 가능**: 잘못된 정보가 발견되거나 추가 정보가 필요할 때, 새로운 브랜치를 생성하여 동적으로 추론 방향을 조정할 수 있다.
- **효율적인 검증 및 보완**: 각 노드에서 IR과 상호작용하며, 불필요한 전체 경로 탐색을 피하고 필요한 부분만 수정하거나 보완한다.
- **신뢰성 있는 최종 정답 도출**: 가장 신뢰도가 높은 경로를 선택하여 최종 정답을 생성함으로써, 높은 정확성과 신뢰성을 제공한다.

<br/>
<br/>

# Experiments and Results



<br/>
<br/>

# Contribution


