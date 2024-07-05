---
title: "[논문리뷰]ReasoningLM: Enabling Structural Subgraph Reasoning in Pre-trained Language Models for Question Answering over Knowledge Graph"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2024-07-04
last_modified_at: 2024-07-04
---

*Jiang, J., Zhou, K., Zhao, W. X., Li, Y., & Wen, J.* (2023, December 30). **ReasoningLM: Enabling Structural Subgraph Reasoning in Pre-trained Language Models for Question Answering over Knowledge Graph**. arXiv.org. [https://arxiv.org/abs/2401.00158](https://arxiv.org/abs/2401.00158)

# Problem Statement
<span style="font-size:110%">1. 구조적 상호작용의 부족 (Lack of Structural Interaction)</span>  
- 모델 아키텍처의 차이로 인해 PLM과 GNN이 느슨하게 통합되는 경우가 많다(예: 관련성 점수 공유).
- 이는 질문(query)과 KG(관련 엔티티로 확장된 서브그래프) 사이의 지식 공유와 세밀한 상호작용을 크게 제한한다.

<span style="font-size:110%">2. 부족한 의미적 지식 (Lack of Semantic Knowledge)</span>  
- GNN 기반 추론 모듈은 주로 서브그래프 구조에 기반하여 추론을 수행한다.
- 이는 PLM에 포함된 풍부한 의미적 지식(text information)을 부족하게 하여, 특히 복잡한 질문에 대한 추론 결과가 덜 효과적일 가능성이 있다.

<span style="font-size:110%">3. 복잡한 구현 과정 (Complex Implementation Process)</span>  
- PLM: text 정보를 학습하여 **query에 대한 understanding이 가능**하지만 KG의 복잡한 구조 정보를 전혀 학습하지 못한다.
- GNN: KG의 구조 정보는 학습하여 **multi-hop reasoning**이 가능하지만, text정보를 활용하지 못해 복잡한 질문에 대한 정답을 추론하지 못한다.
-  모듈의 통합은 실제 구현에서 복잡한 과정을 요구한다.

<br/>
<br/>

# Related Work
<span style="font-size:110%">1. Question Answering(QA)</span>  
- Multi-hop KGQA는 topic entity에서 multi-hop만큼 떨어져있는 정답 엔티티를 찾는 것을 목표로 한다.

<span style="font-size:110%">2. PLM for KG Reasoning</span>  
- PLM을 통한 KGQA는 상식 추론이나 유실된 사실 추론(predicting missing fact)를 하는 것이다.

<br/>
<br/>

# Method
## Model Overview

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/0fe3bb2e-8294-4a0a-99ea-f27fbc15f0f5">
</p>

ReasningLM은 <span style="color:gold">**Question과 Subgraph의 직렬화된 자연어 토큰 시퀀스**</span>를 입력으로 받는다. 트랜스포머 모듈이 각 토큰에대한 임베딩 시퀀스를 출력하고 최종적으로 정답을 찾기위해 subgraph의 hidden representation들만 linear layer를 통과시켜 score를 계산하게된다. 본 논문에서는 트랜스포머 모듈(backbone)로 RoBERTa-base를 사용하였다.

ReasoningLM의 핵심 요소는 두 가지이다.
- Adaptation Tuning Strategy
- Subgraph-Aware Self-Attention

## 1. Adaptation Tuning Strategy
Adaptation Tuning Strategy은 질문과 서브그래프를 추출하기 위한 전략이다. 학습을 위해서 총 2만 개의 synthesized question을 뽑아낸다. 이 때, 서브그래프는 Large-scale KG에 해당하는 Wikidata5M에서 추출한다.

### 1) Subgraph Extraction
추출된 subgraph가 PLM에 제대로 적용되기 위해서는 subgraph들이 대중적으로 사용되는 지식(commonly used knowledge)를 잘 내포하고 있어야 한다. 따라서 인기있는 엔티티(popular entity)를 Wikidata5M에서 추출해 시드 토픽 엔티티(seed topic entity)로 사용하고, KQA Pro[(Cao et al., 2022)](https://aclanthology.org/2022.acl-long.422/)와 같은 방식으로 정답 엔티티와 서브그래프를 추출하게 된다.

먼저 Wikidata5M에서 인기있는 2000개의 토픽 엔티티를 추출한다. 각 토픽 엔티티들을 출발점으로 하여 <span style="color:gold">**randomwalk를 수행하여 Reasoning path를 추출**</span>한다. Reaoning path의 길이는 4-hop을 넘지 않으며 **종점은 반드시 정답(answer) 엔티티**가 되게 만든다. 각 Reaoning path들은 결론적으로 시작점이 토픽 엔티티이고, 종점이 정답 엔티티가 되게된다. 

Reasoning path가 정해지면 이제 앞서말한 KQA Pro논문의 아이디어를 활용하여 subgraph를 추출할 수 있다. Reasoning path의 시작점인 토픽 엔티티를 기준으로 $$k$$-hop 내의 엔티티와 릴레이션을 임의로 추출한다. 그리고 실제로 존재하는 트리플들만 중복은 제거하고 추출하여 하나의 서브그래프를 만든다. 이 때, 서브그래프에는 반드시 reasoning path가 포함이되어야 한다.

<br/>

### 2) Question Synthesis
Reasoning path는 토픽 엔티티와 정답 엔티티를 포함한다. 본 논문에서는 이 reaoning path를 이용해서 자동으로 질문을 만들어내는 방법을 제안한다. 먼저, <span style="color:gold">**질문 생성을 위해 ChatGPT를 사용**</span>하였다. 질문 생성 방식에는 크게 두 가지로 나눠진다.

- 규칙 기반 생성
  - 여러 **일반적인 템플릿**을 수작업으로 작성한다. 이를 토대로 토픽 엔티티와 릴레이션을 질문으로 변환한다.
  - Ex) "What is the <span style="color:lime">\[relation\]</span> of <span style="color:coral">\[entity\]</span>?" ➔ "What is the <span style="color:lime">**capital**</span> of <span style="color:coral">**France**</span>" 

- LLM 기반 질문 생성
  - ChatGPT와 같은 대형 언어 모델을 사용하여 형식과 유창한 표현을 가진 질문을 생성할 수 있다.
  - 총 20,000개의 질문을 생성함
 
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/e968ff6a-ab5d-4d84-b98d-fd53fcc7e077">
</p>

## 2. Subgraph-Aware Self-Attention
### 1) Serialization of Input Sequence

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/aee2e6a1-f9c9-4ed0-8a27-7996dcb2a4f8">
</p>


<br/>
<br/>

# Experiments

<br/>
<br/>

# Contribution

