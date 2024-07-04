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



<br/>
<br/>

# Experiments

<br/>
<br/>

# Contribution

