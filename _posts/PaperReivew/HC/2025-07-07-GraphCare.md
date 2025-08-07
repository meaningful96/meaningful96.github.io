---
title: "[논문리뷰]GraphCare: Enhancing Healthcare Predictions with Personalized Knowledge Graphs"

categories: 
  - HC
  
toc: true
toc_sticky: true

date: 2025-08-07
last_modified_at: 2025-08-07
---

*Pengcheng Jiang, Cao Xiao, Adam Cross, and Jimeng Sun*. “[**GraphCare: Enhancing Healthcare Predictions with Personalized Knowledge Graphs**](https://arxiv.org/abs/2305.12788).” In Proceedings of the 12th International Conference on Learning Representations (ICLR 2024)

# Problem Statement
- Personalized Knowledge Graph-based Clinical Predictive Modeling
- GraphCare는 **환자별 전자의무기록(EHR)** 데이터와 외부 의료 지식 그래프(KG)를 결합하여, 각 환자에게 맞춤화된 Personalized KG를 생성하고 이를 기반으로 다양한 임상 예측(사망률, 재입원, 입원 기간, 약물 추천 등)을 수행한다.

<span style = "font-size:110%">**기존 연구의 한계점**</span>
**[구조적 EHR 및 단순 계층 기반 KG의 한계]**: EHR 내의 구조적 정보만을 사용하거나, simple hierarchy(부모-자식 관계)에만 국한된 KGs를 사용하여 의료 개념 간의 복잡한 관계를 충분히 반영하지 못한다.

**[제한적 Feature 및 외부 지식 연동 부족]**: 기존 Personalized KG 연구들도 대부분 제한적이고 수작업으로 큐레이팅된 feature에 의존하거나, 외부 지식베이스와의 연동이 부족해 예측 성능의 상한이 존재한다.

GraphCare는 LLM 및 외부 biomedical KG의 지식을 환자별 Personalized KG로 융합하여 open-world knowledge의 장점을 극대화하고, 이를 활용한 임상 예측의 성능을 향상시키고자 한다.

# Methodology
## Overview
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.07.07%5DGraphCARE/figure1.png?raw=true">
</p>

GraphCare는 환자별 임상 예측을 위해 외부 지식과 EHR 데이터를 통합하여 Personalized Knowledge Graph를 생성하고, 이를 Bi-attention GNN에 입력해 다양한 임상 예측을 수행한다. 구체적으로, 각 환자의 진단·시술·약물 등 의료 개념별로 LLM과 외부 KG에서 관계(triple)를 추출해 concept-specific KG를 만들고, 이들을 클러스터링하여 의미적으로 정제한 뒤, 환자별 방문 시퀀스에 맞춰 personalized temporal graph로 통합한다. 마지막으로 이 personalized graph를 Bi-attention GNN에 입력하여, 방문과 노드에 대한 attention을 통해 핵심적인 임상 정보를 강조하여 사망률, 재입원, 입원 기간, 약물 추천 등 다양한 healthcare prediction task를 효과적으로 수행한다.

## Step 1. Medical Concept-specific KG 생성
- **입력**: Medical Concept(질병, 시술, 약물) 목록
- **출력**: 각 Concept별 KG
  
1. LLM Prompting을 통해 **각 의료 개념에 대한 triple(지식)을 생성**한다. 프롬프트는 instruction + 예시 + target concept이 포함되며, 여러 번 반복하여 다양한 관계를 확보한다.
2. 기존 **Biomedical KG(UMLS 등)**에서 k-hop subgraph sampling 방식으로 해당 개념 중심의 부분 그래프 추출한다.
3. 앞선 과정을 통해 하나의 거대한 concept-specific KG를 구축한다.
4. 모든 KG 노드/엣지는 word embedding cosine similarity를 이용한 agglomerative clustering(임계값 $$\delta$$)으로 유사 개체/관계를 그룹화하여 global KG로 정제한다.

## Step 2. Personalized KG 합성
- **입력**: 각 환자의 직접 연관 Medical Concept과 해당 concept-specific KG
- **출력**: 환자별 Personalized KG
  
1. 환자 노드(P)를 만들고, 이 환자와 직접 연결된 개념의 노드들과 concept-specific KG들을 합친다.
2. 환자 방문 시퀀스별로 visit-subgraph 생성, 방문 간 temporal 정보(연결 edge)도 통합하여 최종 Personalized KG를 완성한다.


## Step 3. Bi-attention Augmented (BAT) Graph Neural Network
- **입력**: 환자별 Personalized KG
- **출력**: 환자 임상 예측 결과(사망률, 재입원, 입원 기간, 약물 추천 등)

1. 각 노드/엣지 embedding은 word embedding에서 dimension reduction을 거쳐 사용된다.
2. **Node-level/Visit-level Attention**: 노드별, 방문별 attention 가중치를 각각 계산하여 핵심 방문·노드 정보에 집중하도록 함. 방문 attention의 경우, decay coefficient를 도입해 최근 방문에 더 큰 가중치 부여.
3. **Attention Initialization**: LLM knowledge 기반 prior를 이용, 예측 task-feature에 특화된 term("terminal condition" 등) embedding과의 cosine similarity로 attention 초기화.
4. **Convolution Layer**: node/edge feature와 attention을 이용, 각 노드 embedding을 반복적으로 업데이트.
5. **Representation**:
  - Patient-graph embedding: 모든 노드 embedding 평균
  - Patient-node embedding: 직접 연결 medical concept의 embedding 평균
  - Joint embedding: 위 두 embedding의 concat
  - 세 가지 중 task별로 가장 적합한 embedding을 MLP에 입력하여 최종 예측값 산출
