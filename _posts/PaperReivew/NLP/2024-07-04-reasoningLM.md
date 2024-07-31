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

ReasningLM은 <span style="color:red">**Question과 Subgraph의 직렬화된 자연어 토큰 시퀀스**</span>를 입력으로 받는다. 트랜스포머 모듈이 각 토큰에대한 임베딩 시퀀스를 출력하고 최종적으로 정답을 찾기위해 subgraph의 hidden representation들만 linear layer를 통과시켜 score를 계산하게된다. 본 논문에서는 트랜스포머 모듈(backbone)로 RoBERTa-base를 사용하였다.

ReasoningLM의 핵심 요소는 두 가지이다.
- Adaptation Tuning Strategy
- Subgraph-Aware Self-Attention

## 1. Adaptation Tuning Strategy
Adaptation Tuning Strategy은 질문과 서브그래프를 추출하기 위한 전략이다. 학습을 위해서 총 2만 개의 synthesized question을 뽑아낸다. 이 때, 서브그래프는 Large-scale KG에 해당하는 Wikidata5M에서 추출한다.

### 1) Subgraph Extraction
추출된 subgraph가 PLM에 제대로 적용되기 위해서는 subgraph들이 대중적으로 사용되는 지식(commonly used knowledge)를 잘 내포하고 있어야 한다. 따라서 인기있는 엔티티(popular entity)를 Wikidata5M에서 추출해 시드 토픽 엔티티(seed topic entity)로 사용하고, KQA Pro[(Cao et al., 2022)](https://aclanthology.org/2022.acl-long.422/)와 같은 방식으로 정답 엔티티와 서브그래프를 추출하게 된다.

먼저 Wikidata5M에서 인기있는 2000개의 토픽 엔티티를 추출한다. 각 토픽 엔티티들을 출발점으로 하여 <span style="color:red">**randomwalk를 수행하여 Reasoning path를 추출**</span>한다. Reaoning path의 길이는 4-hop을 넘지 않으며 **종점은 반드시 정답(answer) 엔티티**가 되게 만든다. 각 Reaoning path들은 결론적으로 시작점이 토픽 엔티티이고, 종점이 정답 엔티티가 되게된다. 

Reasoning path가 정해지면 이제 앞서말한 KQA Pro논문의 아이디어를 활용하여 subgraph를 추출할 수 있다. Reasoning path의 시작점인 토픽 엔티티를 기준으로 $$k$$-hop 내의 엔티티와 릴레이션을 임의로 추출한다. 그리고 실제로 존재하는 트리플들만 중복은 제거하고 추출하여 하나의 서브그래프를 만든다. 이 때, 서브그래프에는 반드시 reasoning path가 포함이되어야 한다.

<br/>

### 2) Question Synthesis
Reasoning path는 토픽 엔티티와 정답 엔티티를 포함한다. 본 논문에서는 이 reaoning path를 이용해서 자동으로 질문을 만들어내는 방법을 제안한다. 먼저, <span style="color:red">**질문 생성을 위해 ChatGPT를 사용**</span>하였다. 질문 생성 방식에는 크게 두 가지로 나눠진다.

- 규칙 기반 생성
  - 여러 **일반적인 템플릿**을 수작업으로 작성한다. 이를 토대로 토픽 엔티티와 릴레이션을 질문으로 변환한다.
  - Ex) "What is the <span style="color:red">\[relation\]</span> of <span style="color:coral">\[entity\]</span>?" ➔ "What is the <span style="color:red">**capital**</span> of <span style="color:coral">**France**</span>" 

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

학습을 위해 추출된 질문(question)과 reasoning path를 포함하는 서브그래프(subgraph)를 사전 학습된 언어 모델(PLM)에 입력시키기 위해서는 자연어 토큰을 시퀀스로 만들어줘야 한다. 이를 위해 ReasoningLM은 question-subgraph 쌍을 하나의 시퀀스로 구성한다. 이 때, 서브그래프에 존재하는 엔티티와 릴레이션의 이름을 직렬화(serialization) 시켜줘야하는데, 이를 위해 **너비 우선 탐색 알고리즘(Breadth-first search, BFS)**를 사용한다. **BFS**를 사용함으로써 토픽 엔티티에서 가까운 순서대로 시퀀스내에 트리플이 위치하기 때문에 서브그래프의 구조 정보가 보존된다. 위의 그림처럼 결론적으로 \[CLS\]-\[Qusetion\]-\[SEP\]-\[Subgraph\]형태로 하나의 직렬화된 토큰 시퀀스가 PLM의 입력으로 들어간다.

<br/>

### 2) Subgraph-Aware Self-Attention with Masking
사전 학습된 PLM을 fine-tuning함에 있어서 두 가지의 핵심적인 요소가 추가된다. 바로 마스킹(masking)과 어뎁터(adpater)이다.  

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/88a198a4-a40d-4f19-bb5e-f6a37b7db457">
</p>

<span style="font-size:110%">**Masked Attention**</span>  
먼저 Self-attention을 할 때, subgraph-aware하게 만들기 위해 어텐션 스코어 매트릭스에 마스킹을 적용해야한다. 마스킹은 Masked Attention 그림에서와 같이 크게 네 부분으로 구분된다.

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/36053659-ffc7-4b2e-9a01-fb9b8b586c59">
</p>

**A**는 입력으로 들어온 시퀀스에서 질문(question, query)를 구성하는 토큰들 사이의 어텐션에 해당한다. 이는 일반적인 문맥화된 자연어 시퀀스에 해당함으로 마스킹을 하지 않는다. **B**는 서브그래프와 질문 사이의 어텐션이다. 서브그래프 안의 모든 엔티티와 릴레이션의 이름에 대해 질문을 구성하는 토큰들 사이의 연관성을 구하는 것이다. 서브그래프는 트리플로 정보를 가지고 있기 때문에 일반적인 질문과 달리 핵심어들만 가진 자연어 시퀀스이다. 따라서, 각각의 핵심어가 질문에서 얼마정도의 영향력을 모델이 알 수 있기 때문에 마스킹을 진행하지 않는다.

반면 **C** 서브그래프내의 엔티티와 릴레이션들간의 어텐션이며, 일부분만 마스킹을 하지 않는다. Subgraph-aware하게 만드려면 구조적 정보를 줄 수 있는 inductive bias를 주입해야한다. 따라서 ReasoningLM에서는 이 서브그래프-서브그래프 어텐션 사이에서 각 엔티티를 기준으로 $$1$$-hop 거리에 있는 트리플들만 정보를 살려놓고, 나머지는 마스킹을 해버린다. 이렇게 함으로써, 각 엔티티의 이웃이 무엇인지, 직접적으로 연결된 트리플이 무엇인지 모델이 학습 가능하다. 

**D**는 앞선 A, B, C와 달리 모든 부분을 마스킹한다. D는 질문-서브그래프 쌍의 어텐션으로 B와 반대이다. 하지만, B는 모든 정보를 살리는 반면 D는 모든 정보를 죽인다. 추측컨데, D부분을 full-masking하는 이유가 'a', 'the'와 같이 <span style="color:red">**불용어 등이 query에 존재하고, 이들이 서브그래프의 핵심어들과의 연관성을 낮추는 noise가 될 수 있기 때문**에 마스킹하는 것 같다.

<span style="font-size:110%">**Adapter**</span>  
다음으로는 adapter 부분을 추가해 업데이트의 효율성을 고려한다. 어뎁터는 위에 그림에서 오른쪽과 같다. 트랜스포머 내부에 FF layer 다음부분에 추가한다. 본 논문에서는 트랜스포머의 모든 레이어의 파라미터를 업데이트 하는 것이 아닌, 어뎁터만 업데이트해 학습 효율성을 높일 수 있다고 한다.

Adapter는 정확히 어떻게 구성되어 있는지 설명하지 않았다. 2019년 논문 "Parameter-efficient transfer learning for NLP"에서 소개된 어댑터 메커니즘을 차용한 것이다. 어댑터는 Transformer의 내부에 추가되며, 기존의 Transformer 구조를 변경하지 않고 학습 가능한 파라미터를 추가한다. Fine-tuning전 adaptation tuning시 파라미터를 업데이트하는 방식에 따라 두 가지로 구분된다.

- Full-Parameter Tuning (FPT) 
  - Adaptation Tuning시 어댑터를 포함해 모든 레이어를 학습한다.

- Parameter-Efficient Tuning (PET)
  - 어댑터만 업데이트: PET는 전체 모델 파라미터를 업데이트하지 않고, 특정 어댑터 모듈만 업데이트한다. 이는 학습되는 파라미터 수를 줄이고, 메모리와 계산 자원을 절약하면서도 효율적인 학습을 가능하게 한다. → 나머지는 freeze

## Training
**(1). Adaptation Tuning**  
  1. 질문-서브그래프 쌍의 직렬화된 자연어 시퀀스를 PLM에 입력시킨다.  
  2. 트랜스포머를 거쳐 질문-서브그래프 쌍의 hidden representation을 얻는다.  
  3. 이 중, 서브그래프의 hidden representation만 liner layer에 통과시켜 서브그래의 모든 엔티티들의 socre를 얻는다. 이를 $$s$$라 한다.  

<center><span style="font-size:110%">$$s = \text{softmax}(\text{Linear}(\mathbf{H}))$$</span></center>  
<center><span style="font-size:110%">$$\mathcal{L}_{at} = D_{KL}(s, s^{*})$$</span></center>

  4. $$s$$와 Ground-Truth에 대한 one-hot vector인 $$s^{*}$$과 KL-divergence를 최소화되도록 학습한다.
  5. 총 20,000개의 training 질문-서브그래프 쌍으로 adaptation tuning을 실시한다.

**(2) Fine-tuning PLM**
  1. 어댑터만 업데이트를 진행하며, 특정 Dataset의 training set으로 학습을 진행하게된다.

<br/>
<br/>

# Experiments

## Dataset

<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/0ad952f3-feea-4112-bec4-2273b200a9e6">
</p>

- Knowledge Graph
  - Wikidata5M (엔티티: 500만개, 릴레이션: 822개)
- QA dataset
  - WebQSP, CWQ, MQA-1H, MQA-2H, MQA-3H

## Main Result
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/48341ed4-a7cf-407e-8936-40731c957dce">
</p>

- Adaptation Tuning시 Synthesized qustion을 추출하는 방식
  - 규칙 기반 생성: Rule-SYN
  - LLM 기반 생성: LLM-SYN
- Adaptation Tuning시 파라미터 업데이트
  - PET: 어댑터만 학습
  - FPT: 모든 레이어 학습

먼저 ReasoningLM의 경우 fine-tuning시 업데이트의 효율성을 위해 어댑터만 학습한다. 따라서 다른 모델들에 비해 업데이트하는 파라미터 수가 적은 것을 확인할 수 있다. 성능 결과를 보면 ReasoningLM이 MQA-1H를 제외한 모든 데이터셋에서 SOTA인 것을 알 수 있다. 특히 ChatGPT와 같은 LLM보다도 성능이 뛰어나다는 점이 인상깊다. 이는 **QA문제를 풀 때 KG를 활용하는 것이 매우 중요**하다는 것을 말해준다. 반면 MQA-1H에서는 ReasoningLM의 성능이 SOTA모델보다 약간 떨어진다. 이는 ReasoningLM이 1-hop prediction보다 multi-hop reaoning에 강하다는 것을 말해준다.

Ablation study의 일환으로 Adaptation Tuning시 FPT 방식을 채택하고, 정제화된 질문을 생성하는 방식에 따라 성능을 비교하였다. 당연하게도 LLM을 기반으로 질문을 생성하는 것의 성능이 압도적으로 좋았다. 또한 LLM-SYN을 그대로 두고, PET방식으로 학습을 진행하여 성능 비교를 해봤을때, adpatation tuning시 모든 레이어의 파라미터를 업데이트하는 것이 더 좋다는 점을 알 수 있다.

## Extra Experiment
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/11094a0f-8dd3-4979-b940-da067ede3427">
</p>

### 1) Ablation Study

Table 3은 ReasoningLM의 핵심인 Subgraph-Aware Self-Attention(SA)와 Adaptation Tuning(AT)를 제거했을 때의 성능을 비교한 실험이다. 결과적으로 Adaptation Tuning을 제거하였을 때 WebQSP에서 성능이 크게 감소하였다. 반면, 좀 더 어려운 문제를 푸는 CWQ 데이터셋의 경우는 SA의 효과가 훨씬 크다는 것을 알 수 있다.

추측으로는, 더욱 더 전문적인 지식을 포함하는 Knowledge Intensive Task에서는 positive와 negative를 구분해주는 것이 훨씬 중요하며, CommonSense reasoning, Open-Domain QA와 같은 general한 문제에서는 KG를 활용하는 것 자체가 성능 향상에 많이 기여하는 것 같다.

<br/>

### 2) Performance Comparison using Different Backbone
Table 4는 PLM의 모듈을 바꿨을 때의 성능 변화이다. 결론적으롷로 파라미터의 수가 증가할수록 성능이 높아진다. 하지만, 본 실험에서 가장 큰 LM이 RoBERTa-large라는 점에서 아쉽다. sLM이나 LLM을 사용했을 때의 성능이 있었으면 좋았을 것 같다.


<br/>
<br/>

# Limitation and Contribution
<span style="font-size:110%">**Limitations**</span>  
1. 여러 KGQA 데이터셋에서 실험을 수행했지만, commonsense reasoning이나 KGC와 같은 추론 작업에 대한 평가는 부족합니다.
2. Seed가 되는 토픽 엔티티를 어떻게 선정했는지 밝히지 않았다. 즉, 토픽 엔티티를 뽑는 기준이 모호하다.

<span style="font-size:110%">**Contributions**</span>  
1. PLM이 <span style="color:red">**Adaptation Tuning**</span>과 <span style="color:red">**Subgraph-Aware Self-Attention**</span> 메커니즘을 활용하여 질문 이해, 질문과 서브그래프 간의 깊은 상호작용, 서브그래프에 대한 추론을 동시에 모델링할 수 있도록 한다.
2. PLM이 특수한 입력 형식과 주의 메커니즘에 적응할 수 있도록 LLM을 사용하여 KGQA 작업 형식을 위한 <span style="color:red">**자동 데이터 구축 방법을 제안**</span>한다.
