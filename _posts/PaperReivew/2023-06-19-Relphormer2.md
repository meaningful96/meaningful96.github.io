---
title: "[논문리뷰]Relphormer: Relational Graph Transformer for Knowledge Graph Representation"

categories: 
  - PaperReview
  
tags:
  - [KG Completion]
  
toc: true
toc_sticky: true

date: 2023-07-10
last_modified_at: 2023-07-10
---

Bi, Z. (2022, May 22). *Relphormer: Relational Graph Transformer for Knowledge Graph Representations*. arXiv.org. https://arxiv.org/abs/2205.10852

이번 포스팅은 3월 14일 포스팅된 ["Relational Graph Transformer for Knowledge Graph Completion-Relphormer"](https://meaningful96.github.io/paperreview/Relphormer/)의 업데이트 버전이다. 논문 버전이 수정되면서 Ablation Study가 추가되었다.

# Problem Statement

일반적인 그래프와는 다르게 Knowledge Graph는 노드 또는 릴레이션의 타입이 여러가지인 Heterogeneous Graph이다. 자연어 처리 분야에서 Transformer가 압도적인 성능을 보여주면서 Computer Vision등의 여러 분야에 접목하려는 실험이 진행되는 중이다. 마찬가지로 Transformer모델이 Knowledge Graph에도 적용하려는 시도가 있었다.

Transformer는 그래프에 적용하면(i.e., KG-BERT) 모든 노드들의 Attention을 통해 관계를 파악하는 것을 목표로 한다. 하지만, 이럴 경우 그래프에서 중요한 정보 중 하나인 <span style="color:gold">**구조 정보(Structural Information)**</span>를 제대로 반영하지 못한다. 본 논문에서는 3가지 문제점을 제시한다.

<span style ="font-size:110%"><b>1. Heterogeneity for edges and nodes</b></span>      
먼저 **Inductive Bias**라는 개념을 알아야한다. 일반적으로 모델이 갖는 일반화의 오류는 불안정하다는 것(**Brittle**)과 겉으로만 그럴싸 해 보이는 것(**Spurious**)이 있다. 모델이 주어진 데이터에 대해서 잘 일반화한 것인지, 혹은 주어진 데이터에만 잘 맞게 된 것인지 모르기 때문에 발생하는 문제이다. 이러한 문제를 해결하기 위한 것이 바로 Inductive Bias이다. **Inductive Bias**란, <u>주어지지 않은 입력의 출력을 예측하는 것이다. 즉, 일반화의 성능을 높이기 위해서 만약의 상황에 대한 추가적인 가정(Additional Assumptions)이라고 보면 된다.</u> 

- Models are Brittle: 아무리 같은 의미의 데이터라도 조금만 바뀌면 모델이 망가진다.
- Models are Spurious: 데이터의 진정한 의미를 파악하지 못하고 결과(Artifacts)와 편향(Bias)을 암기한다.

논문에서는 <b>기존의 Knowledge Graph Transformer가 함축적인 Inductive Bias를 적용</b>한다고 말한다. 왜냐하면 KG-BERT의 경우 입력이 **Single-Hop Triple**로 들어가기 때문이다. 이럴 경우 1-hop 정보만 받아가므로 <span style = "color:gold">**Knowledge Graph에 구조적인 정보를 반영하는데 제약**</span>이 된다.

<br/>

<span style ="font-size:110%"><b>2. Topological Structure and Texture description</b></span>        
1번 문제와 비슷한 문제이다. 기존의 Transformer 모델은 모든 Entity와 Relation들을 plain token처럼 다룬다. 하지만 Knowledge Graph에서는 엔티티가 **위상 구조(Topological Structure) 정보와 문맥(Text Description) 정보**의 두 유형의 정보를 가지며 Transformer는 오직 Text description만을 이용해 추론(Inference)를 진행한다. 중요한 것은 **서로 다른 엔티티는 서로 다른 위상 구조 정보을 가진다**. 따라서, 마찬가지로 결국 기존의 <span style="color:gold">**Knowledge Graph Trnasformer 모델들은 필수적인 구조 정보를 유실**</span>시킨다.

<span style="font-size:120"><b>➜ How to treat heterogeneous information using Transformer architecture?</b></span>

<br/>

<span style ="font-size:110%"><b>3. Task Optimization Universalty</b></span>    
Knowledge Graph는 기존에 보통 Graph Embedding 모델들에 의해 task를 풀었다. 하지만 이 기존의 방식들의 비효율적인 면은 바로 Task마다 사전에 Scoring function을 각각 다르게 정의해주어야 한다는 것이다. 즉, 다른 <span style="color:gold">**Task object마다 다른 Scoring function을 필요**</span>로 하기 때문에 비효율적이다. 기존의 연구들을 다양한 Task에 대해 통일된 representation을 제시하지 못한다.

<span style="font-size:120"><b>➜ How to unite Knowledge Graph Representation for KG-based tasks?</b></span>


<br/>
<br/>

# Related Work

<span style = "font-size:110%"><b>Knowledge Graph Embedding</b></span>  
KG Representation Learning은 <b>연속적인 저차원의 벡터 공간으로 엔티티와 릴레이션들을 projection하는 것을 타겟</b>으로한다. TransE, TransR, RotatE등의 모델들이 존재한다. 하지만 앞서 말했듯, 서로 다른 Task들에 대해 사전에 정의된 Scoring function을 필요로 한다는 비효율성이 존재한다.  

<span style = "font-size:80%">참고: [Knowledge Graph Completion](https://meaningful96.github.io/graph/cs224w-10/)</span>

<br/>
<br/>

# Method

## 1. Overview

### 1) Model Architecture

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Paper_Reconstruction/assets/111734605/20206831-ee83-4acc-9044-49483c2320a3">
</p>

1) **Triple2Seq**: 엔티티와 릴레이션의 다양성(Heterogeneity)를 대응하고 모델의 입력 시퀀스로서 Contextual Sub-Graph를 Sampling한다.(Dynamic Sampling)
2) **Structured-Enhanced Mechanism**: Structural Information과 Textual Information을 다루기 위함
3) **Masked Knowledge Modeling**: KG Representation Leanrning의 Task들을 통합

<br/>

### 2) Preliminaries & Notations

Knowledge Graphs는 triple($$head, relation, tail$$)로 구성된다. 논문에서는 **Knowledge Graph Completion** Task와 **Knowledge Graph-Enhanced Downstream Task**를 푸는 것을 목표로 한다. 모델을 살펴보기 전 Notation을 살펴봐야 한다.

<p align="center">
<img width="500" alt="1" src="https://user-images.githubusercontent.com/111734605/224572006-9fcb2f52-8504-43c1-b8ef-b04e1cd4db07.png">
</p>

- 주의깊게 봐야할 Notation
  - Relational Graph $$G = (\mathscr{E}, R)$$
  - Node Set $$V = \mathscr{E} \; \cup \; R$$
  - Adjacency Matrix = 요소들이 [0,1] 사이에 있고, 차원이 $$ \vert V \vert \times \vert V \vert$$

- Knowledge Graph Completion
  - Triple $$(v_{subject}, v_{predicate}, v_{object}) = (v_s, v_p, v_o) = T$$  
  - As the label set $$T$$, $$f: T_M,A_G \rightarrow Y$$, $$ Y \in \mathbb{R}^{\vert \mathscr{E} \vert \times \vert R \vert} $$ 로 정의된다.

<br/>

## 2.1 Triple2Seq

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Paper_Reconstruction/assets/111734605/f43ff18e-429e-4361-a92d-56d5f5314f3e">
</p>

Knowledge Graph는 많은 숫자의 **Relational Information**을 포함하고 있기 때문에, 그래프를 직접 direct하게 Transformer 모델에 입력으로 집어넣는 것은 불가능하다. Full-graph-based Transformer의 이러한 단점을 극복하기 위해서 논문에서는 **Triple2Seq**를 제안한다. Triple2Seq는 <span style="color:gold">**Contextualized Sub-Graphs를 입력 시퀀스로 사용해 Local Structure 정보를 인코딩**</span>한다.

### 1) Contextualized Sub-Graph

Triple $$\mathcal{T}$$의 Contextualized sub-graph인 <b>$$\mathcal{T_G}$$</b>은 Sub-graph에 중심에 해당하는 Center Triple <b>$$\mathcal{T_C}$$</b>와 Center Triple을 둘러싼 Surrounding neighborhood triple set <b>$$\mathcal{T_{context}}$$</b>를 포함한다. 이 때, Sub-graph sampling process는 오직 triple level에서만 일어난다. Github에 올라온 코드를 확인해보면 이 Sub-graph의 총 triple수는 변수로 지정되어있고, Triple의 최대 hop수는 1로 정해져 있는 것을 알 수 있다. 따라서 Triple $$\mathcal{T}$$에 둘러싸인 이웃들에 해당하는 $$\mathcal{T_{context}}$$를 샘플링하여 얻을 수 있다. 이를 수식으로 표현하면 다음과 같다.

<span style="font-size:110%"><center>$$\mathcal{T_{context} \; = \; \{ {\mathcal{T} \vert \mathcal{T_i} \in \mathcal{N}}} \}$$</center></span> 
<span style="font-size:110%"><center>$$\mathcal{T_G} \; = \; \mathcal{T_C} \; \cup \; \mathcal{T_{context}}$$</center></span>

<br/>

### 2) Dynamic Sampling

여기서 $$\mathcal{N}$$은 Center Triple $$\mathcal{T_C}$$의 고정된 크기의 이웃 Triple set이다. 논문에선는 Local structural information을 좀 더 잘 뽑아내기 위해 학습 중 <span style="color:gold">**Dynamic Sampling**</span>을 하였다. 이는 <u>각 Epoch마다 같은 Center Triple에 대해 여러개의 Contextualized Sub-graph를 <b>무작위(random)로 선택</b>해 추출하는 방법</u>이다. 

<p align="center">
<img width="600" alt="1" src="https://github.com/meaningful96/Paper_Reconstruction/assets/111734605/2be7eccc-0931-4f1e-97a0-4b667b66ac14">
</p>

Triple2Seq의 결과로 얻는 것이 바로 Contextualized Sub-Graph인 $$\mathcal{T_G}$$이다. 또한 $$\mathcal{T_G}$$의 local structure information은 인접 행렬(Adjacency matrix) $$A_G$$에 저장된다. 이전에 나왔던 논문 중 [HittER: Hierarchical transformers for knowledge graph embeddings](https://meaningful96.github.io/paperreview/HittER/)을 통해 알 수 있는 중요한 사실이 하나 있다. 바로 <u>엔티티-릴레이션(Entity-Relation)쌍에 저장된 정보가 중요하다는 것이다.</u> 이러한 사실을 바탕으로 논문에서는 <span style="color:gold">**엔티티-릴레이션 쌍을 Plain token으로 표현하고 릴레이션을 contextualized sub-graph의 special node**</span>로 간주한다. 이러한 방식으로 엔티티-릴레이션, 엔티티-엔티티 및 릴레이션-릴레이션 쌍을 포함한 노드 쌍 정보를 얻을 수 있다. 이렇게 함으로서 결론적으로 **릴레이션 노드를 special node**로 볼 수 있다는 것이다.

<br/>

### 3) Global Node

Triple2Seq는 결국 Contextualized Sub-graph를 통해 Locality를 뽑아낸다. 이럴 경우 global information에 대한 정보가 부족할 수 있다. 따라서 논문에서는 <span style="color:gold">**Global node**</span>의 개념을 도입한다. global node는 쉽게 말하면 임의의 새로운 엔티티를 만들어 training set에 존재하는 모든 엔티티와 1-hop으로 연결시켜놓은 것이다. 즉 모두와 1-hop으로 연결된 엔티티이다. 하지만, 논문에서는 global node를 training set 전체에다가 연결시킨 것이 아닌, <span style="color:gold">**추출된 Sub-graph에 있는 모든 엔티티와 연결된 엔티티를 의미**</span>한다.

<span style="font-size:110%"><b>Remark 1.</b></span>  
> Triple2Seq는 입력 시퀀스를 만들기위해 contextualized sub-graph를 dynamic sampling한다.
> 결과적으로 Transformer는 Large KG에 대해서도 쉽게 적용될 수 있다.
> Relphormer는 Heterogeneous graph에 초점을 맞춘 모델이며,
> sequential modeling을 위해 문맥화된 하위 그래프(Contextualized sub-graph)에서 edge(relation)를 하나의 Special node로 취급한다.
> 게다가, Sampling process는 성능을 향상시키는 data augmentation operator로 볼 수 있다.

## 2.2 Structure enhanced self attentionPermalink

<p align="center">
<img width="600" alt="1" src="https://github.com/meaningful96/Paper_Reconstruction/assets/111734605/2be7eccc-0931-4f1e-97a0-4b667b66ac14">
</p>

### 1) Attention Bias

트랜스포머는 입력으로 Sequence를 받는다. 이 때, <span style="font-size:105%"><b>Sequential Input가 Fully-Connected Attention Mechanism을 거치면서 Structural Information을 유실</b></span>시킬 수 있다. 그 이유는 Fully-Connected 라는 것은 결국 Dense-layer의 형태이다. 즉, Neural Network를 예로 들면 모든 drop-out이 0인 상태인데 <u><b>한 노드에 대해 다른 모든 노드들과의 attention을 구하므로(구조와 상관없이 모든 노드를 상대하기 때문) 구조 정보가 반영되지 못하는 것</b></u>이다.

이를 극복하기 위해 논문에서는 <span style="color:aqua">**Attention Bias**</span>를 추가로 사용하는 방식을 제안하였다. Attention bias를 통해 <span style="color:gold"><b>Contextualized Sub-Graph 안의 노드쌍들의 구조 정보(Structural information)을 보존</b></span>할 수 있다. 노드 $$v_i$$와 $$v_j$$사이의 attention bias는 <b>$$\phi(i,j)$$</b>로 표기한다.

<p align="center">
<img width="700" alt="1" src="https://github.com/meaningful96/Paper_Reconstruction/assets/111734605/54360008-8682-4822-8ecd-08f71f0eb9a4">
</p>

Triple2Seq에서 샘플링된 Contextualized Sub-graph의 구조 정보는 인접 행렬(Adjacency Matrix) $$A_G$$에 저장된다. 이 때, Sub-graph의 구조 정보를 Normalization한 값을 <b>$$\widetilde{A}$$</b>으로 표기한다. 이러한 사실을 바탕으로 <b>$$\widetilde{A^m}$$($$\widetilde{A}$$ to the $$m$$-th power)</b>를 정의한다. 이는 $$\widetilde{A}$$를 **m번 제곱**한 것(**행렬곱**을 m번 수행 <span style="font-size:80%">ex)$$ \widetilde{A} \; @ \; \widetilde{A}$$</span>) m은 hyperparameter이다. 

여기서 알아야 할 개념이 있는데, 인접 행렬(Adjacency Matrix)의 제곱, 세제곱 등이 가지는 의미이다. 제곱을 예로 설명하면, m = 2인 상황으로, 어떤 노드 $$v_i$$에서 또 다른 노드 $$v_j$$로 이동할 때 2번(m=2)움직여서 갈 수 있는 횟수를 의미한다. 즉, 노드를 순회할 때, m번 이동하여 갈 수 있는 경우의 수가 각 $$\widetilde{A^m}$$의 요소가 된다. 간단하게 코드를 통해 살펴보면 다음과 같다.

<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/Paper_Reconstruction/assets/111734605/4496cab6-f416-48f7-bf9b-3d44c091fae6">
</p>

```python
import networkx as nx
import numpy as np

head = [1,2,3,4]

G = nx.Graph()
G.add_nodes_from(head)
G.add_edges_from([(1,1),(1,2),(1,4),(2,3),(2,4),(3,4)])

## Adjacency Matrix
a1 = np.array([1,1,0,1])
a2 = np.array([1,0,1,1])
a3 = np.array([0,1,0,1])
a4 = np.array([1,1,1,0])

A = np.c_[a1,a2,a3,a4]
```

<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/Paper_Reconstruction/assets/111734605/805a3e21-8847-4e93-96db-7c9a48855df6">
</p>

앞의 실험을 통해 $$\widetilde{A^m}$$이 확실하게 정의되었고, $$f_{structure}$$는 구조 정보를 인코딩하는 **Linear Layer**로 $$\widetilde{A^m}$$을 입력으로 한다. 이를 통해 최종적으로 Attention bias인 $$\phi(i,j)$$가 정의된다. 다시 한 번 강조하지만, <span style="color:gold">**Attention bias는 샘플링된 Contextualized Sub-Graph의 구조 정보를 포착**</span>하는 역할을 하며, 이 부분이 이 논문의 가장 큰 Contribution중 하나이다.

<br/>

### 2) Contrastive Learning Strategy

Dense한 Knowledge Graph(WN18RR보다는 FB15k-237이 relation의 종류가 더 많으므로 더 Dense함)의 경우, 하나의 Center Triple에 대해 많은 수의 Contextualized Sub-Graph를 샘플링하여 학습을 진행하면 정보의 편향이 생길 수 있다. Dynamic Sampling을 할 때, 이런 불안정성을 극복하고자 **Contextual contrastive strategy**를 이용한다. 

> We use the contextualized sub-graph of the same triple in different epochs triple in different epochs to enforce the model to conduct similar predictions.

즉, <span>서로 다른 Epoch에서 같은 Triple의 contextualized sub-graph를 사용해 모델에게 유사도를 학습하는 걸 강요한다. Contextualized sub-graph의 입력 시퀀스를 인코딩하고 hidden vector $$\mathcal{h_{mask}}$$를 현재 Epoch $$t$$에서 $$c_t$$, 마지막 Epoch $$c_{t-1}$$로 가져온다. 마지막으로 Loss를 정의하는데 $$\mathcal{L_{contextual}}$$를 contextual loss로 정의한다. 이 작업을 거쳐 최종 목표는 <span style="color:gold">**서로 다른 sub-graph사이의 차이를 최소화(minimization)하는 것**</span>이다. 

<p align="center">
<img width="700" alt="1" src="https://github-production-user-asset-6210df.s3.amazonaws.com/111734605/252900409-5e4d57c7-c3d3-4692-8acb-c33f8a5bc005.png">
</p>

수식에서도 보이듯이, Contrastive Learning의 형태를 따르며 주의할 것은 빨산색 부분이다. 분모는 $$v_i$$ 노드의 t와 t-1시점의 유사도와 t시점에서 $$v_i$$노드와 $$v_j$$노드의 유사도의 합이다. 이를 통해 알 수 있는 것은 학습의 방향이 서로 다른 sub-graph의 차이를 줄이는 방향으로 간다는 것이다.

<span style="font-size:110%"><b>Remark 2.</b></span>  
> 이 Structure-enhanced Transformer는 모델에 구애받지 않으며 따라서 트랜스포머 아키텍처에 의미론적(semantic) 및 > 구조적 정보(Structure Information)를 주입하는 기존의 접근 방식과 직교한다는 점에 주목해야 한다. 
> Original Graph에서 말 그대로 기존 트랜스포머는 모든 노드에 대한 attention을 수행하므로 구조 정보가 반영되지 못한다.
> 하지만, Structure-Enhanced Transformer를 사용하면 Local Contextualized Sub-graph의 구조와 의미론적 특징의 영향력을 활용할 수 있는 유연성을 제공한다. Local graph 구조에서 유사한 노드간의 정보 교환에 편리하다.

좀 더 쉽게 말하자면, 기존의 atttention operation은 단순히 전체 그래프 안에서 노드와 의미있는 relation사이에서 계산을 진행하는것에 반해, Structure-enhances self attention은 <span style="color:gold">**Contextualized Sub-graph 구조를 이용한 Locality 정보와 Semantic feature들에 대해도 유의미한 영향을 주는 유연성을 이끌어내며 이를 통해 Transformer 모델에 구조적 정보(Structural information)와 의미론적 정보(Semantic feature)를 동시에 줄 수 있다**</span>는 것이 특징이다.

## 2.3 Masked Knowledge Modeling

Masked Knowledge Modeling은 특별한 것이 아닌, Masked Langauge Modeling(MLM)과 유사하다. Knowledge Graph Completion은 $$f: \mathcal{T_M}, A_G \; \rightarrow \; Y$$를 푸는 Task이다. Relphormer에서는 랜덤하게 입력 시퀀스의 토큰들을 마스킹하고, 그 마스킹된 토큰들을 예측한다.

> We randomly mask specific tokens of the input sequences and then predict those masked tokens.

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/224577501-d11de3de-c587-4b19-8832-e6567f4b8573.png">
</p>

Input Contexturalized Sub-Graph node sequence $$\mathcal{T_G}$$가 주어졌을 때, 랜덤하게 Center triple을 마스킹한다. 구체적으로 relation prediction을 할 때는 head나 tail 둘 중에 하나를 마스킹한다. 이를 Triple로 표현하면 $$(v_{h},?,\[MASK\]) or (\[MASK\], ?, v_t)$$이다. 

<p align="center">
<img width="200" alt="1" src="https://github.com/meaningful96/Paper_Reconstruction/assets/111734605/682de0cf-40a9-4d9d-95e3-ab5af8bd5814">
</p>

$$Y$$는 Candidate(후보)이다. Masked Knowledge Modeling이 궁극적으로 풀고자 하는 것은 <span style="color:gold"><b>마스킹된 노드 시퀀스 $$\mathcal{T_M}$$과 Contextualized Sub-graph의 구조 정보를 나타내는 $$A_G$$가 주어졌을 때 Original Triple $$\mathcal{T}$$의 missing part를 찾는 것</b></span>이다. ($$Y \in \mathbb{R^{\vert \mathcal{E} \vert \times \vert \mathcal{R} \vert}}$$)

직관적으로 MKM방식은 기존의 거리를 기반으로 Scoring function을 정의해 학습하는 Graph-Embedding방식과 차이를 보여준다. 

<br/>
<br/>

# Experiment & Result



<br/>
<br/>

# Contribution

# Reference
[Inductive Bias란 무엇일까?](https://re-code-cord.tistory.com/entry/Inductive-Bias%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%BC%EA%B9%8C)  
["Relational Graph Transformer for Knowledge Graph Completion-Relphormer"](https://meaningful96.github.io/paperreview/Relphormer/)  

