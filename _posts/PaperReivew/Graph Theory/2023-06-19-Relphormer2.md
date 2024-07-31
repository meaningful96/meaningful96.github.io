---
title: "[논문리뷰]Relphormer: Relational Graph Transformer for Knowledge Graph Representation"

categories: 
  - GR
  
toc: true
toc_sticky: true

date: 2023-07-10
last_modified_at: 2023-07-10
---

Bi, Z. (2022, May 22). *Relphormer: Relational Graph Transformer for Knowledge Graph Representations*. arXiv.org. https://arxiv.org/abs/2205.10852

이번 포스팅은 3월 14일 포스팅된 ["Relational Graph Transformer for Knowledge Graph Completion-Relphormer"](https://meaningful96.github.io/gr/Relphormer/)의 업데이트 버전이다. 논문 버전이 수정되면서 Ablation Study가 추가되었다.

# Problem Statement

일반적인 그래프와는 다르게 Knowledge Graph는 노드 또는 릴레이션의 타입이 여러가지인 Heterogeneous Graph이다. 자연어 처리 분야에서 Transformer가 압도적인 성능을 보여주면서 Computer Vision등의 여러 분야에 접목하려는 실험이 진행되는 중이다. 마찬가지로 Transformer모델이 Knowledge Graph에도 적용하려는 시도가 있었다.

Transformer는 그래프에 적용하면(i.e., KG-BERT) 모든 노드들의 Attention을 통해 관계를 파악하는 것을 목표로 한다. 하지만, 이럴 경우 그래프에서 중요한 정보 중 하나인 <span style="color:red">**구조 정보(Structural Information)**</span>를 제대로 반영하지 못한다. 본 논문에서는 3가지 문제점을 제시한다.

<span style ="font-size:110%"><b>1. Heterogeneity for edges and nodes</b></span>      
먼저 **Inductive Bias**라는 개념을 알아야한다. 일반적으로 모델이 갖는 일반화의 오류는 불안정하다는 것(**Brittle**)과 겉으로만 그럴싸 해 보이는 것(**Spurious**)이 있다. 모델이 주어진 데이터에 대해서 잘 일반화한 것인지, 혹은 주어진 데이터에만 잘 맞게 된 것인지 모르기 때문에 발생하는 문제이다. 이러한 문제를 해결하기 위한 것이 바로 Inductive Bias이다. **Inductive Bias**란, <u>주어지지 않은 입력의 출력을 예측하는 것이다. 즉, 일반화의 성능을 높이기 위해서 만약의 상황에 대한 추가적인 가정(Additional Assumptions)이라고 보면 된다.</u> 

- Models are Brittle: 아무리 같은 의미의 데이터라도 조금만 바뀌면 모델이 망가진다.
- Models are Spurious: 데이터의 진정한 의미를 파악하지 못하고 결과(Artifacts)와 편향(Bias)을 암기한다.

논문에서는 <b>기존의 Knowledge Graph Transformer가 함축적인 Inductive Bias를 적용</b>한다고 말한다. 왜냐하면 KG-BERT의 경우 입력이 **Single-Hop Triple**로 들어가기 때문이다. 이럴 경우 1-hop 정보만 받아가므로 <span style = "color:red">**Knowledge Graph에 구조적인 정보를 반영하는데 제약**</span>이 된다.

<br/>

<span style ="font-size:110%"><b>2. Topological Structure and Texture description</b></span>        
1번 문제와 비슷한 문제이다. 기존의 Transformer 모델은 모든 Entity와 Relation들을 plain token처럼 다룬다. 하지만 Knowledge Graph에서는 엔티티가 **위상 구조(Topological Structure) 정보와 문맥(Text Description) 정보**의 두 유형의 정보를 가지며 Transformer는 오직 Text description만을 이용해 추론(Inference)를 진행한다. 중요한 것은 **서로 다른 엔티티는 서로 다른 위상 구조 정보을 가진다**. 따라서, 마찬가지로 결국 기존의 <span style="color:red">**Knowledge Graph Trnasformer 모델들은 필수적인 구조 정보를 유실**</span>시킨다.

<span style="font-size:120"><b>➜ How to treat heterogeneous information using Transformer architecture?</b></span>

<br/>

<span style ="font-size:110%"><b>3. Task Optimization Universalty</b></span>    
Knowledge Graph는 기존에 보통 Graph Embedding 모델들에 의해 task를 풀었다. 하지만 이 기존의 방식들의 비효율적인 면은 바로 Task마다 사전에 Scoring function을 각각 다르게 정의해주어야 한다는 것이다. 즉, 다른 <span style="color:red">**Task object마다 다른 Scoring function을 필요**</span>로 하기 때문에 비효율적이다. 기존의 연구들을 다양한 Task에 대해 통일된 representation을 제시하지 못한다.

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
<img width="1000" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/c73fd70f-1111-48c7-b87f-5c205fa4e9ec">
</p>

1) **Triple2Seq**: 엔티티와 릴레이션의 다양성(Heterogeneity)를 대응하고 모델의 Input Sequence로서 Contextual Sub-Graph를 Sampling한다.(Dynamic Sampling)
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
<img width="1000" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/be4bd6fd-a0a4-488e-9f7f-bcbb3e9e8903">
</p>

Knowledge Graph는 많은 숫자의 **Relational Information**을 포함하고 있기 때문에, 그래프를 직접 direct하게 Transformer 모델에 입력으로 집어넣는 것은 불가능하다. Full-graph-based Transformer의 이러한 단점을 극복하기 위해서 논문에서는 **Triple2Seq**를 제안한다. Triple2Seq는 <span style="color:red">**Contextualized Sub-Graphs를 Input Sequence로 사용해 Local Structure 정보를 인코딩**</span>한다.

### 1) Contextualized Sub-Graph

Triple $$\mathcal{T}$$의 Contextualized sub-graph인 <b>$$\mathcal{T_G}$$</b>은 Sub-graph에 중심에 해당하는 Center Triple <b>$$\mathcal{T_C}$$</b>와 Center Triple을 둘러싼 Surrounding neighborhood triple set <b>$$\mathcal{T_{context}}$$</b>를 포함한다. 이 때, Sub-graph sampling process는 오직 triple level에서만 일어난다. Github에 올라온 코드를 확인해보면 이 Sub-graph의 총 triple수는 변수로 지정되어있고, Triple의 최대 hop수는 1로 정해져 있는 것을 알 수 있다. 따라서 Triple $$\mathcal{T}$$에 둘러싸인 이웃들에 해당하는 $$\mathcal{T_{context}}$$를 샘플링하여 얻을 수 있다. 이를 수식으로 표현하면 다음과 같다.

<span style="font-size:110%"><center>$$\mathcal{T_{context} \; = \; \{ {\mathcal{T} \vert \mathcal{T_i} \in \mathcal{N}}} \}$$</center></span> 
<span style="font-size:110%"><center>$$\mathcal{T_G} \; = \; \mathcal{T_C} \; \cup \; \mathcal{T_{context}}$$</center></span>

<br/>

### 2) Dynamic Sampling

여기서 $$\mathcal{N}$$은 Center Triple $$\mathcal{T_C}$$의 고정된 크기의 이웃 Triple set이다. 논문에선는 Local structural information을 좀 더 잘 뽑아내기 위해 학습 중 <span style="color:red">**Dynamic Sampling**</span>을 하였다. 이는 <u>각 Epoch마다 같은 Center Triple에 대해 여러개의 Contextualized Sub-graph를 <b>무작위(random)로 선택</b>해 추출하는 방법</u>이다. 

<p align="center">
<img width="600" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/6bc4842e-ce90-4fcb-8f59-e004e070624d">
</p>

Triple2Seq의 결과로 얻는 것이 바로 Contextualized Sub-Graph인 $$\mathcal{T_G}$$이다. 또한 $$\mathcal{T_G}$$의 local structure information은 인접 행렬(Adjacency matrix) $$A_G$$에 저장된다. 이전에 나왔던 논문 중 [HittER: Hierarchical transformers for knowledge graph embeddings](https://meaningful96.github.io/paperreview/HittER/)을 통해 알 수 있는 중요한 사실이 하나 있다. 바로 <u>엔티티-릴레이션(Entity-Relation)쌍에 저장된 정보가 중요하다는 것이다.</u> 이러한 사실을 바탕으로 논문에서는 <span style="color:red">**엔티티-릴레이션 쌍을 Plain token으로 표현하고 릴레이션을 contextualized sub-graph의 special node**</span>로 간주한다. 이러한 방식으로 엔티티-릴레이션, 엔티티-엔티티 및 릴레이션-릴레이션 쌍을 포함한 노드 쌍 정보를 얻을 수 있다. 이렇게 함으로서 결론적으로 **릴레이션 노드를 special node**로 볼 수 있다는 것이다.

<br/>

### 3) Global Node

Triple2Seq는 결국 Contextualized Sub-graph를 통해 Locality를 뽑아낸다. 이럴 경우 global information에 대한 정보가 부족할 수 있다. 따라서 논문에서는 <span style="color:red">**Global node**</span>의 개념을 도입한다. global node는 쉽게 말하면 임의의 새로운 엔티티를 만들어 training set에 존재하는 모든 엔티티와 1-hop으로 연결시켜놓은 것이다. 즉 모두와 1-hop으로 연결된 엔티티이다. 하지만, 논문에서는 global node를 training set 전체에다가 연결시킨 것이 아닌, <span style="color:red">**추출된 Sub-graph에 있는 모든 엔티티와 연결된 엔티티를 의미**</span>한다.

<span style="font-size:110%"><b>Remark 1.</b></span>  
> Triple2Seq는 Input Sequence를 만들기위해 contextualized sub-graph를 dynamic sampling한다.
> 결과적으로 Transformer는 Large KG에 대해서도 쉽게 적용될 수 있다.
> Relphormer는 Heterogeneous graph에 초점을 맞춘 모델이며,
> sequential modeling을 위해 문맥화된 하위 그래프(Contextualized sub-graph)에서 edge(relation)를 하나의 Special node로 취급한다.
> 게다가, Sampling process는 성능을 향상시키는 data augmentation operator로 볼 수 있다.
>
> Note that with Triple2Seq, which dynamically samples
> contextualized sub-graphs to construct input sequences, Transformers
> can be easily applied to large knowledge graphs. However,
> our approach focuses on heterogeneous graphs and regards edges (relation)
> as special nodes in contextualized sub-graphs for sequential
> modeling. Besides, the sampling process can also be viewed as a data
> augmentation operator which boosts the performance.


## 2.2 Structure enhanced self-attention 

<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/de87859d-492d-48ee-883e-94e98157a9c8">
</p>

### 1) Attention Bias

트랜스포머는 입력으로 Sequence를 받는다. 이 때, <span style="font-size:105%"><b>Sequential Input가 Fully-Connected Attention Mechanism을 거치면서 Structural Information을 유실</b></span>시킬 수 있다. 그 이유는 Fully-Connected 라는 것은 결국 Dense-layer의 형태이다. 즉, Neural Network를 예로 들면 모든 drop-out이 0인 상태인데 <u><b>한 노드에 대해 다른 모든 노드들과의 attention을 구하므로(구조와 상관없이 모든 노드를 상대하기 때문) 구조 정보가 반영되지 못하는 것</b></u>이다.

이를 극복하기 위해 논문에서는 <span style="color:green">**Attention Bias**</span>를 추가로 사용하는 방식을 제안하였다. Attention bias를 통해 <span style="color:red"><b>Contextualized Sub-Graph 안의 노드쌍들의 구조 정보(Structural information)을 보존</b></span>할 수 있다. 노드 $$v_i$$와 $$v_j$$사이의 attention bias는 <b>$$\phi(i,j)$$</b>로 표기한다.

<p align="center">
<img width="700" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/d87d1256-1235-46d0-af2f-e22cd9625f83">
</p>

Triple2Seq에서 샘플링된 Contextualized Sub-graph의 구조 정보는 인접 행렬(Adjacency Matrix) $$A_G$$에 저장된다. 이 때, Sub-graph의 구조 정보를 Normalization한 값을 <b>$$\widetilde{A}$$</b>으로 표기한다. 이러한 사실을 바탕으로 <b>$$\widetilde{A^m}$$($$\widetilde{A}$$ to the $$m$$-th power)</b>를 정의한다. 이는 $$\widetilde{A}$$를 **m번 제곱**한 것(**행렬곱**을 m번 수행 <span style="font-size:80%">ex)$$ \widetilde{A} \; @ \; \widetilde{A}$$</span>) m은 hyperparameter이다. 

여기서 알아야 할 개념이 있는데, 인접 행렬(Adjacency Matrix)의 제곱, 세제곱 등이 가지는 의미이다. 제곱을 예로 설명하면, m = 2인 상황으로, 어떤 노드 $$v_i$$에서 또 다른 노드 $$v_j$$로 이동할 때 2번(m=2)움직여서 갈 수 있는 횟수를 의미한다. 즉, 노드를 순회할 때, m번 이동하여 갈 수 있는 경우의 수가 각 $$\widetilde{A^m}$$의 요소가 된다. 간단하게 코드를 통해 살펴보면 다음과 같다.

<p align="center">
<img width="600" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/73c55cdf-79f8-47d3-a053-7d0d60b8b9ab">
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
<img width="600" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/506fa430-1dd8-48a7-98d0-35fa2c8003b2">
</p>

앞의 실험을 통해 $$\widetilde{A^m}$$이 확실하게 정의되었고, $$f_{structure}$$는 구조 정보를 인코딩하는 **Linear Layer**로 $$\widetilde{A^m}$$을 입력으로 한다. 이를 통해 최종적으로 Attention bias인 $$\phi(i,j)$$가 정의된다. 다시 한 번 강조하지만, <span style="color:red">**Attention bias는 샘플링된 Contextualized Sub-Graph의 구조 정보를 포착**</span>하는 역할을 하며, 이 부분이 이 논문의 가장 큰 Contribution중 하나이다.

<br/>

### 2) Contrastive Learning Strategy

Dense한 Knowledge Graph(WN18RR보다는 FB15k-237이 relation의 종류가 더 많으므로 더 Dense함)의 경우, 하나의 Center Triple에 대해 많은 수의 Contextualized Sub-Graph를 샘플링하여 학습을 진행하면 정보의 편향이 생길 수 있다. Dynamic Sampling을 할 때, 이런 불안정성을 극복하고자 **Contextual contrastive strategy**를 이용한다. 

> We use the contextualized sub-graph of the same triple in different epochs triple in different epochs to enforce the model to conduct similar predictions.

즉, <span>서로 다른 Epoch에서 같은 Triple의 contextualized sub-graph를 사용해 모델에게 유사도를 학습하는 걸 강요한다. Contextualized sub-graph의 Input Sequence를 인코딩하고 hidden vector $$\mathcal{h_{mask}}$$를 현재 Epoch $$t$$에서 $$c_t$$, 마지막 Epoch $$c_{t-1}$$로 가져온다. 마지막으로 Loss를 정의하는데 $$\mathcal{L_{contextual}}$$를 contextual loss로 정의한다. 이 작업을 거쳐 최종 목표는 <span style="color:red">**서로 다른 sub-graph사이의 차이를 최소화(minimization)하는 것**</span>이다. 

<p align="center">
<img width="700" alt="1" src="https://github-production-user-asset-6210df.s3.amazonaws.com/111734605/252900409-5e4d57c7-c3d3-4692-8acb-c33f8a5bc005.png">
</p>

수식에서도 보이듯이, Contrastive Learning의 형태를 따르며 주의할 것은 빨산색 부분이다. 분모는 $$v_i$$ 노드의 t와 t-1시점의 유사도와 t시점에서 $$v_i$$노드와 $$v_j$$노드의 유사도의 합이다. 이를 통해 알 수 있는 것은 학습의 방향이 서로 다른 sub-graph의 차이를 줄이는 방향으로 간다는 것이다.

<span style="font-size:110%"><b>Remark 2.</b></span>  
> 이 Structure-enhanced Transformer는 모델에 구애받지 않으며 따라서 트랜스포머 아키텍처에 의미론적(semantic) 및 > 구조적 정보(Structure Information)를 주입하는 기존의 접근 방식과 직교한다는 점에 주목해야 한다. 
> Original Graph에서 말 그대로 기존 트랜스포머는 모든 노드에 대한 attention을 수행하므로 구조 정보가 반영되지 못한다.
> 하지만, Structure-Enhanced Transformer를 사용하면 Local Contextualized Sub-graph의 구조와 의미론적 특징의 영향력을 활용할 수 있는 유연성을 제공한다. Local graph 구조에서 유사한 노드간의 정보 교환에 편리하다.
>
> It should be noted that our structure-enhanced Transformer
> is model-agnostic and, therefore, orthogonal to existing approaches,
> which injects semantic and structural information into the
> Transformer architecture. In contrast to [34] where attention operations
> are only performed between nodes with literal edges in the
> original graph, structure-enhanced Transformer offers the flexibility
> in leveraging the local contextualized sub-graph structure and influence
> from the semantic features, which is convenient for information
> exchange between similar nodes in the local graph structure.



좀 더 쉽게 말하자면, 기존의 atttention operation은 단순히 전체 그래프 안에서 노드와 의미있는 relation사이에서 계산을 진행하는것에 반해, Structure-enhances self attention은 <span style="color:red">**Contextualized Sub-graph 구조를 이용한 Locality 정보와 Semantic feature들에 대해도 유의미한 영향을 주는 유연성을 이끌어내며 이를 통해 Transformer 모델에 구조적 정보(Structural information)와 의미론적 정보(Semantic feature)를 동시에 줄 수 있다**</span>는 것이 특징이다.

## 2.3 Masked Knowledge Modeling

Masked Knowledge Modeling은 특별한 것이 아닌, Masked Langauge Modeling(MLM)과 유사하다. Knowledge Graph Completion은 $$f: \mathcal{T_M}, A_G \; \rightarrow \; Y$$를 푸는 Task이다. Relphormer에서는 랜덤하게 Input Sequence의 토큰들을 마스킹하고, 그 마스킹된 토큰들을 예측한다.

> We randomly mask specific tokens of the input sequences and then predict those masked tokens.

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/2dfca278-beee-49b2-b3c1-f1156768b771">
</p>

Input Contexturalized Sub-Graph node sequence $$\mathcal{T_G}$$가 주어졌을 때, 랜덤하게 Center triple을 마스킹한다. 구체적으로 relation prediction을 할 때는 head나 tail 둘 중에 하나를 마스킹한다. 이를 Triple로 표현하면 $$(\; v_{h},?,[MASK] \;) \; or \; (\; [MASK], ?, v_t \;)$$이다. 

<p align="center">
<img width="300" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/cd798629-c5fd-45d8-8f8b-cf7be1837f85">
</p>

$$Y$$는 Candidate(후보)이다. Masked Knowledge Modeling이 궁극적으로 풀고자 하는 것은 <b>마스킹된 node sequence $$\mathcal{T_M}$$과 Contextualized Sub-graph의 구조 정보를 나타내는 $$A_G$$가 주어졌을 때 Original Triple $$\mathcal{T}$$의 missing part를 찾는 것</b>이다. 참고로, Y의 shape은 ($$Y \in \mathbb{R^{\vert \mathcal{E} \vert \times \vert \mathcal{R} \vert}}$$)이다.

구체적으로, <span style="color:red">**Contextualized Sub-Graph의 유니크한 구조정 정보를 이용해 Contextual information을 더 잘 통합하기위해 Sequence에서 단 하나의 토큰만 랜덤하게 마스킹**</span>한다. 

직관적으로 masked knowledge modeling은 scoring function을 기반으로 하는 이전 translation distance 방법들과는 확연한 차이를 보여준다. 다만, <u>마스킹을 할 때 Head와 Tail의 인접한 노드(이웃 노드)를 동시에 샘플링할 경우 Link prediction시 심각한 <b>Label leakage</b>를 유발</u>할 수 있다. 주의해야 할 점은 바로 Training 중에는 예측된 마스크 토큰의 구조를 알 수 없다는 것이다. 논문에서는 이 Label Leakgage를 극복해 공평평한 비교(fair comparison)를 보장하기위해 <span style="color:red">**타겟 엔티티(Target entity)의 context node를 제거**</span>하여 training과 testing의 차이를 좁혀준다.    

<span style="font-size:110%"><b>Remark 3.</b></span>  
> Masked Knowledge Modeling은 더 좋은 Link preidction을 위해 적절한 최적화 target을
> 자동으로 찾을 수 있는 매개 변수 scoring function의 근사치일 수 있다.
>  
> The advancement of empirical results (See section 4)
> illustrates that masked knowledge modeling may be a parametric
> score function approximator, which can automatically

## 3. Training and Inference

### 1) Pseudo Code
<span style="font-size:110%"><b>Hypothesis 1.</b></span>  
(Score function approximator) $$\mathcal{T_M}$$을 masked triple이라고 할 때, $$\mathbf{h} \; \in \; \mathbb{R^d}$$는 Relphormer $$\mathcal{M}(\theta)$$에서 multi-head attention을 통해 얻어진 마스킹된 head이다. Vocabulary 토큰 임베딩은 $$W \; \in \; \mathbb{R^{d \times N}}$$이며 $$N \; = \; \vert \mathcal{E} \vert \; + \; \vert \mathcal{R} \vert $$이다. 

Pseudo Code는 다음과 같다.

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/4d2b0251-2e6c-49c8-bfac-87ada18edda3">
</p>

만약 $$\mathcal{T_M} = (v_s, v_p, [MASK])$$이면 tail이 마스킹된 트리플을 의미한다. $$g(\cdot)$$은 multi-head attention layer 모듈이고 $$V_{object} \; \subset \; W$$은 tail이 될 수 있는 후보들의 임베딩을 의미한다. Output Logit은 $$sigmoid(W \mathbf{h})$$이며 이는 근사적으로 $$sigmoid(V_{object} \mathbf{h})$$와 동일하다.

<p align="center">
<img width="300" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/84820e60-7883-4948-837d-6ad26f116f97">
</p>

최종적인 final logit은 위와같다. 하나의 $$f(v_s, v_p,v_{object})$$ term을 고른 후 $$f( \cdot ) \; \approx \; v_{object_i} \, g( \cdot )$$을 이용한다. 이렇게 함으로써 f가 결국 <u>score function처럼 동작하게 되며 결론적으로 Masked knowledge modeling이 score function approximator</u>처럼 된다.

<br/>

### 2) Training and Inference

결론적으로 위의 Pseudo code algorithm과 같이 Relphormer는 동작한다. 학습 중에는 joint optimization을 이용해 masked knowledge loss와 contextual contrastive constrained object를 동시에 최적화한다. 따라서 최종 Loss는 아래와 같이 정의된다. $$\lambda$$는 hyperparameter이고 $$\mathcal{L_{MKM}}$$과 $$\mathcal{L_{contextual}}$$은 각각 masked knowledge loss와 contextual loss이다.

<p align="center">
<img width="300" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/dfb55d7e-c856-45da-ada2-dbc0360520b8">
</p>


<br/>

<span style="font-size:105%"><b>KG Completion</b></span>  
**추론시(Inference)**, multi-sampling strategy를 이용해 예측의 안정성을 향상시킨다. 이 때, $$\mathbf{y_k} \in \mathbb{R^{\vert V \vert \times 1}}$$의 shape을 가지며 하나의 Contextual sub-graph의 예측 결과를 나타내며, $$K$$는 샘플링된 sub-graph의 수를 나타낸다.

<p align="center">
<img width="100" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/dd015b54-677a-4eed-80c3-929cd002be43">
</p>


<br/>

<span style="font-size:105%"><b>Question Answering and Recommendation</b></span>  
Relphomrer에 fine-tuning을 하여 QA task와 추천 시스템에 적용하였다. QA task의 수식은 아래와 같다. $$\mathcal{Q_M}$$은 마스킹된 query이고 $$\mathcal{M(\theta)}$$는 pre-trained된 KG transformer이다. downstream task에 따라서 $$\mathcal{Q_M}$$의 표현은 조금씩 달라질 수 있다. (QA = Question Answering, RS = Recommandataion System)

<p align="center">
<img width="170" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/4381859f-bf18-4662-8aed-dd2a093cbd5a">
</p>


- In QA: $$\mathcal{Q_M}$$ is defined by \[ *question tokens*; \[MASK\] \] = KG에서 정답인 엔티티를 예측
- In RS: $$\mathcal{Q_M}$$ is defined by \[ *items tokens*; \[MASK\] \]

<br/>

<span style="font-size:105%"><b>Model Time Complexity Anaylsis</b></span>  
KG-BERT와 Relphormer의 성능을 비교하기에 앞서 먼저 Time Complexity를 비교하는 실험을 진행하였다. Relphormer의 경우가 KG-BERT에 비해 훨씬 더 좋은 Time Complexity를 보이며 학습과 추론시간에 있어서 차이가 많이 나는 것을 확인할 수 있다. Relphormer는 Masked knowledge modeling을 이용하여 모델이 마스킹된 엔티티나 릴레이션을 예측한다. 비록 Triple2Seq에서 시간이 좀 오래 걸리지만, Relphormer가 여전히 KG-BERT에 비해 우수한 성능을 보인다.

<p align="center">
<img width="600" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/3d1c546c-68a4-42c9-a03d-9792eb2493b8">
</p>

<br/>
<br/>

# Experiment & Result
총 6개의 Banchmark Dataset을 사용
- Knowledge Graph Completion(KGC)
  - WN18RR
  - FB15k-237
  - UMLS   
- Knowledge-Base Qusetion Answering
  - FreeBaseQA
  - WebQuestionSP
- Recommandation
  - MovieLens      

## 1. KG Completion & Relation Preidction

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/f4a94433-1443-48b7-b761-c3b00a61db53">
</p>

Table 3에서 방식이 Relphormer가 baseline들과 비교하여 모든 Dataset에서 경쟁력 있는 성능을 달성할 수 있음을 보여준다. Relphormer는 Hits@1 및 MRR Metric에서 최고의 성능을 달성하고 WN18R에서 Hits@10에서 두 번째로 우수한 성능을 산출했다. QuatE와 같은 이전 SOTA 변환 거리 모델과 비교하여 모든 Metric에서 개선되었다. Relphormer가 WN18R에서 SOTA Transformer 기반 모델 HitER보다 우수하다.

또한 FB15K-237 데이터 세트에서 Relphormer가 대부분의 번역 거리 모델보다 성능이 우수하다. 트랜스포머 기반 모델과 비교했을 때 Relphormer는 Hits@1에서 KG BERT, StAR 및 HittER보다 성능이 가장 우수하다. HitER는 **FB15K-237에서 더 많은 트랜스포머 아키텍처를 명시적으로 활용**하기 때문에 성능을 향상시키며, Relphormer는 여전히 비슷한 성능을 얻습니다. 게다가, 우리는 Relphormer가 UMLS, 특히 Hits@10에서 매우 좋은 성능을 보여준다. Relphormer의 Relational Transformer Framework가 KGC에서 우수한 성능을 만들어낸다. 

<p align="center">
<img width="600" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/c30d94bd-eb04-4491-b911-b0d4a2d3a9db">
</p>

Table 4에서 Relphormer가 baseline들과 비교하여 경쟁력 있는 성능을 얻을 수 있음을 알 수 있다. WN18RR 데이터 세트에서 Relphormer는 이미 모든 baseline을 능가하며, 이는 relation prediction을 위한 Relphormer의 접근 방식이 성능 향상에 직접적임을 보여준다. TransE와 비교하여 Hits@1에서 15.8%, 9.7% 향상되었다. FB15K-237에서 Relphormer의 성능 향상은 특히 Hits@3에서 중요하다. Relphormer는 DistMult보다는 성능이 우수하지만 RotatE보다는 성능이 떨어진다.

## 2. Question-Answering & Recommandation

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/19fd7be1-9a8b-4b89-8359-ed466cc48e1e">
</p>

QA의 경우, Relphormer가 FreebaseQA 및 WebQuestionSP에서 가장 좋은 성능을 보인다.(Figure 3.) HitER에 비해 Relphormer는 FreebaseQA Dataset의 *Full Setting*에서 6.8% 향상되었다. 또한, Relphormer는 WebQuestionSP의 *Full & Filter Setting*에서 2.9% 및 1.4% 향상되었다. Relphormer는 BERT로 쉽게 초기화할 수 있으며 masked knowledge modeling으로 최적화되므로 QA 작업을 위해 <span style="color:red">**Relphormer로 사전 훈련된 표현을 주입하는 것이 효율적이므로 성능이 향상**</span>된다. Relphormer가 훨씬 더 효율적이며, HitER와 같은 일부 KG 표현 모델의 경우 QA 작업을 향상시키기 위해 복잡한 통합 전략을 설계해야한다.

한 가지 간단한 방법은 Pre-trained representation을 Extra QA 모델에 주입하는 것이다. 하지만 Pre-trained된 KG 모델과 Downstream 모델 간의 불일치로 인해 효과를 검증하기가 어렵다. Freebase에 풍부한 Textural 및 Structual information이 있는 FreeBaseQA(Figure 4.)를 이용한 hard sample을 통해 Relphormer가 서로 다른 단어 또는 엔티티 간의 명시적이고 암묵적인 상관관계를 배울 수 있다는 것에 주목해야한다.

<br/>

Recommandation의 경우 Relphormer가 모든 baseline 모델보다 성능이 우수하다(Figure 3). BERT4REC와 비교하여 Relphormer는 Hits@1에서 2%, MRR에서 1% 향상되었다. 또한 Relphormer는 각 노드의 BERT 임베딩 계산 및 aggregation에 의해 구현되는 KG-BERT를 능가한다. Figure 4에서 볼 수 있듯이, 특정 사용자가 긴 목록의 영화를 시청한 경우, 여기서 목표는 다음에 시청할 영화를 예측하는 것이다. 그 영화들 중 *Sleepless in Seattle* 과 *Tin Cup* 은 두 영화의 주제가 모두 로맨스와 코미디에 관한 것이기 때문에 밀접한 상관관계가 있다. 한편, 추가된 KGs의 영화 *Mighty Aphrodite* 도 같은 이유로 *Sleepless in Seattle* 과 *Tin Cup* 에 연결되어 있다.

분명히, <span style="color:red">**KG의 잠재적 노드 관련성은 영화 추천 작업에 도움이 된다**</span>. 이러한 샘플의 경우, Relphormer는 깊은 상관관계를 학습하고 기준 모델보다 더 나은 성능을 산출할 것이다. 전반적으로, Relphromer를 사용한 KG 표현이 Link prediction을 통해 더 나은 본질적인 평가 성능을 수행할 수 있을 뿐만 아니라 잘 학습된 Knowledge Representation을 통해 QA 및 Recommandation의 KG-based downstream task를 촉진할 수 있음을 보여준다.

## 3. Ablation Study 1

<span style="font-size:110%"><b>How do different key modules in the Relphormer framework contribute to the overall performance?</b></span>

### 1) Optimization Object

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/4f38bf80-0c3e-467c-8aac-5a1d33ec6bc0">
</p>

Knowledge Graph에는 Relational Pattern이 있으며, 1-N, N-1 및 N-N 관계와 같은 패턴을 해결할 수 없는 접근 방식도 있다. 예를 들어 특정 엔티티-릴레이션 쌍(ℎ, 𝑟)이 주어지면 일반적으로 tail의 수는 둘 이상이다. Masked Knowledge Modeling(MKM) 없이, 대신 negative log-likelihood을 사용하여 ablation study를 수행하였다. Table 5에서 MKM이 있는 모델은 두 데이터 세트 모두에서 Hit@1에서 더 나은 성능을 낼 수 있지만 WN18R에서 MR의 향상을 달성하지 못한다. 이는 <span style = "color:red">**WN18RR에 충분한 구조적 특징이 없기 때문**</span>일 수 있다. 따라서 rank에 대한 **NLL 기반의 최적화가 더 유리**할 수 있다.

<br/>

### 2) Global Node
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/faab45a3-838e-4616-a8c4-eaa02a93a13b">
</p>
이 실험에서는 global node에 영향력에 대해서 실험한다. *w/o global node*는 global node가 없는 모델을 나타낸다. Figure 5.에서 baseline 모델들과 비교했을 때, global node가 없으면 안좋은 성능을 보여주었다. 이를 통해 <span style="color:red">**global node는 global information을 보존하는데 좋은 솔루션임을 입증**</span>한다.

<br/>

### 3) Case Analysis

Table 5.에서 보여지듯, FB15k-237의 몇몇 Triple을 통해 ablation study를 진행하였다. Relphormer의 경우 Structural information과 Textual feature를 동시에 다뤄 엔티티와 릴레이션 레벨에서의 다양성 문제(Heterogeneity Probelm)를 풀 수 있다. 예를 들어, Graph Embedding 모델 중 하나인 RotatE와 비교하면 RotatE는 그래프의 구조 정보만을 가지고 학습한다. 따라서 Graph의 Textual information에 대해 학습하지 못해 성능이 Relphormer에 비해 뒤쳐진다. 대조적으로 오직 Textual Encoding만을 가지고 학습한 [StAR](https://meaningful96.github.io/paperreview/5StaR/)는 그래프의 Context 정보만을 이용한다. 따라서, 그래프의 구조 정보가 반영되지 못해 Relphormer에 비교하면 성능이 떨어진다.

 ➜ StAR는 Texture Encoding 기반의 모델이지만, StAR(Self-Adp): 앙상블을 통해서 더 좋은 성능을 보여준다. 과연 StAR(Texture Encoding)으로 성능 비교를 하는게 합리적인가?

<br/>

## 4. Ablation Study 2

<span style="font-size:110%"><b>How effective is the proposed Relphormer model in addressing heterogeneity KG Structure and semantic textual description?</b></span>

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/f7b1f539-86a4-4d98-9612-ce15079f2aaa">
</p>

### 1) The number of sampled contextualized sub-graph triples

Figure 6. Triple의 수를 4부터 64까지 바꿔가며 성능을 측정하는 실험을 진행하였다. 그 수가 작을 때는 Center triple과 연결된 적은 수의 contextual node들이 샘플링된다. 이 실험을 통해서 명확한 것은, <span style="color:red">**contextualized subgraph의 크기를 키우는 것이 성능 향상에 직접적인 영향을 준다**</span>는 것이다. 하지만, 만약 Triple의 수가 너무 많아지면 성능은 일정 수준 올라가다가 **Saturation**된다. 논문에서 저자는 그 이유가 neighborhood의 정보가 유용하더라도, <u>너무 많은 unrelated된 정보들이 일종의 noise로 작용</u>해 성능에 부정적인 영향을 끼친다는 것이다. 이런 noise역할을 하는 low-quality node들은 negativ contextual information으로 작용한다.

<br/>

### 2) Structure-Enhanced self-attention

Structure-Enhanced attention의 영향력을 증명하기위해 Figure 5.의 실험을 진행하였다. 모든 모델이 Structure-Enhanced attention을 하지 않으면 성능이 떨어지는 것을 볼 수 있다. 저자는 무작위로 랜덤하게 예시를 가져와 attention matrix를 시각화해 structure-enhanced self-attention의 영향력을 실험하였다. Figure 7.에서와 같이 Structure-Enhanced Self-Attention은 attention weight 결과에 영향을 주는 것을 확인할 수 있다. 구체적으로, <span style="color:red">**Structure-Enhanced Self-Attention과 함께 구조 정보를 주입하는 것은 엔티티들의 거리에 의미적인 상관관계를 포착**</span>한다. 예를 들어, 한 엔티티는 sub-graph내 멀리 떨어진 엔티티를 통해 Structure correlation을 학습할 수 있다.

<br/>
<br/>

# Contribution

1. Transformer 기반의 새로운 모델인 Relphormer를 제안
2. 6개의 Benchmark Dataset에 대하여 기존의 Graph Embedding 모델들과 Transformer 기반 모델들에 비해 우수한 성능을 보여줌
3. <span style ="color:red">Attention bias를 이용해 그래프의 구조적 정보를 보존하고 Knowledge Graph에 적합한 Self-attention mechanism을 제시(**Structure enhanced self-attention**)</span>
    - 특히<span style = "color:green"> $$ \phi(i, j)$$</span>를 제시한 Structure-enhanced Self-attention이 가장 큰 Contribution

# Reference
[Inductive Bias란 무엇일까?](https://re-code-cord.tistory.com/entry/Inductive-Bias%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%BC%EA%B9%8C)  
["Relational Graph Transformer for Knowledge Graph Completion-Relphormer"](https://meaningful96.github.io/paperreview/Relphormer/)  

