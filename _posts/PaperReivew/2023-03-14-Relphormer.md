---
title: Relphormer-Relational Graph Transformer for Knowledge Graph Representation

categories: 
  - PaperReview
  
tags:
  - [KG Completion]
  
toc: true
toc_sticky: true

date: 2023-03-14
last_modified_at: 2023-03-14
---

# 1. Problem Statement

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/224568729-8133ac85-90ce-4ca0-8ec4-c5e9a6cdc78a.png">
</p>

Pure Graph와는 다르게 Knowledge Graph는 여러 가지의 노드 타입이 존재하는 Heterogeneous Graph이다. Transformer가 Computer vision이나, Natural Language Process등의 분야에서 성공적이였다면 아직 **Knowledge Graph modeling에서는 적합한지 의문**이다. 구체적으로 두 가지 nontrivial challenge가 있다.

- **1. Heterogeneity for edges and nodes**
  
  -  Knowledge Graph는 일종의 Semantic-Enrich entity로 구성된 relational graph이다.
    - Multiple edges have different relational information
    
  - 또한 Knowledge Graph는 일종의 Text-Rich network이다.
    -  각각의 노드가 서로 다른 topological structure와 textual description을 갖는다.
    
  - Vanilla Transformer는 <span style = "color:gold">모든 엔티티와 릴레이션을 plain token으로 간주하기 때문에 필수적인 구조 정보가 유실</span>된다.
  
    
  
    ➜<span style = "font-size:120%"> **How to treat heterogeneous information using Transformer architecture?**</span>
  



- **2. Task Optimization University**
  
  - 기존의 연구들은 Knowledge Embedding을 위해 사전에 정의된 scoring function을 사용함.
    - Entity prediction과 Relation prediction에 서로 다른 Optimizing object를 사용(비효율적) 
    
  - 기존 연구들은 다양한 Task에 대해 통일된 Representation을 제시하지 못함.
  
    
  
    ➜ <span style = "font-size:120%">**How to unite Knowledge Graph Representation for KG-based tasks?**</span>
  

# 2. Method

1. KG representation을 통합시키기위함

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/224571721-9fe510b4-85f0-45a7-a91c-ce50a21defbc.png">
</p>
## 0) Preliminaries

Knowledge Graphs는 triple($$head, relation, tail$$)로 구성된다. 논문에서는 **Knowledge Graph Completion** Task와 **Knowledge Graph-Enhanced Downstream Task**를 푸는 것을 목표로 한다. 모델을 살펴보기 전 Notation을 살펴봐야 한다.

<p align="center">
<img width="500" alt="1" src="https://user-images.githubusercontent.com/111734605/224572006-9fcb2f52-8504-43c1-b8ef-b04e1cd4db07.png">
</p>

- 주의깊게 봐야할 Notation
  - Relational Graph $$G = (\mathscr{E}, R)$$
  - Node Set $$V = \mathscr{E} \; \cup \; R$$
  - Adjacency Matrix = 요소들이 [0,1] 사이에 있고, 차원이 $$ㅣVㅣ \times ㅣVㅣ$$

- Knowledge Graph Completion
  - Triple $$(v_{subject}, v_{predicate}, v_{object}) = (v_s, v_p, v_o) = T$$  
  - As the label set $$T$$, $$f: T_M,A_G \rightarrow Y$$, $$ Y \in \mathbb{R}^{ㅣ\mathscr{E}ㅣ \times ㅣRㅣ} $$ 로 정의된다.

## 1) Triple2Seq

Triple2Seq의 목적은 <span style = "color:gold">**Edge들의 Heterogeneity를 풀기 위함이다.(To solve heterogeneity of edges)**</span>  

모델의 Input sequence로 **Contextualized Sub-graphs**를 사용하는 방식이다. Contextualized sub-graph를 사용하여 local structure information을 집어넣을 수 있다. Contextualized Sub-Graph는 $$T_G$$이다.


<br/>
<br/>
<span style = "font-size:110%">$$(1) \; \; T_G = T \; \cup \; T_{context} $$</span>
<br/>
<br/>


이 때, $$T$$는 Center triplet이고, $$T$$의 이웃 노드 집합이 $$T_{context}$$이다.  즉, Contextualized Sub-Graph $$T_G$$는 center triplet과 그 이웃 노드들의 triplet으로 구성되어 있다.


<br/>
<br/>
<span style = "font-size:110%">$$(2) \;\;T_{context} = \{vㅣv = v_s \; or \; v_p \; or \; v_o, \; \exists \;(v_s, v_p, v_o) \; \in \; \mathscr{N} \}$$</span>
<br/>
<br/>


$$\mathscr{N}$$은 $$T$$의 고정된 크기의 이웃 triple의 집합이다.(fixed-size neighborhood triple set of the triple $$T$$)

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/224574321-29e04f94-3b8a-447f-b588-39af76603222.png">
</p>

Triple2Seq에서는 edges(relations)를 일반적인 노드로 여기며, 이렇게 만들어진 local structural information을 갖는 contextualized sub-graph를 모델의 input으로 넣는다. 또한, local structural feature를 더 잘 포착하기 위해 학습하는 동안 <span style = "color:aqua">**Dynamic Sampling Strategy**</span>를 사용한다. 이건 각 Epoch마다 동일한 center triple에 대하여 여러개의 contextualized sub-graph를 랜덤하게 선택하여 사용한다. 즉, 여러 개의 subgraph를 만들고 각 Epoch마다 다르게 사용하여 학습하는 것이다.

## 2) Structure-Enhances Self-attention

### 2.1) global node

Relphormer의 선행 연구인 HittER을 통해 Entity-Relation 쌍의 정보가 Knowledge Graph에 있어서 필수적인 정보라는 것이 밝혀졌다. 앞서 Contextualized Sub-graph를 사용한 Triple2Seq를 통해 Entity-Relation 쌍의 정보와 더불어 Entity-Entity, Relation-Relation쌍의 정보 또한 얻을 수 있다.(이게 가능한 이유는, Contextualized Subgraph에서 relation도 하나의 normal node로 보고 subgraph를 생성하기 때문이다.) 

Knowledge Graph에서 Relation의 수는 압도적으로 Entity수보다 훨씬 적기 때문에 Relation edge로 contextualized subgraph 사이의 globally semantic information을 유지할 수 있다. 수가 더 적기 때문에 각각의 위치가 나타내는 구조적 정보가 효력이 있는 것이다.

논문에서는 추가적으로 Global information을 보존하기 **global node**를 추가한다. global node는 자연어 처리의 pre-training 모델에서 [CLS] 토큰과 유사한 역할을 수행한다. 이 global node를 기존의 contextualized subgraph와 <span style= "color:aqua">학습가능한 가상의 거리(virtual distance) 또는 고정된 거리를 통하여 연결</span>한다.

<br/>
<br/>
<span style = "font-size:110%">$$(3) \; \; \{v_{cls}, v_1, v_2, \cdots, v_i\}$$</span>
<br/>
<br/>

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/224576605-64e7ef8a-a6d9-4312-8b77-fc62dfaab744.png">
</p>

### 2.2) Structure enhanced self attention

Transformer의 input은 Sequential 하고, 이러한 Sequential input의 구조적 정보는 Fully-connected attention을 하면 정보가 손실될 수 있다. 그 이유는 Fully-connected라는 것이 Dense-layer의 형태이고 모든 노드를 한 번에 분석하여 encode하는 것이기 때문에 Sequential input의 구조적 정보가 반영되지 못할 수도 있다는 것이다.

이를 극복하기위해 **Attention Bias**를 추가로 사용하는 방식을 제안하였다. Attention bias를 통해 노드쌍 사이의 구조적 정보를 포착할 수 있다

<br/>
<br/>
<span style = "font-size:110%">$$(4) \; \; a_{ij} =  \frac{(h_iW_Q)(h_jW_K)}{\sqrt{d}} + \phi(i,j), \; \; \; \; \phi(i,j) = f_{structure}(\tilde{A}^1, \tilde{A}^2, \cdots, \tilde{A}^m)$$</span>

- $$\phi(i,j)$$ : Attention bias between node $$v_i$$ and node $$v_j$$
- $$\tilde{A}$$ : Normalized adjacency matrix
- $$f_{structure}$$ : Linear layer
- $$m$$ : Hyperparameter
- $$\tilde{A}^m$$ : Reachable relevance by taking m-steps from one node to the other node

### 2.3) Contrastive learning strategy

모델을 학습하는 동안 **하나의 Center Triple에 대해서만 sub-graph를 사용하면 Inconsistency가 생긴다**. 즉, 하나의 중심 노드에 대한 하위 그래프들에 대해서만 학습되므로 전체적인 그래프의 정보에 대한 <u>비일관적이고 모순적인 정보가 가공</u>된다. 이러한 모순을 해결하기 위해 논문에서 Dynamic sampling을 하면서 동시에 **Contextual Contrastive Strategy**를 사용했다. 

Contextual contrrastive strategy는 모델이 비슷한 예측을 수행하도록 강제하는 것으로 Epoch마다 같은 중심 triple에 대해 다른 Contexualized sub-graph를 사용하는 전략이다. Contextual loss는 다음과 같다.

<br/>
<br/>
<span style = "font-size:110%">$$(5) \;\; \mathscr{L_{contextual} = -log\frac{exp(sim(c_t, c_{t-1}/\tau))}{exp(sim(c_t, c_{t-1}/\tau)) + \sum_{j}exp(sim(c_t, c_{j}/\tau))}}$$</span>

- $$sim(c_t, c_{t-1}/\tau)$$ = Cosine 유사도
- $$c_t$$ t 번째 epoch의 hidden state representation

Input sequence를 인코딩하고 난 후 hidden vector $$h_{mask}$$를 current epoch t에서의 contextual representation $$c_t$$로 취한다. <span style ="color:aqua">Contextual loss의 목적은 **서로 다른 sub graph들 사이의 차이를 최소화** 하는 것</span>이다. $$c_t$$는 다시 말해서 $$h_{mask}$$를 contextual representation 형태로 나타낸 것이고, 이는 다른 중심 트리플들에 속한 t-epoch에서의 hidden state representation이다.

기존의 atttention operation은 단순히 전체 그래프 안에서 노드와 의미있는 relation사이에서 계산을 진행하는것에 반해, <span style = "color:gold">Structure-enhances self attention은 **Contextualized Sub-graph** 구조를 이용한 Locality 정보와 Semantic feature들에 대해도 유의미한 영향을 주는 유연성을 이끌어내며 이를 통해 Transformer 모델에 구조적 정보(Structural information)와 의미론적 정보(Semantic feature)를 동시에 줄 수 있다</span>는 것이 특징이다. 

### 3) Masked Knowledge Modeling

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/224577501-d11de3de-c587-4b19-8832-e6567f4b8573.png">
</p>

기존의 Graph Embedding 모델들은 다른 Task를 풀 때 다른 Scoring function을 사용해야한다. Relphormer에서는 KG representation learning 방식으로 논문에서 **Masked Knowledge Modeling**을 제안한다. Input $$T_G$$를 랜덤하게 Masking한다. 단, Link prediction task와 relation prediction task에서 다른 방식으로 마스킹된다.

- Link Prediction  ➜ $$(v_{src}, v_p, v_{mask})$$ or $$(v_{mask}, v_p, v_{target})$$
- Relation Prediction  ➜ $$(v_{src}, v_{mask}, v_{target})$$

마스킹을 수식화하면 다음과 같다.

<br/>
<br/>
<span style = "font-size:110%">$$(6)\begin{align} \;\; &T_M = MASK(T_G)\\
&Relphormer(T_M, A_G) \rightarrow Y, Y \in \mathbb{R}^{|\mathscr{E}|\times |R|}
\end{align}$$</span>
<br/>
<br/>

Sequence에서 단 **하나의 토큰만 랜덤하게 마스킹**한다. 그 이유는 <span style = "color:gold">Contextualized Sub-graph 의 유니크한 구조적 정보로 Conxtextual information을 더 잘 통합</span>하기 위함이다 .  

다만, 마스킹을 한 후 head와 tail entity의 이웃들을 동시에 sampling하면 label leakage 문제가 발생할 수 있다. (만약 relation이 masking되고 동시에 head와 tail을 추출할경우 둘이 구분이 안될 수도 있다.)  Label leakage를 극복하고 Training과 Test의 간극을 줄이기 위해서는 <span style = "color:aqua"> **target entity의 context node를 제거**하여 공정한 비교(fair comparison)를 보장</span>할 수 있게 만든다.

Masked Knowledge Modeling은 매개 변수의, Parametric한 score function의 approximator이다. 이는 더 나은 Link prediction을 목표로 적합한 최적화값을 자동으로 찾아낸다.

### 4) Optimization and Inference

학습에는 Masked Knowledge loss와 Contrastive Learning Object를 같이 사용한다.(Joint Optimization)

<br/>
<br/>
<span style = "font-size:110%">$$(7) \;\;\mathscr{L}_{all} = \mathscr{L}_{MKM} +\lambda\, \mathscr{L}_{contextual}$$</span>

Reasoning(추론) 중에는 Multi-sampling strategy를 사용한다.

<br/>
<br/>
<span style ="font-size:110%">$$(8) \;\; \tilde{y} = \frac{1}{K} \displaystyle\sum_{k}\bf{y}_k $$</span>

- $$y_k \in \mathbb{R}^{ㅣVㅣ \times 1}$$ : 하나의 Contextualized subgraph의 예측 결과

### 5) Fine-tuning for KG-based Task

KBQA 같은 문제를 풀려면 Fine-tuning을 해야한다. 예를 들어 KBQA의 경우의 수식은 다음과 같다.

- $$ f: Q_M, M(\theta) \rightarrow Y$$ - Fine-tuning for Question Answering Task
<br/>
<br/>

### Pseudo Code

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/224578800-fc733695-8b05-4f43-b38e-6db50a1a2e26.png">
</p>
- Hypothesis (Score function approximator)
    - $$T_M$$이 tail entity가 마스킹된 triplet이고 $$g(\cdot)$$ 함수가 multy-head attention module을 나타내며 $$V_{object} \subset W$$ 인 Tail entity의 후보자 임베딩(candidate embedding)일 때 출력은 $$sigmoid(W\bf{h})$$이고 이는 근사적으로 $$sigmoid{(V_{object}\bf{h})}$$와 동일하다.
        - $$T_M$$: Masked Triplet
        - $$\bf{h}$$: $$h \in \mathbb{R}^d$$, Masked head derived from multi-head attention layer
        - $$W$$: Vocab token embedding, $$W \in \mathbb{R}^{d \times N}$$ & $$N = ㅣmathscr{E}ㅣ + ㅣ\mathscr{R}ㅣ$$ 
    - 수식화
        - <span style = "font-size:110%">$$ sigmoid\displaystyle\sum^{ㅣ\mathscr{E}ㅣ}v_{object_i}g(v_{object}, v_{predicate}, [MASK])$$ </span>이다.
    - 이 때 <span style = "font-size:110%">$$ f(\cdot) \approx v_{object_i}g(\cdot)$$</span>을 score function role로 정의한다. 
    - 이로써, <span style = "color:aqua">Masked knowledge Modeling은 일종의 score function approximator</span>가 된다.

# 3. Experiment

## 1) DataSet

총 6개의 Benchmark Dataset을 사용

- Knowledge Graph Completion(KGC)
  - WN18RR
  - FB15K-237
  - UMLS
- Knowledge-Base Question Answering
  - FreeBaseQA
  - WebQuestionSP
- Recommendation
  - MovieLens

## 2) Result of Knowledge Graph Completion(Link Prediction)

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/224579129-ec48487a-50b5-4d2d-8524-63695b75f869.png">
</p>

- Relphormer가 전체적으로 Translation distance model들보다는 성능이 우수

- **WN18RR**에서는 대체적으로 기존의 Transformer-based model들보다 우수

  - 단 rank 개수가 많아진다고 StAR에비해 비약적인 성능향상이 이루어지지 않음

    

- **FB15k-237**에서 rank의 개수가 낮을때(Hits@1) 다른 모델들에비해 성능이 가장 우수

  - 단, WN18RR과 마찬가지로 rank개수가 많아진다고 비약적인 성능 향상을 보이지 않음

  

- **UMLS**의 경우 가장 우수

## 3) Result of Relation prediction

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/224579674-6cdd9914-4efc-4912-92af-545364564ff4.png">
</p>

두 Dataset 모두 rank개수가 낮으면 Relphormer의 경우 성능이 우수한 것을 볼 수 있다. 

- **WN18RR** Dataset의 경우는 모든 평가지표에서 가장 좋은 성능을 보여줌
- **FB15K-237**의 경우 MRR과 Hit@1에서 가장 우수한 성능은 아니지만 전체적으로 성능이 우수한 것을 볼 수 있다. 또한 Hit@3에서는 가장 성능이 우수하다.

## 4) Question Answering

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/224579948-70c77d56-f8ab-4c16-9aa4-19e7956dc235.png">
</p>
- *Full~, Filter~* setting은 appendix에 있다.
- BERT와 HittER 모델에 비해서 QA Task에서 더 높은 정확성을 보여준다.

## 5) Recommendation

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/224579959-4112a68e-759d-4482-8a2e-cd2aeafef759.png">
</p>
## Ablastion Study 1 : Sub-graph의 수가 성능에 미치는 영향

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/224580077-8eae82fa-e9c3-4442-b611-d6d3716d26f4.png">
</p>

Subgraph의 개수가 4개에서부터 늘어날수록 점점 성능이 좋아지지만 16개 이후로는 결과가 수렴하게된다. 그 이유는, Subgraph의 개수가 아무리 많아져도 결국 Center triplet에 대해 중복된 정보만 생성되기에 성능향상이 일어나지 않는다.

## Ablation Study 2 : structure-enhanced self attention, optimization object, Global node

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/224580501-f267a1ce-18e4-4d40-8d64-3b85d386965b.png">
</p>
- Figure 4
    - Structure-enhanced attention이 없을 때 두 dataset모두 성능 저하가 눈에띄게 나타남
    - Structure-enhanced attention, optimization object, MKM, Global node가 하나씩 없을경우의 성능을 보여줌
- Figure 5
    - (L) Structure-enhanced self attetion 있을때
    - 구조 정보를 Structure-enhanced self attention을 통해 주입해 줌
    - 결과적으로 엔티티들의 거리에 대한 더 좋은 Semantic correlation을 포착함함

## Ablation Study 3 : Inference speed comparison

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/224580539-4ed139f1-43b3-415f-9552-9de66f365498.png">
</p>
- KG-BERT의 가장 큰 문제점은 Time Complexity가 너무 높다는 것이다.
- RelPhormer의 경우 성능면에서도 우수하지만, Time Complexity면에서도 우수함을 보여준다.

# 4. Contribution

1. Transformer 기반의 새로운 모델인 Relphormer를 제안
2. 6개의 Benchmark Dataset에 대하여 기존의 Graph Embedding 모델들과 Transformer 기반 모델들에 비해 우수한 성능을 보여줌
3. <span style ="color:gold">Attention bias를 이용해 그래프의 구조적 정보를 보존하고 Knowledge Graph에 적합한 Self-attention mechanism을 제시(Structure enhanced self-attention)</span>
    - 특히<span style = "color:gold"> $\phi(i, j)$</span>를 제시한 Structure-enhanced Self-attention이 가장 큰 Contribution이다.
