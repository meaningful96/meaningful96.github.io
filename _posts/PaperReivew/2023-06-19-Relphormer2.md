---
title: "[논문리뷰]Relphormer: Relational Graph Transformer for Knowledge Graph Representation"

categories: 
  - PaperReview
  
tags:
  - [KG Completion]
  
toc: true
toc_sticky: true

date: 2023-06-19
last_modified_at: 2023-06-19
---

Bi, Z. (2022, May 22). *Relphormer: Relational Graph Transformer for Knowledge Graph Representations*. arXiv.org. https://arxiv.org/abs/2205.10852

이번 포스팅은 3월 14일 포스팅된 ["Relational Graph Transformer for Knowledge Graph Completion-Relphormer"](https://meaningful96.github.io/paperreview/Relphormer/)의 업데이트 버전이다. 논문 버전이 수정되면서 Ablation Study가 추가되었다.

# Problem Statement

일반적인 그래프와는 다르게 Knowledge Graph는 노드 또는 릴레이션의 타입이 여러가지인 Heterogeneous Graph이다. 자연어 처리 분야에서 Transformer가 압도적인 성능을 보여주면서 Computer Vision등의 여러 분야에 접목하려는 실험이 진행되는 중이다. 마찬가지로 Transformer모델이 Knowledge Graph에도 적용하려는 시도가 있었다.

Transformer는 그래프에 적용하면(i.e., KG-BERT) 모든 노드들의 Attention을 통해 관계를 파악하는 것을 목표로 한다. 하지만, 이럴 경우 그래프에서 중요한 정보 중 하나인 <span style="color:gold">**구조 정보(Structural Information)**</span>를 제대로 반영하지 못한다. 본 논문에서는 3가지 문제점을 제시한다.

<span style ="font-size:110%"><b>Heterogeneity for edges and nodes</b></span>  
먼저 **Inductive Bias**라는 개념을 알아야한다. 일반적으로 모델이 갖는 일반화의 오류는 불안정하다는 것(Brittle)과 겉으로만 그럴싸 해 보이는 것(Spurious)이 있다. 모델이 주어진 데이터에 대해서 잘 일반화한 것인지, 혹은 주어진 데이터에만 잘 맞게 된 것인지 모르기 때문에 발생하는 문제이다. 이러한 문제를 해결하기 위한 것이 바로 Inductive Bias이다. **Inductive Bias**란, <u>주어지지 않은 입력의 출력을 예측하는 것이다. 즉, 일반화의 성능을 높이기 위해서 만약의 상황에 대한 추가적인 가정(Additional Assumptions)이라고 보면 된다.</u> 

- Models are Brittle: 아무리 같은 의미의 데이터라도 조금만 바뀌면 모델이 망가진다.
- Models are Spurious: 데이터의 진정한 의미를 파악하지 못하고 결과(Artifacts)와 편향(Bias)을 암기한다.

논문에서는 <b>기존의 Knowledge Graph Transformer가 함축적인 Inductive Bias를 적용</b>한다고 말한다. 왜냐하면 KG-BERT의 경우 입력이 **Single-Hop Triple**로 들어가기 때문이다. 이럴 경우 1-hop 정보만 받아가므로 <span style = "color:gold">**Knowledge Graph에 구조적인 정보를 반영하는데 제약**</span>이 된다.

<br/>

<span style ="font-size:110%"><b>Topological Structure and Texture description</b></span>    
1번 문제와 비슷한 문제이다. 기존의 Transformer 모델은 모든 Entity와 Relation들을 plain token처럼 다룬다. 하지만 Knowledge Graph에서는 엔티티가 **위상 구조(Topological Structure) 정보와 문맥(Text Description) 정보**의 두 유형의 정보를 가지며 Transformer는 오직 Text description만을 이용해 추론(Inference)를 진행한다. 중요한 것은 **서로 다른 엔티티는 서로 다른 위상 구조 정보을 가진다**. 따라서, 마찬가지로 결국 기존의 <span style="color:gold">**Knowledge Graph Trnasformer 모델들은 필수적인 구조 정보를 유실**</span>시킨다.

<span style="font-size:120"><b>➜ How to treat heterogeneous information using Transformer architecture?</b></span>

<br/>

<span style ="font-size:110%"><b>Task Optimization Universalty</b></span>  

Knowledge Graph는 기존에 보통 Graph Embedding 모델들에 의해 task를 풀었다. 하지만 이 기존의 방식들의 비효율적인 면은 바로 Task마다 사전에 Scoring function을 각각 다르게 정의해주어야 한다는 것이다. 즉, 다른 <span style="color:gold">**Task object마다 다른 Scoring function을 필요**</span>로 하기 때문에 비효율적이다. 기존의 연구들을 다양한 Task에 대해 통일된 representation을 제시하지 못한다.

<span style="font-size:120"><b>➜ How to unite Knowledge Graph Representation for KG-based tasks?</b></span>


<br/>
<br/>

# Related Work



<br/>
<br/>

# Method



<br/>
<br/>

# Experiment & Result



<br/>
<br/>

# Contribution

# Reference
[Inductive Bias란 무엇일까?](https://re-code-cord.tistory.com/entry/Inductive-Bias%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%BC%EA%B9%8C)  
["Relational Graph Transformer for Knowledge Graph Completion-Relphormer"](https://meaningful96.github.io/paperreview/Relphormer/)  

