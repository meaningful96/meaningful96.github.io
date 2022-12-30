---
title: Improving Multi-hop Knowledge Base Question Answering by Learning Intermediate Supervision Signals 

categories: 
  - PaperReview
  
tags:
  - [KBQA]
  
toc: true
toc_sticky: true

date: 2022-12-22
last_modified_at: 2022-12-22 
---

## 1. 논문을 들어가기 앞서 알면 좋은 Basic Knowledge
- [Graph의 개념](https://meaningful96.github.io/datastructure/2-Graph/)
- [Cross Entropy, Jensen-Sharnnon Divergence](https://drive.google.com/file/d/18qhdvC_2B9LG7paPdAONARqj3DWxxa8h/view?usp=sharing)
- [Knowledge Based Learning](https://meaningful96.github.io/etc/KB/)
- [Reward Shaping](https://meaningful96.github.io/etc/rewardshaping/#4-linear-q-function-update)
- [Action Dropout](https://meaningful96.github.io/deeplearning/dropout/#4-test%EC%8B%9C-drop-out)
- [GloVe]()
- [BFS, DFS](https://meaningful96.github.io/datastructure/2-BFSDFS/)
- [Bidirectional Search in Graph](https://meaningful96.github.io/datastructure/3-Bidirectionalsearch/)
- [GNN](https://meaningful96.github.io/deeplearning/GNN/)
- [Various Types of Supervision in Machine Learning](https://meaningful96.github.io/etc/supervision/)
- [End-to-end deep neural network](https://meaningful96.github.io/deeplearning/1-ETE/)
- [NSM(Neural State Machine)]()

## 문제 정의(Problem Set)
### Lack of Supervision signals at Intermediate steps.
Multi-hop Knowledge base question answering(KBQA)의 목표는 Knowledge base(Knowledge graph)에서 여러 홉 떨어져 있는 Answer entity(node)를 찾는 것이다.
기존의 KBQA task는 <span style = "color:aqua">Training 중간 단계(Intermediate Reasoning Step) Supervision signal을 받지 못한다.</span> 다시말해, 
feedback을 final answer한테만 받을 수 있다는 것이고 이는 결국 학습을 unstable하고 ineffective하게 만든다.

<p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/111734605/210034900-0bceb022-2127-41b6-a52c-3c4a9512365d.png">
</p>

Figure 1.  
Qusetion: What types are the film starred by actors in the *nine lives of fritz the cat*?
- Start node(Topic Entity)  = 초록색 노드 
- Final Node(Answer Entity) = 빨간색 노드
- Answer Path    = 빨간색 Path
- Incorrect Path = 파란색 Path, 회색 Path

여기서 중간단계에서 Supervision signal이 부족할 경우 발생하는 경로가 바로 **파란색**이다. 논문에서는 이 경로를 Spurious fowrward path(가짜 경로)라 명칭했다. 

<span style = "font-size:120%">**참고**</span>  
KBQA task에서 Input data
- Ideal Case: <*question, relation path*>
- In this Paper: <*question, answer*>

## Method
- Teacher & Student Network
- Neural State Machine(NSM)
- Bidirectional Reasoning Mechanism

