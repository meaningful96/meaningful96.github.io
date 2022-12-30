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
- Ideal Case: <*question, relation path* >
- In this Paper: <*question, answer* >

<span style = "font-size:120%">**What we need to solve?**</span>  
Intermediate Reasoning Step에 Supervision Signal을 통해 Feedback을 하여 더 잘 Training되게 한다.

## Method
- Teacher & Student Network
- Neural State Machine(NSM)
- Bidirectional Reasoning Mechanism

### 1. Teacher - Student Network
#### Overview  
```
The main idea is to train a student network that focuses on the multi-hop KBQA task itself, while another teacher
network is trained to provide (pseudo) supervision signals (i.e., inferred entity distributions in our task) at 
intermediate reasoning steps for improving the student network.
```
학생 네트워크는 multi-hop KBQA를 학습하는 한편, 선생 네트워크에서는 Intermediate Supervision Signal을 만들어 학생 네트워크로 넘겨준다.
이렇게 함으로써 학생 네트워크에서 더 학습이 잘되게끔 한다.

#### Student Network
선생-학생 네트워크에서 학생 네트워크(Student Network)가 Main model이다. 학생 네트워크의 목표는 Visual question answering으로부터 정답을 찾는 것이다. 
학생 네트워크에서는 NSM(Neural State Machine) 아키텍쳐를 이용한다.

<span style = "font-size:120%">**NSM(Neural State Machine)**</span>  

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/210039872-680ef240-219b-4a2c-9e81-421ab3d22fa5.png">
</p>
  
- Given an image, construct a 'Scene Graph'
- Given a question, extract an 'Instruction Vector'

Input으로 이미지에서 뽑아낸 Scene graph와, 질문에서 뽑아낸 Intruction vector가 Input으로 들어간다.

<span style = "font-size:120%">**Student Network Architecture**</span>    
Student Network은 NSM 아키텐쳐를 바탕으로 구성된다. NSM 아키텍쳐는 Scene Graph와 Instruction Vector를 각각 이미지와 질문으로부터 추출해내면 이걸 Input으로 받아 정답을 찾아내게
된다.

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/209019844-d2d7e641-295f-4721-b589-da131f5dde9d.png">
</p>

- Instruction Component
  1. Natural Language Question이 주어지면 이걸 Series of instruction vector로 바꾸고, 이 Instruction vector는 resoning process를 control한다.
  2. Instruction Component = query embedding + instruction vector
  3. instruction vector의 초기값은 zero vector이다.
  4. GloVe 아키텍쳐를 통해 query 단어들을 임베딩하고, 이를 LSTM 인코더에 넣어 Hidden state를 뽑아낸다.


The input of the instruction
component consists of a query embedding and an instruction vector
from the previous reasoning step. The initial instruction vector is
set as zero vector. We utilize GloVe [26] to obtain the embeddings
of the query words. Then we adopt a standard LSTM encoder to
obtain a set of hidden states 
## Related Work
- Knowledge Base Question Answering
- Multi-hop Reasoning
- Teacher-Student Network
