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
- [NSM(Neural State Machine)](https://meaningful96.github.io/etc/NSM/)

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
<span style ="color:aqua">**Intermediate Reasoning Step에 Supervision Signal을 통해 Feedback을 하여 더 잘 Training**</span>되게 한다.
  
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
학생 네트워크는 multi-hop KBQA를 학습하는 한편, 선생 네트워크에서는 <span style ="color:aqua">Intermediate Supervision Signal</span>을 만들어 학생 네트워크로 넘겨준다.
이렇게 함으로써 학생 네트워크에서 더 학습이 잘되게끔 한다.

#### Student Network
선생-학생 네트워크에서 학생 네트워크(Student Network)가 Main model이다. 학생 네트워크의 목표는 Visual question answering으로부터 정답을 찾는 것이다. 
학생 네트워크에서는 NSM(Neural State Machine) 아키텍쳐를 이용한다.

##### (1) NSM(Neural State Machine)

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

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/210233075-7c40808e-0e59-4c22-981a-ce481268fd48.png">
</p>    
<center><span style = "font-size:80%">Student Network Equation Table</span></center>


##### (2-1) Instruction Component    
1. Natural Language Question이 주어지면 이걸 Series of instruction vector로 바꾸고, 이 Instruction vector는 resoning process를 control한다.  
2. Instruction Component 🡄 query embedding + instruction vector  
3. instruction vector의 초기값은 zero vector이다.  
4. GloVe 아키텍쳐를 통해 query 단어들을 임베딩하고, 이를 LSTM 인코더에 넣어 Hidden state를 뽑아낸다.    
   (Hidden State식 $$ h_l $$이고, $$l$$은 query의 길이)  

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/210257037-542d9aaa-ec19-46e6-be97-9a4d61354f16.png">
</p>     
<center><span style = "font-size:80%">Instruction Component</span></center>  

- query Embedding과 j번째 hidden state를 element wise product해서 Softmax를 먹인다.
  - $$q^{(k)}$$의 식은 Instruction vector에 weighted 처리된 것이다.
  - 즉, 가중치를 곱하여 처리한 것이다.
  - 그러면 Instruction vector에서 영향력 큰 부분만 뽑아내겠다.
  - 즉, query에 큰값이 있는걸 뽑아내는 것 

Insteruction vector를 학습하는데 가장 중요한 것은 매 Time step마다 query의 특정한 부분에 <span style = "font-size:110%">**Attention**</span>을 취하는 것이다.
이러한 과정이 결국 query representation을 동적으로 업데이트 할 수 있게되고 따라서 **이전의 Instruction vector들에 대한 정보를 잘 취합**할 수 있다. 얻은 Instruction
vector들을 리스트로 표현하면 $$[i_{k=1}^j]$$이다. 

##### (2-2)Attention Fuction이란?  

<p align="center">
<img width="" alt="500" src="https://user-images.githubusercontent.com/111734605/210244763-6df0807b-7e7f-4d4a-a73b-f100734ee83e.png">
</p>     
<center><span style = "font-size:80%">Instruction Component</span></center>

어텐션 함수는 Query, Key, Value로 구성된 함수이다.  
<center>$$Attention(Q,K,V) \; = Attention \, - \, Value $$</center>  
<center>
$$\begin{aligned}
Q &: Query  \\
K &: Key\\
V &: Value\\
\end{aligned}$$
</center>

어텐션 함수는 주어진 **'쿼리(Query)'**에 대해 모든 **'키(Key)'**의 유사도를 각각 구합니다. 그리고, 이 유사도를 키(Key)와 매핑되어 있는 각각의 **'값(Value)'**에 반영해줍니다. 그리고 '유사도가 반영된'값을 모두 더해서 리턴하고, 어텐션 값을 반환한다.

##### (3) Reasoning Component

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/210257533-069772df-1a82-4dca-9b02-bc8bcb8bfd00.png">
</p>     
<center><span style = "font-size:80%">Reasoning Component</span></center>  

Reasoning Component(추론 요소)를 구조와 그 수식은 위와 같다. 먼저, Instruction Vector $$i^{(k)}$$를 Instruction Component 과정을 통해 얻었고 이를 Reasoning Component에서
Guide Signal로서 사용가능하다. Reasoning Component의 Input과 Output은 다음과 같다.
- Input : **현재 step의 instruction vector** + **이전 step의 entity distribution와 entitiy embedding**
- Output: entity distribution $$p^{(k)}$$ + entitiy embedding $$e^{(k)}$$
  - Entity Embedding의 초기값인 $$e^{(0)}$$은 2번식이다.
  - $$\sigma$$는 표준편차를 의미(entity distribution이므로)
  - $$<e^{\prime}, r, e>$$는 Triple이라한다. 노드(Entity), 엣지, 노드 순서이다.

<span style = "font-size:110%">**(2)번 식 Entity Embedding의 초기값**</span>  
2번식을 자세히보면 Entity의 임베딩식은 결국 Weight Sum의 표준편차를 구한 것이다. 이전의 연구들과는 다르게 이 논문에서는 **엔티티를 인코딩하는데 <span style ="color:aqua">트리플(노드와 노드, 엣지로 표현된 Relation)의 정보</span>를 적극적으로 사용**한다. 게다가 이렇게 정보를 활용하면 **엔티티 노이즈에 대한 영향력이 줄어든다.** 추론 경로를 따라 중간 엔터티의 경우 이러한 엔터티의 식별자가 중요하지 않기 때문에 e(0)를 초기화할 때 e의 원래 임베딩을 사용하지 않는다. 왜냐하면 중간 엔티티들의 **relation**만이 중요하기 때문이다.

<span style = "font-size:110%">**(3)번 식 Match vector**</span>  
Triple($$<e^{\prime}, r, e>$$)이 주어졌을때 Match vector $$m_{<e^{\prime}, r, e>}^{(k)}$$는 (3)번 식과 같다. Instruction vector와 Edge(Relation)에 가중치를 곱한 값과 Element wise product한 값의 표준편차값이다. 이 식의 의미를 보자면, Match vector라는 것은 결국 <span style = "color:aqua">올바른 Relation을 나타내는, 올바른 Edge에 대해서 더 높은 값을 부여해 엔티티가 그 엣지를 따라가게끔 값을 부여하는 것</span>이다. 따라서, '올바른 Edge를 매칭한다'라는 의미로 Match vector라고 한다. 

<span style = "font-size:110%">**(4)번 식**</span>      
Match vector들을 통해서 올바른 Enge를 찾고난 후 우리는 <span style = "color:aqua">**이웃 Triple들로부터 matching message를 집계(aggregate)**한다. 그리고 마지막 추론 단계에서 얼마나 많은 **어텐션**을 받는지에 따라 **가중치를 할당**</span>한다. $$p_{e^{\prime}}^{(k-1)}$$은 $$e^{\prime}$$는 마지막 추론 스탭에서 Entity에 할당된 확률이다.      
<center>(4)$$\widetilde{e} \, = \, \sum_{<e^{\prime}, r,e> \in {\mathscr{N}_e}}p_{e^\prime}^{(k-1)} \ㅊdot m_{<e^{\prime}, r, e>}^{(k)}$$</center>

<span style = "font-size:110%">**(5)번 식 Entity Embedding Update**</span>    
Entity Embedding은 Feed Forward Neural Network를 통해 업데이트 한다. 이 FFN은 input으로 이전 임베딩 값인 $$e^{k-1}$$와 relation-aggregate 임베딩인 $$\widetilde{e}^{(k)}$$
두 값을 받는다.   
<center>(5)$$e^{(k)} = FFN([e^{(k-1)};\widetilde{e}^{(k)}])$$</center>

<span style = "font-size:110%">**(6)번 식 **</span>    
이러한 프로세스를 통해 relation path(Topic Entity  ➜ Answer Entity)와 질문의 일치 정도(Matching degree with question) 모두  노드 임베딩(Node Embedding)으로 인코딩 될 수 있다.  
<center>(6)$$p^{k} = softmax(E^{(k)^T}w)$$</center>  
- $$E^{(k)}$$는 k번째 step에서 엔티티들의 임베딩 벡터들을 column방향으로 concatenation한 것이다. 
- $$E^{(k)}$$는 결국 (5)번 식으로부터 Update된 Entity Embedding 행렬이다. 
- $$w$$는 Entity Distribution인 $$p^{(k)}$$로부터 유도된 파라미터이다.

  

## Related Work
- Knowledge Base Question Answering
- Multi-hop Reasoning
- Teacher-Student Network
    
