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

# 1. 문제 정의(Problem Set)
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

# 2. Method  
## 1) Modeling 
- Teacher & Student Network
- Neural State Machine(NSM)
- Bidirectional Reasoning Mechanism

## 2) Teacher - Student Network  
### (1) Overview    
```
The main idea is to train a student network that focuses on the multi-hop KBQA task itself, while another teacher
network is trained to provide (pseudo) supervision signals (i.e., inferred entity distributions in our task) at 
intermediate reasoning steps for improving the student network.
```
학생 네트워크는 multi-hop KBQA를 학습하는 한편, 선생 네트워크에서는 <span style ="color:aqua">Intermediate Supervision Signal</span>을 만들어 학생 네트워크로 넘겨준다.
이렇게 함으로써 학생 네트워크에서 더 학습이 잘되게끔 한다.

## 3) Student Network  
선생-학생 네트워크에서 학생 네트워크(Student Network)가 Main model이다. 학생 네트워크의 목표는 Visual question answering으로부터 정답을 찾는 것이다. 
학생 네트워크에서는 NSM(Neural State Machine) 아키텍쳐를 이용한다.

### (1) NSM(Neural State Machine)

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


### (2-1) Instruction Component    
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

### (2-2)Attention Fuction이란?  

<p align="center">
<img width="" alt="500" src="https://user-images.githubusercontent.com/111734605/210244763-6df0807b-7e7f-4d4a-a73b-f100734ee83e.png">
</p>     
<center><span style = "font-size:80%">Instruction Component</span></center>

어텐션 함수는 Query, Key, Value로 구성된 함수이다.  
<center>$$Attention(Q,K,V) \; = Attention \, Value $$</center>  
<center>
$$\begin{aligned}
Q &: Query  \\
K &: Key\\
V &: Value\\
\end{aligned}$$
</center>

어텐션 함수는 주어진 **'쿼리(Query)'**에 대해 모든 **'키(Key)'**의 유사도를 각각 구합니다. 그리고, 이 유사도를 키(Key)와 매핑되어 있는 각각의 **'값(Value)'**에 반영해줍니다. 그리고 '유사도가 반영된'값을 모두 더해서 리턴하고, 어텐션 값을 반환한다.

### (3) Reasoning Component

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
<center>$$(4) \; \widetilde{e} \, = \, \sum_{<e^{\prime}, r,e> \in {\mathscr{N}_e}}p_{e^\prime}^{(k-1)} \cdot m_{<e^{\prime}, r, e>}^{(k)}$$</center>

<span style = "font-size:110%">**(5)번 식 Entity Embedding Update**</span>    
Entity Embedding은 Feed Forward Neural Network를 통해 업데이트 한다. 이 FFN은 input으로 이전 임베딩 값인 $$e^{k-1}$$와 relation-aggregate 임베딩인 $$\widetilde{e}^{(k)}$$
두 값을 받는다.   
<center>$$(5) \; e^{(k)} = FFN([e^{(k-1)};\widetilde{e}^{(k)}])$$</center>

<span style = "font-size:110%">**(6)번 식**</span>    
이러한 프로세스를 통해 relation path(Topic Entity  ➜ Answer Entity)와 질문의 일치 정도(Matching degree with question) 모두  노드 임베딩(Node Embedding)으로 인코딩 될 수 있다.  
<center> $$(6) \; p^{k} = softmax(E^{(k)^T}w)$$</center>  
- $$E^{(k)}$$는 k번째 step에서 엔티티들의 임베딩 벡터들을 column방향으로 concatenation한 것이다. 
- $$E^{(k)}$$는 결국 (5)번 식으로부터 Update된 Entity Embedding 행렬이다. 
- $$w$$는 Entity Distribution인 $$p^{(k)}$$로부터 유도된 파라미터이다.

### (4) Discussion
- Student Network의 NSM model은 Two-fold이다.  
  1. Teacher Network로 부터 <span style = "color:aqua">**중간 엔티티 분포(Intermediate entity distribution)을 Supervision signal로**</span> Student Network에 이용한다 
    - 기존의 KBQA 연구들은 이런 중간 단계에서 엔티티 분포를 이용하지 않음!!
  2. NSM은 주어진 **Knowledge graph**에 대해 엄청난 <span style = "color:aqua">**추론 능력(reasoning capacity)**</span>을 보여주는 GNN 모델이다. 
    - 엔티티 분포와 엔티티 임베딩을 학습하는 것은 결국 GNN의 일반적인 <span style = "color:aqua">"전사 후 집계(*propagate-then-aggregate*)" 메커니즘</span>을 잘 반영해준다.

- NSM은 Scene graph와 instruction vector를 이용해 추상적인 잠재 공간에서 시각적 추론을 하기위한 모델이다. 이를 Multi-hop KBQA에 사용하기 위해 두 가지 방법을 사용하였다.
  1. 엔티티들에 관련된 **관계 임베딩(relation embedding)**을 집계하여 노드 임베딩을 초기화한다.
    - 식 (2), 엔티티의 초기값 방법은 결국 좀 더 유의미한 엔티티의 relation에 초점을 맞춰, 노이즈 엔티티의 영향력을 감소시킨다.
  2. **이전 임베딩** $$e^{(k-1)}$$와 **relation-aggregated 임베딩** $$\widetilde{e}^{(k)}$$와 통합해서 엔티티 임베딩을 업데이트 한다.
    (Original NSM은 두 factor를 각각 모델링함.)
  
## 4) Teacher-Network    
Teacher Network 모델은 Student Network와는 그 존재 목적 자체가 다르다. Teacher Network는 <span stlye = "color:aqua">**중간 추론 단계에서 신뢰가능한 엔티티(reliable entity)를 학습하거나 추론**</span>한다. 참고로, Teacher Network를 학습할때는 Unlabeling 된 데이터들을 사용한다.
 
이러한 이유로 논문에서는 Bidirectional Search 알고리즘을 참고해 <span style = "color:aqua">**Bidirectional reasoning mechanism**</span>을 도입했다. 이 메커니즘을 활용하여
중간 추론 단계에서의 Teacher Network 학습을 향상시켰다. Bidirectional reasoning mechanism을 *forward reasoning*이라고 한다.
  
### (1) Bidirectional Reasoning for Multi-hop KBQA
기존의 Knowledge Graph에서는 Topic entity에서 Answer entity로 한방향 탐색을 통해 정답에 접근했다. 하지만, 논문에서는 **양방향 탐색(Bidirectional Search)**를 응용해 양방향 추론
을 구현했다.  
- Bidirectional Reasoning Mechanism
  1. Topic Entity  ➜ Answer Entity
  2. Answer Entity ➜ Topic Entity

기존의 연구는 모두 1번을 기준으로 진행되었다. 이 논문에서는 2번을 활용한 것이다. 아이디어는 두 추론 프로세스가 중간 단계에서 서로 동기화되도록 하는 것이다. 다시 말해,
forward 방향에서 k번째 엔티티 분포인 $$p_f^{(k)}$$와 backward 방향의 (n-k)번째 엔티티 분포인 $$p_b^{(n-k)}$$일 때, 만약 두 추론 프로세스가 안정적이고 정확하다면 두 분포는
그 값이 비슷하거나 일정할 것이다. ➜ $$p_f^{(k)} \approx p_b^{(n-k)}$$

### (2) Reasoning Architecture  
<p align="center">
<img width="100%" alt="1" src="https://user-images.githubusercontent.com/111734605/210313258-d7bfb2f5-11e8-4bce-8631-105c23e8afce.png">
</p>

### (3) Parallel Reasoning
Figure 3에 (a)번째와 같이 Instruction vector를 공유하지 않고 **서로 다른 NSM**을 사용해 forward와 backward reasoning을 **각각** 진행한다. 두 NSM network는 반드시 Isolated하며
**서로 어떠한 파라미터도 공유하지 않는다.** 단지 그 두 프로세스 사이의 중간 엔티티 분포에 서로 대응 제약(Correspondence Constraint)만 통합하는 것만 고려한다.
  

### (4) Hybrid Reasoning

Hybrid Reasoning 방법에서는 Instruction Component를 공유하고, Cycle Pipeline(원형 파이프라인 모듈)로 구성했다. 또한 대응 제약 외에도, 같은 Instruction Vector를 받는다.
**forward reasoning의 마지막 스텝은 backward reasoning의 첫번째 값이 된다.** 이를 식으로 정리하면 다음과 같다.

<p align="center">
<img width="350" alt="1" src="https://user-images.githubusercontent.com/111734605/210319909-88e3450a-3069-411f-9f3c-9a8823c76433.png">
</p>
  
Figure 3에서 볼 수 있듯이, Parallel reasoning이 좀 더 느슨한 통합을 가진 반면, Hybrid reasoning은 forward와 backward reasoning 과정의 정보 사이에 더 깊은 통합을
필요로 한다. 여기서 주의할 것은, 일반적인 BFS와는 다르게 역방향 추론이 정방향 추론의 완벽한 역과정은 아니라는 것이다. 왜냐하면 두 과정은 서로 다른 semantic(의미론)에 대해 
해당한다. 즉, multi-hop에서 같은 **entity를 같은 edge를 통해 간다고 하더라도, 방향이 반대이면 그 의미는 다르다.**

이러한 점을 고려할때, <span style = "color:aqua">forward의 마지막 추론 단계의 값을 backward의 초기값으로 **재활용**하고</span> 이러한 방식은 결국 backward reasoning에서 forward reasoning에 관한 정보를
더 많이 받는것이되므로 forward reasoning을 추적하는데 더 큰 도움이 된다.
  
## 5) Teacher-Student framework 이용한 학습

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/210330914-04d911e8-85f9-4741-b296-c46344177007.png">
</p>

### (1) Teacher Network 최적화
Teacher Network의 두가지 추론 아키텍쳐는 같은 방식으로 최적화할 수 있다. 이를 1) Reaspning loss 와 2) Correspondence loss이다. 

- Reasoning Loss [식 (9)]
  <center>![image](https://user-images.githubusercontent.com/111734605/210331487-bbdc9df2-2a34-4e91-babb-2828535082fb.png)</center>
  - reasoing loss는 엔티티를 얼마나 정확하게 나타내는가를 의미하며, 이는 두 direction으로 분해된다.
  - $$p_f^{(n)}$$ 와 $$p_b^{(n)}$$은 각각 forward와 backward 추론 프로세스의 마지막 엔티티 분포이다.
  - KL divergence는 asymmetric한 방법이다.
  - $$p_f^{*}$$ 와 $$p_b^{*}$$를 구하기 위해서 원래의 엔티티(ground-truth entity)를 주파수 정규화 엔티티로 변환해야 한다.
  - 더 정확하게는 그래프에서 $$ k $$ 엔티티가 ground-truth entity이면 마지막 분포에 $$\frac{1}{k}$$의 확률이 할당된다. 

- Correspondence Loss [식 (10)]
  - 잰슨-셰넌 divergence를 이용한다. JS Divergence는 symmetric한 방법이다. 이를 Lagrange Multiplier를 이용해 표현하면 (10)식과 같이 된다.
  - 
### (2) Student Network 최적화
NSM 모델을 Student Network 모델에 적용해 forward reasoning을 수행했다. 게다라 reasoning loss를 고려하여, student network의 prediction과 teacher network의 
supervision signal의 loss를 통합한다. 이를 식으로 나타내면  (12)식이 된다.
  
Teacher Network의 최적화가 완료되면 두 추론 프로세스로부터 중간 엔티티 분포(Intermediate Entity Distribution)를 얻게 된다. 이 두 중간 엔티티 분포를
Supervision signal로 여기고 평균을 취하면 (11)식이 된다. 
- $$p_t^{(k)}$$와 $$p_s^{(k)}$$는 k번째 스텝에서 Student network와 Teacher network의 중간 엔티티 분포이다. $$\lambda$$ Lagrange Multiplier다. 

### (3) Discussion
실제로 많은 KBQA 모델들은 중간 추론 단계에서 labeled data는 거의 사용되지 못한다. 즉, Supervision signal이 부족하다. 이 논문의 핵심은, 추가적으로 Labeled data를
사용하지 않고, <span style = "color:aqua">Teacher Network의 **Bidirectional Reasoning Mechanism**을 이용해서 **중간 엔티티 분포**를 만들어내고, 이를 Supervision signal로 Student Network에서 이용하여 학습 효율을 높이는 것</span>이다. 

# Result
  
## 논문을 들어가기 앞서 알면 좋은 Basic Knowledge
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
  
## Related Work
- Knowledge Base Question Answering
- Multi-hop Reasoning
- Teacher-Student Network
    
