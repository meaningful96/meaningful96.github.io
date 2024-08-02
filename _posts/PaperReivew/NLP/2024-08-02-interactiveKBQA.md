---
title: "[논문리뷰]Interactive-KBQA: Multi-Turn Interactions for Knowledge Base Question Answering with Large Language Models"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2024-08-02
last_modified_at: 2024-08-02
---
*Xiong, G., Bao, J., & Zhao, W*. (2024, February 23). **Interactive-KBQA: Multi-Turn Interactions for Knowledge Base Question Answering with Large Language Models**. arXiv.org. [https://arxiv.org/abs/2402.15131](https://arxiv.org/abs/2402.15131)

이 논문에서 제안한 모델은 <span style="color:red">**Prompt-Engineering**</span>와 <span style="color:red">**Knowledge-Base Interaction**</span> 요소를 포함한다.

# Problem Statement
<span style="font-size:110%">**Knowledge Base Question Answering(KBQA)**</span>  
<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/ce5a15d5-0f24-4908-970c-9cad068d574f">
</p>

**Knowledge Base Question Answering(KGQA)**는 지식 베이스(knowledge base)를 활용하여 자연어 질문에 답변하는 기술이다. Knowledge Graph(KG)는 개체와 개체 간의 관계를 구조화된 형태로 표현한 데이터베이스로, 다양한 정보가 체계적으로 정리되어 있습니다. KGQA는 이러한 구조화된 데이터를 이용해 사용자의 질문에 정확하고 효율적으로 답변할 수 있습니다.

예를 들어, "What is the period of the author of *Off on a Comet*?"이라는 질문을 입력으로 받았다. 그리고 주어진 KG에는 "*Off on a Comet*" 이라는 엔티티가 있고, "author"이라는 릴레이션으로 "Jules Verne"가 연결되어있다.

- Triple 1: (*Off on a Comet*, author, Jules Verne)
- Triple 2: (Jules Verne, period, 1828-1905)

이처럼, 두 개의 트리플을 순차적으로 연결하면 질문에 대한 정답(1828-1905)을 찾을 수 있다. 위의 예시는 다시 말해 <span style="color:red">**2-hop 추론(reasoning)**</span> 문제가 되는 것이다.

최근 KBQA 연구들은 1)**정보 검색 기반 방법**(Information Retrieval, IR)과 2)**의믜 분석 기반 방법(Semantic Parsing, SP)** 두 가지로 분류할 수 있다. IR기반 방법은 쿼리를 이해하고 **질문과 관련된 KB의 적절한 서브그래프(subgraph)를 추출**하여, 이 서브그래프로부터 답변을 추출하는 데 중점을 둔다. 반면, SP기반 방법은 자연어 질문을 **실행 가능한 논리적 형식으로 변환**하여 사전 학습된 생성 모델을 활용해 KB와 상호작용하고 답변을 생성한다.



<br/>

<span style="font-size:110%">Limitations of Prior Studies</span>
1. **복잡한 쿼리 처리 문제(Complex Query Handling)**
  - IR 기반 접근 방식은 복잡한 쿼리를 처리하기 어렵다. 예를 들어, 엔티티 타입, 엔티티의 컨셉, **수치적 제약만으로 구성된 질문은 단순한 엔티티 인식 이상의 이해를 요구**한다. 현재 IR 기반 방법은 이러한 복잡한 조건을 처리하는 데 한계가 있다.
  - 예시)
    - 질문: "키가 2m 이상인 농구 선수는 몇 명인가요?"
    - 단순히 "농구 선수"를 인식하는 것 이상으로, "2m"라는 수치적 제약을 이해하고 정답을 찾아야함.
     
2. **의미 분석을 위한 자원 부족 문제(Resource scarcity for Semantic Parsing)**
  - SP 기반 접근 방식은 광범위한 주석이 달린 데이터셋(annotated dataset)을 필요로 하며, 이는 **자원 집약적**이다. 이러한 제약 조건은 SP 방법의 확장성을 제한하고, 추론 과정의 투명성과 해석 가능성에 어려움을 야기한다.
  - 주석이란 description, entity concept, relation type등을 말한다.
  - 예시)
    - Freebase나 Wikidata와 같은 대규모 지식 베이스에 대해 자연어 질문을 논리적 형식으로 변환하는 모델은 학습시키기 위해 수천 개의 주석이 달린 데이터가 필요하다.
    - 이러한 데이터 셋을 수집하고 주석을 추가하는 것은 매우 큰 컴퓨팅 자원을 요구한다.
    - KGC에서 Description이 없는 NELL-995의 성능이 자연어 기반 방식에서 대체적으로 낮은 성능을 보여줌.

3. **대형 언어 모델의 활용 부족 문제(Underutilization of Large Language Models)**
  - LM의 추론 및 소수 예제 학습 능력이 입증되었음에도 불구하고, 기존 KBQA 접근 방식은 이러한 강점을 완전히 활용하지 못했다. 대부분의 현재 방법은 LLM을 단순히 **술어(predicate, relation)를 식별하는 분류기**로 사용하거나 가능한 논리적 형식이나 질문을 생성하는 데 사용한다. LLM을 더 효과적으로 활용하여 KBQA 시스템의 정확성과 해석 가능성을 높일 수 있는 큰 기회가 여전히 남아 있다.

<br/>
<br/>

# Methods
## Problem Formulation
Interactive-KBQA는 **의미 분석(Semantic Parsing, SP)**에 대한 연구이다. Knowledge Base(= Knowledge Graph)는 $$K \in E \times R \times (E \cup L \cup C)$$로 나타낼 수 있다.
- $$E$$ = 엔티티 집합(entity set)
- $$R$$ = 릴레이션 집합(relation set)
- $$C$$ = 클래스 집합(class set)
- $$L$$ = 리터럴 값 집합(literal value set)
  - 리터럴 값이란 KB에서 엔티티와 관계되지 않은 단순한 데이터 값을 의미한다.
  - 엔티티와 달리 고유한 식별자가 없으며, 단순히 데이터의 값을 나타낸다.
  - 예를 들어 어떤 인물의 나이, 이름, 생일 등의 **속성 값**을 말한다. 
- $$p(S \vert Q, K)$$ = Problem Formulation

질문 $$Q$$와 Knowledge Base $$K$$가 주어졌을 때, 질문에 맞게 실행 가능한 SPARQL 표현식 $$S$$를 생성하는 것이다. 이를 수식으로 표현하면 $$p(S \vert Q, K)$$이다. 즉, **Given $$Q, K$$에 대한 $$S$$를 생성하는 확률**로 형식화된다.

## Model Architecture
<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/702f5ee2-758c-4c35-87b0-f7023a8f26cd">
</p>

앞서 말한 세 가지 문제점 i)복잡한 쿼리 처리 문제, ii)의미 분석을 위한 자원 부족 문제, iii)대형 언어 모델의 활용 부족 문제를 해결하기 위해 **KB와 상호작용하기 위한 세 가지 도구**와 **LLM의 추론 능력을 결합**한 Interactive-KBQA 프레임워크를 제안한다. LLM을 에이전트(agent)로, KB를 환경(enviroment)으로 개념화하여 반복적이고 대화 기반의 문제 해결 과정을 촉진한다. 

복잡한 쿼리가 주어지면, LLM은 특정 도구를 통해 KB와 상호작용하기 위한 행동을 생각해내고 제공해야 한다. 이 도구들은 실행 결과를 관찰값으로 반환한다. 구체적으로, Freebase, Wikidata 및 Movie KB와 같은 이질적인 Knowledge Base를 지원하는 일원화된 상호작용 논리를 갖춘 도구들을 제안하였다.

Interactive-KBQA는 KB와 상호작용하기 위한 세 가지 도구를 제안한다. 
  1) SearchNodes(name)  
  2) SearchGraphPatterns(sparql, semantic)  
  3) ExecuteSPARQL(sparql)  

저자들은 복잡한 질문을 범주화하고 각 유형에 대해 완전한 상호작용 과정을 포함한 두 개의 **주석이 달린 예제**를 제공하여 LLM이 작업을 완료하도록 유도하는 맥락 학습 데모로 사용했다. 나아가, 이 연구에서 소개된 방법은 **수동 개입**을 허용했다. 결과적으로, 우리는 작은 데이터 세트를 수동으로 주석 달아 상세한 추론 과정을 포함시켜 저자원 데이터 셋을 만들었다. 마지막으로, 우리는 이 데이터 세트에 대해 Open Source **LLM을 미세 조정(fine-tuning)**했다. 수행된 실험은 이 방법이 저자원 환경(low-resource enviroment)에서 효과적임을 보여주었다. 그리고 이 고품질 데이터 셋을 추가적인 NLP 연구를 위해 공개했다. 

정리하면, Interactive-KBQA 프레임워크는 <span style="color:red">**LLM(대형 언어 모델)을 에이전트로, 지식 베이스(KB)를 환경으로 개념화하여 대화 기반의 문제 해결 과정을 촉진**</span>한다. 전체적인 프로세스는 다음과 같다.

1. **질문 입력**  
  - 사용자가 복잡한 질문을 시스템에 입력  
2. **사고-행동 패러다임**
  - LLM은 입력된 질문을 바탕으로 **판단**하고, 특정 도구를 통해 **행동**을 결정한다.
3. **도구 실행**
4. **관찰, 반복적 상호작용**
  - 논리적 형식을 점진적으로 개선
  - 필요한 경우 수동으로 개입하여 LLM의 출력을 반복적으로 개선
  - 최종 응답 도출
5. **미세 조정**
  - 생성된 데이터 셋을 사용하여 LLM을 미세조정한다.    

## Tools for Knowledge Base
### 1) SearchNodes(name)
이 함수는 주어진 이름(name)을 사용하여 KB에서 **엔티티를 검색**한다. 주된 목적은 <span style="color:red">**엔티티 연결(Entity Linking)**</span>이다. 이 함수는 엔티티의 이름과 함께 설명(description)과 엔티티 타입(type)을 같이 반환한다. 여러 KB에 general하게 적용 가능하다.

<br/>

### 2) SearchGraphPatterns(sparql, semantic)

이 함수는 KB내에서 중요한 **술어(Predicate, Relation)를 식벽하고 순위를 매기는 것**을 목표로 한다. 입력으로는 SPARQL의 쿼리가 필요하다. 

- SPARQL 쿼리 = `SELECT ?e WHERE`

그런 다음 엔티티 `?e`를 중심으로 하는 **one-hop 서브그래프에서 쿼리**를 수행한다. 그 후, semantic 파라미터와 트리플의 술어(predicate)와의 의미적 관련성에 따라 검색된 트리플의 순위를 매긴다. 최종적으로 **top_K개의 트리플을 반환**한다. 이 도구는 context window(= LLM이 한 번에 처리할 수 있는 텍스트의 최대 길이)의 사용을 최적화하기 위해 불필요한 정보를 제거하면서 서브그래프를 정확하게 식별할 수 있게 만든다. 즉, <span style="color:red">**LLM의 제한(=최대 입력 길이)된 입력 안에서 불필요한 정보를 제거한 서브그래프를 입력시키기 위함**</span>이다. 유연한 작업을 지원하며, Freebase의 Compound Value Type(CVT)에 대해 특별히 최적화되었다.

<br/>

### 3) ExecuteSPARQL(sparql)
이 함수는 **임의의 SPARQL 쿼리를 직접 실행**할 수 있도록 하여 뛰어난 **유연성을 보장**한다. 이 도구를 사용하면 <span style="color:red">**사용자가 SPARQL 쿼리를 작성하고 이를 실행**</span>하여 KB에서 필요한 정보를 직접 검색할 수 있다.

<br/>

## Interactive Process
질문 $$Q$$가 주어졌을 때 가장 먼저 프롬프트를 생성한다. 프롬프트(Prompt)는 Instruction인 $$\text{Inst}$$, 예제(Examplar) $$E$$, 질문 $$Q$$를 입력으로 받는다. 논문에서는 대화형 형식으로 두 개의 complete examplar에 수동으로 주석을 달았다.

<center><span style="font-size:105%">$$\text{Prompt} = \{ \text{Inst}, E, Q \}$$</span></center>

또한 각 턴마다 LLM과의 상호작용을 한다. 각 <span style="color:red">**턴 $$T$$마다 LLM이 프롬프트와 이전 히스토리를 바탕으로 액션 $$a_T$$를 생성**</span>한다. 히스토리는 $$H = \{ c_o , a_o , o_0 , \cdots, c_{T_1} , a_{T-1} , o_{T-1} \}$$로 표현한다. **행동** $$a$$는 {SearchNodes, SearchGraphPatterns, ExecuteSPARQL, Done}의 집합으로 도구를 사용하여 실행된다. **관찰** $$o$$는 도구의 실행 결과로 다음 턴의 사고와 행동을 결정하는데 사용된다. $$T$$시점의 **관찰**은 $$o_T = \text{Tool}(a_T)$$로 표현할 수 있다. **사고** $$c$$는 질문을 하위 쿼리로 분해한 것이다. $$c_0$$는 엄격하게 정의하지 않고 LLM이나 사람이 직접 정한다.

<center><span style="font-size:105%">$$a_T = \text{LLM}(\{\text{Prompt}, H \})$$</span></center>

저자들은 이를 바탕으로 사고-행동 패러다임을 고안하였다.
- **사고-행동 패러다임**
  - **사고(Thought)**
    - 질문 $$Q$$가 주어지면, 초기 사고 $$c_0$0$는 이를 **트리플 형식과 유사한 하위 쿼리로 분해**하는 것이다.
    - 첫 번째 라운드를 제외하고, LLM은 관찰을 기반으로 추론 과정을 명확히 설명하는 사고 $$c$$를 생성해야 한다.
    - 이 접근 방식은 의사 결정 과정을 **설명 가능**하게 만드는 것을 목표로 한다.
    - Ex) "What movies has Tom Hanks acted in?" → (Tom Hanks, act in, ?movie)

  - **행동(Action)**
    - 각 턴 $$T$$마다 LLM은 현재 라운드를 종료하는 행동 $$a_T$$를 생성해야 한다.
    - 구문 분석하여 실행한 후, 그 결과를 반환한다. 이 결과를 관찰 $$o$$로 사용한다.
    - LLM은 이 관찰을 바탕으로 대화를 계속할지 종료할지 여부를 결정한다.
    - 만약 $$a_T = \text{Done}$$이면 최종 관찰 $$o_T$$를 답으로 출력한다.
        
## Solutions for Complex Questions
Interactive-KBQA는 상호작용 추론 과정을 주석 달아 LLM(대형 언어 모델)을 통해 추론을 유도하는 것을 포함한다. <span style="color:red">**다양한 유형의 복잡한 질문에 대해 패턴을 식별하고, 상호작용 모드를 설계하며, 고품질의 예제를 라벨링**</span>하는 것이 중요하다.

**멀티 홉 질문(Multi-hop Question)**의 경우 각 단계에서 구체적인 엔티티보다는 특정 술어(=릴레이션)에 집중한다. 그래프 패턴을 SPARQL로 표현하는 것만으로 도구가 작업의 순위를 처리할 수 있도록 충분하다.

**Freebase의 CVT 구조**의 경우, 이러한 구조를 직면했을 때 추론 과정을 명시적으로 설명한다. 또한 별 모양(star-shape)의 CVT 구조를 **여러 개의 단일 홉 관계로 분해**하여 각각 처리한다. 예를 들어, "Tom Hanks가 영화 'Nothing in Common'에서 'David Basner' 역할을 한다"는 문장의 의미는 ("Tom Hanks", film.actor.film -> film.performance.film, "Nothing in Common")와 ("Tom Hanks", film.actor.film -> film.performance.character, "David Basner") 두 개의 트리플로 표현 가능하다.

## Human-Machine Collaborative Annotation
실제로는 모든 질문 유형의 예제를 문맥에 포함하는 것은 두 가지 주요 이유로 불가능하다.
- i)  비용이 크다.
- ii) 입력 토큰이 일정 한계를 초과하면 LLM의 성능이 현저히 저하됨.

이전 연구에서 프로세스 감독(Process Supervision)이 모델의 일반화 능력을 향상시킬 수 있다는 결과가 있었다. Interactive-KBQA는 이 점을 활용한다. 구체적으로 주석자(annotator, 논문에서는 annotator가 사람, LLM 모두 가능)가 행동 $$a_T$$가 비합리적이라고 판단할 때, 이를 수동으로 $$a^{'}_T$$로 수정하고 메세지에 통합해 $$a_{T+1}$$을 생성하게 된다.

<center><span style="font-size:105%">$$a_{T+1} = \text{LLM}(\text{Prmopt}, c_o , a_o , o_0 , \cdots, c_{T_1}, a^{'}_T, o_T^{'})$$</span></center>

이를 통해, LLM이 사고와 행동을 생성하는 각 라운드 후에 중단점을 설정하여 인간 평가자가 이를 검토하고 수락 여부를 결정한다. 일단 수락되면, 프로세스는 계속 진행된다. 만약 거부되면, 주석자는 생성된 사고와 행동을 수정한 후 진행한다. 이 방법론의 핵심은 <span style="color:red">**인간이 데이터에 주석을 다는 과정을 모방**</span>하는 것이다. 주석자가 모델이 환각(hallucination, 관찰에 없는 술어 생성), 사고와 행동 간의 불일치, 또는 정답 경로에서 벗어나는 경우(= 연속 두 라운드가 잘못된 경우)에 개입한다.

<br/>
<br/>

# Experiments

## Dataset
<p align="center">
<img width="400" alt="1" src="https://github.com/user-attachments/assets/d853e522-273d-40c2-a85c-9d397dec413d">
</p>

## Main Result
<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/e7a69990-5fc6-4bd2-aaf4-cae390fd73a4">
</p>

- **실험 방법**
  - **Prompting with GPT-4 Turbo**: GPT-4 Turbo를 사용하여 프롬프트 기반의 QA를 수행.
  - **Fine-tuning with open-LLM**: LLaMA2-7B와 13B를 fine-tuning하여 QA를 수행.

- **실험 결과**
  - 대체적으로 LLM으로 프롬프트 튜닝을 하여 실험한 결과들이 대체적으로 성능이 우수했다.
  - 하지만 CWQ 데이터셋에서는 일부 질문 유형(Compa, Super)에서 fine-tuning을 했을 때가 성능이 더 좋았다.
  

## Additional Experiments
<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/ef38af78-fb8a-48c2-9a4c-35740fe041f3">
</p>

**Table 4: Results of Entity Linking**는 Entity Linking 성능을 비교한 것이다. WebQSP 및 CWQ 데이터셋에서 제안된 방법은 StructGPT와 ToG를 포함한 다른 방법들보다 높은 RHits@1 및 EM 성능을 보였다. **Golden entities**를 사용한 경우 성능이 더욱 향상되었다.

**Table 6:  Impact of Exemplar Number and Average Price (AP)**는 질문 유형 분류기의 성능을 보여준다. 2-shot, 4-shot 설정에서 Interactive-KBQA의 성능이 우수하며, zero-shot설정과 비교하여 성능이 크게 향상되었다.

**Table 7:  Impact of Exemplar Number and Average Price (AP)**은 예제 수와 평균 가격(AP)의 영향을 보여준다. OpenAI의 GPT-4 Turbo와 GPT-3.5 Turbo, 그리고 open-source LLM인 Mistral-7B와 Llama 2의 성능을 평가하였다. Fine-tuning된 모델들이 결론적으로 더 높은 성능을 보여주었다.

<br/>

<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/0212a01c-5d9d-4fbf-824e-3ef039dbe4dd">
</p>

**Table 5: Results on WebQSP and CWQ with Golden Entities** golden entities를 사용한 WebQSP 및 CWQ 데이터셋에서의 결과를 보여준다. 제안된 방법이 두 데이터셋 모두에서 높은 성능을 보여주었다.

**Table 8: Performance of Different Backbone Models**은 다양한 백본 모델의 성능을 보여준다. Error Type에 따라 성능을 분석하였다. Entity Linking, Predicate Search, Reasoning Error, Format Compliance, Hallucination, Other 등의 오류 유형에서 모델의 성능을 평가하였으며 제안된 방법이 대부분의 오류 유형에서 다른 모델들보다 우수한 성능을 보였다.

<br/>
<br/>

# Contributions & Limitations
- **Contribution**
  - **Interactive-KBQA 프레임워크 제안**: LLM의 추론 능력을 활용한 새로운 프레임워크를 제안하여, 다중 턴 상호작용을 통해 의미 분석을 수행할 수 있게 한다.
  - **통합 SPARQL 기반 도구 세트 및 상호작용 논리 설계**: 다양한 복잡한 쿼리를 효율적으로 처리할 수 있는 통합된 SPARQL 기반 도구 세트와 상호작용 논리를 설계했다.
  - **단계별 추론 과정이 포함된 인간 주석 KBQA 데이터셋 공개**: 저자원 데이터셋으로 활용될 수 있는, 단계별 추론 과정을 포함한 인간 주석 KBQA 데이터셋을 공개했다.
  
- **Limitations**
  - **LLM에 의존성**: 프롬프트 학습 기반 접근 방식은 LLM(대형 언어 모델)의 능력에 크게 의존하며, 다중 라운드 대화를 포함하는 시나리오에서는 추론 비용이 상당히 높아진다.
  - **LLM 호출에 따른 비용**: LLM API를 호출할 때 LLM의 출력을 조정하는 것은 비실용적이다.
  - **높은 추론 비용**: 다중 라운드 대화를 포함하는 시나리오에서는 추론 비용이 상당히 높아진다. 이는 실제 적용 시 비용 효율성 측면에서 도전 과제가 될 수 있다.

