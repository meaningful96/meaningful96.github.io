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

### 1)

<br/>
<br/>

# Experiments



<br/>
<br/>

# Limitations & Contributions


