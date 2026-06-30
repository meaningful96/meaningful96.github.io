---
title: "[딥러닝]Evaluation Metric(평가 지표) - (2) 순위 성능 지표"

categories: 
  - DeepLearning

toc: true
toc_sticky: true

date: 2024-06-30
last_modified_at: 2024-06-30
---

# Evaluation Metric

이전 포스터 [\[딥러닝\]Evaluation Metric(평가 지표)](https://meaningful96.github.io/deeplearning/evaluation_metric/#site-nav)에 이어서 순위 기반의 모델(Ranking Based Model)들의 성능을 측정하는 순위 성능 지표에 대해 알아보겠다.

## 2. 순위 기반 모델의 평가 지표

순위 기반 모델(Ranking based Model)들의 평가지표이다. Mean Rank(MR), Mean Reciprocal Rank(MRR), Hits@k, NDCG등이 있다.

### 1) Mean Rank(MR)

**MR**은 매우 간단한 개념이다. 모델이 예측한 샘플들의 <span style="color:red">**순위의 평균**</span>을 의미한다. 수식은 다음과 같다. $$N$$은 테스트한 샘플의 수이고, $$rank_i$$는 $$i$$번째 샘플의 순위이다.

<center><span style="font-size:110%">$$\text{MR} \; = \; \frac{1}{N} \sum_{i=1}^N rank_i$$</span></center> 

예를 들어, 한 학생이 5번의 대회에 참가해 각각 1,3,3,5,2 등을 차지했다고 가정해보자. 이 때의 MR은 (1+3+5+5+2)/5 = 3.2가 된다. 즉, 평균적으로 이 학생은 3.2등을 한 것이다.

<br/>

### 2) Mean Reciprocal Rank (MRR)

**MRR**은 <span style="color:red">**실제 정답의 순위의 역수를 평균**</span> 낸 것이다. 추천 시스템, Knowledge Graph Completion, 정보 검색 등 여러 분야에서 자주 사용된다. **MRR이 1에 가까울수록 모델의 성능이 좋은 것**이다.

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/3b390cdc-d0e7-4886-bde1-caf543e15f49">
</p>

위의 예제는 MRR을 계산하는 방법을 잘 보여준다. User 1의 경우 가장 첫 번째로 유사성이 높은 아이템을 추천 받은 것이 3위이다. 따라서 User 1의 reciprocal rank(순위의 역수)는 1/3이다. 반면 User2와 User3은 처음으로 관련성이 깊은 아이템을 추천 받은 순위가 각각 2위와 1위이다. 따라서 둘의 reciprocal rank는 각각 1/2와 1이 된다. 이를 토대로 MRR을 계산하면 0.61이 된다. MRR은 다음과 같은 장단점을 가진다.

- Pros
  - 계산이 쉽고, 해석이 쉽다.
  - 관련이 깊은 첫 번째 element에 대해서만 집중하기 때문에 user에게 가장 적합한 아이템을 추천해주기에 용이하다. 

- Cons
  - 관련이 깊은 첫 번째 element를 제외하고 나머지 아이템은 고려하지 못한다.
  - user가 여러 아이템(item list, item sequence)를 원하면 사용이 불가능하다.

<br/>

### 3) Hits@k

**Hits@k**는 모든 결과들 중 <span style="color:red">**상위 k개에 실제 정답(true candidate)이 순위안에 들어있는 비율을 계산한 것**</span>이다. @k라는 것은 상위 k개의 랭크를 말한다. 

<center><span style="font-size:110%">$$\text{Hits@k} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(rank_i \leq k)$$</span></center> 

- $$N$$: 테스트 샘플의 총 개수
- $$rank_i$$: $$i$$번째 샘플의 랭크
- $$\mathbb{I}$$: 조건이 참이면 1, 거짓이면 0을 출력하는 Indicator function
- $$k$$: 몇 개의 예측 결과를 고려할 것인지에 대한 기준

다음의 그림은 Hits@k를 구하기 위한 예시이다. 세 명의 사용자(User)가 있으며, 이들은 각각 3개의 아이템을 추천받았다. 추천받은 아이템이 정답과 일치하면 R, 불일치하면 NR로 표시하였다.

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/42f2ea5e-f6c8-4d5b-88ca-c4924c840b55">
</p>

먼저 Hits@1을 구하는 과정이다.
- Hits@1
  - User 1: 추천 순위 1에 NR이 있으므로 Miss
  - User 2: 추천 순위 1에 NR이 있으므로 Miss
  - User 3: 추천 순위 1에 R이 있으므로 Hit
  - Hits@1 = $$\frac{1}{3} = 0.333$$ (세 명의 User 중 순위 1에 정답이 있는 case가 하나이다.)
 
다음으로 Hits@3를 구하는 과정이다.
- Hits@3
  - User 1: 추천 순위 3에 R이 있으므로 Hit
  - User 2: 추천 순위 2와 3에 R이 있으므로 Hit
  - User 3: 추천 순위 1과 2에 R이 있으므로 Hit
  - Hits@3 = $$\frac{3}{3} = 1$$
 
참고로, 이전 포스터의 내용 중 Precision과 Recall에도 @k의 개념이 도입될 수 있다. Precision@k는 상위 $$k$$개의 예측 중에서 실제로 관련성이 있는 항목의 비율을 측정하는 지표이다. 또한 Recall@k는 실제로 관련성이 있는 모든 항목 중에서 상위 $$k$$개의 예측이 얼마나 많은 관련 항목을 포함하는지를 측정하는 지표이다. 이 둘을 수식으로 표현하면 다음과 같다. 

<center><span style="font-size:110%">$$\text{Precision@k} = \frac{1}{k} \sum_{i=1}^{k} \mathbb{I}(\text{relevant}_i)$$</span></center>   
<center><span style="font-size:110%">$$\text{Recall@k} = \frac{1}{N} \sum_{i=1}^{k} \mathbb{I}(\text{relevant}_i)$$</span></center>   

<br/>

### 4) Mean Average Precision(MAP)
**MAP**는 정보 검색, 추천 시스템 등에서 사용되는 평가 지표이다. 이는 여러 쿼리나 사용자에 대한 평균 정확도를 측정하여 시스템의 전반적인 성능을 평가한다. MAP는 세 단계에 걸쳐 계산된다.

- Step 1. **$$\text{Precision@k}$$**를 구한다.

<center><span style="font-size:110%">$$\text{Precision@k} = \frac{1}{k} \sum_{i=1}^{k} \mathbb{I}(\text{relevant}_i)$$</span></center>  

- Step 2. **Average Precision (AP)**를 각 쿼리나 사용자에 대해 계산한다.

<center><span style="font-size:110%">$$\text{AP} = \frac{\sum_{k=1}^{n} (\text{Precision@k} \times \mathbb{I}(\text{relevant}_k))}{\text{number of relevant documents}}$$</span></center>  
  
- Step 3. **MAP**는 모든 쿼리나 사용자의 평균 AP를 계산한다.

<center><span style="font-size:110%">$$\text{MAP} = \frac{1}{Q} \sum_{q=1}^{Q} \text{AP}_q$$</span></center>  

아래 그림은 MAP를 구하는 두 개의 예제이다.

<p align="center">
<img width="500" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/cbe801d2-aaae-419a-88e1-c6a012399dac">
</p>

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/94f977e3-0416-45f9-8780-946cfe0a2578">
</p>

MAP는 다음과 같은 장단점을 가진다.
- Pros
  - Precision(정밀도)-Recall(재현율) 곡선 아래의 복잡한 영역을 나타내는 단일 지표를 제공한다.
  - 리스트 내 추천된 항목들의 순위를 다룬다(항목들을 집합으로 고려하는 것과 대조적).
  - 추천 리스트 상위에 발생한 오류에 더 많은 가중치를 부여한다.

- Cons
  -  세밀한 숫자 평점에는 적합하지 않고, 이진 (관련/비관련) 평점에 적합하다.
  -  세밀한 평점 (1에서 5까지의 척도)에서는, 주어진 평점을 이진 평점으로 변환하기 위해 임계값을 먼저 설정해야 한다. 예를 들어, 평점이 4를 초과하는 경우에만 관련성이 있다고 간주하고, 4 이하인 경우 비관련성으로 간주한다.
    - 이러한 수동 임계값 설정으로 인해 평가 지표에 편향이 발생할 수 있다.
    - 세밀한 정보를 버리게 된다.
 
<br/>

### 5) Normalized Discounted Cumulative Gain (NDCG)
**NDCG**를 이해하기 위해서는 먼저 **DCG**에 대한 개념 이해가 필요하다. DCG는 Discounted Cumulative Gainㅇ의 약자로, 각 문서나 항목의 관련성 점수에 따라 가중치를 부여하여 순위를 평가한다. 높은 순위에 있는 관련성이 높은 문서일수록 더 높은 가치를 가진다. 다음은 DCG의 수식이다. DCG는 문서의 관련성 점수 $$rel_i$$를 고려하여 순위 $$i$$의 문서에 대해 계산한다.

<center><span style="font-size:110%">$$\text{DCG} = \sum_{i=1}^{p} \frac{rel_i}{\log_2(i + 1)}$$</span></center>  

또한 log 기반의 DCG 수식도 존재한다. 로그 함수는 순위에 따라 감소하는 가중치를 부여한다.

<center><span style="font-size:110%">$$\text{DCG} = \sum_{i=1}^{p} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$</span></center>  

다음으로는 **IDCG**에 대해 알아야 한다. IDCG는 Ideal Discounted Cumulative Gain의 약자로, 이상적인 (ideal) 순서로 정렬된 문서 집합의 DCG이다. 즉, 관련성 점수가 가장 높은 문서들이 최상위에 위치한 경우의 DCG이다. IDCG는 NDCG를 계산하기 위해 사용되며, DCG를 이상적인 경우와 비교할 수 있게 해준다. 이 역시 기본 수식과 로그 기반 수식이 존재한다.

<center><span style="font-size:110%">$$\text{IDCG} = \sum_{i=1}^{|REL_p|} \frac{rel_i}{\log_2(i + 1)}$$</span></center>  
<center><span style="font-size:110%">$$\text{IDCG} = \sum_{i=1}^{|REL_p|} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$</span></center>    

이제 이 두 개념을 통해 **NDCG(Normalized Discounted Cumulative Gain)**를 구할 수 있다. NDCG는 DCG를 IDCG로 정규화한 값이다. 이는 <span style="color:red">**실제 결과의 순위를 이상적인 순서와 비교하여 평가**</span>한다. NDCG는 **0과 1 사이의 값**을 가지며, 1에 가까울수록 순위가 이상적임을 의미한다. 수식은 다음과 같다.

<center><span style="font-size:110%">$$\text{NDCG} = \frac{\text{DCG}}{\text{IDCG}}$$</span></center>  

정리하자면, <u>DCG는 문서의 관련성에 따라 순위를 평가하고, IDCG는 이상적인 순서로 정렬된 경우의 DCG를 제공한다. NDCG는 이 두 값을 비교하여 순위의 품질을 평가</u>한다.

다음은 NDCG를 구하는 예시이다.

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/b0ed895f-cf1a-468b-a009-9b05559f0ba0">
</p>

