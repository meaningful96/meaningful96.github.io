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

**MR**은 매우 간단한 개념이다. 모델이 예측한 샘플들의 순위의 평균을 의미한다. 수식은 다음과 같다. $$N$$은 테스트한 샘플의 수이고, $$rank_i$$는 $$i$$번째 샘플의 순위이다.

<center><span style="font-size:110%">$$\text{MR} \; = \; \frac{1}{N} \sum_{i=1}^N rank_i$$</span></center> 

예를 들어, 한 학생이 5번의 대회에 참가해 각각 1,3,3,5,2 등을 차지했다고 가정해보자. 이 때의 MR은 (1+3+5+5+2)/5 = 3.2가 된다. 즉, 평균적으로 이 학생은 3.2등을 한 것이다.

<br/>

### 2) Mean Reciprocal Rank (MRR)

**MRR**은 실제 정답의 순위의 역수를 평균 낸 것이다. 추천 시스템, Knowledge Graph Completion, 정보 검색 등 여러 분야에서 자주 사용된다. <span style="color:gold">**MRR이 1에 가까울수록 모델의 성능이 좋은 것**</span>이다.

<p align="center">
<img width="700" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/3b390cdc-d0e7-4886-bde1-caf543e15f49">
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

**Hits@k**는 모든 결과들 중 상위 k개에 실제 정답(true candidate)이 순위안에 들어있는 비율을 계산한 것이다. @k라는 것은 상위 k개의 랭크를 말한다. 

<center><span style="font-size:110%">$$\text{Hits@k} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(rank_i \leq k)$$</span></center> 

- $$N$$: 테스트 샘플의 총 개수
- $$rank_i$$: $$i$$번째 샘플의 랭크
- $$\mathbb{I}$$: 조건이 참이면 1, 거짓이면 0을 출력하는 Indicator function
- $$k$$: 몇 개의 예측 결과를 고려할 것인지에 대한 기준

다음의 그림은 Hits@k를 구하기 위한 예시이다. 세 명의 사용자(User)가 있으며, 이들은 각각 3개의 아이템을 추천받았다. 추천받은 아이템이 정답과 일치하면 R, 불일치하면 NR로 표시하였다.

<p align="center">
<img width="700" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/42f2ea5e-f6c8-4d5b-88ca-c4924c840b55">
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
**Mean Average Precision(MAP)**는 정보 검색, 추천 시스템 등에서 사용되는 평가 지표이다. 이는 여러 쿼리나 사용자에 대한 평균 정확도를 측정하여 시스템의 전반적인 성능을 평가한다. MAP는 세 단계에 걸쳐 계산된다.

- Step 1. **$$\text{Precision@k}$$**를 구한다.

<center><span style="font-size:110%">$$\text{Precision@k} = \frac{1}{k} \sum_{i=1}^{k} \mathbb{I}(\text{relevant}_i)$$</span></center>  

- Step 2. **Average Precision (AP)**를 각 쿼리나 사용자에 대해 계산한다.

<center><span style="font-size:110%">$$\text{AP} = \frac{\sum_{k=1}^{n} (\text{Precision@k} \times \mathbb{I}(\text{relevant}_k))}{\text{number of relevant documents}}$$</span></center>  
  
- Step 3. **MAP**는 모든 쿼리나 사용자의 평균 AP를 계산한다.

<center><span style="font-size:110%">$$\text{MAP} = \frac{1}{Q} \sum_{q=1}^{Q} \text{AP}_q$$</span></center>  

아래 그림은 MAP를 구하는 두 개의 예제이다.

<p align="center">
<img width="700" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/cbe801d2-aaae-419a-88e1-c6a012399dac">
</p>

<p align="center">
<img width="700" alt="1" src="https://github.com/meaningful96/Blogging/assets/111734605/94f977e3-0416-45f9-8780-946cfe0a2578">
</p>
