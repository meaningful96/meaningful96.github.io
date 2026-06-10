---
title: "[그래프 AI]그래프 머신러닝(딥러닝) - 노드 임베딩(Node Embedding)"
categories: 
  - Graph
  
toc: true
toc_sticky: true

date: 2024-08-03
last_modified_at: 2024-08-03
---
# 그래프 머신러닝(딥러닝)

<figure style="text-align: center; margin: auto;">
  <img width="1000" alt="1" src="https://github.com/user-attachments/assets/d32f9fb9-ad71-404e-a7bd-12726049a08b" style="display: block; margin: auto;">
  <figcaption style="font-size:70%; text-align: center; width: 100%; margin-top: 0px;">
    <em>ref) Stanford University <a href="https://web.stanford.edu/class/cs224w/">CS224W</a></em>
  </figcaption>
</figure>

그래프는 실생활에서 다양하게 사용되고 있다. 인스타그램, 메타, 링크드인과 같은 거대한 소셜 네트워크 안에서 팔로워 관계를 나타내는 소셜 네트워크(Social Network), 단백질간의 상호작용을 나타내는 그래프, 질병과 약물간의 상관 관계를 나타내는 그래프, 전자상거래 시스템에서 사용자와 구매 아이템 간의 상호 관계를 그려 놓은 사용자-아이템 그래프 등이 대표적이다.  

## 1. 그래프 머신 러닝의 종류
<p align="center">
<img width="600" alt="1" src="https://github.com/user-attachments/assets/35cf9e2d-8ea9-4b26-a094-656a7ee9ea34">
</p>

Deep learning 기법을 이용해 그래프의 정보를 학습하는 방법은 크게 세 가지로 구분된다. **전통적인 방식(Traditional Method)**은 그래프의 구조적 패턴을 직접 분석하고 비교하여 그래프 간의 유사성을 측정한다. **노드 임베딩(Node Embedding)**은 노드(= 정점, 엔티티)를 벡터 공간에 임베딩하여 그래프의 구조적 정보를 저차원 표현으로 변환한다. 마지막으로, **그래프 신경망(GNN)**은 그래프 데이터에서 학습을 통해 노드와 그래프의 표현을 최적화하여 다양한 그래프 관련 작업을 수행한다.

## 2. 임베딩을 사용하는 이유.
임베딩의 결과는 그래프의 구조, 정점 간의 관계, 정점이나 부분그래프와 연관된 다른 기타 부가 정보들을 나타낼 수 있어야 한다. 다양한 성질을 잘 파악할 수 있도록 여러가지 임베딩 방법들이 계속 연구되고 있다. 임베딩은 크게 두 가지로 나뉜다.

- **노드 임베딩**: 각 정점(= 노드)를 벡터로 표현한다. 정점 예측(Node Classification), 두 정점 사이의 링크 예측(Link Prediction)등의 문제를 풀거나, 시각화할 때 사용된다.
- **그래프 임베딩**: 그래프 전체를 하나의 벡터로 표현한다. 주로 그래프 예측(Graph Classification)등의 문제를 풀기 위해 사용된다. 예를 들어, 단백질은 각각 다른 구조를 가지고 있으며, 단백질의 구조를 각각 하나의 부분그래프로 구조화할 수 있다. 또한 분자식의 경우도 이에 해당한다.

인접행렬(Adjacent Matrix)대신 임베딩을 사용하는 가장 큰 이유는 **압축된 표현이라서 공간 효율이** 좋기 때문이다. 그래프의 정점의 개수를 $$N$$이라고 할때, 인접행렬의 크기는 $$N \times N$$이다. Wikidata와 같이 정점의 수가 천만 단위에 달하는 초거대 그래프는 인접행렬 자체만으로도 엄청난 공간을 차지한다. 반면, 임베딩은 정점의 표현식이 임베딩의 차원 수로 고정되기 때문에 훨씬 공간 절약을 할 수 있다.

## 3. 임베딩의 Challenge
임베딩을 통해 얻은 결과물은 만족시켜야할 필수요건들이 있다. 대표적으로 세 가지 필수요건이 있다.
- 임베딩이 **그래프의 성질을 잘 나타낸다고 확신**할 수 있어야 한다. 그래프의 연결상태, 주변구조 등을 잘 표현하고 있어야 한다. 임베딩을 사용한 시각화나 예측의 성능은 **임베딩 자체 성능**에 특히 큰 영향을 받기 때문이다.
- 그래프의 크기가 임베딩하는 속도에 주는 영향이 작아야 한다. 소셜 네트워크를 생각하면 알 수 있듯이 일반적으로 다뤄지는 그래프는 엄청 크다. **큰 그래프를 효율적**으로 다룰 수 있는 방법은 좋은 임베딩 방법의 필수요소이다.
- 가장 큰 어려움은 **임베딩 차원을 결정**하는 것이다. 차원을 늘리면 많은 정보를 담을 수는 있지만 성능의 증가에 비해 알고리즘의 시간복잡도나 공간복잡도가 크게 증가한다. 대부분의 연구들이 128 ~ 256사이로 설정을 한다.

<br/>
<br/>

# 노드 임베딩(Node Embedding)
## 1. 정의
<p align="center">
<img width="600" alt="1" src="https://github.com/user-attachments/assets/fc71d824-338d-452e-88f7-316a8494c726">
</p>

노드 임베딩(Node Embedding)은 정점을 <span style="color:red">**저차원의 벡터 공간**</span>에 임베딩하여 그래프의 구조적 정보를 저차원 표현으로 변환하는 방법이다. 이는 **정점 간의 관계**와 그래프의 **구조적 정보를 보존**하면서 벡터 공간에서 표현하는 방법이다. 정점 임베딩의 목적은 그래프에서 수행되는 다양한 머신러닝 작업(예: 정점 분류(Node Classification), 링크 예측(Link Prediction), 그래프 클러스터링(Graph Clustering) 등)을 효과적으로 처리하기 위함이다. 각 정점의 각자 특정한 벡터로 임딩되며, 비슷한 노드간의 벡터는 가깝게 임베딩된다.

## 2. 유사도(Similarity)
임베딩 공간에서 비슷한 위치로 맵핑되기 위해서는 벡터와 벡터 간의 **유사도**를 수학적으로 정의해야 한다. 가장 대표적인 방법은 두 벡터의 내적을 이용하는 것이다. 이외에도 코사인 유사도(Cosine Similarity), L2-norm 유사도 등이 존재한다.
- **내적 유사도**
  - 두 임베딩의 내적. 두 임베딩의 크기와 각도를 모두 고려.
  - Similarity = $$sim(u, v) = Z_u^ \cdot Z_v = \vert Z_u \vert \vert Z_v \vert cos(\theta)$$  
- **코사인 유사도**
  - 두 임베딩 이루는 각도만을 고려. 
  - Similarity = $$sim(u, v) = cos(\theta)$$ 
- **L2-norm 유사도**
  - 두 임베딩의 **크기**만을 고려 
  - Similarity = $$sim(u, v) = \vert Z_u \vert \vert Z_v \vert$$ 

## 3. Transductive VS Inductive
<p align="center">
<img width="600" alt="1" src="https://github.com/user-attachments/assets/e6504963-29ea-4e78-a48f-e8cbb7722a56">
</p>

노드 임베딩을 학습하는 방식에는 **Transductive**하게 하는 방식과 **Inductive**하게 하는 방식 두 가지가 존재한다.

- **Transductive**
  - 학습 결과: <span style="color:red">**노드 임베딩**</span> $$z_v$$
  - 학습 중 보지 못한 정점(Unseen node)에 대한 임베딩을 생성하지 못한다.

- **Inductive**
  - 학습 결과: <span style="color:red">**Encoder**</span>(= function)
  - 학습 중 보지 못한 정점(Unseen node)에 대한 임베딩을 생성할 수 있다.

일반화(Generalization)의 관점에서 Inductive Setting이 학습 중 등장하지 않은 정점에도 대응할 수 있으므로 일반화 능력이 더 좋다고 할 수 있다. 하지만, Inductive setting에서 학습하는 것이 대체적으로 더 어렵다.

## 4. Shallow Encoders
<p align="center">
<img width="300" alt="1" src="https://github.com/user-attachments/assets/00ac847e-e1a7-4fff-b27f-bda7c18d1b63">
</p>

이 그림은 Shallow Encoder의 작동 방식을 보여준다. Shallow Encoder는 노드 임베딩의 가장 기본적인 형태이다. 이 단일 은닉층은 정점 ($$u$$)를 임베딩하여 ($$ \mathbf{z}_u $$)로 매핑한다. 구체적으로, 정점 ($$u$$)와 정점 ($$v$$)의 임베딩을 찾은 후, 두 임베딩의 내적(dot product)을 계산하여 유사도를 구한다. 이는 다음 수식으로 표현된다. 여기서 ($$\mathcal{N}_R(u)$$)는 노드 ($$u$$)의 $$R$$-hop 이웃 집합을 나타낸다:

<center>$$
\mathbf{z}_u = f(\mathbf{z}_v, v \in \mathcal{N}_R(u))
$$</center>

이 알고리즘의 한계는 여러 가지가 있다. 첫째, **추정해야 하는 파라미터의 개수가 네트워크 내의 노드의 개수, 즉 ($$O(\vert V \vert)$$)와 동일**하다. 둘째, 노드(=정점) 사이의 어떤 **파라미터도 공유하지 않으며**, 모든 노드는 자신만의 고유한 임베딩을 가진다. 이는 모델이 각 노드에 대해 별도의 임베딩을 학습해야 한다는 것을 의미한다. 셋째, 학습 과정에서 **보지 못한 노드에 대한 임베딩을 생성할 수 없다**는 점이다. 이는 모델이 귀납적(inductive)이지 않고, **전이적(transductive)**이라는 것을 의미한다. 마지막으로, 이 알고리즘은 **노드 특성을 통합하지 못하므로**, 각 노드의 개별적 특성을 고려하지 않는다.

Shallow Encoder는 단순한 구조로 인해 계산 효율성이 높을 수 있으나, 위와 같은 제한사항으로 인해 복잡한 그래프 구조를 처리하는 데 한계가 있다. 특히, 새로운 노드(=정점)나 보지 못한 데이터에 대해 일반화하는 능력이 부족하며, 노드 간의 파라미터 공유가 없기 때문에 확장성이 떨어진다. 이러한 이유로, Shallow Encoder는 특정 용도에 적합할 수 있으나, 더 복잡하고 일반화 가능한 모델을 필요로 하는 경우에는 한계가 명확하다. 이는 일반적인 노드 임베딩의 단점과 일맥상통한다.


## 5. 노드 임베딩의 단점
<p align="center">
<img width="500" alt="1" src="https://github.com/user-attachments/assets/d40de70c-2653-4388-941c-743f4d82c2a9">
</p>
1. 노드 임베딩은 **Transductive**하다. 학습 중 등장한 정점들의 임베딩 벡터를 생성해야 하므로, 학습하지 못한 임베딩은 생성할 수 없다.
2. $$O(\vert V \vert d)$$의 파라미터가 필요하다. $$\vert V \vert$$는 정점의 개수이고 $$d$$는 차원수이다.
  - 각 정점은 파라미터를 공유하지 않는다.
  - 각 정점이 유니크한 임베딩을 가진다.
  - 정점 개수에 대해 Scale-Up이 안된다.
  - 따라서 초거대 그래프에서는 모든 노드 임베딩을 구하기 위해 매우 많은 컴퓨팅 자원을 요구하므로 Scalability가 떨어진다고 할 수 있다.

3. 정점의 feature가 통합되지 못한다.
  - 각 feature(age, gender, region, …)이 따로 임베딩 벡터를 만듬.

이러한 문제점을 극복하기 위해 나온 것이 바로 그래프 신경망(Graph Neural Network)이다. 그래프 신경망은 정점의 feature(Node feature)와 그래프의 구조 정보를 받아 MLP로 학습하는 방법이며, MLP의 파라미터를 공유할 수 있다. 이는 다음 포스팅에서 다루도록 하겠다. 



<br/>
<br/>

# 노드 임베딩의 대표적인 연구
## 1. DeepWalk
**DeepWalk**\[1\]는 랜덤 워크(random walk)를 사용해서 임베딩을 생성한다. 랜덤 워크는 그래프에서 정점들의 무작위로 걸어다니는 알고리즘이다. 한 정점에서 시작해서 임의의 이웃으로 가고 또 그 정점의 임의의 이웃으로 정해진 횟수만큼 이동한다. 이렇게 이동했을 때 방문한 정점들의 나열한 것을 랜덤 워크라고 한다. DeepWalk는 샘플링, Skim-graph 학습, 임베딩 계산의 세 단계로 구성된다.

\[가장 Naive한 형태의 랜덤 워크 알고리즘.\]
```python
import random
import networkx as nx

def random_walk(graph, start_node, walk_length):
    walk = [start_node]
    current_node = start_node
    
    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(current_node))
        if neighbors:
            next_node = random.choice(neighbors)
            walk.append(next_node)
            current_node = next_node
        else:
            break  # No more neighbors to walk to
    
    return walk
```

<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/3f46fd42-a283-4671-b3db-7f7b223d990e">
</p>

<span style="font-size:110%">**샘플링**</span>    
입력된 그래프에서 랜덤 워크를 샘플링한다. 각 정점에서 출발하는 랜덤 워크 경로를 여러 개(약 40개)씩 생성한다.

<span style="font-size:110%">**Skip-gram 학습**</span>  
위에서 생성된 랜덤 워크는 Word2Vec에서 문장에 대응된다. Skip-gram 신경망은 원-핫 인코딩된 정점 벡터를 입력받고 근처에 있는 정점을 잘 예측하도록 학습한다. 이 과정에서 앞뒤로 10개의 정점을 고려한다.

<span style="font-size:110%">**임베딩 계산**</span>  
DeepWalk는 랜덤 워크를 임의로 생성하기 때문에, 각각의 랜덤 워크 내에서는 인접한 정점들을 잘 반영할 수 있지만, 전체 그래프 구조를 고려했을 때는 정점의 주변 이웃 관계를 완벽하게 반영하지 못할 수 있다. (Node2vec이 이를 개선하고자 함.)

## 2. Node2vec
**Node2vec**\[2\]은 DeepWalk에서 랜덤 워크를 생성하는 방법을 발전시킨 기법이다. Node2vec은 두 개의 주요 파라미터 $$p$$와 $$q$$를 도입하여 **랜덤 워크를 생성할 때의 유연성을 높인다**.
- **파라미터 $$p$$**: 직전 정점으로 돌아올 가능성을 조절한다. $$p$$ 값이 작으면, 랜덤 워크가 최근에 방문했던 정점으로 다시 돌아올 확률이 높아져서 지역 탐색을 강화한다.
- **파라미터 $$q$$**: 직전 정점으로부터 얼마나 멀어질지를 조절한다. $$q$$ 값이 크면, 랜덤 워크가 새로운 영역을 탐색할 확률이 높아진다.

이를 통해, $$p$$는 **주변을 얼마나 잘 탐색하는지**를 조절하고, $$q$$는 랜덤 워크가 **새로운 곳을 얼마나 잘 발견하고 탐색**하는지를 조절한다. 참고로 $$p=q=1$$일 때, node2vec과 DeepWalk는 동일해진다. Node2vec은 그래프의 지역적 탐색과 글로벌 탐색을 모두 효과적으로 수행할 수 있는 임베딩을 생성하는 것을 목표로 한다. 이 방법은 두 가지 주요 탐색 전략을 혼합한다.
- **DFS(깊이 우선 탐색)** 기반 랜덤 워크: 정점의 지역적인 이웃 구조를 잘 반영한다.
- **BFS(너비 우선 탐색)** 기반 랜덤 워크: 정점의 글로벌 구조를 잘 반영한다.

Novd2vec도 DeepWalk와 마찬가지로 학습 과정이 세 단계로 구성된다. 램덤 워크 생성, Skip-graph학습, 임베딩 계산 순서대로 진행된다.

<span style="font-size:110%">**램덤 워크 생성**</span>    
파라미터 $$p$$와 $$q$$를 사용하여 각 정점에서 시작하는 랜덤 워크를 생성한다.

<span style="font-size:110%">**Skip-gram 학습**</span>  
생성된 랜덤 워크는 word2vec에서의 문장에 해당하며, Skip-gram 신경망은 원-핫 인코딩된 정점 벡터를 입력으로 받아 근처에 있는 정점을 예측하도록 학습된다. 이때 앞뒤로 10개의 정점을 고려한다.

<span style="font-size:110%">**임베딩 계산**</span>  
임베딩 벡터는 신경망의 은닉층 출력값을 사용하여 계산된다. Node2vec은 주어진 그래프의 모든 정점에 대해 이 임베딩을 계산한다.

## 3. SDNE
**SDNE(Structural Deep Network Embedding)**는 랜덤 워크를 사용하지 않기 때문에 이전의 두 방법과는 크게 다른 접근법이다. 이 방법을 소개하는 이유는 다양한 영역에서 성능이 안정적이기 때문이다. SDNE의 장점은 한 단계 이웃 관계와 두 단계 이웃 관계를 잘 반영한다는 점이다. 각각의 의미는 아래와 같다.

- 한 단계 이웃 관계는 연결된 두 정점 간의 비슷한 정도를 나타내며, 상대적으로 좁은 관계를 의미한다. 예를 들어, 어떤 논문이 다른 논문을 인용했다면 비슷한 내용을 다뤘다는 것을 의미하듯이, 두 정점이 연결되어 있으면 두 정점의 임베딩은 서로 비슷해야 한다.
- 두 단계 이웃 관계는 각 정점의 주변 구조가 얼마나 비슷한지를 나타내며, 상대적으로 넓은 관계를 의미한다. 예를 들어, 지수가 ('킬링 이브', '스캄 프랑스', '리틀 드러머 걸')을 좋아하고, 지은이가 ('킬링 이브', '스캄 프랑스', '체르노빌')을 좋아하면 둘의 취향은 비슷할 가능성이 높다.

그래프의 인접 행렬의 한 행을 그 행에 대응되는 정점의 인접 벡터라고 부르는데, 정의에 의해 이웃한 정점들에 해당하는 좌표의 값이 1이고 나머지는 0인 벡터가 된다. 정점의 인접 벡터가 아래 그림에 나오는 오토인코더의 입력값이다. 이 오토인코더를 바닐라 오토인코더라고 부르며, 두 단계 이웃 관계를 학습하게 된다.

<p align="center">
<img width="600" alt="1" src="https://github.com/user-attachments/assets/93004bfd-11ca-433c-8f77-bd6ba6449d0f">
</p>

양쪽에서 나온 임베딩의 거리를 계산하고 이 값을 신경망의 손실 함수에 포함한다. 선으로 연결된 모든 두 정점 쌍에 대해 거리를 계산하여 더함으로써 모델이 한 단계 이웃 관계를 잘 표현할 수 있게 한다.

SDNE 모델의 전체 손실 함수는 오토인코더의 손실 함수와 위에서 계산한 거리를 더한 값으로 정의된다. 이를 통해 SDNE는 한 단계 이웃 관계와 두 단계 이웃 관계를 모두 잘 표현할 수 있다.

## 4. Graph2vec

<p align="center">
<img width="600" alt="1" src="https://github.com/user-attachments/assets/b4acea91-771d-4fd1-a973-2451408f01ee">
</p>

마지막으로 소개할 방법은 그래프 전체를 벡터로 변환하는 것입니다. 즉, 벡터 하나가 그래프 전체를 나타낸다. **Graph2vec**은 doc2vec의 skip-gram 아이디어를 기초로 한다. 간단히 요약하면, 문서(document)의 아이디를 입력값으로 받고 문서에 포함된 단어를 예측하기 위한 학습 방법이다. 자세한 설명은 이 글을 참조하길 바란다. Graph2vec은 아래 세 단계로 구성된다.

<span style="font-size:110%">**그래프의 모든 뿌리가 있는 부분그래프(rooted subgraphs)를 샘플링하고 레이블을 다시 지정한다.**</span>  
그 결과로 위 그림에서 $$(m)$$개의 부분그래프를 얻는다. 여기에서 뿌리가 있는 부분그래프는 특정 정점(뿌리)을 중심으로 정해진 거리 안에 있는 정점의 집합이다.

<span style="font-size:110%">**skip-gram 모델을 학습한다.**</span>  
문서는 단어(혹은 문장)의 집합이고 그래프는 부분그래프의 집합이기 때문에 그래프를 문서와 비슷하게 볼 수 있다. 이런 관점으로 입력된 그래프에 부분그래프가 있을 확률을 계산하는 skip-gram 모델을 학습한다. 입력 그래프는 원-핫 인코딩된 벡터의 형태로 들어온다.

<span style="font-size:110%">**그래프의 아이디가 원-핫 인코딩 되어 들어오면 임베딩을 계산한다.**</span>  
DeepWalk와 마찬가지로 은닉층의 결과가 임베딩이 된다.

모델이 부분그래프를 예측하도록 설계되어 있기 때문에 두 그래프가 비슷한 부분그래프를 가지면, 즉 비슷한 구조를 가지면 비슷한 임베딩이 나오게 된다.


# Reference
\[1\] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. Deepwalk: online learning of social representations. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, KDD ’14. ACM  
\[2\] Aditya Grover and Jure Leskovec. 2016. node2vec: Scalable feature learning for networks.  
\[3\] Daixin Wang, Peng Cui, and Wenwu Zhu. 2016. Struc- tural deep network embedding. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16, page 1225–1234, New York, NY, USA. Association for Computing Machinery.  
\[4\]Annamalai Narayanan, Mahinthan Chandramohan, Rajasekar Venkatesan, Lihui Chen, Yang Liu, and Shantanu Jaiswal. 2017. graph2vec: Learning distributed representations of graphs.  
\[5\] [그래프 임베딩 요약](https://medium.com/watcha/%EA%B7%B8%EB%9E%98%ED%94%84-%EC%9E%84%EB%B2%A0%EB%94%A9-%EC%9A%94%EC%95%BD-bc2732048999)
