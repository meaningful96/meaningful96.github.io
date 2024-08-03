---
title: "[그래프이론]What is the Graph?"
categories: 
  - Graph
  
toc: true
toc_sticky: true

date: 2024-08-03
last_modified_at: 2024-08-03
---

# 그래프(Graph)의 정의

<p align="center">
<img width="500" alt="1" src="https://github.com/user-attachments/assets/e28c01d8-02a9-428d-a908-7122c3ae19af">
</p>

그래프는 정점(Node, Vertex) 간선(Edge)으로 이루어진 자료 구조이다. 간선의 <span style="color:red">**방향성 유무**</span>에 따라 방향성 그래프(Directed Graph)와 무방향 그래프(Undirected Graph)로 구분된다. 그래프는 여러 형태가 존재하며, 그래프를 설명하기 위한 여러 가지 용어들이 있다.

<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/e5922a75-5ee6-4d1f-b4f3-ad7874491c58">
</p>

특히 인접, 차수 등의 용어는 그래프의 특성을 나타내는데 중요하다. 일반적인 그래프는 모든 연결성을 보여주는 행렬을 **인접 행렬(Adjacent Matrix)**이라고 하고, 하이퍼그래프(Hyper-Graph)에서의 연결성을 보여주는 행렬을 **연결 행렬(Incident Matrix)**라고 한다. 아래의 예시로 용어들을 설명할 수 있다.

<p align="center">
<img width="500" alt="1" src="https://github.com/user-attachments/assets/51513102-c594-45e8-afaa-aaf74f98dd4e">
</p>

위의 그래프는 정점이 A,B,C,D로 네 개인 그래프이다. 이 때, A 정점의 **차수는 곧 A와 연결되어 있는 간선의 수**를 나타낸다. 여기서 중요한 점은 차수는 연결된 정점의 수가 아닌 간선의 수라는 점이다. 일반적인 그래프들은 보통 간선에 레이블링이 되어 있거나, 지식 그래프(Knowledge Graph)의 경우 자연어로 된 정보를 포함하고 있다. 다시 말해, 간선이 여러 가지 타입을 가지게 된다. 이로 인해 두 정점이 여러 간선으로 연결될 수 있다. 따라서 <span style="color:red">**차수는 연결된 간선의 수**</span>로 정의한다. 경로는 두 정점 사이의 정점들을 나열한 것이다. 그리고 이 때 경로 상의 정점의 수를 경로 길이(Path length)라고 한다.

<br/>

# 그래프 표현 방법
<p align="center">
<img width="400" alt="1" src="https://github.com/user-attachments/assets/409da418-b80b-4731-9af8-11d8751475e2">
</p>
그래프를 표현하는 방법으로는 크게 인접 행렬(Adjacent Matrix)와 인접 리스트(Linked List)를 이용하는 방법이 있다. Python으로 그래프를 만드는 방법 또한 다양하며 대표적으로 Python의 기본 자료구조인 `dict`를 이용하는 방법과 `networkx`라는 패키지를 이용해 만들 수 있다.

## 1. 인접 행렬(Adjacent Matrix)
<p align="center">
<img width="600" alt="1" src="https://github.com/user-attachments/assets/f027407c-6d38-460f-8df3-dcebcd92fa36">
</p>

정점의 개수가 $$N$$일 때, $$N \times N$$ 행렬로 표현된다. 연결된 정점들의 행렬 성분은 $$1$$, 연결성이 없으면 $$0$$으로 채워 넣는다. 인접 행렬은 이차원 배열을 이용하는 방식이다. 이 방식은 **구현이 단순하지만, 인접 리스트에 비해 느리다**. 인접 행렬을 이용하여 그래프를 표현하는 방법은 두 가지 특징을 가진다.

- 특징
  - 정점의 개수가 $$N$$인 그래프는 항상 $$N^2$$의 메모리 공간이 필요하다.
  - 무방향 그래프의 인접 행렬은 대칭 행렬이다.
 
## 2. 인접 리스트(Linked List)
<p align="center">
<img width="600" alt="1" src="https://github.com/user-attachments/assets/0e1d5ebb-120e-48e9-937e-d01208410b5d">
</p>
인접 리스트(Linked List)로 표현해서, 각 정점에 인접하게 연결되어 있는 정점들을 순서에 상관없이 이어준다. 인접 행렬보다 **빠르지만 구현이 복잡**하다. 인접 리스트는 다음과 같은 특징을 가진다.

- 특징
  -	모든 정점을 인접 리스트에 저장한다. 정점의 번호만 알면 이 번호를 배열의 인덱스로 하여 각 정점의 리스트에 쉽게 접근 가능.
  -	무방향 그래프(Undirected Graph)에서 $$(h, t)$$의 간선은 두 번 저장됨.
  -	$$N$$개의 배열, $$2E$$개의 정점이 필요
  -	트리에서는 루트 노드에서 다른 모든 정점로 접근이 가능; Tree class 불필요
  -	그래프에선 특정 정점에서 다른 모든 정점로 접근이 불가능; Graph class 필요

## 인접 행렬 vs 인접 리스트
- **인접 행렬(Adjacent Matrix)**: 간선의 수가 많은 밀집 그래프(Dense Graph)에 적합.
  - **Pros**
    - 두 정점 사이 간선의 존재 여부를 바로 알 수 있음($$M_{i,j}$$를 $$O(1)$$안에 알 수 있음).
    - 정점의 차수는 $$O(N)$$안에 알 수 있다. = 인접 배열의 $$i$$번째 행 또는 열의 합.
  - **Cons**
    - 모든 관계를 기록함으로 정점의 개수가 많을수록 불필요한 메모리 낭비가 일어남.
    - 어떤 정점에 인접한 정점들을 찾기 위해서는 모든 정점을 순회해야 함.
    - 그래프에 존재하는 모든 간선의 수는 $$O(N^2)$$이다(인접 행렬 전체를 조사).

- 인접 리스트: 간선 수가 적은 희소 그래프(Sparse Graph)에 적합.


