---
title: Chapter 7. Tree & Heap(힙)

categories: 
  - DataStructure
tags:
  - [DataStructure, Tree, Heap, Nonlinear Structure]

toc: true
toc_sticky: true

date: 2022-12-09
last_modified_at: 2022-12-09
---

## 1. Tree(트리)
### 1) Tree(트리)란?
연결리스트나 힙, 스택등의 앞서 공부한 자료구조들은 모두 선형 자료구조(Linear Data Structure)이다. 

반면, 트리는 <span style = "color:aqua">부모(parents)-자식(child) 관계를 **계층적**으로 표현</span>한 **비선형 자료구조(Nonlinear Data Structure)**이다.


<p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/111734605/206685637-e37173d4-ed51-4931-b595-498f496f5f4d.png">
</p>

### 2) Tree 구조에서 쓰이는 용어

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/206690396-257ca115-c14e-46df-af4f-75d745d1ef43.png">
</p>

### 3) 이진 트리(Binary Tree)
이진 트리(Binary Tree)란, 모든 노드의 **자식 노드가 2개**를 넘지 않는 트리를 말한다.  
대부분의 실제 사용되는 트리 구조는 이진 트리 구조이다.

<p align="center">
<img width="500" alt="1" src="https://user-images.githubusercontent.com/111734605/206693671-eb9ea74d-83bd-4e53-a896-685cc417b82d.png">
</p>

### 4) 이진 트리의 표현법
이진 트리를 표현하는 방법으로는 크게 **리스트**를 이용하는 방법과 **연결 리스트를 클래스를 정의**하는 방법이 있다.

#### (1) 하나의 리스트로 표현  

<p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/111734605/206695097-ab5dc200-e65e-454b-acd4-9ba5fab26e7c.png">
</p>


