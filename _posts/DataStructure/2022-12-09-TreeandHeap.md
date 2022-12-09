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
**레벨 0부터** 차례대로, **왼쪽에서 오른쪽 순서**로 작성한다. 자식노드가 없는 경우는 None으로 작성한다.  

`장점`
이렇게 트리를 표현하면, `상수시간의 연산`으로 자식노드와 부모노드를 찾을 수 있다.

왜냐하면 현재 노드의 인덱스번호를 알고 있다면, 자식노드와 부모노드의 인덱스번호를 계산할 수 있기 때문이다.
* 자식노드
  - 왼쪽 자식노드: $$(Index)  \times  2 + 1$$
  - 오른쪽 자식노드: $$(Index)  \times  2 + 2$$ 
* 부모 노드  
  - $$\frac{(Index) -1}{2}$$의 **몫**
  - (Index - 1)//2

```
부모 노드의 인덱스가 0 (a 노드)
이때 왼쪽 자식노드는 A[0*2 + 1] = A[1]이다. (b노드)

부모 노드의 인덱스가 3일때 (노드 c)
오른쪽 자식노드는 A[2*2+2] = A[6]이다. (f노드)
```

`단점`  
불필요한 `메모리 낭비`가 발생한다. 
- 연산 시간 ⇋ 메모리  `Trade-off`

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/206702907-46470b0e-a3a0-4287-860d-67a68536d3a9.png">
</p>  

노드가 실제로는 비어있지만, 하나의 리스트로 표현해야 하기에, 그 빈 노드에 `None`을 채워넣게 되고  
그에따라 차지하는 메모리는 증가한다.

#### (2) 리스트를 중복해서 표현

<p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/111734605/206698717-8d1e749c-ae71-443b-a9e1-58507cadd18c.png">
</p>  

[루트, [루트의 **왼쪽** 부트리], [루트의 오른쪽 부트리]]형식으로 재귀적으로 정의

#### (3) 연결 리스트로 표현

<p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/111734605/206700248-d884b578-da17-4ef8-a7dc-fb15ca0c4369.png">
</p>  
각 노드가 key, parent, left, right 에 대한 정보를 가진다. 단 루트 노드는 제외다.(루트 노드는 부모 노드가 없다.)


## 2. Heap(힙)
### 1) 힙의 개념
모양 성질과 힙 성질을 만족하는 리스트에 저장된 값의 시퀀스이다.

```
모양 성질: 
        1. 마지막 레벨을 제외한 각 레벨의 노드가 모두 채워져 있어야 한다.
        2. 마지막 레벨에선 노드들이 왼쪽부터 채워져야한다.

힙 성질:
        1. 루트 노드를 제외한 모든 노드에 저장된 값(key)은 자신의 부모노드의 
           보다 크면 안된다.
 ```  
 <span style = "color:aqua">보통 리스트의 데이터는 힙성질, 모양성질을 갖춘 데이터가 주어지지 않는다.</span>  
 따라서 힙 성질을 갖춘 데이터가 되도록 리스트의 데이터들을 **재정렬**해 주어야 한다.
 - makeheep() 함수 이용
  
  
