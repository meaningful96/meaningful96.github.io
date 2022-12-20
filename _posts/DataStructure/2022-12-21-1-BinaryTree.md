---
title: Chapter 8. Binary Tree & Binary Search Tree(이진트리, 이진 탐색트리)

categories: 
  - DataStructure
tags:
  - [DataStructure, Tree, Binary Tree, Nonlinear Structure]

toc: true
toc_sticky: true

date: 2022-12-21
last_modified_at: 2022-12-21
---

## 1. 이진 트리
### 1) 이진 트리(Binary Tree)의 정의
이진 트리는 자식의 노드가 2개 이하인 트리이다. 일반적으로 자식 노드가 많으면 많을수록 유용한데, 삽입과 삭제연산이 매우 복잡해진다. 따라서 이진 트리를 많이 사용한다.
- 최대 자식 노드의 수와 연산 복잡도는 **Trade-off**

이진트리를 표현하는 방법 중 가장 먼저 공부한 방법은 배열 또는 리스트에 저장하는 방식이다. 이는 앞선 포스팅인 [Heap 자료구조](https://meaningful96.github.io/datastructure/TreeandHeap/)에서 다뤘다.
힙(Heap)은 이진 트리의 특이 케이스로, 이진 트리의 모양 성질과 힙성질 모두를 만족시켜야 하기때문에 <span style = "color:aqua">**makeheap**</span>이라는 함수를 구현했다.
- Python에서는 **headq 모듈**을 이용해 사용가능

하지만, 이렇게 배열 또는 리스트로 나열을하면은 메모리 관점에서 굉장한 낭비가 된다. 이는 연결 리스트를 이용하면 해결 가능하다.

### 2) 연결 리스트를 이용한 표현
연결리스트는 하나의 노드에 링크와 키(value)로 구성된 자료구조이다. 이진트리를 링크와 키로 표현하면, 총 세 개의 링크와 한 개의 키값이 필요하다.
- 링크: 부모링크(Parents link), 왼쪽 자식링크(Left Child), 오른쪽 자식링크(Right Child)
- key: 값

<p align="center">
<img width="110%" alt="1" src="https://user-images.githubusercontent.com/111734605/208725348-debdd633-2f59-4f23-8314-7741a39ee5da.png">
</p>

### 3) Python 구현

#### (1) Node Class 정의  
```python
class Node: #초기값은 None
    def __init__(self,key=None,parent=None,left=None,right=None):
        self.key = key
        self.parent = parent
        self.left = left
        self.right = right

        #print시 출력할 것
    def __str__(self):
        return str(self.key)
```

#### (2) 순회  
Binary 클래스에서 선언된 노드들의 key값을 모두 출력하고 싶을때는 각 노드들을 방문하는 일정한 규칙인 <span style = "color:aqua">**순회(Traversal)**</span>을 이용한다.  
순회에는 총 3가지 방법이 있다.
- preorder : MLR 방식
- inoder   : LMR 방식
- postorder: LRM 방식
(M = 자기 자신, L = 왼쪽 자식노드, R = 오른쪽 자식 노드)
**이 방식은 각 노드들에서** <span style = "color:aqua">**재귀적**</span>**으로 적용한다.**

