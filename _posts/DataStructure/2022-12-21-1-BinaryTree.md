---
title: Chapter 8.1 Binary & Binary Search Tree(이진트리, 이진 탐색트리)

categories: 
  - DataStructure
tags:
  - [DataStructure, Tree, Binary Tree, Nonlinear Structure]

toc: true
toc_sticky: true

date: 2022-12-21
last_modified_at: 2022-12-21
---

## 1. 이진 트리(Binary Tree)
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

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/208728948-d7a76e9b-7799-4ffd-99a9-d083641edf5d.png">
</p>  
(왼: preorder, 중: inorder, 오: postorder)

##### 순회 구현  
```python
class Node:
  def __init__(self, key):
    self.key = key
    self.parent = self.left = self.right = None
  def __str__(self):
    return str(self.key)
    
  def preorder(self): #현재 방문중인 노드 == self
    if self != None: # MLR
      print(self.key)
      if self.left:
        self.left.preorder()
      if self.right:
        self.right.preorder()
        
  def inorder(self):
    if self != None: # LMR
      if self.left:
        self.left.inorder()
      print(self.key)
      if self.right:
        self.right.inorder()
        
  def postorder(self):
    if self != None: # LRM
      if self.left:
        self.left.postorder()
      if self.right:
        self.right.postorder()
      print(self.key)
```  
print 문의 위치에 따라 MLR인지, LMR인지, LRM인지 나뉜다.

## 2. 이진 탐색 트리(Binary Search Tree)
### 1) 이진 탐색 트리(Binary Search Tree)의 정의
이진트리에서 search 연산을 보다 더 효율적으로 하기위한 자료구조이다. 영어로 <span style = "color:aqua">**Binary Search Tree, BST**</span>라고 부른다. 

이진 탐색트리를 만들기 위해서는 두 조건을 **반.드.시** 만족해야 한다.  
```

1. 각 노드의 왼쪽 subtree의 키값은 부모노드의 key값보다 작거나 같아야한다. 
2. 오른쪽 subtree의 키값은 노드의 key값보다커야한다.

```
여기서 반드시라는 것은, 두 조건이 <span style = "color:aqua">**모든 노드들에 대해**</span>적용되야 한다는 것이다. 

이 조건을 만족하게되면, 각 부모 노드를 기준으로 작은 값들이 왼쪽 subtree에, 큰 값들이 오른쪽 subtree에 저장된다.

이 상태로 search연산을 하게되면 트리와 마찬가지로 Level by Level로 이동하며 찾기 때문에, 시간복잡도가 <span style = "color:aqua">**트리의 높이**</span>에 따라 결정된다.  
즉, 시간 복잡도가 트리 높이에 dependent하다.

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/208811015-2c820b36-d078-412e-90c6-28bd8c31e5b5.png">
</p>

```
search(19)
1. 15 < 19 이므로(19가 크다) 오른쪽 subtree  
2. 20 > 19 이므로(19가 작다) 왼쪽 subtree
3. 17 < 19 이므로(19가 크다) 오른쪽 subtree
4. 19 = 19 이므로 search 완료

search(5)
1. 15 > 5 이므로(5가 작다) 왼쪽 subtree   
2. 4  < 5 이므로(5가 크다) 오른쪽 subtree
3. 오른쪽 subtree는 None: 찾는 값 없음.
```

### 2) Python 구현

#### (1) BTS 클래스 정의 코드
BTS 클래스에는 root와 size 정보, iterator가 정의된다.

```python
class BST:
    def __init__(self):
        self.root = None
        self.size = 0
    
    def __len__(self):
        return self.size

    def __iter__(self):
        return self.root.__iter___() 
```

#### (2) find_loc함수 구현  
BTS에서 search나 insert를 하기위해서는 현재 찾고자하는 키값이 어디있는지 찾아야한다. 이를 `find_loc()`메서드로 구현할 수 있다.

1. 만약
- key값 노드가 있다.  ➜ 해당 노드 리턴
- key값 노드가 없다.  ➜ 그 노드가 삽입되어야하는 **부모노드**를 리턴

2. Search 연산에서 if
  - search 연산의 리턴값 = search 값  ➜ 그 노드가 있다.
  - search 연산 의리턴값 ≠ search 값  ➜ 그 노드가 없다.

3. Insert 연산에서 if
  - insert 연산의 리턴값 = insert 값 ➜ 이미 있는 값이다.
  - insert 연산의 리턴값 ≠ insert 값 ➜ 둘이 다르면 부모 노드가 리턴된 것이므로 해당 부모노드의 자식노드로 insert값을 삽입한다.
 
```python
def find_loc(self,key):
    #트리 크기가 0일때
    if self.size == 0:
        return None

    # 부모노드부터 find시작
    p = None
    v = self.root

    while v: #v에 값이 있어야함
        #v의 key값이 찾고자하는 key값과 같으면 해당노드리턴
        if v.key == key:
            return v

        #다르면 p에는 현재노드를, v에는 key값과 비교해 left or right 부여
        elif v.key < key:
            p = v
            v = v.right
        else:
            p = v
            v = v.left
    
    #while문에서 return되지 않으면 p에 leaf(트리의 끝부분)이 저장되어있다.
    return p 
```

#### (2) search(탐색)
```python
def search(self,key):
    #p에는 
    #find_loc으로 찾았다면 해당키의 현재 노드가,
    #찾지 못했다면 해당키가 들어가야하는 부모노드가 할당
    p = self.find_loc(key)

    #p에 찾은 key의 노드가 들어가있다면 p.key == key
    if p.key == key:
        #해당 노드 리턴
        return p

    #p에 부모노드가 할당되어 p.key가 매개변수 key와 다르다면    
    else:
        #None리턴
        return None
```

#### (3) insert(삽입)
```python
def insert(self,key):
    #p에는 
    #find_loc으로 찾았다면 해당키의 현재 노드가,
    #찾지 못했다면 해당키가 들어가야하는 부모노드가 할당
    p = self.find_loc(key)

    #p가 None이라면 BST가 비어있는 경우라서 추가해야만하는 경우
    #p.key와 key가 다르다면 p에는 새로운 key를 만들 부모노드가 할당
    #즉, insert 하는 경우
    if p == None or p.key != key:
        #해당 key로 새로운 노드 선언
        v = Node(key)
        
        #p가 None이라면 BST의 root노드를 새로 정의해줘야함
        if p == None:
            self.root = v

        else:
            #새로운 노드의 부모를 p로 설정
            v.parent = p

            #부모노드와 새로운노드의 key를 비교해 연결
            if p.key >= key:
                p.left = v
            else:
                p.right = v
        
        #insert 성공시, BST의 크기 1 증가
        self.size += 1
        return v

    # BST가 비어있지 않으면서 p.key와 key가 같다면
    # BST에 이미 insert하고자하는 key값이 존재하는 것
    # 즉, insert하지 않는경우
    else:
        print('key is already in tree')
        # p에 insert하고자하는 key값의 노드가 할당되어있음
        # 중복허용하지 않으면 return None
        return p
```

#### (4) delete(삭제)   
삭제 방법은 두 가지 방법이 있다.
- merging
- copying

1. **merging**
x라는 노드를 지울 때, x노드가 키값을 찾아야하는데, `find_loc`함수로 키값의 위치를 찾아 **merging**함수에 넘겨주면된다.

제거 후 만약 삭제노드가 leaf노드가 아닌경우, 연결된 subtree노드들의 부모 노드를 다시 설정해 주어야 한다.
- 여기서 merging 과 copying의 차이점이 생김
- mearging은 삭제된 노드의 왼쪽 subtree를 삭제된 노드로 옮김
- 왼쪽 subtree에서 가장 큰 노드 m의 오른쪽 자식 노드로 오른쪽 subtree를 연결한다.

<p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/111734605/208818157-169f0763-caad-481d-9f11-6b4d800459f1.png">
</p>

위 방식을 적용할 때 2가지 경우로 나뉜다.

- 삭제할 x가 루트노드인 경우
  - left subtree에서 가장 큰 값 m의 right subtree로 삭제한 x의 right subtree가 위친한다. 그리고 left subtree의 최상단 노드가 새로운 root 노드가 된다.
  - 만약 left subtree가 None이라면 x의 right subtree가 새로운 루트가되고, right subtree의 최상단 노드가 새로운 root 노드가 된다.

- 삭제할 x가 루트노드가 아닌 경우
  - x의 부모노드에 left subtree를 연결한다. 그리고 left subtree에서 가장 큰값 m에 right subtree를 연결한다.
  - 만약 left subtree가 None이라면 right subtree가 x의 부모노드에 직접 연결된다.

**Psuedo 코드**  
코드로 구현할 때 고려사항으로 <span style = "color:aqua">**left subtree의 존재유무**</span>와 <span style = "color:aqua">**삭제한 노드 x가 root인지 아닌지**</span>로 나뉜다.
```python
def deleteByMerging(self, x):
    a = x.left
    b = x.right
    pt = x.parent
    #c =  #x 자리를 대체할 노드
    #m =  #Left에서 가장 큰 노드

    #left subtree가 None일때
    if a == None:
        #x자리에 right subtree b를 직접 넣는다.
        c = b
        

    #left subtree가 None이 아닐때
    else:
        c = a

        #a에서 가장 큰 key값 가진 노드 m 찾기
        m = a
        while m.right:
            m = m.right
        
        # right subtree가 존재한다면 
        if b:
            # b와 m을 연결
            b.parent = m
            m.right = b

    
    #삭제한 x노드의 부모가 None인경우
    if pt == None:
        #x가 root노드였으므로, c를 root로 업데이트
        if c:
            c.parent = None
        self.root = c

    #x노드의 부모가 None이 아닌경우
    else:
        # x자리를 대체할 노드 c에 pt를 연결한다.
        if c:
            c.parent = pt

        # pt와 c의 key값을 비교해 연결한다.
        if pt.key > c.key:
            pt.left = c
        else:
            pt.right = c

    self.size -= 1

    # 리턴은 다음에 다시 설명
 ```

2. **copying**

merging과는 다르게 **copying**은 <span style = "color:aqua">left subtree에서 가장 큰값 m을 찾아 삭제된 노드 x의 자리에 대체</span>한다.

m이 left subtree에서 가장 큰 값이므로 m의 right subtree는 없다는 것이 보장되므로,
m의 right subtree로 x의 right subtree를 연결한다.

m의 원래 left subtree와 m의 부모노드가 끊긴 상태이므로, 이 둘을 연결한다.
m이 x의 위치로 올라갔으므로 m과 left subtree의 최상단 노드를 연결한다.

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/208823286-34544296-cb8d-4af7-8169-d8f0a3dc53ee.png">
</p>

**Pseudo 코드**
```python
def deleteByCoping(self,key):
    a = x.left
    b = x.right
    pt = x.parent
    #c =  #x 자리를 대체할 노드
    #m =  #Left에서 가장 큰 노드

    if a == None:
        #a가 None이면 m이 없으므로 x의 right subtree를 직접대체해야한다.
        c = b
    else:
        #a가 None이 아니라면 m 이 존재한다.
        m = a
        while m.right:
            m = m.right
        #m이 x의 자리를 대체한다.
        c = m

        #만약 x의 right subtree가 존재한다면 둘을 연결
        if b:
            m.right = b
            b.parent = m

        #m이 존재할때, m의 right subtree는 없는 것이 보장되나, left subtree는 있을수도있다.
        if m.left:
            #m의 parent와 연결한다.
            m.left.parent = m.parent.right
            m.parent.right = m.left


    #삭제한 x가 root인경우
    if pt == None:
        if c:
            c.parent = None
        self.root = c


    #x가 root가 아닌경우
    else:
        # x자리를 대체할 노드 c에 pt를 연결한다.
        if c:
            c.parent = pt

        # pt와 c의 key값을 비교해 연결한다.
        if pt.key > c.key:
            pt.left = c
        else:
            pt.right = c
```

### 3) 시간 복잡도
search 연산은 최악의 경우, 노드의 가장 깊은 곳까지 비교해야하므로 트리의 높이에 비례하는 수행시간을 가진다. ➜O(h)

insert 연산또한 search연산을 이용하므로 O(h)가 걸린다.

delete연산 또한 m을 찾기위해 최악의 경우 h만큼 비교하기 때문에 O(h) 시간이 걸린다.

트리의 높이 h는 최악의 경우 right subtree로만 연결되기때문에 n-1까지 가능하므로
모든 연산시간은 <span style = "color:aqua">O(n)</span> 만큼 걸린다.

따라서 트리의 높이를 최적화해야 탐색 수행시간이 줄어든다.
이를 위해 균형이진탐색트리(AVL)가 존재한다.
