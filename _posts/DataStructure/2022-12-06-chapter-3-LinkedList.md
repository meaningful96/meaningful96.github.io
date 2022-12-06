---
title: Chapter 3. Linked List(연결 리스트)

categories:
  - DataStructure
tags:
  - [DS, DataStructure, Stack]

toc: true
toc_sticky: true

date: 2022-11-28
last_modified_at: 2022-11-28 
---

## 1. Linked List(연결 리스트) 
### 1) 연결 리스트의 개념
- 연결 리스트는 각 노드가 한 줄로 연결되어 있는 자료 구조다.
- 각 노드는 (데이터(Key), 포인터(Link)) 형태를 가진다.
- 포인터: 다음 노드의 메모리 주소를 가리키는 목적으로 사용된다.

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/205756932-f43ea33b-dec8-4672-8101-db5141b6fc74.png">
</p>

- 즉, **연속적으로 메모리 할당을 받는것이 아니기 때문에** 현재 값의 다음값을 알기 위해 <span style = "color:aqua">**현재값(1.Key)**</span>과 다음값이 저장된 <span style = "color:aqua">**주소(2. Link)**</span>를 알아야 한다.
- 연결 리스트를 이용하면 다양한 자료구조를 구현할 수 있다.
- Python은 연결 리스트를 활용하는 자료구조를 제공한다.

### 2) 연결 리스트와 배열의 차이
#### (1) 연산 비교
- 배열: 특정 위치의 데이터를 삭제할 때, 일반적인 배열에서는 <span style = "color:aqua">O(N)</span>만큼의 시간이 소요된다.
- 열결 리스트: 단순히 연결만 끊어주면 됨. <span style = "color:aqua">O(1)</span>

#### (2) Array(배열)에서 삽입, 삭제  
**삽입**  
<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/205771610-8ac341cc-ecc2-46fc-b6a6-470e47236fa3.png">
</p>
- 최대 n개를 한 칸씩 밀어야 하기 때문에 O(N)이다.

**삭제**   
<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/205769632-2401b3f5-51b6-459c-9a35-cb10a30e3f2c.png">
</p>
- 최대 n개를 한 칸씩 당겨야 하기 때문에 O(N)이다.

#### (3) Linked List(연결 리스트)에서 삽입, 삭제
**삽입**  
<p align="center">
<img width="800" alt="1" src="">
</p>
- 데이터의 주소가 연속적으로 할당된것이 아니기 때문에 단순히 연결만 끊고 집어 넣어 포인터를 연결해주면 된다. 
- 하지만, 삭제하려는 데이터를 찾으려면 Head에서부터 포인터를 찾아서 따라가야하므로 최대 N개의 포인터를 따라가야한다.
- 따라서 O(N)

**삭제**

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/204198904-9d275b2b-db3e-40f5-b248-3f6e60e82bd8.png">
</p>

### 3) 시간 복잡도 구분

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/204203088-f1a85e22-9b92-48d9-9fe0-ce1f61179102.png">
</p>

## 2. 한반향 연결 리스트(단순 연결 리스트, Singly Linked List)
앞서 보여줬던 노드 하나당 하나의 링크를 가지는 연결 리스트이다. 링크는 다음 노드의 주소를 가리킨다.

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/205756932-f43ea33b-dec8-4672-8101-db5141b6fc74.png">
</p>

### 1) 메서드(Method)

- append(self, data): 가장 뒤에 노드 삽입  
- show(self): 모든 노드를 하나씩 출력  
- search(self, index): 특정 인덱스의 노드 찾기  
- insert(self, index, data): 특정 인덱스에 노드 삽입, O(1)  
  - 참고로 Insert 자체는 O(1)이지만,
  - 그 인덱스까지 Search가 수반된다.
  - 따라서, Insert[O(1)] + Search[O(N)] = O(N)이 된다. 
  - 단, <span style = "color:aqua">Insert 자체는 상수시간</span>이다.
  - 배열의 경우는 Insert가 O(N)  
- remove(self, index): 특정 인덱스의 노드 삭제
  - 마찬가지로 Remove 연산 자체는 O(1), 상수시간이다.

### 2) Python 

**[Input]**
```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    # 가장 뒤에 노드 삽입
    def append(self, data):
        # 헤드(head)가 비어있는 경우
        if self.head == None:
            self.head = Node(data)
            return
        # 마지막 위치에 새로운 노드 추가
        cur = self.head
        while cur.next is not None:
            cur = cur.next
        cur.next = Node(data)

    # 모든 노드를 하나씩 출력
    def show(self):
        cur = self.head
        while cur is not None:
            print(cur.data, end=" ")
            cur = cur.next

    # 특정 인덱스(index)의 노드 찾기
    def search(self, index):
        node = self.head
        for _ in range(index):
            node = node.next
        return node

    # 특정 인덱스(index)에 노드 삽입
    def insert(self, index, data):
        new = Node(data)
        # 첫 위치에 추가하는 경우
        if index == 0:
            new.next = self.head
            self.head = new
            return
        # 삽입할 위치의 앞 노드
        node = self.search(index - 1)
        next = node.next
        node.next = new
        new.next = next

    # 특정 인덱스(index)의 노드 삭제
    def remove(self, index):
        # 첫 위치를 삭제하는 경우
        if index == 0:
            self.head = self.head.next
            return
        # 삭제할 위치의 앞 노드
        front = self.search(index - 1)
        front.next = front.next.next


linked_list = LinkedList()
data_list = [3, 5, 9, 8, 5, 6, 1, 7]

for data in data_list:
    linked_list.append(data)

print("전체 노드 출력:", end=" ")
linked_list.show()

linked_list.insert(4, 4)
print("\n전체 노드 출력:", end=" ")
linked_list.show()

linked_list.remove(7)
print("\n전체 노드 출력:", end=" ")
linked_list.show()

linked_list.insert(7, 2)
print("\n전체 노드 출력:", end=" ")
linked_list.show()
```
**[Output]**
```python
전체 노드 출력: 3 5 9 8 5 6 1 7 
전체 노드 출력: 3 5 9 8 4 5 6 1 7 
전체 노드 출력: 3 5 9 8 4 5 6 7 
전체 노드 출력: 3 5 9 8 4 5 6 2 7 
```

## 3. 양방향 연결 리스트(Doubly Linked List)
### 1) 양방향 연결 리스트란?
- 양방향 연결 리스트는 한방향 연결 리스트의 단점을 보완한다.
- 한방향 연결 리스트의 경우, Search연산에서 시간 복잡도가 O(N)이다.

<p align="center">
<img width="400" alt="1" src="https://user-images.githubusercontent.com/111734605/205992528-f70bdece-b20f-4de6-b928-59355db93b43.png">
</p>

- 위의 그림은 한방향 연결리스트이다. 이때, tail 노드만 안다고 prev 노드는 알 수 없다. Why?
- 만약 tail 노드를 지우고 싶다.
  - tail 노드에서 prev 노드로 가는 링크 없다.
  - Head부터 따라가야 함
  - O(N) <span style = "color:aqua">**Bad!!**</span>
 
 ### 2) 양방향 연결 리스트의 장단점
 #### (1) Pros
 - List에 Node가 주어져 양방향으로 navigate가능
 - prev Node 주소가 없어도 삭제 가능!!
 #### (2) Cons
 - Extra pointer 필요, 더 많은 공간필요
 - 삽입, 삭제 시간 조금 더 필요함!!
 
 <p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/111734605/205992424-f7cb851d-e9f1-4256-ba65-2a01a0a197ce.png">
</p>

### 3) 원형 양방향 연결 리스트(Circular Doubly Linked List)
#### (1)원형 연결리스트의 빈 리스트
**Dummy Node**
