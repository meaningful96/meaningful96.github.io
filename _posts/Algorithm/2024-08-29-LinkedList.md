---
title: "[알고리즘]연결 리스트(Linked List)"

categories: 
  - Algorithm

toc: true
toc_sticky: true

date: 2024-08-28
last_modified_at: 2024-08-28
---

# 연결 리스트(Linked List)

<p align="center">
<img width="600" alt="1" src="https://github.com/user-attachments/assets/36c4fd96-9f6f-4cd8-8f84-880cc91ae4b0">
</p>

**연결 리스트(Linked List)**는 컴퓨터 과학에서 사용하는 기본적인 선형 자료 구조 중 하나로, 각 요소가 데이터와 다음 요소를 참조하는 정보를 포함하는 **노드(node)**로 구성된다. 이 자료 구조는 배열과는 다르게 데이터의 동적 추가와 삭제가 상대적으로 쉬운 특징이 있다. 그러나 특정 위치의 노드를 검색하기 위해 처음부터 차례대로 접근해야 하므로, 검색 속도는 **배열(Array)보다 느리다는 단점**이 있다.

<p align="center">
<img width="700" alt="1" src="https://github.com/user-attachments/assets/55e553b8-4006-44a5-99a9-89951d23a86b">
</p>


## 연결 리스트의 핵심 요소
- **노드(Node)**: 연결 리스트의 기본 단위로, 데이터를 저장하는 **데이터 필드**와 다음 노드를 가리키는 **링크 필드**로 구성된다.
- **포인터**: 각 노드 안에서, 다음이나 이전의 노드와의 연결 정보를 가지고 있는 공간이다.
- **헤드(Head)**: 연결 리스트에서 가장 처음 위치하는 노드를 가리키며, 리스트 전체를 참조하는 데 사용된다.
- **테일(Tail)**: 연결 리스트에서 가장 마지막 위치하는 노드를 가리키며, 이 노드의 링크 필드는 **NULL**을 가리킨다.

연결 리스트는 단일 연결 리스트 외에도 **양방향 연결 리스트(Doubly linked list)**나 **원형 연결 리스트(Circular linked list)**와 같이 여러 형태로 확장될 수 있다. 예를 들어, 양방향 연결 리스트는 각 노드가 이전 노드와 다음 노드를 모두 참조할 수 있으며, 원형 연결 리스트는 마지막 노드가 처음 노드를 참조하여 원형 구조를 형성한다.



이와 같은 구조적 특징 때문에 연결 리스트는 데이터의 추가나 삭제가 빈번히 일어나는 상황에 적합하다. 반면, 특정 위치의 데이터를 빠르게 접근해야 하는 경우에는 배열이 더 효율적이다. 따라서 연결 리스트와 배열은 각기 다른 장단점을 가지고 있으며, 사용 목적에 따라 적합한 자료 구조를 선택하는 것이 중요하다.

## 연결 리스트의 Basic Operation
- **Traversing(순회)**: 리스트의 모든 노드를 순서대로 방문하는 연산
- **Searching(검색)**: 특정 데이터를 가진 노드를 찾는 연산
- **Inserting(삽입)**: 새로운 노드를 리스트에 추가하는 연산
- **Deleting(삭제)**: 리스트에서 특정 노드를 제거하는 연산

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    # 가장 뒤에 노드 삽입
    def Inserting(self, data):
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
    def Traversing(self):
        cur = self.head
        while cur is not None:
            print(cur.data, end=" ")
            cur = cur.next

    # 특정 인덱스(index)의 노드 찾기
    def Searching(self, index):
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
        node = self.Searching(index - 1)
        next = node.next
        node.next = new
        new.next = next

    # 특정 인덱스(index)의 노드 삭제
    def Deleting(self, index):
        # 첫 위치를 삭제하는 경우
        if index == 0:
            self.head = self.head.next
            return
        # 삭제할 위치의 앞 노드
        front = self.Searching(index - 1)
        front.next = front.next.next
```
\[**실행문**\]

```python
linked_list = LinkedList()
data_list = [3, 5, 9, 8, 5, 6, 1, 7]

for data in data_list:
    linked_list.Inserting(data)

print("전체 노드 출력:", end=" ")
linked_list.Traversing()

linked_list.insert(4, 4)
print("\n전체 노드 출력:", end=" ")
linked_list.Traversing()

linked_list.Deleting(7)
print("\n전체 노드 출력:", end=" ")
linked_list.Traversing()

linked_list.insert(7, 2)
print("\n전체 노드 출력:", end=" ")
linked_list.Traversing()
```
```bash
전체 노드 출력: 3 5 9 8 5 6 1 7 
전체 노드 출력: 3 5 9 8 4 5 6 1 7 
전체 노드 출력: 3 5 9 8 4 5 6 7 
전체 노드 출력: 3 5 9 8 4 5 6 2 7
```


# Reference
\[1\] Lecture: ITG6022 Computational Problem Solving  
