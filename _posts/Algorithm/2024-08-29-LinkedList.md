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
<img width="500" alt="1" src="https://github.com/user-attachments/assets/36c4fd96-9f6f-4cd8-8f84-880cc91ae4b0">
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
## 양방향 연결 리스트(Doubly Linked List)
<p align="center">
<img width="600" alt="1" src="https://github.com/user-attachments/assets/4a9c7f56-a75b-4674-9bb5-7edd84b1a068">
</p>

**양방향 연결 리스트(Doubly Linked List)**는 **각 노드가 두 개의 링크**(다음 노드를 가리키는 **next**와 이전 노드를 가리키는 **prev**)를 가지는 연결 리스트이다. 이 구조 덕분에 리스트 내의 **어떤 노드에서든 양방향으로 이동**할 수 있다. 또한, 리스트의 **첫 번째 노드와 마지막 노드를 항상 추적**할 수 있어 데이터 조작이 유연하다.

- **양방향 연결 리스트의 핵심 요소**
  - **노드(Node)**: 데이터를 저장하는 필드와, 다음 노드를 가리키는 **next 포인터**, 이전 노드를 가리키는 **prev 포인터**를 가진다.
  - **헤드(Head)**: 리스트의 첫 번째 노드를 가리키며, **prev 포인터**는 항상 **NULL**을 가리킨다.
  - **테일(Tail)**: 리스트의 마지막 노드를 가리키며, **next 포인터**는 항상 **NULL**을 가리킨다.

### 양방향 연결 리스트의 장단점
- **양방향 탐색 가능**: 리스트 내에서 **양방향으로 이동**할 수 있으므로, 어떤 노드에서든 쉽게 앞뒤로 탐색할 수 있다.
- **노드 삭제의 유연성**: 이전 노드의 주소를 알지 않아도 특정 노드를 삭제할 수 있다. 이는 노드의 **prev 포인터** 덕분이다.

### 양방향 연결 리스트의 단점
- **추가적인 메모리 사용**: 각 노드는 **두 개의 포인터**(`next`와 `prev`)를 가지므로, 더 많은 메모리가 필요하다.
- **삽입 및 삭제의 복잡성**: 노드를 삽입하거나 삭제할 때 **더 많은 포인터 연산**이 필요하므로, 작업 시간이 길어질 수 있다.

양방향 연결 리스트는 **양방향 연결**을 통해 더 많은 유연성을 제공하지만, 메모리와 성능 면에서 추가적인 비용이 발생한다. 이러한 이유로, 데이터의 **양방향 탐색이 빈번히 필요한 경우**에 적합하다. 또한, 양방향 연결 리스트는 **원형 양방향 연결 리스트(Circular Doubly Linked List)**로 확장될 수 있다. 이 경우, 마지막 노드의 **next 포인터**가 첫 번째 노드를 가리키고, 첫 번째 노드의 **prev 포인터**가 마지막 노드를 가리켜 리스트가 원형으로 연결된다. 이 구조는 리스트의 양쪽 끝에서부터 빠르게 데이터 접근이 필요한 경우 유용하다.

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None  # 양방향을 위해 이전 노드를 가리키는 포인터 추가


class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None  # 리스트의 마지막 노드를 추적하기 위해 tail 추가

    # 가장 뒤에 노드 삽입
    def insert_at_end(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node

    # 모든 노드를 하나씩 출력 (앞에서 뒤로)
    def traverse_forward(self):
        cur = self.head
        while cur is not None:
            print(cur.data, end=" ")
            cur = cur.next
        print()

    # 모든 노드를 하나씩 출력 (뒤에서 앞으로)
    def traverse_backward(self):
        cur = self.tail
        while cur is not None:
            print(cur.data, end=" ")
            cur = cur.prev
        print()

    # 특정 인덱스(index)의 노드 찾기
    def search(self, index):
        node = self.head
        for _ in range(index):
            node = node.next
        return node

    # 특정 인덱스(index)에 노드 삽입
    def insert_at_index(self, index, data):
        new_node = Node(data)
        # 첫 위치에 추가하는 경우
        if index == 0:
            new_node.next = self.head
            if self.head is not None:
                self.head.prev = new_node
            self.head = new_node
            if self.tail is None:  # 리스트가 비어 있었던 경우
                self.tail = new_node
            return
        
        # 삽입할 위치의 앞 노드
        prev_node = self.search(index - 1)
        next_node = prev_node.next

        prev_node.next = new_node
        new_node.prev = prev_node
        new_node.next = next_node

        if next_node is not None:
            next_node.prev = new_node
        else:
            self.tail = new_node  # 새 노드가 마지막 노드인 경우 tail 갱신

    # 특정 인덱스(index)의 노드 삭제
    def delete_at_index(self, index):
        if self.head is None:
            return  # 리스트가 비어 있는 경우

        # 첫 위치를 삭제하는 경우
        if index == 0:
            self.head = self.head.next
            if self.head is not None:
                self.head.prev = None
            else:
                self.tail = None  # 삭제 후 리스트가 비게 된 경우
            return

        # 삭제할 위치의 앞 노드
        node_to_delete = self.search(index)
        prev_node = node_to_delete.prev
        next_node = node_to_delete.next

        prev_node.next = next_node

        if next_node is not None:
            next_node.prev = prev_node
        else:
            self.tail = prev_node  # 삭제한 노드가 마지막 노드인 경우 tail 갱신

```

## Runner (or Second Pointer) Technique
**Runner (or Second Pointer) Technique**는 **두 개의 포인터를 사용**하여 연결 리스트를 탐색하는 기법이다. 이는 효율적으로 리스트를 탐색하거나 특정 문제를 해결하는 데 유용하다. 이 방법은 빠른 포인터와 느린 포인터를 동시에 사용하여 리스트 내의 패턴을 탐지하거나 리스트의 특정 요소를 찾아낸다.

- **Fast** 포인터: 느린 포인터보다 더 빠르게 이동한다. 일반적으로 한 번에 두 노드씩 이동한다.
- **Slow** 포인터: 한 번에 한 노드씩 이동한다

Runner (or Second Pointer) Technique는 리스트의 중간 지점을 찾고, 사이클을 검출하며, 리스트 병합 및 교차 문제를 해결하는 데 사용된다.
- **중간 지점 찾기**: 리스트를 순회하면서 중간에 위치한 노드를 효율적으로 찾는다.
- **사이클 검출**: 리스트를 탐색하며 사이클이 있는지 확인한다. 빠른 포인터가 느린 포인터를 따라잡으면 사이클이 존재함을 의미한다.
- **리스트 병합 및 교차 문제 해결**: 두 리스트를 특정 패턴으로 병합하거나 재정렬할 때 사용된다.

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    # 리스트 끝에 노드 삽입
    def insert(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        last = self.head
        while last.next:
            last = last.next
        last.next = new_node

    # 리스트의 중간 노드를 찾기 위한 Runner Technique
    def find_middle(self):
        slow = self.head
        fast = self.head

        # 빠른 포인터는 두 칸씩, 느린 포인터는 한 칸씩 이동
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        # 느린 포인터가 중간 노드를 가리킴
        return slow.data

    # 리스트를 출력
    def print_list(self):
        current = self.head
        while current:
            print(current.data, end=" ")
            current = current.next
        print()
```

# Reference
\[1\] Lecture: ITG6022 Computational Problem Solving  
