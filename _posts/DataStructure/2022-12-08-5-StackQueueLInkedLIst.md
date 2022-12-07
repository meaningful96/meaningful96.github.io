---
title: Chapter 5. Stack과 Queue, Deque를 Linked List로 구현하기
categories: 
  - DataStructure
tags:
  - [DataStruture, Stack, Queue, Deque, Linked List]

toc: true
toc_sticky: true

date: 2022-12-08
last_modified_at: 2022-12-08
---

## 1. Stack with Linked List

### 1) 스택에 쓰이는 연산

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/206262118-299a7d22-1f04-4dca-b0a5-ec46b0f16bb9.JPG">
</p>

- 파이썬의 기본적인 리스트 자료형은 append()와 pop()메서드를 제공
- append와 pop 모두 시간 복잡도는 O(1)
- 따라서 일반적으로 Stack을 구현할때는 List를 사용한다.

### 2) 리스트를 이용한 스택 구현

```python
class Stack:
    def __init__(self):
        self.stack = []

    def push(self, data):
        # 마지막 위치에 원소 삽입
        self.stack.append(data)

    def pop(self):
        if self.is_empty():
            return None
        # 마지막 원소 추출
        return self.stack.pop()

    def top(self):
        if self.is_empty():
            return None
        # 마지막 원소 반환
        return self.stack[-1]

    def is_empty(self):
        return len(self.stack) == 0
```

### 3) 연결 리스트로 스택 구현
- 연결 리스트로 스택을 구현하면 **삽입(push)와 삭제(pop)연산에 있어서** <span style="color:aqua">**O(1)**</span>**의 시간 복잡도(Time Complexity)**를 보장한다.
- 연결 리스트로 구현할 때는 **머리(head)**를 가리키는 하나의 포인터만 가진다.
- **머리(head)**: 남아있는 원소 중 **가장 마지막에 들어 온 데이터**를 가리키는 포인터

- **삽입**할 때는 머리(head) 위치에 데이터를 넣는다
- **삭제**할 때는 머리(head) 위치에서 데이터를 꺼낸다.

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/206263388-a2be0da4-4f60-4c81-bd9a-e2a4a72769ce.png">
</p>

**[Input]**
```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class Stack:
    def __init__(self):
        self.head = None

    # 원소 삽입
    def push(self, data):
        node = Node(data)
        node.next = self.head
        self.head = node

    # 원소 추출하기
    def pop(self):
        if self.is_empty():
            return None

        # 머리(head) 위치에서 노드 꺼내기
        data = self.head.data
        self.head = self.head.next

        return data

    # 최상위 원소(top)
    def top(self):
        if self.is_empty():
            return None
        return self.head.data

    # 먼저 추출할 원소부터 출력
    def show(self):
        cur = self.head
        while cur:
            print(cur.data, end=" ")
            cur = cur.next

    # 스택이 비어있는지 확인
    def is_empty(self):
        return self.head is None

## 클래스를 실행해보기, for문을 이용한 generator를 이용해 실행
stack = Stack()
arr = [9, 7, 2, 5, 6, 4, 2]
for x in arr:
    stack.push(x)
stack.show()
print()

while not stack.is_empty():
    print(stack.pop())
```

**[Output]**

```python
2 4 6 5 2 7 9 
2
4
6
5
2
7
9
```

## 2. Queue with Linked List  
### 1) 큐에 쓰이는 연산  
- 큐는 먼저 삽입된 데이터가 먼저 추출되는 자료구조
  - Ex) 게임 대기 큐는 먼저 대기한 사람이 먼제 게임에 매칭된다.
  - Ex) 은행 대기열은 먼저 번호표를 뽑은 사람이 먼저 업무를 본다.

- enqueue(삽입), dequeue(삭제)
- front 연산은 dequeue를 했을 때, 실제 `self.items`의 value를 삭제하는 대신, `self.front_index`를 증가시킴으로써 dequeue한 값을 함수 상에서 얻는 값 취급을 했다.   
  front연산은 삭제하지 않고 리턴만 하기때문에 `self.items[self.fornt_index]`를 리턴해주기만 하면 된다.

### 2) 리스트를 이용한 큐 구현

```python
class Queue():
    def __init__(self):
        self.items = []
        self.front_index = 0

    def enqueue(self, value):
        self.items.append(value)

    def dequeue(self):
        if self.front_index == len(self.items):
            print('queue empty')
            return None
        else:
            returnvalue = self.items[self.front_index]
            self.front_index += 1
            return returnvalue
   
    def front(self):
        if self.front_index == len(self.items):
            print('queue is empty')
        else:
            returnvalue = self.items[self.front_index]
            return returnvalue
```

### 3) 연결 리스트를 이용한 큐 구현
- 연결 리스트로 큐을 구현하면 **삽입(push)와 삭제(pop)연산에 있어서** <span style="color:aqua">**O(1)**</span>**의 시간 복잡도(Time Complexity)**를 보장한다.
- 연결 리스트로 구현할 때는 **머리(head)**와 **꼬리(tail)** 두 개의 포인터를 가진다.
- **머리(head)**: 남아있는 원소 중 **가장 먼저 들어온 데이터**를 가리키는 포인터 
- **꼬리(tail)**: 남아있는 원소 중 **가장 마지막에 들어 온 데이터**를 가리키는 포인터

- **삽입**할 때는 꼬리(tail) 위치에 데이터를 넣는다.
- **삭제**할 때는 머리(head) 위치에서 데이터를 꺼낸다.

**[Input]**  
```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class Queue:
    def __init__(self):
        self.head = None
        self.tail = None

    def enqueue(self, data):
        node = Node(data)
        if self.head == None:
            self.head = node
            self.tail = node
        # 꼬리(tail) 위치에 새로운 노드 삽입
        else:
            self.tail.next = node
            self.tail = self.tail.next

    def dequeue(self):
        if self.head == None:
            return None

        # 머리(head) 위치에서 노드 꺼내기
        data = self.head.data
        self.head = self.head.next

        return data

    def show(self):
        cur = self.head
        while cur:
            print(cur.data, end=" ")
            cur = cur.next


queue = Queue()
data_list = [3, 5, 9, 8, 5, 6, 1, 7]

for data in data_list:
    queue.enqueue(data)

print("\n전체 노드 출력:", end=" ")
queue.show()

print("\n[원소 삭제]")
print(queue.dequeue())
print(queue.dequeue())
print(queue.dequeue())

print("[원소 삽입]")
queue.enqueue(2)
queue.enqueue(5)
queue.enqueue(3)

print("전체 노드 출력:", end=" ")
queue.show()
```

**[Output]**  
```python
전체 노드 출력: 3 5 9 8 5 6 1 7 
[원소 삭제]
3
5
9
[원소 삽입]
전체 노드 출력: 8 5 6 1 7 2 5 3
```

## 3. Deque with Linked List
### 1) 덱의 연산
- 덱(Deque) = 스택(Stack) + 큐(Queue)
- 포인터 변수가 더 많이 필요하기 때문에, 메모리는 상대적으로 더 많이 필요하다.
- Python에서는 <span style = "color:aqua">**큐(queue)의 기능**</span>**이 필요할 때 간단히 덱(deque)을 사용**한다.
- 데이터의 삭제와 삽입 모두 시간 복잡도는 O(1)
  - Python에서는 **덱(deque)** 라이브러리를 사용할 수 있다.
  - 아래의 모든 매서드는 최악의 경우 시간 복잡도 O(1)을 보장한다.
  - **우측 삽입**: append()
  - **좌측 삽입**: appendleft()
  - **우측 추출**: pop()
  - **좌측 추출**: popleft() 
<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/206266942-32e865cd-8564-4fdf-b34e-0f142c2ce26b.JPG">
</p>

### 2) 리스트를 이용한 덱 구현

```python
class deque():
    def __init__(self):
        self.items = []
        self.front_value = 0



    def push(self, value):
        self.items.append(value)

    def pushleft(self, value):
        self.items.insert(self.front_value,value)

    def pop():
        if len(self.items) == self.front_value:
            print('queue is empty')
            return None
        else:
            return self.items.pop()

    def popleft():
        if len(self.items) == self.front_value:
            print('queue is empty')
            return None
        else:
            x = self.items[self.front_value]
            self.front_value += 1
            return x
```
- 파이썬 라이브러리를 사용했을 때

```python
from collections import deque


d = deque()
arr = [5, 6, 7, 8]
for x in arr:
    d.append(x)
arr = [4, 3, 2, 1]
for x in arr:
    d.appendleft(x)
print(d)

while d:
    print(d.popleft())

arr = [1, 2, 3, 4, 5, 6, 7, 8]
for x in arr:
    d.appendleft(x)
print(d)

while True:
    print(d.pop())
    if not d:
        break
    print(d.popleft())
    if not d:
        break
```

### 3) 연결 리스트를 이용해 덱 구현
- 덱(deque)을 연결 리스트로 구현하면 **삽입(push)와 삭제(pop)연산에 있어서** <span style="color:aqua">**O(1)**</span>**의 시간 복잡도(Time Complexity)**를 보장한다.
- 연결 리스트로 구현할 때는 **앞(front)**과**뒤(rear)** 두 개의 포인터를 가진다.
- **앞(front)**: **가장 좌측**에 있는 데이터를 가리키는 포인터
- **뒤(rear)** : **가장 우측**에 있는 데이터를 가리키는 포인터
  - 기본적인 파이썬 리스트 자료형은 큐(queue)의 기능을 제공하지 않는다.
  - 가능하다면 파이썬에서 제공하는 덱(deque) 라이브러리를 사용한다.
  - 큐의 기능이 필요할 때는 덱 라이브러리를 사용하는 것을 추천한다.
  - 삽입과 삭제에 대하여 모두 시간 복잡도는 O(1)이 요구된다.

**[Input]**
```python
class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None


class Deque:
    def __init__(self):
        self.front = None
        self.rear = None
        self.size = 0

    def appendleft(self, data):
        node = Node(data)
        if self.front == None:
            self.front = node
            self.rear = node
        else:
            node.next = self.front
            self.front.prev = node
            self.front = node
        self.size += 1

    def append(self, data):
        node = Node(data)
        if self.rear == None:
            self.front = node
            self.rear = node
        else:
            node.prev = self.rear
            self.rear.next = node
            self.rear = node
        self.size += 1

    def popleft(self):
        if self.size == 0:
            return None
        # 앞에서 노드 꺼내기
        data = self.front.data
        self.front = self.front.next
        # 삭제로 인해 노드가 하나도 없는 경우
        if self.front == None:
            self.rear = None
        else:
            self.front.prev = None
        self.size -= 1
        return data

    def pop(self):
        if self.size == 0:
            return None
        # 뒤에서 노드 꺼내기
        data = self.rear.data
        self.rear = self.rear.prev
        # 삭제로 인해 노드가 하나도 없는 경우
        if self.rear == None:
            self.front = None
        else:
            self.rear.next = None
        self.size -= 1
        return data

    def front(self):
        if self.size == 0:
            return None
        return self.front.data

    def rear(self):
        if self.size == 0:
            return None
        return self.rear.data

    # 앞에서부터 원소 출력
    def show(self):
        cur = self.front
        while cur:
            print(cur.data, end=" ")
            cur = cur.next


d = Deque()
arr = [5, 6, 7, 8]
for x in arr:
    d.append(x)
arr = [4, 3, 2, 1]
for x in arr:
    d.appendleft(x)
d.show()

print()
while d.size != 0:
    print(d.popleft())

arr = [1, 2, 3, 4, 5, 6, 7, 8]
for x in arr:
    d.appendleft(x)
d.show()

print()
while True:
    print(d.pop())
    if d.size == 0:
        break
    print(d.popleft())
    if d.size == 0:
        break
```

**[Output]**
```python
1 2 3 4 5 6 7 8 
1
2
3
4
5
6
7
8
8 7 6 5 4 3 2 1 
1
8
2
7
3
6
4
5
```
