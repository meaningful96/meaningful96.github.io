---
title: Chapter 2. Stack

categories:
  - DataStructure
tags:
  - [DS, DataStructure, Stack]

toc: true
toc_sticky: true

date: 2022-11-28
last_modified_at: 2022-11-28 
---

## 1. Stack
### 1) Stack이란?
Stack은 한 쪽 끝에서만 자료를 넣거나 뺼 수 있는, **아래에서부터 저장이 되고 최근에 들어온 값부터 제거가 되는** 선형으로
나열된 자료구조이다. 비유하자면 '프링글스 통'을 예로 들 수 있다. 과자를 먹 을때는 제일 위에서부터 꺼내 먹지만, 과자가 공장에서
생산되어 만들어질 때는 가장 아래쪽부터 쌓인다.

이러한 일반적인 후입선출 구조를 LIFO라고 한다.  
- LIFO : Last-In-First-Out

### 2) Stack의 연산, Method
- push()
  스택에 원소를 추가한다.
- pop()
  스택 가장 **위**의 원소를 삭제하고 그 원소를 return한다.
- peek()
  스택 가장 위에 있는 원소를 return한다. (삭제하지는 않는다.)
- empty()
  스택이 비어있다면 1(True), 아니면 0(False)를 반환한다.
- len()
  원소의 개수를 return
  
이 메서드들을 이용해서 Python 코드를 짜면 다음과 같다.
```python
class Stack:
    #리스트를 이용한 스택 구현
    def __init__(self):
        self.top = []
    #스택 크기 반환
    def __len__(self) -> bool :
        return len(self.top)
    
    #구현함수
    #스택에 원소 삽입
    def push(self, item):
        self.top.append(item)
    #스택 가장 위에 있는 원소를 삭제하고 반환   
    def pop(self):
        if not self.isEmpty():
            return self.top.pop(-1)
        else:
            print("Stack underflow")
            exit()
    #스택 가장 위에 있는 원소를 반환
    def peek(self):
        if not self.isEmpty():
            return self.top[-1]
        else:
            print("underflow")
            exit()
    #스택이 비어있는 지를 bool값으로 반환
    def isEmpty(self) -> bool :
        return len(self.top)==0
```

### 3) Stack을 이용한 예제 

<p align="center">
<img width="911" alt="image" src="https://user-images.githubusercontent.com/111734605/204209749-d1bca502-827c-4494-85f0-aa9e1ab2be57.png">
</p>

<p align="center">
<img width="911" alt="image" src="https://user-images.githubusercontent.com/111734605/204210246-00c9734c-da36-4844-978d-badfe0a1313c.png">
</p>

<p align="center">
<img width="404" alt="image" src="https://user-images.githubusercontent.com/111734605/204210392-cb6aa294-5fef-4b82-8f6e-cb1f9878797f.png">
</p>
