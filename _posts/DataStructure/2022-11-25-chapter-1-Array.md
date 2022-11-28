---
title: Chapter 1. Array & List

categories:
  - DataStructure
tags:
  - [DS,DataStructure,Array, List]

toc: true
toc_sticky: true

date: 2022-11-25
last_modified_at: 2022-11-25 
---

## 1. Array
### 1) Array(배열)
배열이란 같은 성질을 갖는 항목들의 집합이다. 같은 특성을 갖는 원소들이 <span style = "color:aqua">**순서대로**</span> 구성된 선형 자료 구조이며, 메모리 상에 연속적으로
데이터가 저장된 순차 리스트에 해당한다. 순차적으로 저장된 데이터를 참조하는 데에는 <span style = "color:aqua">**Index**</span>가 사용된다.

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/203910344-cf37af79-f47e-463d-973c-a981d238aa59.png">
</p>

### 2) Array의 특징
- 고정된 크기를 갖는다.(데이터의 수가 정해져 있다.) 즉, 사용할 수 있는 메모리의 크기가 정해져 있다.
  - Ex) Array.size = 10, 내부 데이터가 5개만 있더라도 실제 배열의 크기는 10 => 메모리 낭비 가능성 존재
- 논리적 저장 순서 = 물리적 저장 순서
- 추가적으로 소모되는 메모리 양(오버헤드)이 거의 없음. (Why? 데이터의 양에 맞게 크기를 정해놓고 사용하기 때문)
- 삽입 & 삭제의 경우 O(N)
- 원소에 접근 O(1)
- 기억 장소를 미리 확보해야 한다.
- 높은 Cache Hit Rate를 보여준다.
  - Cache Hit Rate란 원하는 정보가 캐시 메모리에 기억되어 있을 때를 적중(Hit) 아닐 때를 실패했다고 한다. 적중률 = 적중 횟수/총 접근 
  - Array는 참조된 주소와 인접한 주소의 내용이 다시 참조되는 특성인 공간 지역성(Spacial Locality)가 좋아 Cache Hit Rate가 높은 것이다.

### 3) Array의 연산

#### (1) 요소 확인 & 요소 변경  
**확인**  
확인(Search)는 특정 인덱스에 있는 요소를 확인하는 것이다. 즉, 특정 인덱스만 있으면 바로 확인 가능하기에 O(1)의 Time complexity를 가진다.

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/203910426-80ef361f-434c-4d3d-b879-f344ca329c21.png">
</p>
  
**변경**  
변경 역시 특정 인덱스의 값 만을 바꾸는 것이다. 정확히 말하면 특정 인덱스에 할당된 주소를 바꾸는 것이다.

- 확인 : O(1)
- 변경 : O(1)

#### (2) 원소 삽입 및 삭제  
**삭제**  
pop(3)    # 3번 index의 값을 삭제한다.  
여기서 중요한 것은 3번 인덱스가 삭제되었으면 그 다음에 있던 인덱스들을 한칸식 앞으로 밀린다는 것이다.

<span style = "color: aqua">Before</span>  

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/203910705-64aae3ad-b6a2-47eb-818a-b12615cec892.png">
</p>

<span style = "color: aqua">After pop(3)</span> 

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/203910748-083eaa8b-3134-4a1d-a82c-77fdaee3e377.png">
</p>

어떠한 원소를 삭제할 때 평균적으로 $\frac{N}{2}$ 만큼 떙겨야 하므로 삭제의 시간 복잡도는 <span style = "color: aqua">**O(N)**</span>이다.  

**삽입**  
크기가 10인 배열에 7개의 데이터가 들어 있다고 하자. 인덱스 3위치에 데이터 19를 추가한다면, 인덱스 3 우측에 있는 데이터들을 한 칸씩 우측으로 밀리게 된다.

<span style = "color: aqua">Before</span>  

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/203910798-3389bfc7-006f-4f0f-b76b-0145a97cefee.png">
</p>

<span style = "color: aqua">After insert(3,19)</span>  

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/203910833-0db7705a-e7ac-4726-9511-3ff1c6e5e365.png">
</p>

역시 마찬가지로, 평균적으로 $\frac{N}{2}$ 만큼 우측으로 밀어야 하므로 시간 복잡도는 <span style = "color: aqua">**O(N)**</span>이다.

- 삽입 : O(N)
- 삭제 : O(N)

## 2. List(리스트)
### 1) List(리스트) 개념
자료구조 관점에서 바라보면 배열 또한 리스트에 포함되지만, 프로그래밍 언어의 관점에서 리스트란 배열이 가지고 있는 인덱스라는 장점을 버리고, 빈틈없는 데이터의 적재라는 장점을
취한 인터페이스다. 즉, Linked List나 ArrayList 등의 선형 자료 구조를 구현할 때 사용되는 추상 자료형이다.

- <span style = "color: aqua">List = 같은 특성을 갖는 원소들이 **순서대로** 구성된 집합인 선형 자료 구조이다.</span>
- 이 리스트는 Python에서 기본적으로 제공하는 List와는 다르다.

하지만, 데이터가 메모리 상에 연속적으로 저장되진 않기 때문에 순차 리스트가 아닌 연결 리스트(Linked-List)라고 부른다.
- 연결 리스트에는 여러 종류의 구조가 존재한다.
- 단일 연결 리스트(Singly Linked List), 이중 연결 리스트(Doubly Linked List), 원형 연결 리스트(Circular Linked List)...

### 2) List의 
#### (1) 원소 확인 및 변경
단일 연결 리스트로 정리함.

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/204198899-1478e79f-cfd0-464f-8c2e-ed223161aa47.png">
</p>

**확인**
순차 접근 방식을 사용하기 때문에 처음부터 순차적으로 탐색한다. 이 때 **딸기**를 찾는다고하자. 처음부터 시작해서 순차적으로 데이터에 접근하기에, 사과를 거쳐 배를 거치고 딸기는 찾는다. 이렇게되면 결론적으로 Time Complexity는 <span style = "color: aqua">O(N)</span>이 된다.

**변경**
변경은 간단하다. 만약 딸기를 파인애플로 변경한다고 하면, 딸기에 해당하는 데이터에 접근한 이후 그 곳에서 Value값만 바꾸면된다. 따라서 Time Complexity는 <span style = "color: aqua">O(1)</span>이다.  
- Search의 Time Complexity는 O(N)이다.
- Change의 Time Complexity는 O(1)이다.

따라서, 만약 확인과 접근을 둘 다 수행한다고하면 O(N+1)이고 O(N+1) = O(N)이라고 할 수 있다.

#### (2) 원소 삽입 및 삭제

**삭제**
이번엔 파인애플을 삭제한다고 해보자. 삭제의 Mechanism은 간단하다. 삭제하는 Link값에 연결된 선을 바꿔주면된다. 즉, 배에 해당하는 Link를 바나나의 Link와 연결시켜주면 되기 때문에 
Time Complexity는 O(1)이다. 하지만, 파인애플을 삭제하기 위해서는 Serch과정이 포함되기에 결론적으로 삭제 연산의 Time Complexity는 <span style = "color: aqua">O(N)</span>이다.
삭제의 경우 당연히 리스트의 크기는 줄어든다.

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/204198902-4ed3db93-21ec-46f5-86f0-138c089d1e39.png">
</p>

**삽입**
삽입의 경우도 Serch과정을 거친후 새로운 Link를 끼워넣는 것과 마찬가지다. 즉 양 옆의 링크를 끊고 새로운 값이 저장된 Node의 Link로 연결시키면 된다. 이 경우도 결국
Time Complexity는 <span style = "color: aqua">O(N)</span>이다. 삽입의 경우 결국 리스트의 크기는 늘어난다.(즉 리스트의 크기는 배열과 다르게 가변적이다.)

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/204198904-9d275b2b-db3e-40f5-b248-3f6e60e82bd8.png">
</p>

- Insert의 Time Complexity는 O(N)이다.
- Delete의 Time Complexity는 O(N)이다.

#### (3) 메모리 상에 연속성
Int형 자료의 크기는 4Byte이다. 이 때 배열의 각 주소는 100,104,108,112 순으로 연속적으로 주소가 할당된다.  
반면 리스트의 경우 100,204,336,540등으로 주소가 연속적으로 할당되지 않는 특징이 있다. 

## 3. Array VS Array-List VS Linked-list
<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/204203088-f1a85e22-9b92-48d9-9fe0-ce1f61179102.png">
</p>

- ArrayList는 쉽게 말하면 Python의 기본 자료구조인 List를 말한다.
- 크기가 가변적인 배열이 필요하다면 ArrayList(PythonList)
- Linked List의 삭제,추가의 시간 복잡도는 O(N)이지만 Array나 ArrayList보다 빠름
  - 데이터의 **삭제, 추가**가 많을 때 => LinkedList
  - 데이터의 **접근**이 많을때        => ArrayList, Array

- Array, ArrayList는 인덱스가 있지만 LinkedList는 없다
  - 인덱스가 필요하면     => ArrayList
  - 인덱스가 필요 없다면  => LinkedList
