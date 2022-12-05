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
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/205756932-f43ea33b-dec8-4672-8101-db5141b6fc74.png">
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
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/205768579-f9e4e93b-9465-48ee-86f3-6e1d7301187d.png">
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
