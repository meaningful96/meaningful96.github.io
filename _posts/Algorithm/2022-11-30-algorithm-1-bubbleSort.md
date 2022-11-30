---
title: Chapter 1.1 Bubble Sort(버블 정령)

categories: 
  - Algorithm
tags:
  - [Algorithm, Sort, Bubble Sort]

toc: true
toc_sticky: true

date: 2022-11-30
last_modified_at: 2022-11-30 
---

## 1. 여러 가지 정렬(Sorting) 방법  
아래의 링크를 참조하여 공부.
- [버블 정렬](https://gmlwjd9405.github.io/2018/05/06/algorithm-bubble-sort.html)
- [삽입 정렬](https://gmlwjd9405.github.io/2018/05/06/algorithm-insertion-sort.html)
- [병합 정렬](https://gmlwjd9405.github.io/2018/05/08/algorithm-merge-sort.html)
- [선택 정렬](https://gmlwjd9405.github.io/2018/05/06/algorithm-selection-sort.html)
- [셸 정렬](https://gmlwjd9405.github.io/2018/05/08/algorithm-shell-sort.html)
- [퀵 정렬](https://gmlwjd9405.github.io/2018/05/10/algorithm-quick-sort.html)
- [힙 정렬](https://gmlwjd9405.github.io/2018/05/10/algorithm-heap-sort.html)

## 2. 버블 정렬의 개념
### 1) 파이썬에서 변수 바꾸기
정렬과 검색 알고리즘에서는 두 변수의 값을 바꾸는 경우가 많다. Python에서는 다음과 같은 방법으로 변수 변환이 이루어진다.
```python
var1 = 1
var2 = 2
var1, var2 = var2, var1
print(var1, var2)

##출력
2 1
```

### 2) 버블 정렬
버블 정렬(Bubble sort)는 <span style = "color:aqua">**가장 간편하지만 속도가 가장 느린**</span> 알고리즘이다. 버블 정렬은 비눗방울이 공중에 서서히 떠오르듯 **리스트**안에서 
**가장 큰 값**을 반복적으로 옮긴다. 버블 정렬의 최악의 Case의 시간 복잡도(Time Complexity)는 **O(N)**이다.

#### 버블 정렬의 작동 원리 이해하기
버블 정렬은 **패스**(pass)라는 과정을 반복한다. 리스트의 크기가 N일 때 버블 정렬의 패스 개수는 N-1이다. 

**첫 번쨰 패스**
리스트에 든 요소들을 작은 값부터 큰 값으로 정렬하려고 할 때, 첫 번째 패스의 목표는 **리스트의 가장 큰 값을** <span style = "color:aqua">**맨 오른쪽**</span>으로 보내는 것이다.
패스가 진행됨에 따라 가장 큰 값이 서서히 이동하는 것을 볼 수 있다. 버블 정렬은 서로 붙어 있는 이웃끼리 값을 비교한다. 인접한 두 값 중 왼쪽 값이 더 크다면 서로의 값을 뒤바꾸고, 오
른쪽으로 한 칸 이동한다. 

<p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/111734605/204766459-c2067707-c3ef-4bc2-a0f6-d700d1ab5ee1.png">
</p>


