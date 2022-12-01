---
title: Chapter 1.1 Bubble Sort

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
**가장 큰 값**을 반복적으로 옮긴다. 버블 정렬의 최악의 Case의 시간 복잡도(Time Complexity)는 **O($$N^2$$)**이다.

#### 버블 정렬의 작동 원리 이해하기
버블 정렬은 **패스**(pass)라는 과정을 반복한다. 리스트의 크기가 N일 때 버블 정렬의 패스 개수는 N-1이다. 

**첫 번쨰 패스**
리스트에 든 요소들을 작은 값부터 큰 값으로 정렬하려고 할 때, 첫 번째 패스의 목표는 **리스트의 가장 큰 값을** <span style = "color:aqua">**맨 오른쪽**</span>으로 보내는 것이다.
패스가 진행됨에 따라 가장 큰 값이 서서히 이동하는 것을 볼 수 있다. 버블 정렬은 서로 붙어 있는 이웃끼리 값을 비교한다. 인접한 두 값 중 왼쪽 값이 더 크다면 서로의 값을 뒤바꾸고, 오
른쪽으로 한 칸 이동한다. 

**Ex1)**
<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/204766459-c2067707-c3ef-4bc2-a0f6-d700d1ab5ee1.png">
</p>

**Ex2)**
<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/204768406-7eb80781-055d-4be6-9940-0eb3e5e959de.png">
</p>

### 3) 파이썬으로 구현하기  
#### Ex1) 첫 번째 패스 구현  
**[Input]**  
```python
list = [23,21,22,24,23,27,26]
lastElementsIndex = len(list) - 1
print(0, list)
for idx in range(lastElementsIndex):
    if list[idx] > list[idx + 1]:
        list[idx], list[idx + 1] = list[idx + 1], list[idx]
    print(idx + 1, list )
```
**[Output]**  
```python
0 [23, 21, 22, 24, 23, 27, 26]
1 [21, 23, 22, 24, 23, 27, 26]
2 [21, 22, 23, 24, 23, 27, 26]
3 [21, 22, 23, 24, 23, 27, 26]
4 [21, 22, 23, 23, 24, 27, 26]
5 [21, 22, 23, 23, 24, 27, 26]
6 [21, 22, 23, 23, 24, 26, 27]
```
첫 번쨰 패스가 마무리되면 가장 큰 값이 리스트의 맨 오른쪽에 위치하게 된다. 그 다음은 두번 째 패스인데, 두 번째 패스의 목표는 당연하게도 리스트에서 두 번째로 가장 큰 값을 맨 오른쪽
끝에서 두 번째(index = 1)로 옮기는 것이다. 이를 N-1번 반복 수행하므로 다음과 같이 코드를 일반화 할 수 있다.

#### Ex2) 재사용이 편리하도록 함수로 구현한 버블 정렬 코드
**[Input]**  
```python
def BubbleSort(list):
    # 리스트에 담긴 데이터를 순서대로 정렬합니다.
    lastElementsIndex = len(list) - 1
    for passNo in range(lastElementsIndex, 0, -1): # range(시작, 끝, 간격)
        for idx in range(passNo):
            if list[idx] > list[idx + 1]:
                list[idx], list[idx + 1] = list[idx + 1], list[idx]
    return list
a = [23,21,22,24,23,27,26]
# print(BubbleSort(a))
```

**[Output]**
```python
[21, 22, 23, 23, 24, 26, 27]
```

#### Ex3) 최적화  
이전 패스에서 앞뒤 자리 비교(swap)이 한 번도 일어나지 않았다면 정렬되지 않는 값이 하나도 없었다고 간주할 수 있다. 따라서 이럴 경우, 이후 패스를 수행하지 않아도 된다.
```python
Initial: [1, 2, 3, 5, 4]

 Pass 1: [1, 2, 3, 4, 5] => Swap 있었음
                      *
 Pass 2: [1, 2, 3, 4, 5] => Swap 없었음
                   *  *
=> 이전 패스에서 swap이 한 번도 없었으니 종료
```

**[Input]**
```python
def BubbleSort(list):
    for i in range(len(list) - 1):
        swapped = False
        for j in range(i):
            if list[j] > list[j + 1]:
                list[j], list[j + 1] = list[j + 1], list[j]
                swapped = True
        if not swapped:
            break
```

#### Ex4) 최적화 2
이전 패스에서 앞뒤 자리 비교(swap)가 있었는지 여부를 체크하는 대신 마지막으로 앞뒤 자리 비교가 있었던 index를 기억해두면 다음 패스에서는 그 자리 전까지만 정렬해도 된다. 
따라서 한 칸씩 정렬 범위를 줄여나가는 대신 한번에 여러 칸씩 정렬 범위를 줄여나갈 수 있다.
```python
Initial: [3, 2, 1, 4, 5]

 Pass 1: [2, 1, 3, 4, 5] => 마지막 Swap 위치가 index 1
             ^        *
 Pass 2: [1, 2, 3, 4, 5] => 중간 패스 skip하고 바로 index 1로 보낼 값 찾기
          ^     *  *  *
 ```
 
**[Input]**
```python
def BubbleSort(list):
    end = len(list) - 1
    while end > 0:
        last_swap = 0
        for i in range(end):
            if list[i] > list[i + 1]:
                list[i], list[i + 1] = list[i + 1], list[i]
                last_swap = i
        end = last_swap
```

## 3. 버블 정렬 알고리즘의 특징
- 버블 정렬은 점점 큰 값들을 뒤에서 부터 앞으로 하나씩 쌓여 나가게 때문에 후반으로 갈수록 정렬 범위가 하나씩 줄어들게 됩다.
  왜냐하면, 다음 패스에서는 이전 패스에서 뒤로 보내놓은 가장 큰 값이 있는 위치 전까지만 비교해도 되기 때문이다.
- 제일 작은 값을 찾아서 맨 앞에 위치시키는 선택 정렬과 비교했을 때 정반대의 정렬 방향을 가진다.
- 다른 정렬 알고리즘에 비해서 자리 교대(swap)가 빈번하게 일어나는 경향을 가지고 있다. 
  예를 들어, 선택 정렬의 경우 각 패스에서 자리 교대가 딱 한번만 일어난다.
- 최적화 여지가 많은 알고리즘입니다. 예를 들어, 위 그림에서 Pass 5는 생략할 수 있는 패스이다. 
  왜냐하면 Pass 4에서 한 번도 자리 교대가 일어나지 않았기 때문이다.
  
## 4. 버블 정렬의 성능 분석
### 1) 공간 복잡도(Space Complexity)
버블 정렬은 별도의 추가 공간을 사용하지 않고 주어진 배열이 차지하고 있는 공간 내에서 값들의 위치만 바꾸기 때문에 <span style = "color:aqua">**O(1)**</span>의 공간 복잡도를 가진다.

### 2) 시간 복잡도(Time Complexity)
시간 복잡도는 우선 루프문을 통해 맨 뒤부터 맨 앞까지 모든 인덱스에 접근해야 하기 때문에 기본적으로 <span style = "color:aqua">**O(N)**</span>을 시간을 소모하며, 하나의 루프에서는 인접한 값들의 대소 비교 및 자리 교대를 위해서 O(N)을 시간이 필요하게 됩니다. 따라서 거품 정렬은 총 <span style = "color:aqua">**O($$N^2$$)**</span>의 시간 복잡도를 가지는 정렬 알고리즘입니다.

하지만, 버블 정렬은 부분적으로 정렬되어 있는 배열에 대해서는 최적화를 통해서 성능을 대폭 개선할 수 있으며, 완전히 정렬되어 있는 배열이 들어올 경우, **O(N)**까지도 시간 복잡도를 향상시킬 수 있습니다.

### 3) 루프 개수
- 외부 루프: 패스를 의미한다. 예를 들어, 첫 번째 패스는 외부 루프를 처음 1회 실행하는 것과 같다.
- 내부 루프: 패스 내에서 가장 높은 값을 오른쪽으로 이동시킬 때까지 값들을 반복적으로 비교하는 과정이다. 첫 번째 패스는 총 N-1번, 두 번째 패스는 N-2번을 반복하는 식으로 패스 횟수가
  올라감에 따라 값을 비교하는 횟수가 감소한다.
  
### 4) Summery
- Time Complexity:  O($$N^2$$)
- Space Complexity: O(1)
