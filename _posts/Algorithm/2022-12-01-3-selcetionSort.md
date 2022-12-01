---
title: Chapter 1.3 Selection Sort(선택 정렬)

categories: 
  - Algorithm
tags:
  - [Algorithm, Sort, Selection Sort]

toc: true
toc_sticky: true

date: 2022-12-01
last_modified_at: 2022-12-01 
---

## 1. Selection Sort(선택 정렬)  

아래의 링크를 참조하여 공부.
- [버블 정렬](https://gmlwjd9405.github.io/2018/05/06/algorithm-bubble-sort.html)
- [삽입 정렬](https://gmlwjd9405.github.io/2018/05/06/algorithm-insertion-sort.html)
- [병합 정렬](https://gmlwjd9405.github.io/2018/05/08/algorithm-merge-sort.html)
- [선택 정렬](https://gmlwjd9405.github.io/2018/05/06/algorithm-selection-sort.html) #
- [셸 정렬](https://gmlwjd9405.github.io/2018/05/08/algorithm-shell-sort.html)
- [퀵 정렬](https://gmlwjd9405.github.io/2018/05/10/algorithm-quick-sort.html)
- [힙 정렬](https://gmlwjd9405.github.io/2018/05/10/algorithm-heap-sort.html)
- 
### 1) 선택 정렬의 기본 메커니즘  
- 제자리 정렬(in-place sorting) 알고리즘의 하나  
  - 입력 배열(정렬되지 않은 값들) 이외에 다른 추가 메모리를 요구하지 않는 정렬 방법  
  - 해당 순서에 원소를 넣을 위치는 이미 정해져 있고, 어떤 원소를 넣을지 선택하는 알고리즘  
  - 첫 번째 순서에는 첫 번째 위치에 가장 최솟값을 넣는다.  
  - 두 번째 순서에는 두 번째 위치에 남은 값 중에서의 최솟값을 넣는다.  

- 과정 설명  
  (1) 주어진 배열 중에서 최솟값을 찾는다.  
  (2) 그 값을 맨 앞에 위치한 값과 교체한다(패스(pass)).  
  (3) 맨 처음 위치를 뺀 나머지 리스트를 같은 방법으로 교체한다.  
  (4) 하나의 원소만 남을 때까지 위의 1~3 과정을 반복한다.  

### 2) 선택 정렬의 개념
선택 정렬은 첫 번째 자료를 두 번째 부터 마지막 자료까지 **차례대로** 비교하여 가장 작은 값을 찾아 첫 번째에 놓고, 두 번째 자료를 세 번째 자료부터 마지막 자료까지 차례대로 비교하여
그 중 가장 작은 값을 찾아 두 번째 위치에 놓는 과정을 반복한다. 첫 번째 패스 이후 가장 작은 값의 자료가 맨 앞에 오게 되므로 그 다음 회전에서는 두 번째 자료를 가지고 비교한다. 마찬
가지로 3회전에서는 세 번째 자료를 정렬한다.

<p align="center">
<img width="500" alt="1" src="https://user-images.githubusercontent.com/111734605/204979075-003c4833-8dcf-488e-8433-e7000ff5c641.png">
</p>

- 1회전:
    - 첫 번째 자료 9를 두 번째 자료부터 마지막 자료까지와 비교하여 가장 작은 값을 첫 번째 위치에 옮겨 놓는다. 이 과정에서 자료를 4번 비교한다.
- 2회전:
    - 두 번째 자료6을 세 번째 자료부터 마지막 자료까지와 비교하여 가장 작은 값을 두 번째 위치에 옮겨 놓는다. 이 과정에서 자료를 3번 비교한다.
- 3회전:
    - 세 번째 자료 7을 네 번째 자료부터 마지막 자료까지와 비교하여 가장 작은 값을 세 번째 위치에 옮겨 놓는다. 이 과정에서 자료를 2번 비교한다.
- 4회전:
    - 네 번째 자료 9와 마지막에 있는 7을 비교하여 서로 교환한다.

### 2) Python Code

#### Ex1)  
**[Input]**  

```python
import numpy as np

def SelectionSort(list):
    for i in range(len(list) - 1):
        min_idx = i
        for j in range(i + 1, len(list)):
            if list[j] < list[min_idx]:
                min_idx = j
        list[i], list[min_idx] = list[min_idx], list[i]



if __name__ == "__main__":    
    list_1 = np.random.randint(1,30,10)
    print(list_1)
    SelectionSort(list_1)
    print(list_1)               
```

**[Output]**  
```python
# case 1
[16  1  9 11 18  1 26 29  5 15] # input
[ 1  1  5  9 11 15 16 18 26 29] # SelectionSort(input)

# case 2
[ 6 14 20 10 10  6 14 21  2 26] # input
[ 2  6  6 10 10 14 14 20 21 26] # SelectionSort(input)
```
#### Ex2)  
선택 정렬은 필요한 **교환 횟수를 최소화한 버블 정렬의 개량 버전**이다. 버블 정렬에서는 각 패스마다 가장 큰 값을 오른쪽으로 한 칸씩 움직이므로 크기가 N인 배열에서 교환은 N-1번 발생
한다. 하지만 선택 정렬에서는 각 패스마다 가장 큰 값을 찾아내 맨 오른쪽으로 **바로** 이동시킨다. 첫 번째 패스 종료 후 가장 큰 값은 맨 오른쪽에, 두 번째 패스 종료 후 그 다음 큰 값은
오른쪽에서 두 번째 자리에 위치하게 된다. 알고리즘이 진행됨에 따라 이후의 값들은 그 크기에 맞는 합당한 위치로 옮겨진다. 마지막 값은 N-1번째 패스 종류 후에 제 위치로 옮겨진다.  

- 선택 정렬은 N개의 요소를 N-1번 패스를 사용해 정렬한다.

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/205014295-7b7e2736-84af-4b2a-89c8-a16178683ebf.png">
</p>

**[Input]**  
```python
def SelectionSort(list):
    for fill_slot in range(len(list) - 1, 0, -1):
        max_index = 0
        for location in range(1, fill_slot + 1):
            if list[location] > list[max_index]:
                max_index = location
        list[fill_slot], list[max_index] = list[max_index], list[fill_slot]
    return list

## Input
if __name__ == "__main__":
    InputList  = [70,15,25,19,34,44]
    print(InputList)
    OutputList = SelectionSort(InputList)
    print(OutputList)
```

**[Output]**  
```python
[70, 15, 25, 19, 34, 44] # print(InputList)
[15, 19, 25, 34, 44, 70] # print(OutputList)
```

## 2. 선택 정렬 알고리즘 특징
### 1) 특징  
- 장점  
  - 자료 이동 횟수가 미리 결정된다.  
- 단점  
  - 안정성을 만족하지 않는다.  
  - 즉, 값이 같은 레코드가 있는 경우에 상대적인 위치가 변경될 수 있다.  

### 2) 시간 복잡도(Time Complexity)  
시간 복잡도를 계산하면

- 비교 횟수  
  - 두 개의 for 루프의 실행  
  - 외부 루프: (N-1)번  
  - 내부 루프(최솟값 찾기): N-1, N-2, ..., 2, 1 번  
- 교환 횟수  
  - 외부 루프의 실행 횟수와 동일. 즉, 상수 시간 작업  
  - 한 번 교환하기 위하여 3번의 이동이 발생 3(N-1)   
 
따라서 시간 복잡도는 <span style = "color:aqua">**O($$N^2$$)**</span>

- Time Complexity: **O(N^2)**
- 버블 정렬과 시간 복잡도는 동일하지만, 교환을 더 적게 하기 때문에 평균 성능은 더 뛰어나다

<p align = "center">
<img width="690" alt="image" src="https://user-images.githubusercontent.com/111734605/204984038-9021521d-48d4-41f3-b2af-6216ed965e52.png">
</p>
