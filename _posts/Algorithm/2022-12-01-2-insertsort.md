---
title: Chapter 1.2 Insert Sort

categories: 
  - Algorithm

tags:
  - [Insert Sort, Sort, Algorithm]

toc: true
toc_sticky: true

date: 2022-12-01
last_modified_at: 2022-12-01 
---

## 1. Insert Sort(삽입 정렬)
아래의 링크를 참조하여 공부.
- [버블 정렬](https://gmlwjd9405.github.io/2018/05/06/algorithm-bubble-sort.html)
- [삽입 정렬](https://gmlwjd9405.github.io/2018/05/06/algorithm-insertion-sort.html)
- [병합 정렬](https://gmlwjd9405.github.io/2018/05/08/algorithm-merge-sort.html)
- [선택 정렬](https://gmlwjd9405.github.io/2018/05/06/algorithm-selection-sort.html)
- [셸 정렬](https://gmlwjd9405.github.io/2018/05/08/algorithm-shell-sort.html)
- [퀵 정렬](https://gmlwjd9405.github.io/2018/05/10/algorithm-quick-sort.html)
- [힙 정렬](https://gmlwjd9405.github.io/2018/05/10/algorithm-heap-sort.html)

### 1) 삽입 정렬의 개념
삽입 정렬의 기본 아이디어는 자료 구조에서 데이터 포인트를 하나씩 빼내어 올바른 위치에 집어넣는 과정을 반복하는 것이다. 먼저 첫 번째 단계에서는 맨 왼쪽에 위치한 두 데이터 포인트를 서로
비교하고 값의 크기에 따라 정렬한다. 그 다음 범위를 확장하여 세 번째 데이터 포인트를 가져온다. 앞에서 결정된 두 개의 데이터 포인트의 적절한 위치를 결정한다. 이 과정을 모든 데이터 포인
트가 제 위치에 삽입될 때까지 반복한다.

<p align = 'center'>
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/204966097-326231c5-fb90-461a-81d7-ed36183dad44.png">
</p>

### 2) Python 코드 실습
#### Ex1)

**[Input]**
```python
import numpy as np

def InsertSort(list):
    for i in range(1, len(list)):
        j = i - 1
        elements_next = list[i]
        while(list[j]> elements_next) and (j>=0):
            list[j+1] = list[j]
            j = j - 1
        list[j + 1] = elements_next
    return list

if __name__ == "__main__":
    # np.random.seed(5)
    list_1 = np.random.randint(0,30,10)
    Out = InsertSort(list_1)
    print(Out)
```

**[Output]**
list_1을 random한 정수들로, size가 10인 리스트를 만들었으므로(정확히 말하면 Array가 생성됨) 실행마다 출력값이 다르다. 하지만, 모두 다 삽입 정렬이 제대로 된 것을 볼 수 있다.
큰 값이 오른쪽으로 가게된다.
```python
[ 2  5  6  9 13 14 15 16 26 27]
[ 1  3  4  7 11 11 12 13 23 28]
[ 0  4  4  5  9 14 14 16 19 26]
```
#### Ex2)
**[Input]**  
```python
def InsertSort(arr):
    for end in range(1, len(arr)):
        for i in range(end, 0, -1):
            if arr[i - 1] > arr[i]:
                arr[i - 1], arr[i] = arr[i], arr[i - 1]
```

### 2) 더 자세한 예로 삽입 정렬을 살펴보자

<p align = 'center'>
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/204969624-4004bc90-4722-4835-86cf-9b72ddcbed95.png">
</p>

- 1회전: 두 번째 자료인 5를 Key로 해서 그 이전의 자료들과 비교한다.  
         Key 값 5와 첫 번째 자료인 8을 비교한다. 8이 5보다 크므로 8을 5자리에 넣고 Key 값 5를 8의 자리인 첫 번째에 기억시킨다.  
- 2회전: 세 번째 자료인 6을 Key 값으로 해서 그 이전의 자료들과 비교한다.    
         Key 값 6과 두 번째 자료인 8을 비교한다. 8이 Key 값보다 크므로 8을 6이 있던 세 번째 자리에 기억시킨다.   
         Key 값 6과 첫 번째 자료인 5를 비교한다. 5가 Key 값보다 작으므로 Key 값 6을 두 번째 자리에 기억시킨다.   
- 3회전: 네 번째 자료인 2를 Key 값으로 해서 그 이전의 자료들과 비교한다.  
         Key 값 2와 세 번째 자료인 8을 비교한다. 8이 Key 값보다 크므로 8을 2가 있던 네 번째 자리에 기억시킨다.  
         Key 값 2와 두 번째 자료인 6을 비교한다. 6이 Key 값보다 크므로 6을 세 번째 자리에 기억시킨다.  
         Key 값 2와 첫 번째 자료인 5를 비교한다. 5가 Key 값보다 크므로 5를 두 번째 자리에 넣고 그 자리에 Key 값 2를 기억시킨다.  
- 4회전: 다섯 번째 자료인 4를 Key 값으로 해서 그 이전의 자료들과 비교한다.  
         Key 값 4와 네 번째 자료인 8을 비교한다. 8이 Key 값보다 크므로 8을 다섯 번째 자리에 기억시킨다.  
         Key 값 4와 세 번째 자료인 6을 비교한다. 6이 Key 값보다 크므로 6을 네 번째 자리에 기억시킨다.    
         Key 값 4와 두 번째 자료인 5를 비교한다. 5가 Key 값보다 크므로 5를 세 번째 자리에 기억시킨다.  
         Key 값 4와 첫 번째 자료인 2를 비교한다. 2가 Key 값보다 작으므로 4를 두 번째 자리에 기억시킨다.  

## 2. 삽입 정렬(Insert Sort) 알고리즘의 특징

### 1)특징
- 선택/거품 정렬은 패스가 거듭될 수록 탐색 범위가 줄어드는 반면에 삽입 정렬은 오히려 점점 정렬 범위가 넚어진다.
- 큰 크림에서 보았을 때 바깥 쪽 루프는 순방향, 안 쪽 루프는 역방향으로 진행하고 있다.

### 2)장점
- 안정한 정렬 방법
- 레코드의 수가 적을 경우 알고리즘 자체가 매우 간단하므로 다른 복잡한 정렬 방법보다 유리할 수 있다.
- 대부분위 레코드가 이미 정렬되어 있는 경우에 매우 효율적일 수 있다.

### 3)단점
- 비교적 많은 레코드들의 이동을 포함한다.
- 레코드 수가 많고 레코드 크기가 클 경우에 적합하지 않다.

- 단순(구현 간단)하지만 비효율적인 방법
  - **삽입 정렬**, 선택 정렬, 버블 정렬
- 복잡하지만 효율적인 방법
  - 퀵 정렬, 힙, 병합 정렬, 기수 정렬

## 3. 삽입 정렬의 시간 복잡도(Time Complexity)와 공간 복잡도(Space Complexity) 
<p align = "center">
<img width="800" alt="image" src="https://user-images.githubusercontent.com/111734605/204970630-457b087f-26af-487c-9491-366b1d04fcb7.png">
</p>

대상 리스트가 이미 정렬된 상태라면 삽입 정렬은 매우 빠르게 동작합니다. 이 경우 삽입 정렬의 시간 복잡도는 선형 즉 O(N)이다. 이 경우가 아주 이상적인 경우이다.

반대로 리스트를 순회할 때마다 모두 요소를 옮겨야만 하는 최악의 경우에는 어떨까? 만약 리스트의 모든 요소를 순회할 때마다 모든 요소를 옮겨야만 하는 최악의 경우가 나올 수 있다.
알고리즘의 시간 복잡도는 이 Worst Case에서의 알고리즘 수행 시간이므로 O($$N^2$$)이다.
$$w(N)$ = $\sum_{i=1}^{N-1} i$ = $\frac{N(N-1)}{2}$ = $\frac{N^2 - N}{2}$$ 
$$w(N)$ \approx = $\frac{N^2)}{2}$ = $ O(N^2)$$



