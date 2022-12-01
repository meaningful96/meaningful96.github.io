---
title: Upper Triangular matrix(상삼각행렬)

categories: 
  - py
tags:
  - [py.Python]

toc: true
toc_sticky: true

date: 2022-12-01
last_modified_at: 2022-12-01 
---

## Upper Triangular Matrix
### 1) 개념

<p align = 'center'>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/111734605/204948256-bb5bd215-b23d-48be-8489-feb8bc79bc95.png">
</p>

삼각행렬은 대각선을 기준으로 위에 부분만 그 값이 0이 아니고, 대각선 밑의 값은 모두 0인 행렬이다.

### 2) Python 코드

np.triu 함수의 사용법은 array와 k값을 차례대로 인자로 적어주면 되며, k를 생략할 경우 기본 값은 0으로 지정된다.k의 의미는 가장 윗줄부터 0의 개수가 k개부터 시작한다는 의미로 
보시면 되며, 아래로 1줄씩 내려갈 때 마다 0의 개수가 1개씩 많아지는 원리이다.  

#### Ex1)
**[Input]**  
```python
import numpy as np

a = np.array([[1, 2, 3, 4], 
             [5, 6, 7, 8],
             [9, 10, 11, 12],
             [13, 14, 15, 16]])


a1 = np.triu(a)
a2 = np.triu(a, k = 1)
a3 = np.triu(a, k = -1)
print(a1)
print(a2)
print(a3)
```

**[Output]**  
```python
print(a1)         print(a2)         print(a3)
[[ 1  2  3  4]    [[ 0  2  3  4]    [[ 1  2  3  4]
 [ 0  6  7  8]     [ 0  0  7  8]     [ 5  6  7  8]
 [ 0  0 11 12]     [ 0  0  0 12]     [ 0 10 11 12]
 [ 0  0  0 16]]    [ 0  0  0  0]]    [ 0  0 15 16]]
```

#### Ex2)
**[Input]**
```python
import numpy as np

a = np.array([[1,2,3,4],
              [5,6,7,8],
              [10,11,12,13],
              [14,15,16,17]])

b1 = np.triu(a, k = 0)
b2 = np.triu(a, k = 1)
b3 = np.triu(a, k = -1)
b4 = np.triu(a, k = 4)

print(b1)
print(b2)
print(b3)
print(b4)
```
**[Ouput]**
```python
print(b1)         print(b2)         print(b3)         print(b4)
[[ 1  2  3  4]    [[ 0  2  3  4]    [[ 1  2  3  4]    [[0 0 0 0]
 [ 0  6  7  8]     [ 0  0  7  8]     [ 5  6  7  8]     [0 0 0 0]
 [ 0  0 12 13]     [ 0  0  0 13]     [ 0 11 12 13]     [0 0 0 0]
 [ 0  0  0 17]]    [ 0  0  0  0]]    [ 0  0 16 17]]    [0 0 0 0]]
 ```
