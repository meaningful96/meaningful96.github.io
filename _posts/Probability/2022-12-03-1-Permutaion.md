---
title: Chapter 0. 순열과 조합(Permutation and Combination)

categories: 
  - Probability
tags:
  - [Math, AI Math, Probability]

toc: true
toc_sticky: true

date: 2022-12-03
last_modified_at: 2022-12-03 
---

## 1. Machine Learning Model을 확률적으로 이해하기

<p align="center">
<img width="500" alt="1" src="https://user-images.githubusercontent.com/111734605/205427933-1d7f77a1-0273-4d68-98b4-a52ce5c71b18.png">
</p>

- 우리는 𝑁개의 학습 데이터로 기계학습 모델을 학습한다.
- 일반적으로 기계학습 모델의 출력은 확률 형태를 띤다.
  - Ex1) 이미지 분류 모델 𝜃가 이미지 𝑥에 대해서 75% 확률로 고양이일 것이라고 예측했다.
  - Ex2) 글 쓰기 모델 𝜃는 나는 밥을 “ 이후에 먹었다 라는 단어가 나올 확률을 42% 로 예측했다.

## 2. 순열과 조합
### 1) 경우의 수
확률론의 가장 기본적이고 중요한 개념중 하나이다. 1회의 시행에서 미래에 일어날 수 있는 사건의 가짓수가 n개라고 할 때, 그 사건의 경우의 수를 n이라고 한다.
#### Ex)
2, 5, 7의 공을 뽑을 경우 이 3개의 공은 $$3!$$의 경우의 수를 보여줄 수 있다. 첫번째 공에 가능한 게 2, 5, 7의 세가지이고, 두번째 공에 가능한 건 각 공마다 두
가지(2의 경우 5와 7)이므로 3에 2를 곱하고, 마지막 공으로 가능한 건 6가지 경우마다 한가지씩(2->5의 경우 7)이므로 $$3\times 2\times 1$$을 하면 가능한 경우의 수가 나온다.
그럼 전체 경우의 수인 $$7 \times 6 \times 5$$을 이렇게 나오는 숫자인 $$3!$$로 나눠주면 가능한 "경우위 수"인 35가 나오게 된다.

이를 2, 5, 7의 세 개의 공을 뽑을 확률이라면<span style = "font-size:150%">$$\frac{3!}{7 \times 6 \times 5} = \frac{6}{210} = \frac{1}{35}$$</span>

### 2) 순열(Permutation)

서로 다른 n개의 원소에서 r개를 <span style = "color:aqua">**중복없이 순서에 상관있게**</span> 선택하는 혹은 나열하는 것을 순열(permutation)이라고 한다.

<span style = "font-size:150%">$$_nP_r = \frac{n!}{(n-r)!}$$</span>

**[Input]**  
```python
from itertools import permutations

arr = ['A','B','C']

# 원소 중에서 2개를 뽑는 모든 순열 계산
result1 = list(permutations(arr, 1))
result2 = list(permutations(arr, 2))
result3 = list(permutations(arr, 3))

print(result1)
print(result2)
print(result3)
```  
**[Output]**  
```python
# print(result1)
[('A',), ('B',), ('C',)] 

# print(result2)
[('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]

# print(result3)
[('A', 'B', 'C'), ('A', 'C', 'B'), ('B', 'A', 'C'), ('B', 'C', 'A'), ('C', 'A', 'B'), ('C', 'B', 'A')]
```

### 3) 조합(Combination)
서로 다른 n개의 원소를 가지는 어떤 집합 에서 <span style = "color:aqua">**순서에 상관없이 r개의 원소를 선택**</span>하는 것이며, 이는 n개의 원소로 이루어진 집합에서 r개의 원소로 이루어진 부분집합을 만드는 것 혹은 찾는 것과 같다. 

<span style = "font-size:150%">$$_nC_r = _nC_{n-r}=\frac{_nP_r}{r!} =\frac{1}{r!} \times \frac{n!}{(n-r)!}$$</span>

**[Input]**  
```python
from itertools import combinations

arr = ['A','B','C']

## Combination 조합 4C2 = (4x3)/(2x1)
arr = ['A','B','C']
result4 = list(combinations(arr, 1))
result5 = list(combinations(arr, 2))
result6 = list(combinations(arr, 3))

print(result4)
print(result5)
print(result6)
```  
**[Output]**  
```python
# print(result4)
[('A',), ('B',), ('C',)]

# print(result5)
[('A', 'B'), ('A', 'C'), ('B', 'C')]

# print(result6)
[('A', 'B', 'C')]
```
### 3) 중복 순열(Permutation with Repetition)
서로 다른 𝑛개에서 <span style = "color:aqua">**중복을 포함해 𝑟개를 뽑아 특정한 순서**</span>로 나열 하는 것을 의미한다.

-세 개의 숫자 1, 2, 3 을 이용해 만들 수 있는 두 자리 자연수의 개수는
  - 단 , 같은 숫자를 여러 번 사용하거나 , 사용하지 않은 숫자가 있어도 괜찮다  
  - 모든 경우 11 12 13 21 22 23 31 32 33  
  - 경우의 수 32=9  

<span style = "font-size:150%">$$_n\Pi_r = n^r$$</span>

**[Input]**
```python
from itertools import product

arr = ['A','B','C']

# 원소 중에서 2개를 뽑는 모든 중복 순열 계산
result1 = list(product(arr,repeat = 2))
result2 = list(product(arr,repeat = 3))
result3 = list(product(arr,repeat = 4))

print(result1)
print(result2)
print(result3)

print(len(result1))
print(len(result2))
print(len(result3))
```

**[Output]**  
```python
# print(result1)
[('A', 'A'), ('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'B'), ('B', 'C'), ('C', 'A'), ('C', 'B'), ('C', 'C')]
# print(result2)
[('A', 'A', 'A'), ('A', 'A', 'B'), ('A', 'A', 'C'), ('A', 'B', 'A'), ('A', 'B', 'B'), ('A', 'B', 'C'), ('A', 'C', 'A'), ('A', 'C', 'B'), ('A', 'C', 'C'), ('B', 'A', 'A'), ('B', 'A', 'B'), ('B', 'A', 'C'), ('B', 'B', 'A'), ('B', 'B', 'B'), ('B', 'B', 'C'), ('B', 'C', 'A'), ('B', 'C', 'B'), ('B', 'C', 'C'), ('C', 'A', 'A'), ('C', 'A', 'B'), ('C', 'A', 'C'), ('C', 'B', 'A'), ('C', 'B', 'B'), ('C', 'B', 'C'), ('C', 'C', 'A'), ('C', 'C', 'B'), ('C', 'C', 'C')]
# print(result3)
[('A', 'A', 'A', 'A'), ('A', 'A', 'A', 'B'), ('A', 'A', 'A', 'C'), ('A', 'A', 'B', 'A'), ('A', 'A', 'B', 'B'), ('A', 'A', 'B', 'C'), ('A', 'A', 'C', 'A'), ('A', 'A', 'C', 'B'), ('A', 'A', 'C', 'C'), ('A', 'B', 'A', 'A'), ('A', 'B', 'A', 'B'), ('A', 'B', 'A', 'C'), ('A', 'B', 'B', 'A'), ('A', 'B', 'B', 'B'), ('A', 'B', 'B', 'C'), ('A', 'B', 'C', 'A'), ('A', 'B', 'C', 'B'), ('A', 'B', 'C', 'C'), ('A', 'C', 'A', 'A'), ('A', 'C', 'A', 'B'), ('A', 'C', 'A', 'C'), ('A', 'C', 'B', 'A'), ('A', 'C', 'B', 'B'), ('A', 'C', 'B', 'C'), ('A', 'C', 'C', 'A'), ('A', 'C', 'C', 'B'), ('A', 'C', 'C', 'C'), ('B', 'A', 'A', 'A'), ('B', 'A', 'A', 'B'), ('B', 'A', 'A', 'C'), ('B', 'A', 'B', 'A'), ('B', 'A', 'B', 'B'), ('B', 'A', 'B', 'C'), ('B', 'A', 'C', 'A'), ('B', 'A', 'C', 'B'), ('B', 'A', 'C', 'C'), ('B', 'B', 'A', 'A'), ('B', 'B', 'A', 'B'), ('B', 'B', 'A', 'C'), ('B', 'B', 'B', 'A'), ('B', 'B', 'B', 'B'), ('B', 'B', 'B', 'C'), ('B', 'B', 'C', 'A'), ('B', 'B', 'C', 'B'), ('B', 'B', 'C', 'C'), ('B', 'C', 'A', 'A'), ('B', 'C', 'A', 'B'), ('B', 'C', 'A', 'C'), ('B', 'C', 'B', 'A'), ('B', 'C', 'B', 'B'), ('B', 'C', 'B', 'C'), ('B', 'C', 'C', 'A'), ('B', 'C', 'C', 'B'), ('B', 'C', 'C', 'C'), ('C', 'A', 'A', 'A'), ('C', 'A', 'A', 'B'), ('C', 'A', 'A', 'C'), ('C', 'A', 'B', 'A'), ('C', 'A', 'B', 'B'), ('C', 'A', 'B', 'C'), ('C', 'A', 'C', 'A'), ('C', 'A', 'C', 'B'), ('C', 'A', 'C', 'C'), ('C', 'B', 'A', 'A'), ('C', 'B', 'A', 'B'), ('C', 'B', 'A', 'C'), ('C', 'B', 'B', 'A'), ('C', 'B', 'B', 'B'), ('C', 'B', 'B', 'C'), ('C', 'B', 'C', 'A'), ('C', 'B', 'C', 'B'), ('C', 'B', 'C', 'C'), ('C', 'C', 'A', 'A'), ('C', 'C', 'A', 'B'), ('C', 'C', 'A', 'C'), ('C', 'C', 'B', 'A'), ('C', 'C', 'B', 'B'), ('C', 'C', 'B', 'C'), ('C', 'C', 'C', 'A'), ('C', 'C', 'C', 'B'), ('C', 'C', 'C', 'C')]

print(len(result1)) # 9
print(len(result2)) # 27
print(len(result3)) # 81
```


### 4) 중복 조합(Combination with Repetition)
서로 다른 𝑛개에서 <span style = "color:aqua">**중복을 포함해 순서를 고려하지 않고**</span> 𝑟개를 뽑는 것을 의미한다.


<span style = "font-size:150%">$$_nH_r = \frac{(r+n-1)!}{r!(n-1)!} = _{r+(n-1)}C_r$$</span>

#### Ex)
$$_3H_4 = \frac{6!}{4!2!} = _{4+3}C_4 = _7C_4 = _7C_3 = \frac{7\times6\times5}{3\times2\times1}$$

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/205431055-144f5cc0-806b-4399-8969-b97d740401e1.png">
</p>

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/205431078-cae6af64-72e9-4332-b4f5-e04700e94232.png">
</p>

**[Input]**
```python
from itertools import combinations_with_replacement

arr = ['A','B','C']

# 원소 중에서 2개를 뽑는 모든 중복 조합 계산
result1 = list(combinations_with_replacement(arr, 2))
result2 = list(combinations_with_replacement(arr, 3))
result3 = list(combinations_with_replacement(arr, 4))

print(result1)
print(result2)
print(result3)
```

**[Output]**  
```python
# print(result1)
[('A', 'A'), ('A', 'B'), ('A', 'C'), ('B', 'B'), ('B', 'C'), ('C', 'C')]

# print(result2)
[('A', 'A', 'A'), ('A', 'A', 'B'), ('A', 'A', 'C'), ('A', 'B', 'B'), ('A', 'B', 'C'), ('A', 'C', 'C'), ('B', 'B', 'B'), ('B', 'B', 'C'), ('B', 'C', 'C'), ('C', 'C', 'C')]

# print(result2)
[('A', 'A', 'A', 'A'), ('A', 'A', 'A', 'B'), ('A', 'A', 'A', 'C'), ('A', 'A', 'B', 'B'), ('A', 'A', 'B', 'C'), ('A', 'A', 'C', 'C'), ('A', 'B', 'B', 'B'), ('A', 'B', 'B', 'C'), ('A', 'B', 'C', 'C'), ('A', 'C', 'C', 'C'), ('B', 'B', 'B', 'B'), ('B', 'B', 'B', 'C'), ('B', 'B', 'C', 'C'), ('B', 'C', 'C', 'C'), ('C', 'C', 'C', 'C')]

print(len(result1)) # 6
print(len(result2)) # 10
print(len(result3)) # 15
```

