---
title: map함수와 lambda 함수

categories:
  - py
tags:
  - [py,python, map, lambda]

toc: true
toc_sticky: true

date: 2022-11-30
last_modified_at: 2022-11-30 
---

## 1. Map 함수
<span style = "color: aqua">**map(function, iterable)**</span>  

- map함수의 변수는 두 가지이다. 첫 번째는 function을, 두 번째 변수는 반복 가능한 자료형(List, Tuple 등)이 할당된다.  
- map 함수의 return값은 map객체 이므로 해당 자료형을 List나 Tuple로 형 변환시켜주어야 한다.  
- 함수의 동작은 두 번째 인자로 들어온 반복 가능한 자료형을 함수에 하나씩 대입해서 함수를 실행하는 함수인 것이다.
- **map(적용시킬 함수, 적용할 값들)**

### Ex1) map함수를 사용하는 것과 아닌것의 차이
```python
# 리스트에 값을 하나씩 더해서 새로운 리스트를 만드는 작업
myList = [1, 2, 3, 4, 5]

# for 반복문 이용
Result1 = []
for val in myList:
    result1.append(val + 1)

print(Result1)
print(f'Result1 : {Result1}')

# map 함수 이용
def add_one(n):
    return n + 1

Result2 = list(map(add_one, myList))  # map반환을 list 로 변환
print(Result2)
print(f'result2 : {Result2}')

##출력
[2,3,4,5,6]
Result1 : [2,3,4,5,6]
[2,3,4,5,6]
Result2 : [2,3,4,5,6]
```

### Ex2) 리스트와 map함수
```
import math # ceil함수 사용하려고 호출

## 예제1) 리스트의 값을 정수 타입으로 변환
Res1 = list(map(int, [1.1,2.2,3.3,4.4,5.5]))
print(Res1)
print(f'Res1 : {Res1}')
  
## 예제2) 리스트의 값을 제곱
def func_power(x):
    return x**2
 
Res2 = list(map(func_power, [1,2,3,4,5]))
print(Res2)
print(f'Res2 : {Res2}')

## 리스트의 값 소수점 올림
Res3 = list(map(math.ceil, [1.1,2.2,3.3,4.7,5.6]))
print(Res3)
print(f'Res3 : {Res3}')

## 출력
[1, 2, 3, 4, 5]
Res1 : [1, 2, 3, 4, 5]
[1, 4, 9, 16, 25]
Res2 : [1, 4, 9, 16, 25]
[2, 3, 4, 5, 6]
Res3 : [2, 3, 4, 5, 6]
```

### Ex3) 람다함수를 이용
```python
# map과 lambda

# 일반적인 함수 이용
def func_power2(x):
  return x**2
  
Res1 = list(map(func_power2, [5,4,3,2,1]))
print(Res1)
print(f'Res1 : {Res1}')

# 람다 함수
Res2 = list(map(lambda x: x**2, [5,4,3,2,1]))
print(Res2)
print(f'Res2 : {Res2}')

##출력
[25, 16, 9, 4, 1]
Res1 : [25, 16, 9, 4, 1]
[25, 16, 9, 4, 1]
Res2 : [25, 16, 9, 4, 1]
```

## lambda 함수
def 를통해 함수를 정의할 수 있고, 이렇게 정의할 경우 언제든지 함수 이름만 이용해서 여러 번 호출해서 사용할 수 있다. 반면, 람다 함수는 일종의 일회성 함수인 것이다.
lambda라고 선언을 함과 동시에 그 뒤에 함수식을 정의해주면 일회성으로 그 함수를 사용하겠다는 것을 의미한다.

lambda 인자: 표현식  
lambda 라는 키워드를 입력하고 뒤에는 매개변수(인자)를 입력하고 콜론(:)을 넣은다음에 그 매개변수(인자)를 이용한 동작들을 적으면 된다.
예를 들면 인자로 들어온 값에 2를 곱해서 반환한다고 하면 lambda x : x * 2  이다.

### Ex1) lambda 함수와 filter 함수
#### filter 함수  
<span style = 'color: aqua'>**filter(함수, 리스트나 튜플)**</span>

- 첫번째 인자에는 두번째 인자로 들어온 리스트나 튜플을 하나하나씩 받아서 필터링할 함수를 넣는다.
- 두번쨰 인자에는 리스트나 튜플을 집어 넣는다.

예를들어 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 까지의 리스트가 있을때 짝수들만 filter 함수를 이용해서 리스트를 다시 만든다고 했을때, filter 함수와 람다를 이용할 수 있다.

```python
# 1. 일반 함수 버전
def is_even(x):
    return x % 2 == 0
 
 
Res1 = list(filter(is_even, range(10)))  # [0 ~ 9]
print(Res1)
 
 
# 2.  람다 함수 버전
Res2 = list(filter((lambda x: x % 2 == 0), range(10)))  # [0 ~ 9]
print(Res2)

##출력
[0,2,4,6,8]
[0,2,4,6,8]
```
 
