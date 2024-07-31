---
title: "[Python]유사 딕셔너리 defaultdict()란?"

categories:
  - py

toc: true
toc_sticky: true

date:  2023-05-05
last_modified_at: 2023-05-05
---
# Collection모듈 & defaultdict() 함수
Python의 내장 자료구조인 <span style = "color:red">**사전(dictionary)**</span>를 사용하다 보면 <u><b>어떤 키(key)에 대한 값(value)이 없는 경우가 종종 발생</b></u>하고 따라서
이에 대한 처리를 해야하는 경우가 자주 발생한다. 

## 일반적인 사전 기본값 처리
다음의 알파벳 글자의 수를 세어서 사전에 저장해주는 함수를 보며 예시를 든다.
```python
def countLetters(word):
    counter = {}
    for letter in word:
        if letter not in counter:
            counter[letter] = 0
        counter[letter] += 1
    return counter
```
`for`루프 안에 `if` 조건절을 통해서 `counter`사전에 어떤 글자가 key로 존재하지 않는 경우, 해당 키에 대한 기본값을 0으로 세팅해주고 있다. 이러한 패턴은 파이썬에서 dictionary를 사용
할 때 상당히 자주 접할 수 있는데, 코드 가독성 측면에서는 이렇게 사소한 처리가 주요 흐름을 파악하는데 방해가 되기도 한다.

## dict.setdefault
위에처럼 조건문을 피할 수 있도록 파이썬의 dictionray 자료구조는 `setdefault` 함수를 제공한다. setdefualt(key, default value)로 구성된다.
```python
def countLetters(word):
    counter = {}
    for letter in word:
        counter.setdefault(letter, 0)
        counter[letter] += 1
    return counter
```
이 방식의 유일한 단점은 for 문을 iteration도는 동안 매번 setdefault함수가 호출된다는 것이다.

## Better Way: collection.defaultdict

파이썬의 내장 모듈인 `collection`의 defaultdict은 생성자로 기본값을 생성해주는 함수를 넘기면, <span style = "color:red">**모든 키에 대해서 값이 없는 경우 자동으로 생성자의 인자로 넘어온 함수를 호출하여 
모든 결과값으로 설정**</span>해준다.
```python
from collections import defaultdict

def countLetters(word):
    counter = defaultdict(int)
    for letter in word:
        counter[letter] += 1
    return counter
```

## 활용: 사전 기본값으로 빈 리스트 셋팅
collection.defaultdict를 활용하여 데이터를 특정 기준에 의해 카테고리롤 묶는 경우를 들 수 있다.

**[Ex 1)]**
```python
from collections import defaultdict

def groupWords(words):
    grouper = defaultdict(list)
    for word in words:
        length = len(word)
        grouper[length].append(word)
    return grouper
```

**[Ex 2]**
```python
from collections import defaultdict

def groupWords(words):
    grouper = defaultdict(set)
    for word in words:
        length = len(word)
        grouper[length].add(word)
    return grouper
```

## Reference
[[파이썬] 사전의 기본값 처리 (dict.setdefault / collections.defaultdict)]("https://www.daleseo.com/python-collections-defaultdict/")  
[파이썬[Python] defaultdict(기본값 있는 dictionary) - collections 모듈]("https://appia.tistory.com/218")
