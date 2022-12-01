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
### 1) 선택 정렬의 기본 메커니즘  
- 제자리 정렬(in-place sorting) 알고리즘의 하나  
  - 입력 배열(정렬되지 않은 값들) 이외에 다른 추가 메모리를 요구하지 않는 정렬 방법  
  - 해당 순서에 원소를 넣을 위치는 이미 정해져 있고, 어떤 원소를 넣을지 선택하는 알고리즘  
  - 첫 번째 순서에는 첫 번째 위치에 가장 최솟값을 넣는다.  
  - 두 번째 순서에는 두 번째 위치에 남은 값 중에서의 최솟값을 넣는다.  

- 과정 설명
  \lowercase\expandafter{\romannumeral1} 주어진 배열 중에서 최솟값을 찾는다.  
  \lowercase\expandafter{\romannumeral2} 그 값을 맨 앞에 위치한 값과 교체한다(패스(pass)).  
  \lowercase\expandafter{\romannumeral3} 맨 처음 위치를 뺀 나머지 리스트를 같은 방법으로 교체한다.  
  \lowercase\expandafter{\romannumeral4} 하나의 원소만 남을 때까지 위의 1~3 과정을 반복한다.  
 
