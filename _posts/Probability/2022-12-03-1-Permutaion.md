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
확률론의 가장 기본적이고 중요한 개념중 하나이다. 1회의 시행에서 미래에 일어날 수 있는 사건의 가짓수가 nn개라고 할 때, 그 사건의 경우의 수를 n이라고 한다.
#### Ex)
2, 5, 7의 공을 뽑을 경우 이 3개의 공은 $3!$의 경우의 수를 보여줄 수 있다. 첫번째 공에 가능한 게 2, 5, 7의 세가지이고, 두번째 공에 가능한 건 각 공마다 두
가지(2의 경우 5와 7)이므로 3에 2를 곱하고, 마지막 공으로 가능한 건 6가지 경우마다 한가지씩(2->5의 경우 7)이므로 $3\times 2\times 1$을 하면 가능한 경우의 수가 나온다.
그럼 전체 경우의 수인 $7 \times 6 \times 5$을 이렇게 나오는 숫자인 $3!$로 나눠주면 가능한 "경우위 수"인 35가 나오게 된다.

이를 2, 5, 7의 세 개의 공을 뽑을 확률이라면 $\frac{3!}{7 \times 6 \times 5} = \frac{6}{210} = \frac{1}{35}$
### 2)
