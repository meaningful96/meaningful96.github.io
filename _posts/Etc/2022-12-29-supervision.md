---
title: Various Types of Supervision in Machine Learning(Supervision의 여러 종류)

categories:
  - DeepLearning
tags:
  - [DL,ML, DeepLearning]

toc: true
toc_sticky: true

date: 2022-12-29
last_modified_at: 2022-12-29 
---

## Learning의 종류
데이터 기반 학습 모델(Data driven machine learning model)  Labeling된 샘플의 사용에 따라 분류된다.  
- **Supervised(지도)**: 데이터에 Labeling된 정보들이 있는, 유의미한 값을 지니고 있는 데이터들을 학습하는 방법이다.
- **Unsupervised(비지도)**: 학습해야하는 데이터들이 Labeling 되어있지 않아 정보가 없는 상태로 학습하는 방법이다.
- **Semi-Supervised(반지도)**: Labeling된 데이터들과 Labeling이 되어있지 않은 데이터들 모두를 이용해 학습하는 방법이다.

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/209854306-a466af61-cf9b-4ee4-9a57-56f04c0dc847.png">
</p>

## Supervision의 목적과 종류

보통 모델의 최고 성능은 Supervised model에서 나온다. 그러나 Label이 지정된 표본은 종종 converge하는데 비용이 많이 든다.
이 문제를 해결하기 위해 몇 가지 접근법이 제안되었다.

Goal: 이러한 접근법은 Label이 지정된 샘플을 생성하는 비용을 줄이고자 한다.

### Self Supervision(자체 감독)  
이 경우 레이블은 데이터 자체에서 추출됩니다. 예를 들어, 언어 모델을 훈련시키기 위해서는 일련의 단어들이 특징이고 다음 단어들이 레이블로 필요하다.

### Distant Supervision(원격 감독)  
레이블이 없는 데이터 샘플 세트와 레이블을 추론하는 데 사용할 수 있는 외부 소스가 있다. 예를 들어, 일련의 문서가 있고 어떤 문서에 결혼한 사람의 이름이 포함되어 있는지 
확인하려고 한다고 가정한다. 기혼자 데이터베이스를 사용하여 각 문서에 "참" 또는 "거짓" 태그를 붙일 수 있다. 생성된 데이터 세트는 분류기를 훈련시키는 데 사용될 수 있다.

### Weak Supervision(약한 감독)
일련의 휴리스틱, 함수, 분포 및 도메인 지식을 사용하여 분류기에 노이즈가 많은 레이블을 제공할 수 있다. 분류기는 각 리소스에서 제공하는 이러한 노이즈 레이블을 교육에 사용한다. 
- 휴리스틱(heuristics)
  휴리스틱 또는 발견법이란 불충분한 시간이나 정보로 인하여 합리적인 판단을 할 수 없거나, 체계적이면서 합리적인 판단이 굳이 필요하지 않은 상황에서 사람들이 빠르게 사용할 수 
  있게 보다 용이하게 구성된 간편추론의 방법이다.

### Self Training(자가 학습)
이 전략에서 우리는 라벨링된 샘플과 라벨링되지 않은 샘플 세트를 가지고 있다. 레이블이 지정된 샘플을 사용하여 모델을 교육합니다. 그런 다음 모델은 레이블이 지정되지 않은 샘플에 
주석을 달기 위해 사용된다. 레이블이 지정된 샘플과 함께 이러한 예측 레이블은 원래 모델을 교육하는 데 사용된다. 또 다른 접근법은 다양한 유형의 기능을 사용하여 서로 다른 분류기를 
훈련시키는 것이다. 그런 다음 이러한 분류기는 서로를 위한 레이블을 제공한다. 이 접근법을 "공동 훈련"이라고 한다.

### Active Learning(능동 학습)
기계 학습 모델에 레이블을 제공하는 것은 종종 비용이 많이 든다. 최소 레이블링된 표본을 최대한 활용하기 위해 분류기는 레이블링된 경우 제공할 정보 측면에서 최적의 표본을 식별하도록 
요구된다. 이러한 샘플은 (도메인 전문가가) 주석을 달고 훈련을 위해 모델에 피드백된다. 이 프로세스는 필요한 성능이 달성될 때까지 반복된다.

## Reference
[Various Types of Supervision in Machine Learning](https://medium.com/@behnamsabeti/various-types-of-supervision-in-machine-learning-c7f32c190fbe)
