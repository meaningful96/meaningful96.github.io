---
title: 0.Cross Validation

categories:
  - ML
tags:
  - [ML,DL,Machine learning, Deep Learning]

toc: true
toc_sticky: true

date: 2022-11-21
last_modified_at: 2022-11-21 
---

## Cross Validation
### 1. Dataset에 관하여
Machine Learning과 Deep Learning에 있어서 가지고 있는 데이터를 전처리하고 가고하는 과정은 매우 중요하다. 전처리 과정을 거친 데이터들은 training할 준비가 된 데이터들이다.
여기서 가지고 있는 Dataset을 모두 training에 활용하지 않는다. Dataset을 왜 모두 tarining에 쓰지 않을까? 가장 근본적이고, 표면적인 이유는 결국 모든 Data들은 '돈'과 직결
되기 때문이다.

1) 새로운 data는 결국 돈!! money !!
2) 따라서 가지고 있는 Dataset을 쪼개서 활용하는 것이 바람직하다.

### 2. Training set, Test set
