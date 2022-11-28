---
title: Robust의 의미

categories:
  - Etc
tags:
  - [Etc]

toc: true
toc_sticky: true

date: 2022-11-28
last_modified_at: 2022-11-28
---

## 1. Robust의 의미
흔히 통계학이나 Mahcine Learning, Deep Learning 공부를 하다보면 Robust 라는 단어를 자주 접하게 된다. Robust의 사전적 의미는 <span style ="color:aqua">견고한, 굳건한, 강직한</span>
이다. 그리고 공부를 하다 접하게 되는 문장은 다음과 같다.

- **'Data가 Robust하다'** 
- **'Robust한 방법으로 추론하였다'** 

### Robust의 의미 in DataScience  
Data를 바라보는 관점에서 Robust하다는 것은 결국 <span style = "color:aqua">**"극단값에 예민하지 않은, 민감하지 않은"**</span>으로 해석할 수 있다. 

예를 들어 어떤 데이터들이 각각 7,9,10,11,13 이라고 하자. 이 다섯 개의 값의 평균은 $\frac{7+9+10+11+13}{5}$ 이므로 10이 된다. 이 경우에 각각의 데이터들이 평균에서 멀리 안 떨어지고 
가깝게 분포되어 있는 것을 볼 수 있다.

```python
import numpy as np
import matplotlib.pyplot as plt

## Data Fitting

Data = np.array([7,9,13,11,10])
DataIndex = np.array([1,2,3,4,5])

Data_Coupled_Index_with_Data = np.column_stack([DataIndex,Data]) #(1,7), (2,9), (3,13), (4,11), (5,10)
Mean = np.mean(Data)

f1 = plt.figure()
ax1 = plt.axes()
ax1.plot(DataIndex, Data, 'ro')


## Line of Mean

Mline_X = np.linspace(1,5,100)
Mline_Y = np.ones([100,1])*Mean
ax1.plot(Mline_X,Mline_Y,'g')
```

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/204193820-8b84f8e9-f345-45bc-ab32-56eb96656538.png">
</p>

만약 이 상황에서 하나의 데이터가 잘못 Mapping되어 7,9,13,<span style = "color:aqua">**100**</span>,10이 되었다고 가정하자. 그러면

```python
## Data Fitting

Data = np.array([7,9,13,500,10])
DataIndex = np.array([1,2,3,4,5])

Data_Coupled_Index_with_Data = np.column_stack([DataIndex,Data]) #(1,7), (2,9), (3,13), (4,11), (5,10)
Mean = np.mean(Data)

f1 = plt.figure()
ax1 = plt.axes()
ax1.plot(DataIndex, Data, 'ro')


## Line of Mean

Mline_X = np.linspace(1,5,100)
Mline_Y = np.ones([100,1])*Mean
ax1.plot(Mline_X,Mline_Y,'g')
```

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/204194556-6268ebc5-6d35-4666-8f77-066eebdfb7c6.png">
</p>

위의 Plot 처럼 바뀌게된다. 평균이 10에서 107.8로 바뀌고 Outlier에 의해 모든 데이터들이 평균과 많이 차이가나게 되는 것을 볼 수 있다. 따라서 이 상태로 데이터들의 평균은 107.8입니다
라고하면 부적절해 보인다.

### 이상값(Ideal Value)
이상값이란 <span style = "color:aqua">**"다른 값에 비해 지나치게 크거나 작은값"**</span>을 말한다. 위의 두 번째 경우 이상값을 평균값으로 하기엔 부적합하다. 이 경우 이상값으로
중앙값을 사용하는 것이 좀 더 데이터의 특성을 잘 반영한다.(중앙값은 10이다.)

## 2. 결론
결론적으로 <span style = "color:aqua">**Robust**</span>하다는 것의 의미는 이상값에 더 적은 영향을 주는 값을 말한다. 
- Robust = 이상값들에 Insensitive한, Less sensitive한 것을 의미한다.
