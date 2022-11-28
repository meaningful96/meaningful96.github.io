---
title: 카테고리랑 페이지 만들기

categories:
  - Etc
tags:
  - [Etc]

toc: true
toc_sticky: true

date: 2022-11-25
last_modified_at: 2022-11-25 
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
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/204193820-8b84f8e9-f345-45bc-ab32-56eb96656538.png">
</p>

만약 이 상황에서 하나의 데이터가 잘못 Mapping되어 7,9,13,<span style = "color:aqua">**100**</span>,10이 되었다고 가정하자. 그러면

```python
## Data Fitting

Data = np.array([7,9,13,100,10])
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
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/204194103-65e2a363-1055-4a26-94d4-520ea4c7554d.png">
</p>

위의 P,ot
