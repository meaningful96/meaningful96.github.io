---
title: "[데이터 전처리] min-MAX Scailing, Standard Scailing"

categories: 
  - DeepLearning
tags:
  - [Data Preprocessing]

toc: true
toc_sticky: true

date: 2023-04-24
last_modified_at: 2023-04-24
---

# 데이터 전처리 과정에서 스케일링(Scailing)이 필요한 이유?
여러 가지 변수들을 동시에 이용해 학습하는 경우가 있는데, 이 때 크기의 gap이 너무 크면 변수가 미치는 영향력이 제대로 표현되지 않을 수 있다. 따라서 모든 변수의 범위를 같게 해주는 **스케일링(Scailing)** 과정이 필요하다. 여기서 중요한 점은, <span style = "color:gold">**스케일링은 분포의 모양을 바꿔주지 않는다**</span>는 사실이다. 

대표적인 스케일링 방식으로 **min-Max Scailing**과 **Standard Scailing**이 있다.

## 1. min-Max Scailing

<p align="center">
<img width="300" alt="1" src="https://user-images.githubusercontent.com/111734605/234765198-531ec9c3-9907-44c9-8cd2-78e311222db8.png">
</p>

- 변수의 범위를 바꿔주는 정규화 스케일링 기법 \[0, 1\]
- 이상 값 존재에 민감
- 회귀(Regression)에 적합

### 1) Basis

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
mu = 0.0
sigma = 1.0

x = np.linspace(-8, 8, 1000)
y = 5*(1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x-mu)**2 / (2 * sigma**2))


f1 = plt.figure()
ax1 = plt.axes()
ax1.plot(x,y,'r-', label = "Before", alpha = 0.7)


## min-Max Rescaling

min_val = np.min(y)
max_val = np.max(y)

rescale_minMax = (y - min_val)*((1-0)/(max_val -min_val)) + 0.
ax1.plot(x,rescale_minMax,'b-', label = "After", alpha = 0.7)
ax1.legend(loc='upper right')
plt.show()
```

이 방식으로 하면 x축의 범위는 그대로이고, y축의 범위만 바뀐다.

<p align="center">
<img width="500" alt="1" src="https://user-images.githubusercontent.com/111734605/234766932-dcb5e9e0-1ced-41c2-9446-031ce6a4a472.png">
</p>

<br/>

### 2) Scikit-learn 라이브러리 사용

- Scikit-learn 패키지를 이용해서 만들 수 있다.
- <span style = "color:gold">**Scailing 값을 조정하는 과정이기 때문에 수치형 변수에만 적용**</span>

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.close("all")

np.random.seed(0)
mu = 0.0
sigma = 1.0

x = np.linspace(-8, 8, 1000)
y = 5*(1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x-mu)**2 / (2 * sigma**2))

from sklearn.preprocessing import MinMaxScaler

X = np.vstack([x,y]).T

from sklearn.preprocessing import MinMaxScaler

X = np.vstack([x,y]).T

for i in range(len(y)):
    # MinMaxScaler 선언 및 Fitting
    mMscaler = MinMaxScaler()
    mMscaler.fit(X)
    
    # 데이터 변환
    mMscaled_data = mMscaler.transform(X).T
  

f2 = plt.figure(2)
ax2 = plt.axes()
ax2.plot(mMscaled_data[0],mMscaled_data[1], "g-", label = "after")
ax2.legend(loc='upper right')
ax2.set_title("figure 2")
plt.show()


f3 = plt.figure(3)
ax3 = plt.axes()
ax3.plot(x,y,'r-', label = "Before", alpha = 0.7)
ax3.plot(mMscaled_data[0],mMscaled_data[1], "g-", label = "after")
ax3.set_title("figure 3")
ax3.legend(loc='upper right')
plt.show()
```

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/234771538-1c8c890f-b865-4b7b-8d4c-25f81980ff3c.png">
</p>


