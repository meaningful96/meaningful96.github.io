---
title: "[데이터 전처리] min-MAX Scailing, Standard Scailing"

categories: 
  - DeepLearning

toc: true
toc_sticky: true

date: 2023-04-24
last_modified_at: 2023-04-24
---

# 데이터 전처리 과정에서 스케일링(Scailing)이 필요한 이유?
여러 가지 변수들을 동시에 이용해 학습하는 경우가 있는데, 이 때 크기의 gap이 너무 크면 변수가 미치는 영향력이 제대로 표현되지 않을 수 있다. 따라서 모든 변수의 범위를 같게 해주는 **스케일링(Scailing)** 과정이 필요하다. 여기서 중요한 점은, <span style = "color:red">**스케일링은 분포의 모양을 바꿔주지 않는다**</span>는 사실이다. 

대표적인 스케일링 방식으로 **min-Max Scailing**과 **Standard Scailing**이 있다.

## 1. min-Max Scailing(Normalization)

<p align="center">
<img width="200" alt="1" src="https://user-images.githubusercontent.com/111734605/234765198-531ec9c3-9907-44c9-8cd2-78e311222db8.png">
</p>

- 변수의 범위를 바꿔주는 정규화 스케일링 기법 \[0, 1\]
- 이상 값 존재에 민감
- 회귀(Regression)에 적합
- 딥러닝에서 말하는 <span style = "color:green">**Normalization(정규화)이 보통 min-Max Scailing**</span>을 말한다.

### 1) Basis

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




## min-Max Rescaling

min_val = np.min(y)
max_val = np.max(y)
x_rescale = np.linspace(0,1,1000)
rescale_minMax = (y - min_val)*((1-0)/(max_val -min_val)) + 0.

f1 = plt.figure(1)
ax1 = f1.add_subplot(131)
ax1.plot(x,y,'r-', label = "Before", alpha = 0.8)
ax1.plot(x,rescale_minMax,'b-', label = "y only", alpha = 0.8)
ax1.legend(loc='upper right')
ax1.set_title("figure 1: origin VS y-only")
plt.show()

ax2 = f1.add_subplot(132)
ax2.plot(x,rescale_minMax,'b-', label = "y only", alpha = 0.8)
ax2.plot(x_rescale,rescale_minMax,'g-', label = "x & y", alpha = 0.8)
ax2.legend(loc='upper right')
ax2.set_title("figure 2: y-only VS Final(x & y)")
plt.show()

ax3 = f1.add_subplot(133)
ax3.plot(x_rescale, rescale_minMax, 'g-', label = "x & y", alpha = 0.8)
ax3.legend(loc='upper right')
ax3.set_title("figure3: Final(x & y)")
plt.show()
```

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/234786087-36664899-e17b-4c9e-843b-1ba8a158d8fc.png">
</p>

<br/>

### 2) Scikit-learn 라이브러리 사용

- Scikit-learn 패키지를 이용해서 만들 수 있다.
- <span style = "color:red">**Scailing 값을 조정하는 과정이기 때문에 수치형 변수에만 적용**</span>

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

### Using sklearn

from sklearn.preprocessing import MinMaxScaler

X = np.vstack([x,y]).T

for i in range(len(y)):
    # MinMaxScaler 선언 및 Fitting
    mMscaler = MinMaxScaler()
    mMscaler.fit(X)
    
    # 데이터 변환
    mMscaled_data = mMscaler.transform(X).T
  
    
f2 = plt.figure(2)

ax4 = f2.add_subplot(121)
ax4.plot(x,y,'r-', label = "Before", alpha = 0.7)
ax4.plot(mMscaled_data[0],mMscaled_data[1], "g-", label = "after")
ax4.set_title("figure 4: Orgin VS Normalization")
ax4.legend(loc='upper right')
plt.show()

ax5 = f2.add_subplot(122)
ax5.plot(mMscaled_data[0],mMscaled_data[1], "g-", label = "after")
ax5.legend(loc='upper right')
ax5.set_title("figure 5: Normalization")
plt.show()
```

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/234786319-74974ad0-ffd5-47e3-ac7a-81c0407d719e.png">
</p>

## 2. Standard Scaling, Standardization

- 데이터를 표준 정규 분포화 시키는 z-score 정규화
- 변수의 평균을 0으로, 표준 편차를 1로 만들어 주는 표준화 스케일링 기법
- 이상 값 존재에 민감하며 여기서 이상 값이란 Outlier를 의미
- 분류 모델에 보다 더 적합
- 딥러닝에서 말하는 <span style = "color:green">**Standardization(표준화)이 보통 Standard Scaling**</span>을 말한다.

<p align="center">
<img width="200" alt="1" src="https://user-images.githubusercontent.com/111734605/234773396-0cd24834-d2f7-42ca-beb3-437aa14979c7.png">
</p>

### 1) Basis

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.close("all")

# 평균이 5 표준편차가 10인 가우시안 분포
np.random.seed(0)
mu = 5.0
sigma = 10.0

x = np.linspace(-25, 35, 10000)
y = 100*(1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x-mu)**2 / (2 * sigma**2))

# Standardization 시작
## y값 표준화
y_mean = np.mean(y)
y_var = np.var(y)
y_std = np.std(y)

y_standard = (y - y_mean)/y_std

## x값 표준화
x_standard = np.linspace(0,1,10000)

f1 = plt.figure(1)
ax1 = f1.add_subplot(131)
ax1.plot(x, y, 'r-', label = "origin", alpha = 0.8)
ax1.plot(x, y_standard, 'b-', label = "y-only", alpha = 0.8)
ax1.legend(loc="upper right")
ax1.set_title("figure 1: origin VS y-only")
plt.show()

ax2 = f1.add_subplot(132)
ax2.plot(x, y_standard, 'b-',label = "y-only", alpha = 0.8)
ax2.plot(x_standard, y_standard, 'g-', label = "Final", alpha = 0.8)
ax2.legend(loc="upper right")
ax2.set_title("figure 2: y-only VS Final(x & y)")
plt.show()

ax3 = f1.add_subplot(133)
ax3.plot(x_standard, y_standard, 'g-', label = "Final", alpha = 0.8)
ax3.legend(loc="upper right")
ax3.set_title("figure3: Final(x & y)")
plt.show()
```
```python
## y와 y_standard 값 비
             y  y_standard
0     0.044318   -1.161807
1     0.044398   -1.161750
2     0.044478   -1.161693
3     0.044558   -1.161635
4     0.044639   -1.161577
       ...         ...
9995  0.044639   -1.161577
9996  0.044558   -1.161635
9997  0.044478   -1.161693
9998  0.044398   -1.161750
9999  0.044318   -1.161807

[10000 rows x 2 columns]
```

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/234786650-752dea5c-d1e0-407a-92e7-f6c7fc58e99b.png">
</p>

<br/>

### 2) Scikit-learn 라이브러리 사용

- Scikit-learn 패키지를 이용해서 만들 수 있다.
- <span style = "color:red">**Scailing 값을 조정하는 과정이기 때문에 수치형 변수에만 적용**</span>

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.close("all")

# 평균이 5 표준편차가 10인 가우시안 분포
np.random.seed(0)
mu = 5.0
sigma = 10.0

x = np.linspace(-25, 35, 10000)
y = 100*(1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x-mu)**2 / (2 * sigma**2))

## Using sklearn

from sklearn.preprocessing import StandardScaler

X = np.vstack([x,y]).T

# StandardScaler 선언 및 Fitting
sdscaler = StandardScaler()
sdscaler.fit(X)

# 데이터 변환
sdscaled_data = sdscaler.transform(X).T # plot하기 쉽게 Transpost한 것
""
# 데이터 프레임으로 저장
sdscaled_data = pd.DataFrame(sdscaled_data.T)

f2 = plt.figure(2)

ax4 = f2.add_subplot(121)
ax4.plot(x,y,'r-', label = "Before", alpha = 0.7)
ax4.plot(sdscaled_data[0],sdscaled_data[1], "g-", label = "after")
ax4.set_title("figure 4: Orgin VS Standardization")
ax4.legend(loc='upper right')
plt.show()

ax5 = f2.add_subplot(122)
ax5.plot(sdscaled_data[0],sdscaled_data[1], "g-", label = "after")
ax5.legend(loc='upper right')
ax5.set_title("figure 5: Standardization")
plt.show()
```

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/234789610-6998fe11-7a9f-4939-a88b-b90dee20ea16.png">
</p>




