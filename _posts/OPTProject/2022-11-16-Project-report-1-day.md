---
title: Day 1. 데이터 시각화 해보기

categories:
  - optproject
tags:
  - [Optimization,Project]

toc: true
toc_sticky: true

date: 2022-11-23
last_modified_at: 2022-11-23
---

## 1. Data Load with Python

```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1) Data Load

dfLoad= pd.read_csv("https://raw.githubusercontent.com/meaningful96/OptimizationProject/main/Dataset/mag_array_inliers.txt",
                    sep ="\s+", names = np.arange(1,19) ) # Dataframe으로 받기 위해 열마다 이름 정해줌

#----------------------------------------------------------------------------------------------------------------------------#

# Step 2) 각각의 Tensor마다 쪼개기 위해 동적 변수만들어 할당함.
for i in range(1,19):
    if i/3 > 1: 
        if i % 3 == 1:
            globals()['x{}'.format(i//3 + 1)] = (dfLoad[i])
        if i % 3 == 2:
            globals()['y{}'.format(i//3 + 1)] = (dfLoad[i])
        if i % 3 == 0:
            globals()['z{}'.format(i//3)] = (dfLoad[i])
    elif  i == 1:
        x1 = dfLoad[i]
    elif i == 2:        
        y1 = dfLoad[i]
    else:
        z1 = dfLoad[i]
#----------------------------------------------------------------------------------------------------------------------------#
 
# Step 3) subplot 만들어서 각 tensor별로 data plot
f1 = plt.figure(1)
ax1 = f1.add_subplot(231,projection = '3d')
ax1.plot(x1,y1,z1,'b.')
ax2 = f1.add_subplot(232,projection = '3d')
ax2.plot(x2,y2,z2,'g.')
ax3 = f1.add_subplot(233,projection = '3d')
ax3.plot(x3,y3,z3,'r.')
ax4 = f1.add_subplot(234,projection = '3d')
ax4.plot(x4,y4,z4,'c.')
ax5 = f1.add_subplot(235,projection = '3d')
ax5.plot(x5,y5,z5,'m.')
ax6 = f1.add_subplot(236,projection = '3d')
ax6.plot(x6,y6,z6,'y.')    

#----------------------------------------------------------------------------------------------------------------------------#

# Step 4) 한 화면에 겹쳐서 출력
f2 = plt.figure(2)
f2_ax2 = plt.axes(projection = '3d')
f2_ax2.plot(x1,y1,z1,'r.')
f2_ax2.plot(x2,y2,z2,'b.')
f2_ax2.plot(x3,y3,z3,'g.')
f2_ax2.plot(x4,y4,z4,'c.')
f2_ax2.plot(x5,y5,z5,'m.')
f2_ax2.plot(x6,y6,z6,'y.')

```

## 2. Data 시각화

### 1) Subplot

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/203575579-11b56fa0-d7a5-41ef-b7e3-0be82e3676db.png">
</p>

### 2) Mapping at once

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/203575585-3828b927-b184-4f20-9bbb-b8f4418dc650.png">
</p>
