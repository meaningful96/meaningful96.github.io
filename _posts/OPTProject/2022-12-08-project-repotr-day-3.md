---
title: Day 3. K matrix Training

categories: 
  - optproject
tags:
  - [Optimization, Project Levenberg]

toc: true
toc_sticky: true

date: 2022-12-08
last_modified_at: 2022-12-08 
---

```python
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
import pandas as pd
import numpy as np

# Step 1) Data Load
np.random.seed(1)
dfLoad= pd.read_csv("https://raw.githubusercontent.com/meaningful96/OptimizationProject/main/Dataset/mag_array_inliers.txt",
                    sep ="\s+", names = np.arange(1,19) ) # Dataframe으로 받기 위해 열마다 이름 정해줌
## 이 데이터는 mj값임 따라서 구하는 것은 K(Upper Triangle 값), x_hat(자기장 방향), b(bias)
dfLoad = dfLoad.to_numpy().T
#----------------------------------------------------------------------------------------------------------------------------#

# Step 2) 각각의 Tensor마다 쪼개기 위해 동적 변수만들어 할당함.
for i in range(0,18):
    if i/3 >= 1: 
        if i % 3 == 0:
            globals()['x{}'.format(i//3 + 1)] = (dfLoad[i])
        if i % 3 == 1:
            globals()['y{}'.format(i//3 + 1)] = (dfLoad[i])
        if i % 3 == 2:
            globals()['z{}'.format(i//3 + 1)] = (dfLoad[i])

    elif  i == 0:
        x1 = dfLoad[i]
    elif i == 1:        
        y1 = dfLoad[i]
    else:
        z1 = dfLoad[i]
#----------------------------------------------------------------------------------------------------------------------------#
angle_1 = [] # 세타
angle_2 = [] # 파이
for i in range(0,489):
    angle_1.append(np.random.rand()*2*np.pi)
    angle_2.append(np.random.rand()*2*np.pi)
x_n1 = np.sin(angle_1)*np.cos(angle_2)
y_n1 = np.sin(angle_1)*np.sin(angle_2)
z_n1 = np.cos(angle_1)
lamb = 0.1
tol = 1e-6
#----------------------------------------------------------------------------------------------------------------------------#

while True:
    Xj = np.column_stack([x_n1, y_n1,z_n1]).T
    mj1 = np.column_stack([x1,y1,z1]).T
    K = np.triu(np.random.randn(3,3))
    rj1 = K.dot(Xj) -mj1
    Jk1 = np.kron(Xj, np.eye(3)).T
    Hessian1 = 2*Jk1.T@Jk1
    middle1 = np.linalg.inv(Hessian1 + 0.1*np.eye(9))
    Delta_K_tmp = middle1@Jk1.T@rj1.T.reshape(1467,1)
    Delta_K = Delta_K_tmp.reshape(3,3)
    K -= np.triu(Delta_K)
    Model_quality_1 = 2*(rj1.T.reshape(1,1467)).dot(Jk1).dot(Delta_K.T.reshape(9,1))
    if Model_quality_1 <= tol:
        break
#----------------------------------------------------------------------------------------------------------------------------#   
```
