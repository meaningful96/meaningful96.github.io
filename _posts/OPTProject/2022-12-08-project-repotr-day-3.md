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
# Deep Learning for AI engineer
"""
Created on Youminkk


Nil sine magno vitae labore dedit mortalibus
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
import pandas as pd
import numpy as np

# Step 1) Data Load

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
i = 0
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
    i += 1
    print(i)
    if Model_quality_1 <= tol:
        break
#----------------------------------------------------------------------------------------------------------------------------#

result = K@Xj
print(result[1])
f3 = plt.figure(3)
ax1 = f3.add_subplot(121,projection = '3d')
ax1.plot(result[0],result[1],result[2],'g.')
ax2 = f3.add_subplot(122, projection = '3d')
ax2.plot(x1,y1,z1,'b.')  

#----------------------------------------------------------------------------------------------------------------------------#
def costfunction1(self, K, Xj, b, mj):
    K = np.triu(np.random.randn(3,3))
    angle_1 = [] # 세타
    angle_2 = [] # 파이
    for i in range(0,489):
        angle_1.append(np.random.rand()*2*np.pi)
        angle_2.append(np.random.rand()*2*np.pi)
    x_n1 = np.sin(angle_1)*np.cos(angle_2)
    y_n1 = np.sin(angle_1)*np.sin(angle_2)
    z_n1 = np.cos(angle_1)
    Xj = np.column_stack([x_n1, y_n1,z_n1]).T
    mj1 = np.column_stack([x1,y1,z1]).T
    
def cost_grad1(self, K, Xj, b, mj):
    Jk1 = np.kron(Xj, np.eye(3)).T
    
```
<p align = "center">
<img width="510" alt="image" src="https://user-images.githubusercontent.com/111734605/206401338-4beb869c-b568-4068-b74d-ca2c9c244fab.png">
</p>

K를 먼저 찾기위해 Bias와 x는 고정된 값으로 진행하였다. 이때, x는 magnetometer의 방향 벡터이고, 우리가 가지고 있는 정답 레이블이 구형이므로 Sphrecal coordinate로 만들었다.
그리고 K의 초기값을 가우시안 분포를 따르는 랜덤 변수로 주었다. 가우시안 분포로 준 이유는, 중심 극한 정리에 의한것이다.

<p align = "center">
<img width="1062" alt="image" src="https://user-images.githubusercontent.com/111734605/206402021-09d93f66-ccb5-49b7-8c03-f723d05ed668.png">
</p>

결과가 타원형으로 나왔다.


**모델 퀄리티 다시 구해!!
