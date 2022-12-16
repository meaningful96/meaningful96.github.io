---
title: Day 11. Joint Optimization으로 문제 풀기

categories: 
  - optproject
tags:
  - [Optimization, Project Levenberg]

toc: true
toc_sticky: true

date: 2022-12-14
last_modified_at: 2022-12-14 
---

왜...왜 Cost가 감소하지 않고 기하급수적으로 증가하는 거지.......

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
# np.random.seed(1)
dfLoad= pd.read_csv("https://raw.githubusercontent.com/meaningful96/OptimizationProject/main/Dataset/mag_array_inliers.txt",
                    sep ="\s+", names = np.arange(1,19) ) # Dataframe으로 받기 위해 열마다 이름 정해줌
## 이 데이터는 mj값임 따라서 구하는 것은 K(Upper Triangle 값), x_hat(자기장 방향), b(bias)
dfLoad = dfLoad.to_numpy().T

plt.close("all")

#----------------------------------------------------------------------------------------------------------------------------#

# vector normalization
def norm(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    x /= r
    y /= r
    z /= r
    return x,y,z

# Step 2) 각각의 Tensor마다 쪼개기 위해 동적 변수만들어 할당함.
for i in range(0,18):
    if i/3 >= 1:
        if i % 3 == 0:
            globals()['x{}'.format(1)] = (dfLoad[i])
        if i % 3 == 1:
            globals()['y{}'.format(1)] = (dfLoad[i])
        if i % 3 == 2:
            globals()['z{}'.format(1)] = (dfLoad[i])
    elif  i == 0:
        x1 = dfLoad[i]
    elif i == 1:
        y1 = dfLoad[i]
    else:
        z1 = dfLoad[i]
b = np.array([0,0,0],dtype = float ).reshape(3,1)

#----------------------------------------------------------------------------------------------------------------------------#

x_n1 = x1 - np.mean(x1)
y_n1 = y1 - np.mean(y1)
z_n1 = z1 - np.mean(z1)
r_n1 = np.sqrt(x_n1**2 + y_n1**2 + z_n1**2)
x_n1 /= r_n1
y_n1 /= r_n1
z_n1 /= r_n1
lamb = 0.1
tol = 1e-4
i = 0
p = 0

#----------------------------------------------------------------------------------------------------------------------------#
#### Joint ####

# K 최적화
# K = np.triu(np.random.randn(3,3))*
K = np.array([[1,2,3],
              [0,4,5],
              [0,0,6]])
Xj = np.column_stack([x_n1, y_n1,z_n1]).T
mj1 = np.column_stack([x1,y1,z1]).T


# X 최적화
j_x = np.zeros((1467,1467))
j=0


# b 최적화
j_b = np.zeros((1467,3))
rj = K.dot(Xj) +b -mj1
rj_origin = np.copy(rj)
count = 0

Cost = []
while True:
    Xj = np.column_stack([x_n1, y_n1,z_n1]).T
    mj1 = np.column_stack([x1,y1,z1]).T
    rj1 = K.dot(Xj) -mj1+b
    rj1_copy = np.copy(rj1)
    j_k = np.kron(Xj, np.eye(3)).T
    
    for iterations in range(489):
        for tt in range(3):
            for kk in range(9):
                j_k[3*iterations+tt][kk] = j_k[3*iterations+tt][kk]/ (np.linalg.norm(Xj.T[iterations]))
    
    
    for i in range (489):
        for k in range (3):
            for n in range(3):
                j_x[3*i+k][3*i+n] = (K/np.linalg.norm(Xj.T[i])@(np.eye(3)-((Xj.T[i].reshape(3,1)@Xj.T[i].reshape(1,3))/(np.linalg.norm(Xj.T[i],2))**2)))[k][n]
                ## float object가 있으면 np.dot으로 연산 불가능
    
    for i in range(489):
        for j in range(3):
            j_b[3*i+j][j] = 1


    Concat_J = np.column_stack([j_k,j_x,j_b])
    middle_1 = -np.linalg.inv(Concat_J.T.dot(Concat_J) + lamb*(1479))
    middle_2 = Concat_J.T.dot(rj.T.reshape(1467,1))
    delta_param = middle_1.dot(middle_2).reshape(3,493)
    delta_K = delta_param.T[:3].T
    delta_Xj = delta_param.T[3:492].T
    delta_b = delta_param.T[492].T
    
    
    K = np.triu(K + delta_K)
    Xj = Xj + delta_Xj
    b = (b.T + delta_b).T
    rj1 = K.dot(Xj) - mj1 + b
    
    # Model quality 분모 : q(델타x) - q(0)
    Model_qulity_mother1 = 2*(rj1.T.reshape(1,1467)).dot(Concat_J.dot(delta_param.T.reshape(1479,1))) 
    Model_qulity_mother2 = (Concat_J.dot(delta_param.T.reshape(1479,1))).T.dot(Concat_J.dot(delta_param.T.reshape(1479,1))) 
    Model_qulity_mother  = Model_qulity_mother1 + Model_qulity_mother2
    
    # Model qulity 분자 : f(x + 델타x) - f(x): 업데이트 후 residual 제곱 - 업데이트 전 residual 제곱
    Model_qulity_son1 = np.sum(rj1[0].T.dot(rj1[0].T) + rj1[1].T.dot(rj1[1].T) + rj1[2].T.dot(rj1[2].T) )
    Model_qulity_son2 = np.sum(rj1_copy[0].T.dot(rj1_copy[0].T) + rj1_copy[1].T.dot(rj1_copy[1].T) + rj1_copy[2].T.dot(rj1_copy[2].T) )
    Model_qulity_son  = Model_qulity_son1 + Model_qulity_son2 
    
    # Model qulity
    Model_qulity = Model_qulity_son/Model_qulity_mother
    
    # Cost visualization을 위해 iteration별 cost 모아두기
    Cost.append(Model_qulity_son1)
    
    ## LM algorithm Tolerance
    
    Cost_before = Model_qulity_son1
    
    rho = Model_qulity
    Cost_prev = np.sum(rj[0].T.dot(rj[0].T) + rj[1].T.dot(rj[1].T) + rj[2].T.dot(rj[2].T) )
    p += 1
    
    if rho > lamb:
        rj = rj1
    
    
    if rho < 0.25:
        lamb = min(4*lamb, 1e+14)
    if rho > 0.75:
        lamb = max(lamb/2, 1e-14)
    
    if p >= 2:
        
        func_tol = abs(Cost[p-1] - Cost[p-2])
    
        
        print(p, func_tol)   
        ## Converge
        if func_tol <1e-3:
            break         

## Visualization

result = (K.dot(Xj) + b)
f1 = plt.figure(1)
ax1 = f1.add_subplot(111, projection = '3d')
ax1.plot(result[0], result[1], result[2], 'r.')
ax1.plot(x1, y1, z1, 'b.')

```
