---

title: "[Pytorch]TuckER decomposition"

categories: 
  - Pytorch

  
toc: true
toc_sticky: true

date: 2023-09-04
last_modified_at: 2023-09-04
---

# TuckER
```python
"""
Created on meaningful96

DL Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def L2_norm(matrix):
    return F.normalize(matrix.float(), p=2, dim=1)



tucker_matrix = torch.randn(5,5,5)
a = torch.randn(7,5)
b = torch.randn(7,5)
c = torch.randn(7,5)
a, b, c = L2_norm(a), L2_norm(b), L2_norm(c)

x1 = torch.tensordot(tucker_matrix, a, dims = ([0],[1]))
x1 = x1.permute(2,1,0)
x2 = torch.tensordot(x1, b, dims = ([1],[1]))
x2 = x2.permute(0,2,1)
x3 = torch.tensordot(x2, c, dims = ([2],[1]))

X = x3
print(X)

pred = torch.sigmoid(X)

print(pred)

#%%
import torch
import torch.linalg as LA

def tucker_decomposition(tensor, ranks):
    # Mode-1 unfolding
    unfold_1 = tensor.unfold(0, tensor.shape[0], tensor.shape[0]).permute(2, 0, 1).reshape(tensor.shape[0], -1)
    _, s1, v1 = LA.svd(unfold_1)

    # Mode-2 unfolding
    unfold_2 = tensor.unfold(1, tensor.shape[1], tensor.shape[1]).permute(2, 0, 1).reshape(tensor.shape[1], -1)
    _, s2, v2 = LA.svd(unfold_2)

    # Mode-3 unfolding
    unfold_3 = tensor.unfold(2, tensor.shape[2], tensor.shape[2]).reshape(tensor.shape[2], -1)
    _, s3, v3 = LA.svd(unfold_3)

    # Reduce the dimensions of the factor matrices based on given ranks
    A = v1[:, :ranks[0]]
    B = v2[:, :ranks[1]]
    C = v3[:, :ranks[2]]

    core = torch.einsum('ijk,il,jm,kn->lmn', tensor, A, B, C)

    return core, [A, B, C]

# Define a tensor
tensor = torch.randn(10, 10, 10)

# Decompose
core, factors = tucker_decomposition(tensor, [5, 5, 5])
```
