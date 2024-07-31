---

title: "[Pytorch]다차원 Tensor의 쌓기, cat & stack"

categories: 
  - Pytorch

  
toc: true
toc_sticky: true

date: 2023-03-06
last_modified_at: 2023-03-06
---
# Pytorch에서 다차원 텐서 곱
## 1. torch.cat VS torch.stack
```python
import torch

# (2, 3) 사이즈 2차원 텐서 2개 생성
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

b = torch.tensor([[7, 8, 9],
                  [10, 11, 12]])
```

### 1) cat함수
cat함수는 <span style = "color:red">**원하는 dimension 방향으로 텐서를 나란히**</span> 쌓아준다. 따라서 `torch.cat()`을 선언할 때 괄호 안에 dimenstion을 설정해 줘야 한다. 
예를 들어, 크기가 (x,b,c)인 텐서와 (y,b,c)인 텐서가 있다. 이 때 dim = 1인 b와 dim = 2인 c는 동일해야 한다. 이 경우 dim = 0 방향으로 concatenation을 진행할 수 있다.

torch.cat([(x,b,c), (y,b,c)], dim=0)을 선언해주면 결론적으로 크기가 (x+y, b, c)인 텐서가 출력된다.
```python
torch.cat([a, b], dim = 0)
'''
tensor([[ 1,  2,  3],
        [ 4,  5,  6],
        [ 7,  8,  9],
        [10, 11, 12]]), size = (2+2, 3) = (4, 3)
'''

torch.cat([a, b], dim = 1)
'''
tensor([[ 1,  2,  3,  7,  8,  9],
        [ 4,  5,  6, 10, 11, 12]]), size = (2, 3+3) = (2, 6)
'''
```
<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/235976058-d23f9b75-401c-4547-9e17-6655f3baf957.png">
</p>

<br/>

### 2) Stack함수
cat함수는 원하는 dimension방향으로 텐서를 나란히 쌓아준다. 반면 stack함수는 <span style = "color:red">**텐서를 새로운 차원에 차곡차곡**</span>쌓아준다. 
예를 들어, (x,y,z)사이즈를 가지는 텐서에 dim = 2에 텐서 3개를 쌓는다면 (x,y,3,z)가 된다.(**같은 사이즈의 텐서끼리만 쌓을 수 있다.**)

```python
import torch

a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

b = torch.tensor([[7, 8, 9],
                  [10, 11, 12]])

print(torch.stack([a, b], dim = 0))
'''
tensor([[[ 1,  2,  3],
         [ 4,  5,  6]],

        [[ 7,  8,  9],
         [10, 11, 12]]]), size = (2, 2, 3)'''

print(torch.stack([a, b], dim = 1))
'''
tensor([[[ 1,  2,  3],
         [ 7,  8,  9]],

        [[ 4,  5,  6],
         [10, 11, 12]]]), size = (2, 2, 3)'''

print(torch.stack([a, b], dim = 2))
'''
tensor([[[ 1,  7],
         [ 2,  8],
         [ 3,  9]],

        [[ 4, 10],
         [ 5, 11],
         [ 6, 12]]]), size = (2, 3, 2)'''

```

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/236613569-481af5a6-d401-4d09-8ccc-bcb7485c2bb1.png">
</p>
