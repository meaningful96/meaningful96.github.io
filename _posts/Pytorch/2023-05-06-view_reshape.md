---

title: "[Pytorch]차원 재구성 - view(), reshape(), transpose(), permute()"

categories: 
  - Pytorch

  
toc: true
toc_sticky: true

date: 2023-05-06
last_modified_at: 2023-05-06
---

# 차원 재구성
## 1. view(), reshape()
Pytorch의 reshape()과 view()는 둘 다 텐서의 모양을 변경하는 데에 사용될 수 있다. 그러나 둘 사이에는 약간의 차이가 존재한다.

- view(): 기존의 데이터와 같은 메모리 공간을 공유하며 stride 크기만 변경하여 보여주기만 다르게 한다. 그래서 **contiguous해야만 동작**하며, 아닌 경우 에러가 발생한다.
- reshape(): 가능하면 input의 view를 반환하고, 안되면 **contiguous**한 텐서로 copy하고 view를 반환한다.

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/236616856-dba8f755-29f5-47d3-9388-6cd356b56ddb.png">
</p>

```python
import torch
import numpy as np

t = np.zeros((4,4,3)) # 0으로 채워진 4x4x3 numpy array 생성
ft = torch.FloatTensor(t) # 텐서로 변환
print(ft.shape)

a = ft.view([-1,3]) # 원래 4x4x3=48이고, [-1,3] 모양으로 변환: ?x3 = 48이다. 따라서 ? = 16
b = ft.view([2,-1]) # 원래 4x4x3=48이고, [2,-1] 모양으로 변환: 2x? = 48이다. 따라서 ? = 24
print(a.shape) # torch.Size([16, 3])
print(b.shape) # torch.Size([2, 24])
#------------------------------------------------------------------------------------------#
import torch
import numpy as np

r = np.zeros((5, 5, 10))
fr = torch.FloatTensor(r)
print(fr.shape)

c = fr.reshape(10, 5, 5)
d = fr.reshape(1, -1)
print(c.shape) #torch.Size([10, 5, 5])
print(d.shape) #torch.Size([1, 250])
```

## 2. transpose(), permute()
permute(), transpose()는 유사한 방식으로 작동한다.
- transpose()는 딱 두 개의 차원을 맞교환할 수 있다. 
- permute()는 모든 차원들을 맞교환할 수 있다.

```python
x = torch.rand(16,32,3)
y = x.transpose(0,2) # 0과 2번째 차원을 교환
z = x.permute(1,2,0) # 0번째 차원 자리에 2번째 차원, 1번째 차원은 그대로, 2번째 차원자리는 0번째 차원이 들어옴
print(x.shape) # torch.Size([16,32,3])
print(x.shape) # torch.Size([3,32,16])
print(x.shape) # torch.Size([32,3,16])
```

## 3. Contiguous의 의미
```
A contiguous array is just an array stored in an unbroken block of memory
: to access the next value in the array, we just move to the next memory address.
```
즉, 연속 어레이(Contiguous array)란 메모리 안에 끊어지지 않은 block안에 저장되어 있는 어레이이다. 따라서 어레이의 다음 값에 접근하기 위해, 메모리의 다음 주소로 이동하면 된다. 
예를 들어, 2D Array가 있다고 하자
```python
import numpy as np
arr = np.arange(12).reshape(3,4)
```

![image](https://user-images.githubusercontent.com/111734605/236617998-64e87d6e-8e42-4dc5-9018-1206ec0639dc.png)

이렇게 모양이 잡힌 행렬은 컴퓨터 메모리 안에서는 다음과 같아 저장된다.

![image](https://user-images.githubusercontent.com/111734605/236618037-e3673b6f-a65f-49d2-be10-cbcfcd3d7426.png)

이렇게 **행(row)** 방향으로 저장된 형태를 <span style = "color:red">**C contiguous**</span> array라고 한다. 반면, Transpos를 할 수도 있다.
```python
arr = arr.T
```
이럴 경우 C contiguity는 상실된다. 왜냐하면 행 방향으로의 접근 가능한 주소가 남아있지 않기 때문이다.

![image](https://user-images.githubusercontent.com/111734605/236618166-d29803f5-97fc-4e9e-8339-bed83f94176e.png)

이런 경우를 <span style = "color:red">**Fortran contiguous**</span>라고 하며, 메모리가 **열(Column)** 방향으로 저장된 것을 말한다.

이처럼, <span style=  "color:red">**Contiguous라는 것은 tensor에서 바로 옆에 있는 요소가 실제로 메모리상에서 서로 인접해있느냐**</span>를 의미한다.



