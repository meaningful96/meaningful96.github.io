---

title: "[Pytorch]ConstantPad2d로 텐서를 상수 값으로 패딩하기"

categories: 
  - Pytorch

  
toc: true
toc_sticky: true

date: 2023-05-07
last_modified_at: 2023-05-07
---

# ConstantPad2d

Pytorch의 ConstantPad2d 연산을 사용하여 <span style="color:red">**텐서를 상수 값으로 패딩**</span>할 수 있다. 이 연산에서 가장 일반적인 문제는 입력 텐서가 3D 또는 4D 형식이어야 하므로 1차원 텐서에는 사용할 수 없다. 
또한 패딩 크기는 튜플 또는 단일 숫자 형태여야 한다.

이러한 문제를 방지하려면 텐서의 각 축에 대한 패딩 크기와 패딩 유형(상수,반사,리플리케이션)을 지정할 수 있는 `torch.nn.functional.pad()` 메서드를 사용할 수 있다. 
또한 torch.nn.ConstantPad2d()또는 nn.ConstantPad3d 메서드를 사용하여 동일한 결과를 얻을 수 있다. 마지막으로 numpy의 np.pad()메서드를 사용하여 텐서를 상수 값으로 패딩할 수도 있다.

<span style="font-size:110%">`class torch.nn.ConstantPad2d(padding, value)`</span>  
입력 탠서 경계를 상수 값으로 채운다.  
`N` 치수 패딩의 경우 `torch.nn.functional.pad()`를 사용해야한다.

- Parameters
  - 패딩(int, tuple) 패딩의 크기이다. `int`인 경우 모든 경계에서 동일한 패딩을 사용한다.
  - tuple인 경우(padding left, padding right, padding top, padding bottom)
  - Input: ($$N,C,H_{in},W_{in}$$)
  - Output: ($$N,C,H_{out},W_{out}$$)   
  - 식1: $$H_{out} = H_{in}$$ + padding_top + padding_bottom
  - 식2: $$W_{out} = W_{in}$$ + padding_left + padding_right

## Example 1

```python
m = nn.ConstantPad2d(2, 3.5)
input = torch.randn(1, 2, 2)
print(input)
#################################################################
tensor([[[ 1.6585,  0.4320],
         [-0.8701, -0.4649]]])
#################################################################


print(m(input))
#################################################################
tensor([[[ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
         [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
         [ 3.5000,  3.5000,  1.6585,  0.4320,  3.5000,  3.5000],
         [ 3.5000,  3.5000, -0.8701, -0.4649,  3.5000,  3.5000],
         [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
         [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000]]])
##################################################################

# 다른면에 다른 패딩 사용
m = nn.ConstantPad2d((3, 0, 2, 1), 3.5)
print(m(input))
##################################################################
tensor([[[ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
         [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
         [ 3.5000,  3.5000,  3.5000,  1.6585,  0.4320],
         [ 3.5000,  3.5000,  3.5000, -0.8701, -0.4649],
         [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000]]])
##################################################################
```

## Example 2
```python
import torch
import torch.nn as nn

# Example 2D tensor (image) with shape (batch_size, channels, height, width)
input_tensor = torch.rand(1, 3, 4, 4)

# Define the amount of padding for each side (left, right, top, bottom)
padding = (1, 2, 1, 2)

# Create the ConstantPad2d module
constant_pad_layer = nn.ConstantPad2d(padding, value=0.0)

# Apply padding to the input tensor
output_tensor = constant_pad_layer(input_tensor)

# Output tensor will have shape (1, 3, 7, 8) after padding
print(output_tensor.shape)
# torch.Size([1, 3, 7, 7])
```
