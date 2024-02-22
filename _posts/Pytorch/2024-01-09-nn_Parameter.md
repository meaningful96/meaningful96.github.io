---

title: "[Pytorch]nn.Parameter로 학습 가능한 변수 만들기"

categories: 
  - Pytorch

toc: true
toc_sticky: true

date: 2024-01-09
last_modified_at: 2024-01-09
---

# nn.Parameter 란?

- `nn.Module`을 상속받는 클래스테서 사용할 수 있으며, 이름 그대로 Parameter를 정의하는 것이다.

> Parameters are Tensor subclasses, that have a very special property when used with Module s - when they’re assigned as Module attributes they are automatically added to
> the list of its parameters, and will appear e.g. in `parameters()` iterator. Assigning a Tensor doesn’t have such effect. This is because one might want to cache some temporary
> state, like last hidden state of the RNN, in the model. If there was no such class as Parameter, these temporaries would get registered too.

- 입력으로 Tensor와 Gradient를 할지 말지 결정하는 Requires Grad를 Bool값으로 받는다.

좀 더 직관적으로 설명하면, Linear Transformation을 할 때 그 식은 Y = WX + b인데, 여기서 학습하는 파라미터는 W, b이다.
- `nn.Module`안에 미리 만들어진 tensor를 보관 가능하다.
- `Tensor`를 사용하지 않고, `nn.Parameter`를 쓰는 이유는 다음과 같다.
  - `Tensor`
    - Gradient 계산 X
    - 값 업데이트 X
    - 모델 저장시 값 저장 X
  - `Parameter`
    - Gradient 계산 O
    - 값 업데이트 O
    - 모델 저장시 값 저장
  - `Buffer`
    - Gradient 계산 X
    - 값 업데이트 X
    - 모델 저장시 값 저장 O


## Example 1)
```py
# SimKGC
inv_t = torch.nn.Parameter(torch.tensor( 1 / args.t).log(), requires_grad = args.finetune_t)
```

## Example 2)
```py
import torch
from torch import nn
from torch.nn.parameter import Parameter

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = Parameter(torch.ones(out_features, in_features))
        self.b = Parameter(torch.ones(out_features))

    def forward(self, x):
        output = torch.addmm(self.b, x, self.W.T)

        return output

x = torch.Tensor([[1, 2],
                  [3, 4]])

linear = Linear(2, 3)
output = linear(x)

# torch.Tensor([[4, 4, 4],
#              [8, 8, 8]]):
```

물론 `Tensor`도 `my_param = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)` 이런식으로 `requires_grad = True` 로 주어 학습 가능한 파라미터로 만들 수 있다.
