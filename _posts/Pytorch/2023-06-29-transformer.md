---
title: "[Pytorch] Transformer 구현하기"

categories: 
  - Pytorch
  
tags:
  - [DL, Pytorch, Transformer]
  
toc: true
toc_sticky: true

date: 2023-05-07
last_modified_at: 2023-05-07
---

논문 리뷰: [\[논문리뷰\]Transformer: Attention Is All You Need]("https://meaningful96.github.io/paperreview/01-Transformer/")

# Why Transformer?

Transformer의 가장 큰 contribution은 <span style = "color:gold">**기존의 RNN 모델이 불가능했던 병렬 처리를 가능**</span>케했다는 것이다.. GPU를 사용함으로써 얻는 가장 큰 이점은 병렬 처리를 한다는 것. RNN(Recurrent Neural Network)은 recursive하기 때문에 병렬 연산이 불가능하다. 
다시 말해 Next layer의 입력이 이전 layer의 hidden state를 받아야 하기 때문이다. Recurrent network를 사용하는 이유는 sequential할 데이터를 처리하기 위함인데, sequential하다는 것은 등장 시점(또는 위치)를 하나의 정보로 취급한다는 것이다. 
따라서 Context vector를 앞에서부터 순차적으로 생성해내고, 그 Context Vector를 이후 시점에서 활용하는 방식으로 구현한다. 즉, 이후 시점의 연산은 앞 시점의 연산에 의존적이다.

따라서 앞 시점의 연산이 끝나지 않을 경우, 그 뒤의 연산을 수행할 수 없다. 이러한 이유로 RNN 계열의 model은 병렬 처리를 수행할 수 없다. 또한 RNN기반의 모델들(RNN, LSTM, Seq2Seq…)의 단점 중 하나는, 하나의 고정된 사이즈의 context vector에 정보를 압축한다는 사실이다. 이럴 경우 필연적으로 입력이 길어지면 정보 손실이 가속화된다. 또한, sequential data의 특성상 위치에 따른 정보가 중요한데, 이러한 위치 정보가 손실되는 Long term dependency가 발생한다. 

<br/>
<br/>

# Model 구조
## 1. Overview

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/DSKUS_Project/assets/111734605/b4052cb7-3c59-427f-b2db-b0cd0f3bc2f3">
</p>

Transformer는 전형적인 Encoder-Decoder 모델이다. 즉, 전체 모델은 Encoder와 Decoder 두 개의 partition으로 나눠진다.  Transformer의 입력은 Sequence 형태로 들어간다. 또한 출력도 마찬가지로 Sequence를 만들어 낸다. 

- Encoder
    - 2개의 Sub layer로 구성되어 있으며, 총 6개의 층으로 구성되어 있다.(N=6)
    - 두 개의 Sub-layer는 **Multi-head attention**과 **position-wise fully connected feed-forward network**이다.
    - Residual connection 존재, Encoder의 최종 Output은 512 차원이다.($$d_{model} = 512$$)
- Decoder
    - 3개의 Sub layer로 구성, 총 6개의 층으로 구성(N=6)
    - 세 개의 Sub layer는 **Masked Multi-head attention**, **Multi-head attention**, **position-wise fully connected feed-forward network**이다.
    - Residual Connection 존재

## 2. Encode-Decoder

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/DSKUS_Project/assets/111734605/b8174595-a0ff-4184-8e65-5d9581609fcb">
</p>

간단하게 정리하면 <span style = "color:gold">**Encoder**</span>의 역할은 <u>문장(Sentence)를 받아와 하나의 Vector를 생성</u>해내는 함수이며 이 과정을 흔히 **Encoding**이라고 한다. 이렇게 Encoding을 통해 생성된 벡터를  Context라고 한다. 
반면 <span style = "color:gold">**Decoder**</span>의 역할은 Encoder와 반대이다. <u>Context와 right shift된 문장을 입력으로 받아 sentence를 생성</u>해낸다. 이 과정을 Decoding이라고 한다.

```python
class Transformer(nn.Module):

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def encode(self, x):
        out_encoder = self.encoder(x)
        return out


    def decode(self, z, c):
        out_decoder = self.decode(z, c)
        return out


    def forward(self, x, z):
        context = self.encode(x)
        y = self.decode(z, context)
        return y
```

## 3. Encoder

### 1) Encoder 구조

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/DSKUS_Project/assets/111734605/4f76e416-6bc6-4f61-ac65-aa9479d67e7f">
</p>

Encoder는 N=6이다. 즉 6개의 층이 쌓여져 있다.  이러한 구조를 통해 할 수 있는 사실은, Input와 Output의 shape이 똑같다는 사실이다. 다시 말해 <u>입출력에 있어서 shape은 완전히 동일한 matrix가되며</u> Encoder block은 shape에 대해 멱등하다 할 수 있다.

- 멱등성(Idempotent): 연산을 여러 번 적용하더라도 결과가 달라지지 않는 성질, 연산을 여러 번 반복하여도 한 번만 수행된 것과 같은 성질

층을 여러 개로 구성하는 이유는 사실 간단하다. Encoder의 입력으로 들어오는 <span style = "color:gold">Input sequence에 대해 더 넓은 관점에서의 Context를 얻기 위함</span>이다. 
더 넓은 관점에서의 context라는 것은 더 추상적인 정보이다. 두 개의 sub graph로 이루어진 Encoder block 하나가 낮은 수준의 context를 생성해내는 반면(하나의 측면에서만 그 문장에 집중하게 됨), 
여러 개의 block을 이용하면 더 많은 context가 쌓이고 쌓여 결론적으로 양질의 context 정보가 저장되게 된다. 

```python
class Encoder(nn.Module):

    def __init__(self, encoder_block, N):  # N: Encoder Block의 개수
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(N):
            self.layers.append(copy.deepcopy(encoder_block))


    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
```

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/DSKUS_Project/assets/111734605/033133ae-424c-424f-8e0b-71aa4f0e792c">
</p>

전통적인 Langauge Model의 경우 입력 시퀀스에 대해 Input Embedding matrix만 만들어 모델의 입력으로 보냈다. 하지만, Transformer의 경우는 입력 시퀀스의 각각의 Token들에 대해 위치 정보까지 주기위해 Positional Encoding도 이용한다. 
단, 이 **Positional Encoding**은 <u>각 단어의 상대적인 위치 정보를 네트워크</u>에 입력하는 것이며 sin 또는 cos함수로 이루어져있다.

<br/>

### 2) Sub-Layer1: Multi-head Attention

Encoder block의 첫 번째 Sub layer에 해당하는 것은 Multi-head attention이다. 


<br/>

### 3) Sub-Layer2: Position-wise Feed Forward Neural Network(FFNN)

