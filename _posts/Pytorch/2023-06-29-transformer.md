---
title: "[Pytorch] Traansformer 구현하기"

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

트랜스포머의 가장 큰 contribution은 <span style = "color:gold">**기존의 RNN 모델이 불가능했던 병렬 처리를 가능**</span>케했다는 것이다.. GPU를 사용함으로써 얻는 가장 큰 이점은 병렬 처리를 한다는 것. RNN(Recurrent Neural Network)은 recursive하기 때문에 병렬 연산이 불가능하다. 
다시 말해 Next layer의 입력이 이전 layer의 hidden state를 받아야 하기 때문이다. Recurrent network를 사용하는 이유는 sequential할 데이터를 처리하기 위함인데, sequential하다는 것은 등장 시점(또는 위치)를 하나의 정보로 취급한다는 것이다. 
따라서 Context vector를 앞에서부터 순차적으로 생성해내고, 그 Context Vector를 이후 시점에서 활용하는 방식으로 구현한다. 즉, 이후 시점의 연산은 앞 시점의 연산에 의존적이다.

따라서 앞 시점의 연산이 끝나지 않을 경우, 그 뒤의 연산을 수행할 수 없다. 이러한 이유로 RNN 계열의 model은 병렬 처리를 수행할 수 없다. 또한 RNN기반의 모델들(RNN, LSTM, Seq2Seq…)의 단점 중 하나는, 하나의 고정된 사이즈의 context vector에 정보를 압축한다는 사실이다. 이럴 경우 필연적으로 입력이 길어지면 정보 손실이 가속화된다. 또한, sequential data의 특성상 위치에 따른 정보가 중요한데, 이러한 위치 정보가 손실되는 Long term dependency가 발생한다. 

<br/>
<br/>

# Model 구조
## 1. Overview

<p align="center">
<img width="600" alt="1" src="https://github.com/meaningful96/DSKUS_Project/assets/111734605/b4052cb7-3c59-427f-b2db-b0cd0f3bc2f3">
</p>

트랜스포머는 전형적인 Encoder-Decoder 모델이다. 즉, 전체 모델은 Encoder와 Decoder 두 개의 partition으로 나눠진다.  트랜스포머의 입력은 Sequence 형태로 들어간다. 또한 출력도 마찬가지로 Sequence를 만들어 낸다. 

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

간단하게 정리하면 <span style = "color:gold">**Encoder**</span>의 역할은 <u>문장(Sentence)를 받아와 하나의 벡터터를 생성</u>해내는 함수이며 이 과정을 흔히 **Encoding**이라고 한다. 이렇게 Encoding을 통해 생성된 벡터를  Context라고 한다. 
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

전통적인 Langauge Model의 경우 입력 시퀀스에 대해 Input Embedding matrix만 만들어 모델의 입력으로 보냈다. 하지만, 트랜스포머의 경우는 입력 시퀀스의 각각의 토큰들에 대해 위치 정보까지 주기위해 Positional Encoding도 이용한다. 

<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/DSKUS_Project/assets/111734605/1742a87c-f7dd-4e0a-ae81-dad8981807bb">
</p>

단, 이 **Positional Encoding**은 <u>각 단어의 상대적인 위치 정보를 네트워크</u>에 입력하는 것이며 sin 또는 cos함수로 이루어져있다. 

<br/>

### 2) Sub-Layer1: Multi-head Attention

#### Attention의 이해
Encoder block의 첫 번째 Sub layer에 해당하는 것은 Multi-head attention이다. Attention mechanism을 이루는 방법에는 여러 가지가 있지만, 트랜스포머의 경우는 <span style = "color:gold"><b>Scaled Dot-Product Attention</b></span>을 병렬적으로 여러 번 수행한다. 트랜스포머이후 Scaled Dot-Product attention 방식을 통상적으로 attention이라고 사용한다.

Attention이 그럼 무슨 역할을 하는 건지를 이해하는 것이 중요하다. Attention Mechanism을 사용하는 목적은 생각보다 간단하다. <span style = "color:gold"><b>토큰들이 서로서로 얼마나 큰 영향력을 가졌는지를 구하는 것</b></span>이다.


- **Self-Attention** = 한 문장 내에서 토큰들의 attention을 구함.
- **Cross-Attention** = 서로 다른 문장에서 토큰들의 attention을 구함. 

#### RNN vs Self-Attention
RNN 계열의 모델들을 다시 생각해보면, 이전 시점까지 나온 토큰들의 hidden state 내부에 이전 정보들을 저장한다. 하지만 순차적으로 입력이 들어가기 때문에 모든 토큰을 동시에 처리하는 것이 불가능하다. 다시 말해 $$h_i$$를 구하기 위해서는 $$h_0, h_1, h_2, \cdots, h_{i_1}$$까지 모두 순서대로 거쳐야 구할 수 있다는 것이다. 이러한 이유로 Input sequence의 길이가 길어지면, 오래된 시점의 토큰들의 의미는 점점 더 퇴색되어 제대로 반영이 되지 못하는 **Long term dependency**가 발생하는 것이다.

반면 Self-Attention의 경우는 한 문장내에서 기준이 되는 토큰을 바꿔가며 모든 토큰에 대한 attention을 <u>행렬 곱을 통해 한 번에 계산</u>한다. 이 행렬 곱 계산이 가능하기에 병럴 처리가 손쉽게 가능하다. 즉, 문장에 n개의 토큰이 있다면 $$n \times n$$ 번 연산을 수행해 모든 토큰들 사이의 연관된 정도를 한 번에 구해낸다. 중간 과정 없이 direct하게 한 번에 구하므로 토큰간의 의미가 퇴색되지 않는다.

#### Attention 구하기(Query, Key, Value)
Attention을 계산할 때는 **Query, Key, Value** 세 가지 벡터가 사용되며 각각의 정의는 다음과 같다.

- Query(쿼리): 현재 시점의 Token, 비교의 주체
- Key(키): 비교하려는 대상, 객체이다. 입력 시퀀스의 모든 토큰
- Value(벨류): 입력 시퀀스의 각 토큰과 관련된 실제 정보를 수치로 나타낸 실제 값, Key와 동일한 토큰을 지칭

예를 들어 I am a teacher라는 문장이 있다. 이 문장을 가지고 Attention을 구한다고 하면 다음과 같이 정리할 수 있다. 

- if Query = 'I'
  - Key = 'I', 'am', 'a', 'teacher'
  - Query-key의 사이의 연관성을 구한다 = Attention

그러면 Query, Key, Value 이 세 벡터가 어떤식으로 추출되는지도 알아야한다. 이 벡터들은 입력으로 들어오는 Token embedding을 <span style = "color:gold">**Fully Connected Layer(FC)**</span>에 넣어 생성된다. 세 벡터를 생성해내는 FC layer는 모두 다르기 때문에 self-attention에서는 <u>Query, Key, Value를 위한 3개의 서로 다른 FC layer가 존재</u>한다. 각각이 개별적으로 구해지는 것과는 달리
**세 벡터의 Shape은 동일**하다. (<span style = "font-size:110%">$$d_{key} = d_{value} = d_{query} = d_k$$</span>)

#### Scaled Dot-Product Attention

<p align="center">
<img width="300" alt="1" src="https://github.com/meaningful96/DSKUS_Project/assets/111734605/33421eba-f11e-4de4-95a4-6d94793e259a">
</p>

Scaled Dot-Product Attention의 메커니즘은 위의 그림과 같다. 먼저 Query와 Key 벡터의 행렬곱을 수행하고 Scaling을 한 후 Softmax를 통해 확률값으로 만들어 버린다. 이후 이 값을 Value와 곱하면된다.

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/DSKUS_Project/assets/111734605/d08aabf7-c352-4894-bf06-aca3e15c2719">
</p>

좀 더 계산과정을 명확하게 보기위해 한 단어와 단어 사이의 attention을 구하는 과정을 집중해본다. 위에처럼 $$d_k = 3$$인 경우라고 가정하고 FC layer에의해 이미 $$Q, K, V$$가 모두 구해졌다고 가정하고 1번 그림처럼 나타낼 수 있다. 위의 메커니즘과 같이 Query와 Key의 행렬곱을 수행해야 한다. 이 때 Scailing을 포함한 이 행렬곱의 결과를 <span style = "color:gold">**Attention Score**</span>라고 한다.

Scailing을 하는 이유는 과연 무엇일까? 그 이유는 사실 간단하다. 행렬 곱의 결과인 attention energy값이 너무 큰 경우 Vanishing Graident현상이 발생할 수 있기 때문이다. 하지만 Scailing을 단순한 상수이므로 행렬곱 연산 결과로 나온 Score의 차원에 영향을 미치지 않는다. 앞서 본 경우는 1:1 관계에서의 attention을 구한 것이다. Self-Attention은 1:N의 관계에서 진행되므로 이를 확장하면 다음과 같다.

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/DSKUS_Project/assets/111734605/b2f39097-4112-4d6c-a40a-9ef06ecc486d">
</p>

먼저 $$Q, K, V$$를 다시 정의해준다. Query의 경우는 비교의 주체이므로 하나의 토큰을 의미하기에 그대로 둔다. 반면 Key와 Value는 비교를 하려는 대상이므로 입력 시퀀스내의 모든 토큰들에 대한 정보를 가지고 있어야 하므로, 각가그이 토큰 임베딩을 Concatenation한 형태로 출력된다. 따라서 $$K, V$$는 행렬로 표현되고 그 크기는 $$n \times d_k$$이다. 이를 통해 마찬가지로 행렬곱을 진행하면 <u>Attention-Score는 전체 토큰 수만큼의 score가 concatenation된 벡터로 출력</u>된다. 다시 말해 Query의 토큰과 입력 시퀀스 내의 모든 토큰들과의 attention score를 각각 계산한 뒤 concatenation한 것이다.

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/DSKUS_Project/assets/111734605/1ca18d2f-5191-4119-bfd1-ba3238caf92e">
</p>

행렬곱 결과 구해진 Attention Score를 이용해 최종적으로 일종의 Weight를 만들어야 한다. 이 때, <span style = "color:gold">Weight로 변환하는 가장 좋은 방법은 그 값을 <b>확률(Probability)</b>로 만드는 것</span>이다. 확률로 만들기위해 논문에서는 **SoftMax function**을 이용했다. 이렇게 Softmax를 통해 구해진 <span style = "color:gold">**Attention Weight(Probability)**</span>을 토큰들의 실질적 의미를 포함한 정보인 Value와 행렬곱을 해준다.(참고로 Attention Weight의 합은 확률이므로 1이다.)

이로써 최종적으로 Query에 대한 <span style = "color:gold">**Attention Value**</span>가 나오게 된다. 여기서 알아야 할 중요한 포인트는 연산의 최종 결과인 <u>Query의 Attention Value의 크기(차원)이 Input Query 임베딩과 동일</u>하다는 것이다. Attention Mechanism입장에서 입력은 Query, Key, Value로 세 가지이지만, 의미상으로 Semantic한 측면에서 고려하면 출력이 Query의 attention이므로 입력도 Query로 생각할 수 있다.

<p align="center">
<img width="500" alt="1" src="https://github.com/meaningful96/DSKUS_Project/assets/111734605/94b93dcd-ed63-4b2b-917f-d5c2954c4aa0">
</p>

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/DSKUS_Project/assets/111734605/9fcccd13-ebad-407c-8641-9fc95b6757f4">
</p>

앞서 구한 과정은 모두 하나의 Query에 대해서 1:1, 1:N 관계로 확장하며 구한 것이다. 또한 한 번의 행렬 연산으로 구해진 것이다. 하지만 실제로 Query역시 모든 토큰들이 돌아가면서 각각의 토큰들에 대한 Query attention value를 구해야 하므로 Concatenation을 이용해 **행렬**로 확장해야한다. 이를 그림으로 표현하면 위와 같다. 

<br/>

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/DSKUS_Project/assets/111734605/881383b6-b941-4d58-8b7e-7334bec0772c">
</p>

<br/>

행렬로 확장해 Attention을 진행하면 위와 같이 된다. 최종적으로 Query에 대한 Attention value역시 행렬로 출력된다. 다시 한 번 강조하면 **Self-Attention은 Input Query(Q)의 Shape에 대해 멱등(Idemopotent)**하다.

<p align="center">
<img width="650" alt="1" src="https://github.com/meaningful96/DSKUS_Project/assets/111734605/bb93c589-6630-4fb3-8d84-dec0c0c1f828">
</p>
<center><span style = "font-size:80%">멱등(Idemopotent)성을 설명하는 그림</span></center>

<br/>
<br/>

Self-Attention의 과정을 수식으로서 정리하면 아래와 같이 정리할 수 있다.

<p align="center">
<img width="600" alt="1" src="https://github.com/meaningful96/DSKUS_Project/assets/111734605/80a2f928-b4bb-48da-bd70-11332f0142ea">
</p>

#### Masked Self-Attention(Masking)
Scaled Dot-Product Attention을 설명하면서 한 부분을 설명하지 않았다. 바로 Masking이다. Masking을 하는 <span style = "color:gold">**이유는 특정 값들을 가려서 실제 연산에 방해가 되지 않도록 하기 위함**</span>이다. Masking에는 크게 두 가지 방법이 존재한다. Padding Masking(패딩 마스킹)과 Look-ahead Masking(룩 어헤드 마스킹)이다. 

<span style = "font-size:110%">패딩(Padding)</span>  
mini-batch마다 입력되는 문장은 모두 다르다. 이 말을 다시 해석하면, 입력되는 모든 문장의 길이는 다르다. 그러면 모델은 이 <u>다른 문장 길이를 조율해주기 위해 모든 문장의 길이를 동일하게 해주는 전처리 과정이 필요</u>하다. 짧은 문장과 긴 문장이 섞인 경우, 짧은 문장을 기준으로 연산을 해버리면 긴 문장에서는 일부 손실이 발생한다. 반대로, 긴 문장을 기준으로 연산을 해버리면 짧은 문장에서 Self-Attention을 할 경우 연산에 오류가 발생한다.(토큰 개수 부족)

따라서 짧은 문장의 경우 0을 채워서 문장의 길이를 맞춰줘야 한다. 중요한 것은 0을 채워주지만 그 zero Token들의 경우 실제로 의미를 가지지 않는다. 따라서 <span style = "color:gold">**실제 attention 연산시에도 제외할 필요**</span>가 있다. 숫자 0의 위치를 체크해주는 것이 바로 패딩 마스킹(Padding Masking)이다.

<span style = "font-size:110%">패딩 마스킹(Padding Masking)</span> 

Scaled Dot-Product Attention을 구현할 때 어텐션 함수에서 mask를 인자로 받아 이 값에다 아주 작은 음수값을 곱해 Attention Score행렬에 더해준다.

```python
def scaled_dot_product_attention(query, key, value, mask):
... 중략 ...
    logits += (mask * -1e9) # 어텐션 스코어 행렬인 logits에 mask*-1e9 값을 더해주고 있다.
... 중략 ...
```

이건 Input Sentence에 \[PAD\] 토큰이 있을 경우 어텐션을 제외하기 위한 연산이다. \[PAD\]가 포함된 입력 문장의 Self-Attention을 구하는 과정은 다음과 같다.

<p align="center">
<img width="600" alt="1" src="https://github.com/meaningful96/DSKUS_Project/assets/111734605/8116f845-e6d6-4da3-a2e2-1798414e215d">
</p>

/[PAD\]는 실제로는 아무 의미가 없는 단어이다. 그래서 트랜스포머에선 key의 경우 \[PAD\] 토큰이 존재할 경우 유사도를 구하지 않도록 마스킹(Masking)을 해준다. Attention에서 제외하기 위해 값을 가리는 행위가 마스킹이다. <u>Attention Score 행렬에서 행에 해당하는 문장은 Query이고 열에 해당하는 문장은 Key</u>이며 key에 \[PAD\]가 있는 열 전체를 마스킹한다.

마스킹을 하는 방법은 사실 간단한데, 매우 작은 음수값을 넣어주면된다. 이 Attention Score가 SoftMax함수를 지나 Value 행렬과 곱해지는데, SoftMax 통과시 PAD부분이 0에 매우 가까운 값이 되어 유사도를 구할 때 반영이 되지 않는다.

<span style = "font-size:110%">룩어헤드 마스킹(Look-Ahead Masking</span> 

RNN이나 트랜스포머, GPT는 문장을 입력받을 때 단방향으로 학습한다. 즉, 하나의 방향으로만 문장을 읽고 트랜스포머는 RNN가 달리 한 step에 모든 문장을 나타내는 행렬이 들어가기 때문에 추가적인 마스킹이 필요하다.

Masked Self-Attention을 하는 이유는, 학습과 추론과정에 정보가 새는(Information Leakage)것을 방지하기 위함이다. 트랜스포머에서 마스킹된 Self Attention은 모델이 <u>한 번에 하나씩 출력 토큰을 생성할 수 있도록 하면서 모델이 미래의 토큰이 아닌 이전에 생성된 토큰에만 주의를 기울이도록 하기 위함</u>이다. 이를 더 자세히 말하자면, Encoder-Decoder로 이루어진 모델들의 경우 입력을 순차적으로 전달받기 때문에 t + 1 시점의 예측을 위해 사용할 수 있는 데이터가 t 시점까지로 한정된다. 하지만 트랜스포머의 현재 시점의 출력값을 만들어 내는데 미래 시점의 입력값까지 사용할 수 있게되는 문제가 발생하기 때문이다.

이 이유는 트랜스포머의 초기값, 1 Epoch을 생각해보면 이해하기 쉽다. 처음에 입력으로 들어가 인코더를 거친 값이 디코더로 들어가는데, 디코더로 들어가는 또 다른 입력은 이전 Epoch에서의 출력 임베딩값이다. 하지만 1 Epoch에서는 과거의 값은 존재하지 않아 초기에 설정해준 값이 들어간다. 즉, 1 Epoch에서 이미 출력값을 입력으로 요구하기 때문에 시점이 미래라 할 수 있는 것이고, 결국은 현재의 출력 값을 예측하는데 미래의 값을 이용한다고 말할 수 있다. 이러한 문제를 방지하기 위해 **Look-Ahead Mask** 기법이 나왔다. 

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/227343660-9676f01e-c7d1-4973-b005-6db96d06753a.png">
</p>

트랜스포머에서는 기존의 연산을 유지하며 Attentio Value를 계산할 때 i<j인 요소들은 고려하지 않는다. Attention(i,j)에서 여기서 i는 Query의 값이고, j는 Value의 값이다. 이를 그림으로 표현하면 위와 같다. 디테일하게 Atttention Score를 계산한 행렬의 대각선 윗부분을 -∞로 만들어 softmax를 취했을 때 그 값이 0이되게 만든다. 즉, Masking된 값의 Attnetion Weight는 0이된다. 이렇게 함으로서 Attention Value를 계산할 때 미래 시점의 값을 고려하지 않게된다. 

<br/>

### 3) Sub-Layer2: Position-wise Feed Forward Neural Network(FFNN)

# Reference
[마스킹| 패딩 마스크(Padding Mask), 룩 어헤드 마스킹(Look-ahead masking)]("https://velog.io/@cha-suyeon/%EB%A7%88%EC%8A%A4%ED%82%B9-%ED%8C%A8%EB%94%A9-%EB%A7%88%EC%8A%A4%ED%81%ACPadding-Mask-%EB%A3%A9-%EC%96%B4%ED%97%A4%EB%93%9C-%EB%A7%88%EC%8A%A4%ED%82%B9Look-ahead-masking")  
[pytorch로 구현하는 Transformer (Attention is All You Need)]("https://cpm0722.github.io/pytorch-implementation/transformer")  
[The Annotated Transformer]("http://nlp.seas.harvard.edu/2018/04/03/attention.html")
