---
title: "[Pytorch] Traansformer 구현하기"

categories: 
  - Pytorch
  
tags:
  - [DL, Pytorch, Transformer]
  
toc: true
toc_sticky: true

date: 2023-03-01
last_modified_at: 2023-03-01
---

논문 리뷰: [\[논문리뷰\]Transformer: Attention Is All You Need]("https://meaningful96.github.io/nr/01-Transformer/")

# Why Transformer?

트랜스포머의 가장 큰 contribution은 <span style = "color:red">**기존의 RNN 모델이 불가능했던 병렬 처리를 가능**</span>케했다는 것이다.. GPU를 사용함으로써 얻는 가장 큰 이점은 병렬 처리를 한다는 것. RNN(Recurrent Neural Network)은 recursive하기 때문에 병렬 연산이 불가능하다. 
다시 말해 Next layer의 입력이 이전 layer의 hidden state를 받아야 하기 때문이다. Recurrent network를 사용하는 이유는 sequential할 데이터를 처리하기 위함인데, sequential하다는 것은 등장 시점(또는 위치)를 하나의 정보로 취급한다는 것이다. 
따라서 Context vector를 앞에서부터 순차적으로 생성해내고, 그 Context Vector를 이후 시점에서 활용하는 방식으로 구현한다. 즉, 이후 시점의 연산은 앞 시점의 연산에 의존적이다.

따라서 앞 시점의 연산이 끝나지 않을 경우, 그 뒤의 연산을 수행할 수 없다. 이러한 이유로 RNN 계열의 model은 병렬 처리를 수행할 수 없다. 또한 RNN기반의 모델들(RNN, LSTM, Seq2Seq…)의 단점 중 하나는, 하나의 고정된 사이즈의 context vector에 정보를 압축한다는 사실이다. 이럴 경우 필연적으로 입력이 길어지면 정보 손실이 가속화된다. 또한, sequential data의 특성상 위치에 따른 정보가 중요한데, 이러한 위치 정보가 손실되는 Long term dependency가 발생한다. 

<br/>
<br/>

# Model 구조
## 1. Overview

<p align="center">
<img width="600" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/c9b6d97f-efff-4d24-a35b-1fd3c8ceffaf">
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
<img width="800" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/c409b3b3-50ca-43f8-9534-a1fd23a91b0b">
</p>

간단하게 정리하면 <span style = "color:red">**Encoder**</span>의 역할은 <u>문장(Sentence)를 받아와 하나의 벡터터를 생성</u>해내는 함수이며 이 과정을 흔히 **Encoding**이라고 한다. 이렇게 Encoding을 통해 생성된 벡터를  Context라고 한다. 
반면 <span style = "color:red">**Decoder**</span>의 역할은 Encoder와 반대이다. <u>Context와 right shift된 문장을 입력으로 받아 sentence를 생성</u>해낸다. 이 과정을 Decoding이라고 한다.

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

#### Encoder

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/dd49408a-2d44-4a69-9c41-d008c8c837db">
</p>

Encoder는 N=6이다. 즉 6개의 층이 쌓여져 있다.  이러한 구조를 통해 할 수 있는 사실은, Input와 Output의 shape이 똑같다는 사실이다. 다시 말해 <u>입출력에 있어서 shape은 완전히 동일한 matrix가되며</u> Encoder block은 shape에 대해 멱등하다 할 수 있다.

- 멱등성(Idempotent): 연산을 여러 번 적용하더라도 결과가 달라지지 않는 성질, 연산을 여러 번 반복하여도 한 번만 수행된 것과 같은 성질

층을 여러 개로 구성하는 이유는 사실 간단하다. Encoder의 입력으로 들어오는 <span style = "color:red">Input sequence에 대해 더 넓은 관점에서의 Context를 얻기 위함</span>이다. 
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
<img width="800" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/84218d5b-8ce6-47e3-b4dd-837d72b9ff2a">
</p>

전통적인 Langauge Model의 경우 입력 시퀀스에 대해 Input Embedding matrix만 만들어 모델의 입력으로 보냈다. 하지만, 트랜스포머의 경우는 입력 시퀀스의 각각의 토큰들에 대해 위치 정보까지 주기위해 Positional Encoding도 이용한다. 

<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/9bd553f9-6d2e-4689-b596-55999bb77fc6">
</p>

단, 이 **Positional Encoding**은 <u>각 단어의 상대적인 위치 정보를 네트워크</u>에 입력하는 것이며 sin 또는 cos함수로 이루어져있다. 

#### Encoder Block

Encoder Block은 크게 Multi-Head Attention Layer, Position-wise Feed-Forward Layer로 구성된다. 
```python
class EncoderBlock(nn.Module):

    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention 
        self.position_ff = position_ff


    def forward(self, x):
        out = x
        out = self.self_attention(out)
        out = self.position_ff(out)
        return out
```

<br/>

### 2) Sub-Layer1: Multi-head Attention

#### Attention의 이해
Encoder block의 첫 번째 Sub layer에 해당하는 것은 Multi-head attention이다. Attention mechanism을 이루는 방법에는 여러 가지가 있지만, 트랜스포머의 경우는 <span style = "color:red"><b>Scaled Dot-Product Attention</b></span>을 병렬적으로 여러 번 수행한다. 트랜스포머이후 Scaled Dot-Product attention 방식을 통상적으로 attention이라고 사용한다.

Attention이 그럼 무슨 역할을 하는 건지를 이해하는 것이 중요하다. Attention Mechanism을 사용하는 목적은 생각보다 간단하다. <span style = "color:red"><b>토큰들이 서로서로 얼마나 큰 영향력을 가졌는지를 구하는 것</b></span>이다.


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

그러면 Query, Key, Value 이 세 벡터가 어떤식으로 추출되는지도 알아야한다. 이 벡터들은 입력으로 들어오는 Token embedding을 <span style = "color:red">**Fully Connected Layer(FC)**</span>에 넣어 생성된다. 세 벡터를 생성해내는 FC layer는 모두 다르기 때문에 self-attention에서는 <u>Query, Key, Value를 위한 3개의 서로 다른 FC layer가 존재</u>한다. 각각이 개별적으로 구해지는 것과는 달리
**세 벡터의 Shape은 동일**하다. (<span style = "font-size:110%">$$d_{key} = d_{value} = d_{query} = d_k$$</span>)

#### Scaled Dot-Product Attention

<p align="center">
<img width="300" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/4dd98ab9-0096-4722-86b0-a480c0a1354d">
</p>

Scaled Dot-Product Attention의 메커니즘은 위의 그림과 같다. 먼저 Query와 Key 벡터의 행렬곱을 수행하고 Scaling을 한 후 Softmax를 통해 확률값으로 만들어 버린다. 이후 이 값을 Value와 곱하면된다.

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/d7bc87fc-940a-4112-9616-e99c53a1cb8e">
</p>

좀 더 계산과정을 명확하게 보기위해 한 단어와 단어 사이의 attention을 구하는 과정을 집중해본다. 위에처럼 $$d_k = 3$$인 경우라고 가정하고 FC layer에의해 이미 $$Q, K, V$$가 모두 구해졌다고 가정하고 1번 그림처럼 나타낼 수 있다. 위의 메커니즘과 같이 Query와 Key의 행렬곱을 수행해야 한다. 이 때 Scailing을 포함한 이 행렬곱의 결과를 <span style = "color:red">**Attention Score**</span>라고 한다.

Scailing을 하는 이유는 과연 무엇일까? 그 이유는 사실 간단하다. 행렬 곱의 결과인 attention energy값이 너무 큰 경우 Vanishing Graident현상이 발생할 수 있기 때문이다. 하지만 Scailing을 단순한 상수이므로 행렬곱 연산 결과로 나온 Score의 차원에 영향을 미치지 않는다. 앞서 본 경우는 1:1 관계에서의 attention을 구한 것이다. Self-Attention은 1:N의 관계에서 진행되므로 이를 확장하면 다음과 같다.

<p align="center">
<img width="600" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/9c328e6d-13d3-4487-b80f-b18159227f04">
</p>

먼저 $$Q, K, V$$를 다시 정의해준다. Query의 경우는 비교의 주체이므로 하나의 토큰을 의미하기에 그대로 둔다. 반면 Key와 Value는 비교를 하려는 대상이므로 입력 시퀀스내의 모든 토큰들에 대한 정보를 가지고 있어야 하므로, 각가그이 토큰 임베딩을 Concatenation한 형태로 출력된다. 따라서 $$K, V$$는 행렬로 표현되고 그 크기는 $$n \times d_k$$이다. 이를 통해 마찬가지로 행렬곱을 진행하면 <u>Attention-Score는 전체 토큰 수만큼의 score가 concatenation된 벡터로 출력</u>된다. 다시 말해 Query의 토큰과 입력 시퀀스 내의 모든 토큰들과의 attention score를 각각 계산한 뒤 concatenation한 것이다.

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/4d31fb25-de5e-4bc9-8c89-a0f599820abd">
</p>

행렬곱 결과 구해진 Attention Score를 이용해 최종적으로 일종의 Weight를 만들어야 한다. 이 때, <span style = "color:red">Weight로 변환하는 가장 좋은 방법은 그 값을 <b>확률(Probability)</b>로 만드는 것</span>이다. 확률로 만들기위해 논문에서는 **SoftMax function**을 이용했다. 이렇게 Softmax를 통해 구해진 <span style = "color:red">**Attention Weight(Probability)**</span>을 토큰들의 실질적 의미를 포함한 정보인 Value와 행렬곱을 해준다.(참고로 Attention Weight의 합은 확률이므로 1이다.)

이로써 최종적으로 Query에 대한 <span style = "color:red">**Attention Value**</span>가 나오게 된다. 여기서 알아야 할 중요한 포인트는 연산의 최종 결과인 <u>Query의 Attention Value의 크기(차원)이 Input Query 임베딩과 동일</u>하다는 것이다. Attention Mechanism입장에서 입력은 Query, Key, Value로 세 가지이지만, 의미상으로 Semantic한 측면에서 고려하면 출력이 Query의 attention이므로 입력도 Query로 생각할 수 있다.

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/5292fb72-e760-4ad5-9884-2b72405afaf4">
</p>

앞서 구한 과정은 모두 하나의 Query에 대해서 1:1, 1:N 관계로 확장하며 구한 것이다. 또한 한 번의 행렬 연산으로 구해진 것이다. 하지만 실제로 Query역시 모든 토큰들이 돌아가면서 각각의 토큰들에 대한 Query attention value를 구해야 하므로 Concatenation을 이용해 **행렬**로 확장해야한다. 이를 그림으로 표현하면 위와 같다. 

<br/>

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/94e7763e-ff8e-4c14-93d6-7a6881a6af7f">
</p>

<br/>

행렬로 확장해 Attention을 진행하면 위와 같이 된다. 최종적으로 Query에 대한 Attention value역시 행렬로 출력된다. 다시 한 번 강조하면 **Self-Attention은 Input Query(Q)의 Shape에 대해 멱등(Idemopotent)**하다.

<p align="center">
<img width="650" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/f296ac48-0042-47cf-82f9-5b443502b732">
</p>
<center><span style = "font-size:80%">멱등(Idemopotent)성을 설명하는 그림</span></center>

<br/>
<br/>

Self-Attention의 과정을 수식으로서 정리하면 아래와 같이 정리할 수 있다.

<p align="center">
<img width="300" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/a17d9919-9f83-4a74-9c49-e5e453363562">
</p>

```python
def calculate_attention(query, key, value, mask):
    # query, key, value: (n_batch, seq_len, d_k)
    # mask: (n_batch, seq_len, seq_len)
    d_k = key.shape[-1]
    attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, (n_batch, seq_len, seq_len)
    attention_score = attention_score / math.sqrt(d_k)
    if mask is not None:
        attention_score = attention_score.masked_fill(mask==0, -1e9)
    attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, seq_len, seq_len)
    out = torch.matmul(attention_prob, value) # (n_batch, seq_len, d_k)
    return out
```
- Input의 경우 실제로는 한 문장이 아니라 mini-batch이기 때문에 $$Q, K, V$$에 `n_batch`차원이 추가된다. 
- `calculate`의 인자로 받는 mask는 pad mask인데 이는 다음부분에 다룬다.

<br/>

#### Masked Self-Attention(Masking)
Scaled Dot-Product Attention을 설명하면서 한 부분을 설명하지 않았다. 바로 Masking이다. Masking을 하는 <span style = "color:red">**이유는 특정 값들을 가려서 실제 연산에 방해가 되지 않도록 하기 위함**</span>이다. Masking에는 크게 두 가지 방법이 존재한다. Padding Masking(패딩 마스킹)과 Look-ahead Masking(룩 어헤드 마스킹)이다. 

<span style = "font-size:110%"><b>패딩(Padding)</b></span>  
mini-batch마다 입력되는 문장은 모두 다르다. 이 말을 다시 해석하면, 입력되는 모든 문장의 길이는 다르다. 그러면 모델은 이 <u>다른 문장 길이를 조율해주기 위해 모든 문장의 길이를 동일하게 해주는 전처리 과정이 필요</u>하다. 짧은 문장과 긴 문장이 섞인 경우, 짧은 문장을 기준으로 연산을 해버리면 긴 문장에서는 일부 손실이 발생한다. 반대로, 긴 문장을 기준으로 연산을 해버리면 짧은 문장에서 Self-Attention을 할 경우 연산에 오류가 발생한다.(토큰 개수 부족)

따라서 짧은 문장의 경우 0을 채워서 문장의 길이를 맞춰줘야 한다. 중요한 것은 0을 채워주지만 그 zero Token들의 경우 실제로 의미를 가지지 않는다. 따라서 <span style = "color:red">**실제 attention 연산시에도 제외할 필요**</span>가 있다. 숫자 0의 위치를 체크해주는 것이 바로 패딩 마스킹(Padding Masking)이다.

<span style = "font-size:110%"><b>패딩 마스킹(Padding Masking)</b></span> 

Scaled Dot-Product Attention을 구현할 때 어텐션 함수에서 mask를 인자로 받아 이 값에다 아주 작은 음수값을 곱해 Attention Score행렬에 더해준다.

```python
def scaled_dot_product_attention(query, key, value, mask):
... 중략 ...
    logits += (mask * -1e9) # 어텐션 스코어 행렬인 logits에 mask*-1e9 값을 더해주고 있다.
... 중략 ...
```

이건 Input Sentence에 \[PAD\] 토큰이 있을 경우 어텐션을 제외하기 위한 연산이다. \[PAD\]가 포함된 입력 문장의 Self-Attention을 구하는 과정은 다음과 같다.

<p align="center">
<img width="600" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/016f5f29-4c61-4681-b7bb-4ef4293b6899">
</p>

/[PAD\]는 실제로는 아무 의미가 없는 단어이다. 그래서 트랜스포머에선 key의 경우 \[PAD\] 토큰이 존재할 경우 유사도를 구하지 않도록 마스킹(Masking)을 해준다. Attention에서 제외하기 위해 값을 가리는 행위가 마스킹이다. <u>Attention Score 행렬에서 행에 해당하는 문장은 Query이고 열에 해당하는 문장은 Key</u>이며 key에 \[PAD\]가 있는 열 전체를 마스킹한다.

마스킹을 하는 방법은 사실 간단한데, 매우 작은 음수값을 넣어주면된다. 이 Attention Score가 SoftMax함수를 지나 Value 행렬과 곱해지는데, SoftMax 통과시 PAD부분이 0에 매우 가까운 값이 되어 유사도를 구할 때 반영이 되지 않는다.

<span style = "font-size:110%"><b>룩어헤드 마스킹(Look-Ahead Masking)</b></span> 

RNN이나 트랜스포머, GPT는 문장을 입력받을 때 단방향으로 학습한다. 즉, 하나의 방향으로만 문장을 읽고 트랜스포머는 RNN가 달리 한 step에 모든 문장을 나타내는 행렬이 들어가기 때문에 추가적인 마스킹이 필요하다.

Masked Self-Attention을 하는 이유는, **학습과 추론과정에 정보가 새는(Information Leakage)것을 방지**하기 위함이다. 트랜스포머에서 마스킹된 Self Attention은 모델이 <u>한 번에 하나씩 출력 토큰을 생성할 수 있도록 하면서 모델이 미래의 토큰이 아닌 이전에 생성된 토큰에만 주의를 기울이도록 하기 위함</u>이다. 이를 더 자세히 말하자면, Encoder-Decoder로 이루어진 모델들의 경우 입력을 순차적으로 전달받기 때문에 t + 1 시점의 예측을 위해 사용할 수 있는 데이터가 t 시점까지로 한정된다. 하지만 트랜스포머의 현재 시점의 출력값을 만들어 내는데 미래 시점의 입력값까지 사용할 수 있게되는 문제가 발생하기 때문이다.

이 이유는 트랜스포머의 초기값, 1 Epoch을 생각해보면 이해하기 쉽다. 처음에 입력으로 들어가 인코더를 거친 값이 디코더로 들어가는데, 디코더로 들어가는 또 다른 입력은 이전 Epoch에서의 출력 임베딩값이다. 하지만 1 Epoch에서는 과거의 값은 존재하지 않아 초기에 설정해준 값이 들어간다. 즉, <u>1 Epoch에서 이미 출력값을 입력으로 요구하기 때문에 시점이 미래라 할 수 있는 것</u>이고, 결국은 현재의 출력 값을 예측하는데 미래의 값을 이용한다고 말할 수 있다. 이러한 문제를 방지하기 위해 **Look-Ahead Mask** 기법이 나왔다. 

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/227343660-9676f01e-c7d1-4973-b005-6db96d06753a.png">
</p>

트랜스포머에서는 기존의 연산을 유지하며 Attentio Value를 계산할 때 i<j인 요소들은 고려하지 않는다. Attention(i,j)에서 여기서 i는 Query의 값이고, j는 Value의 값이다. 이를 그림으로 표현하면 위와 같다. 디테일하게 <span style = "color:red">**Atttention Score를 계산한 행렬의 대각선 윗부분을 -∞로 만들어 softmax를 취했을 때 그 값이 0**</span>이되게 만든다. 즉, Masking된 값의 Attnetion Weight는 0이된다. 이렇게 함으로서 Attention Value를 계산할 때 미래 시점의 값을 고려하지 않게된다. 

#### Multi-head Attention

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/227325573-f5ca67b9-3b5a-4e51-bab8-dbeefffa36e8.png">
</p>

트랜스포머의 특징 중 하나는 Multi-head attention을 수행한다는 것이다. 한 Encoder, Decoder Layer마다 1회씩 수행하는 것이 아니라 병렬적으로 $$h$$회 각각 수행한 뒤 그 결과를 종합해 사용한다. 이렇게 하는 이유는 다양한 Attention을 반영해 더 좋은 성능을 내기 위함이다. 논문에서는 head의 개수가 총 **8개**이며 Q,K,V를 위한 FC Layer가 3개에서 $$3 \times 8 = 24$$개가 필요하게 된다. 출력은 Single self-attention의 경우 <b>$$n \times d_k$$</b>의 shape을 가진다. head가 8개가 되면서 이 <span style = "color:red"><b>출력 차원은 $$n \times (d_k \times h)$$로 바뀌게</b></span> 된다.(n은 토큰 개수, 사실상 seq_len). 논문에서는 <b>$$d_{model}  = d_k \times h$$</b>로 정의한다.

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/0f7c9caf-5b94-4600-b0b3-16bb2745b5bf">
</p>

실제 연산은 <span style = "color:red">**병렬로 한 step에서 한 번에 수행**</span>되어 더 효율적인 방식으로 구현된다.

<br/>

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/82f7b89c-bf35-4e43-b869-21b47731fa4a">
</p>

```python
class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, h, qkv_fc, out_fc):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.k_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.v_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.out_fc = out_fc              # (d_model, d_embed)

def forward(self, *args, query, key, value, mask=None):
        # query, key, value: (n_batch, seq_len, d_embed)
        # mask: (n_batch, seq_len, seq_len)
        # return value: (n_batch, h, seq_len, d_k)
        n_batch = query.size(0)

        def transform(x, fc):  # (n_batch, seq_len, d_embed)
            out = fc(x)        # (n_batch, seq_len, d_model)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h) # (n_batch, seq_len, h, d_k)
            out = out.transpose(1, 2) # (n_batch, h, seq_len, d_k)
            return out

        query = transform(query, self.q_fc) # (n_batch, h, seq_len, d_k)
        key = transform(key, self.k_fc)     # (n_batch, h, seq_len, d_k)
        value = transform(value, self.v_fc) # (n_batch, h, seq_len, d_k)

        out = self.calculate_attention(query, key, value, mask) # (n_batch, h, seq_len, d_k)
        out = out.transpose(1, 2) # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        out = self.out_fc(out) # (n_batch, seq_len, d_embed)
        return out
```
먼저 생성자를 살펴보면 `qkv_fc`인자로 <b>$$d_{embed} \times d_{model}$$</b>의 weight matrix를 갖는 FC Layer를 호출받아 멤버 변수로 Q, K, V에 대해 각각 `copy.deepcopy`를 호출해 저장한다. `deepcopy`를 호출하는 이유는 실제로는 서로 다른 weight를 갖고 별개로 사용되게 하기 위해서이다. copy를 하지않으면 항상 같은 Q, K, V 얻게 된다. `out_fc`는 attention 계산 이후 거쳐가는 FC Layer로 <b>$$d_{model} \times d_{embed}$$</b>의 weight matrix를 갖는다.

`forward()` 부분은 가장 핵심적인 부분이며 반드시 이해해야 한다. 인자로 받는 `query`, `key`, `value`는 실제 $$Q, K, V$$ 행렬이 아닌, input sentence embedding이며 shape은 <b>(n_batch $$\times$$ seq_len $$\times \; d_{embed}$$)</b>이다. 이를 3개의 서로 다른 FC Layer에 넣어 $$Q, K, V$$를 구하는 것이다. 이 셋을 별개의 인자로 받는 이유는 Decoder에서 활용하기 위함이다. `mask`는 한 문장에 대해 <b>(seq_len $$\times$$ seq_len)</b>의 shape를 가진다. 여기서 mini-batch까지 고려하면 <b>(n_batch $$\times$$ seq_len $$\times$$ seq_len)</b>가 된다.

`transform`은 $$Q, K, V$$를 구하는 함수이다. 그렇기 때문에 입력의 shape은 <b>(n_batch $$\times$$ seq_len $$\times \; d_{embed}$$)</b>이고, 출력의 shape도 <b>(n_batch $$\times$$ seq_len $$\times \; d_{embed}$$)</b>이다. 하지만 실제로는 단순히 FC Layer만 거쳐가는 것이 아닌 추가적인 변형이 일어난다. 우선 $$d_{model}$$을 $$h$$와 $$d_k$$로 분리하고, 각각을 하나의 차원으로 분리한다. 따라서 shape이 <b>(n_batch $$\times$$ seq_len $$\times h \times \; d_k $$)</b>가 된다. 그 다음 transpose해 <b>(n_batch $$\times \; h \; \times$$ seq_len $$\times \; d_k$$)</b>로 변환한다. 이러한 작업을 수행하는 이유는 위에서 만든 `calculate_attention()`이 입력으로 받고자 하는 shape이 <b>(n_batch $$\times \cdots \; \times$$ seq_len $$\times \; d_k $$)</b>이기 때문이다. 다시 한 번 `calculate_attention()` 살펴보면 아래와 같다. 


```python
def calculate_attention(self, query, key, value, mask):
    # query, key, value: (n_batch, h, seq_len, d_k)
    # mask: (n_batch, 1, seq_len, seq_len)
    d_k = key.shape[-1]
    attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, (n_batch, h, seq_len, seq_len)
    attention_score = attention_score / math.sqrt(d_k)
    if mask is not None:
        attention_score = attention_score.masked_fill(mask==0, -1e9)
    attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, h, seq_len, seq_len)
    out = torch.matmul(attention_prob, value) # (n_batch, h, seq_len, d_k)
    return out
```

우선 $$d_k$$를 중심으로 $$Q, K$$사이의 행렬곱 연산을 수행하기 때문에 $$Q, K, V$$의 마지막 dim은 반드시 $$d_k$$여야만 한다. 또한 attention_score의 shape는 마지막 두 dimension이 반드시 <b>(seq_len $$\times$$ seq_len)</b>이어야만 masking이 적용될 수 있기 때문에 $$Q, K, V$$의 마지막 직전 dim(`.shape[-2]`)는 반드시 seq_len이어야만 한다. 

`forward()`로 돌아와서 `calculate_attention()`을 사용해 attention을 계산하고 나면 그 shape은<b>(n_batch $$\times \; h \times$$ seq_len $$\times \; d_k$$)</b>이다. <span style = "color:red"><b>Multi-head Attention</b></span> Layer 역시 shape에 대해 멱등(Idempotent)해야만 하기 때문에 출력의 shape은 입력과 같은 <b>(n_batch $$\times \; h \times$$ seq_len $$\times \; d_k$$)</b>여야만 한다. 이를 위해 $$h$$와 seq_len의 순서를 뒤바꾸는 `.transpose(1,2)` 메서드를 수행하고 다시 $$h$$와 $$d_k$$를 $$d_{model}$$로 결합한다. 이후 FC Layer를 거쳐 $$d_{model}$$을 $$d_{embed}$$로 변환하게 된다.

```python
class EncoderBlock(nn.Module):

    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff


    def forward(self, src, src_mask):
        out = src
        out = self.self_attention(query=out, key=out, value=out, mask=src_mask)
        out = self.position_ff(out)
        return out

```

다시 ENcoder block을 보면 pad mask는 외부에서 생성할 것이므로 Encoder block의 `forward()`에서 인자로 받는다. 따라서 `forward()`의 최종 인자는 `x`와 `mask`가 된다. 한편, 이전에는 Multi-Head Attention Layer의 `forward()`의 인가자 `x` 1개로 가정하고 코드를 작성했지만, 실제로는 `query`, `key`, `value`와 함께 `mask`도 인자로 받아야 함으로 수정해야한다.

```python
class Encoder(nn.Module):

    def __init__(self, encoder_layer, n_layer):  # n_layer: Encoder Layer의 개수
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_layer))

             
    def forward(self, src, src_mask):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)
        return out
```
`mask`인자를 받기위해 Encoder Block뿐만 아니라 Encode도 역시 수정을 해줘야한다. `forward()`에 `mask`인자를 추가하고, 이를 각 sublayer의 `forward()`에 넘겨준다.

```python
class Transformer(nn.Module):

    ...

    def encode(self, src, src_mask):
        out = self.encoder(src, src_mask)
        return out


    def forward(self, src, tgt, src_mask):
        encoder_out = self.encode(src, src_mask)
        y = self.decode(tgt, encoder_out)
        return y

    ...
```

Transformer class 역시 수정해야 한다. `forward()`의 인자에 `src_mask`를 추가하고, `encoder`의 `forward()`에 넘겨준다.

#### Pad Mask Code with Pytorch

```python
def make_pad_mask(self query, key, pad_idx=1):
    # query: (n_batch, query_seq_len)
    # key: (n_batch, key_seq_len)
    query_seq_len, key_seq_len = query.size(1), key.size(1)

    key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)
    key_mask = key_mask.repeat(1, 1, query_seq_len, 1)    # (n_batch, 1, query_seq_len, key_seq_len)

    query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
    query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (n_batch, 1, query_seq_len, key_seq_len)

    mask = key_mask & query_mask
    mask.requires_grad = False
    return mask
```

앞서 생략된 pad masking을 생성하는 `make_pad_mask()`이다. 인자로는 `qurey`와 `key`를 받는데, 각각 <b>(n_batch $$\times$$ seq_len)</b>의 shape을 가진다. <u><b>embedding을 획득하기도 전 token sequence상태로 들어오는 것</b></u>이다. 여기서 `<pad>`의 인덱스를 의마하는 `pad_idx`와 일치하는 token들은 모두 0, 그 외에는 모두 1인 mask를 생성한다. pad masking은 개념적으로 Encoder 내부에 위치하는게 아닌, `Transformer` class의 메서드로 위치시킨다.

<br/>

### 3) Sub-Layer2: Position-wise Feed Forward Neural Network(FFNN)

#### Position-Wise Feed Forward Layer

```python
class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, fc1, fc2):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1   # (d_embed, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = fc2 # (d_ff, d_embed)


    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

```

단순하게 2개의 FC Layer를 갖는 Layer이다. FC Layer는 ($$d_{model} \; \times \; d_{ff}$$)와 ($$d_{ff} \; \times \; d_{embed}$$)의 weight matrix를 갖는다. 즉, Feed Forward Layer역시 shape에 대해 멱등(Idempotent)하다. 다음 Encoder Block에게 shape를 유지 한 채 넘겨줘야 하기 때문이다. FFNN은 다시 말해서 Multi-head Attention Layer의 출력을 입력으로 받아 연산하고, 다음 Encoder Block에게 Output을 넘겨준다. 논문에서는 첫번째 FC Layer의 출력에 `ReLU()`를 적용한다.

<span style = "font-size:120%"><center>$$ max(0, \; xW_1 \; + \; b_1)W_2 \; + \; b_2 $$</center></span>

#### Residual Connection

<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/57df6b4b-e115-47d4-8093-432b4260289a">
</p>

Residual Connection은 단순하다. 다음 Layer로 넘길 때 원래 입력과 더해주어 $$y \; = \; f(x)$$ 에서 $$y \; = \; f(x) \; + \; x$$ 로 변경하는 것이다. 이로써 <span style = "color:red">**Back Propagation 도중 발생할 수 있는 Vanishing Gradient 현상을 방지**</span>할 수 있다. 보통은 여기에 Layer Normalization과 DropOut까지 추가하는게 일반적이다.

```python
class ResidualConnectionLayer(nn.Module):

    def __init__(self):
        super(ResidualConnectionLayer, self).__init__()


    def forward(self, x, sub_layer):
        out = x
        out = sub_layer(out)
        out = out + x
        return out
```
`forward()`에서 `sub_layer`까지 인자로 받는 구조이다.

따라서 Encode Block의 코드도 바꿔야한다. `residuals`에 Residual Connection Layer를 2개 생성한다. `forward()`에서 `residual[0]`은 `multi_head_attention_layer`를 감싸고, `residual[1]`은 `position_ff`를 감싸게 된다.

```python
class EncoderBlock(nn.Module):

    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(2)]


    def forward(self, src, src_mask):
        out = src
        out = self.residuals[0](out, lambda out: self.self_attention(query=out, key=out, value=out, mask=src_mask))
        out = self.residuals[1](out, self.position_ff)
        return out
```
Residual Connection Layer의 `forward()`에 `sub_layer`를 전달할 때에는 대개 해당 layer 자체를 넘겨주면 되지만, 필요한 경우에 따라 lambda 식의 형태로 전달할 수도 있다. 대표적으로 Multi-Head Attention Layer와 같이 `forward()`가 단순하게 `x`와 같은 인자 1개만 받는 경우가 아닐 때가 있다.


<center><span style = "font-size:80%">Encoder 구조 정리</span></center>

## 4. Decoder

### 1) Decoder 구조

#### Decoder

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/f84e6f84-8528-42d3-b9cd-84995bb4fe07">
</p>

가장 처음에 트랜스포머의 전체 구조를 이야기할 때 봤던 Decoder의 구조이다. context와 Some Sentence를 input으로 받아 Output Sentence를 출력한다. context는 Encoder의 출력이다. 트랜스포머 모델델의 목적을 다시 상기시켜 보자. input sentence를 받아와 output sentence를 만들어내는 model이다. 대표적으로 번역과 같은 task를 처리할 수 있을 것이다. 영한 번역이라고 가정한다면, Encoder는 context를 생성해내는 것, 즉 input sentence에서 영어 context를 압축해 담아내는 것을 목적으로 하고, Decoder는 영어 context를 활용해 한글로 된 output sentence를 만들어내는 것을 목적으로 한다. 

디코더는 추가적으로 다른 Sentence를 더 받는데 이 Sentence를 왜 받아야하며 또한 이 Sentence가 무엇인지 알아야한다. 참고로 Decoder에는 총 3개의 Sublayer가 있다.

```python
class Decoder(nn.Module):

    def __init__(self, decoder_block, n_layer):
        super(Decoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.n_layer)])


    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        for layer in self.layers:
            out = layer(out, encoder_out, tgt_mask, src_tgt_mask)
        return out
```

<b>Context</b>    
Decoder의 입력으로 context와 sentence가 있다. context는 Encoder에서 생성된 것이다. 명심해야 할 것은 <u><b>Encoder 내부에서 Multi-head Attention Layer나 Position-Wise Feed-Forward Layer 모두 shape에 대해서 멱등(Idempotent)</b></u>했다는 것이다. 때문에 이 두 Layer로 구성된 Encoder block도 shape에 대해 반드시 멱등(Idempotent)하다. <span style = "color:red"><b>Encoder의 출력이 context이다</b></span>. context가 Decoder의 입력으로 들어가고 이 shape은 결국 Encoder의 입력과 같은 것이다.

<b>Teacher Forcing</b>    
Decoder의 입력에 추가적으로 들어오는 sentence를 이해하기 위해서는 **Teacher Forcing**의 개념을 알아야 한다. RNN 기반의 모델이던, 트랜스포머기반의 모델이든 이 모델들이 풀고자 하는 task는 결국 sentence generation, 새로운 문장을 생성해내는 것이다. 학습을 하는 과정에서 만약 이 모델들에게 random한 초깃값을 주면 학습이 제대로 이뤄지지 않을 수 있다.(random하게 초기화된 입력은 실제로 의미상 어떠한 문맥 정보도, 토큰의 의미 정보도 포함하고 있지 않기 때문) 첫 단추가 잘 못 끼어진 모델은 Epoch마다 이상한 값을 출력해내고, 그 데이터로 다시 학습하기 때문에 결론적으로 성능에 악영향을 끼치게 된다. 단순한 신경망 모델들의 경우는 이러한 현상을 방지하기위해 Xavier initialization같은 초기화 기법을 도입한다. 트랜스포머머에서는 이러한 현상을 예방하고자 Teacher forcing을 사용하게 된다.

Teacher forcing이란 Supervised Learning에서 <span style = "color:red">**label data를 input으로 활용**</span>하는 것이다. 즉, 학습 시 초기값을 ground truth로 주는 것이다. RNN을 예로 번역 모델을 만든다고 할 때, 학습 과정에서 모델이 생성해낸 토큰을 다음 토큰 생성 때 사용하는 것이 아닌, 실제 label data의 토큰을 사용하게 되는 것이다.

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/39316842-af58-41ba-a35b-d417d3f8ae2a">
</p>

정확도 100%를 달성하는 Ideal한 모델의 경우를 생각했을 때 위와 같다. 예상대로 RNN이전 cell의 출력을 활용해 다음 cell에서 토큰을 정상적으로 생성해낼 수 있다. 하지만, 이런 모델은 실제할 수 없다.

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/3e93d960-4b74-4e10-9236-2c4cdf3e11ee">
</p>

실제로는, 특히나 모델 학습 초창기에는 위처럼 잘못된 토큰을 생성해내고, 그 이후 계속적으로 잘못된 토큰이 생성될 것이다. 즉, 초기값이 랜덤해 부정확하기 때문에 그 다음 순차적으로 학습이 일어나는 RNN이 제대로된 학습을 하지 못하게되는 것이다. 초반에 하나의 토큰이라도 잘못 도출되어 그 이후 토큰들이 잘못 생성되면 학습의 정확성을 높이기 어렵다. 따라서 이를 위해 Labeling된 data를 이용하는 Teaching Forcing을 사용한다.

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/df96334b-ef6e-4b73-b15b-b0a337086b67">
</p>

Teaching Forcing은 실제 Labeled data(Ground Truth)를 RNN cell의 입력으로 사용하는 것이다. 정확히는 Ground Truth의 \[:-1\]로 slicing한 것이다(마지막 토큰인 \[EOS\] 토큰을 제외하는 것). 이를 통해서 모델이 잘못된 토큰을 생성해내더라도 이후 제대로 된 토큰을 생성해내도록 유도할 수 있다. 하지만, 이는 모델 학습 과정에서 Ground Truth, 즉 정답을 사용한 것이므로 일종의 **Cheating**이 된다. 따라서 <span style="color:red">**Test를 할 때는 Ground Truth를 데이터셋에서 제거해주고 진행**</span>해야 한다. 또한 실제로는 데이터셋에 Ground Truth가 포함되어 있어야만 가능한 것이기에 Test나 실제로 Real-World에 Product될 때에는 모델이 생성해낸 이전 토큰을 사용하게 된다. 이처럼 학습 과정에과 실제 사용에서의 괴리가 발생하지만, 모델의 비약적 성능 향상에 직접적으로 영향을 준다. Teaching Forcing은 **Encoder-Decoder 구조 모델에서 많이 사용하는 기법**이다.

<b>Teacher Forcing in Transformer (Subsequent Masking)</b>  
Teacher Forcing 개념을 이해하고 나면 트랜스포머의 Decoder에 입력으로 들어오는 문장은 ground truth\[:-1\]의 문장일 것이다. 하지만 이런 방식으로 Teaching Forcing이 트랜스포머에 그대로 적용되지 못한다. 앞서 든 예시는 RNN이고, RNN은 동시에 모든 토큰을 처리하는 것이 아닌 이전 출력값을 다음 cell의 입력으로 사용하는 순차적인 모델이기 때문이다. 하지만, 트랜스포머는 행렬곱 연산을 통해 한 번에 모든 토큰을 처리한다. 즉, Multi-head attention을 통해 얻는 가장 큰 장점인 **병렬 연산**이 가능하다는 장점이 있다는 것이다. 병렬 연산을 위해 ground truth의 embedding을 행렬로 만들어 입력으로 사용하면 Decoder에서 현재 출력해내야 하는 토큰의 정답을 알고 있는 상황이 발생한다.

따라서 **Masking**을 적용해야 한다. <span style="color:red"><b>$$i$$번째 토큰을 생성해낼 때, $$1 \; ~ \; i-1$$의 토큰은 보이지 않도록 처리\[Masking\]</b></span>를 해야한다.

```python
def make_subsequent_mask(query, key):
    # query: (n_batch, query_seq_len)
    # key: (n_batch, key_seq_len)
    query_seq_len, key_seq_len = query.size(1), key.size(1)

    tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8') # lower triangle without diagonal
    mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)
    return mask
```

`mask_subsequent_mask()`는 `np.tril()`을 사용해 lower traiangle을 생성한다. 아래는 `query_seq_len`과 `key_seq_len`이 모두 10일 때, `np_tril()`의 결과이다.

```python
[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
```

0번째 토큰은 자기 자신밖에 못 본다. 1~n번째 토큰은 0으로 가려져 있으며, 1번재 토큰은 0, 1번째 토큰밖에 보지 못한다. 즉, 1번째 토큰의 입장에서 2~n번째 토큰은 모두 masking되어 있는 것이다. Decoder 역시 **pad**와 **masking**을 모두 수행해야 한다. `make_tgt_mask()`는 다음과 같다. `make_subsequent_mask()`와 `make_tgt_mask()`는 `make_src_mask()`와 같이 `Transformer`에 메서드로 작성한다.

```python
def make_tgt_mask(self, tgt):
    pad_mask = self.make_pad_mask(tgt, tgt)
    seq_mask = self.make_subsequent_mask(tgt, tgt)
    mask = pad_mask & seq_mask
    return pad_mask & seq_mask
```

트랜스포로 다시 돌아가보자. 기존에는 Encoder에서 사용하는 pad mask(`src_mask`)만이 `forward()`을 구해야 했다면, 이제는 Decoder에서 사용할 subsequent + pad mask (`tgt_mask`)도 구해야 한다. `forward()` 내부에서 Decoder의 forward()를 호출할 때 역시 변경되는데, tgt_mask가 추가적으로 인자로 넘어가게 된다.

```python
class Transformer(nn.Module):

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def encode(self, src, src_mask):
        out = self.encoder(src, src_mask)
        return out


    def decode(self, tgt, encoder_out, tgt_mask):
        out = self.decode(tgt, encoder_out, tgt_mask)
        return out


    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        encoder_out = self.encode(src, src_mask)
        y = self.decode(tgt, encoder_out, tgt_mask)
        return y

    ...
```

#### Decoder Block

Decoder 역시 Encoder와 마찬가지로 $$N$$개의 Decoder Block이 겹겹이 쌓인 구조이다. 이 때 주목해야 하는 점은 <span style = "color:red"><b>Encoder에서 넘어오는 context가 각 Decoder Block마다 입력으로 주어진다는 것</b></span>이며 그 외에는 Encoder와 차이가 없다.


<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/a415ef8c-8f5b-481c-bd24-35ed440462b4">
</p>

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/4360de0b-d250-46de-a2f2-959c50c6b0ce">
</p>

그리고 각각의 Decoder Block은 다음과 같다. 참고로 트랜스포머에는 총 3가지의 Attention이 존재한다.

1. Encoder Self-Attention
2. Maksed Decoder Self-Attention
3. Encoder-Decoder Attention(Cross-Attention)

<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/DataStructure_and_Algorithm/assets/111734605/46bb4d86-23e6-4a8d-af33-3fa768f43f8f">
</p>

Decoder Block은 Encoder Block과 달리 Multi-head Attention Layer가 2개가 존재한다. 첫번째 layer는 Self-Multi-Head Attention Layer라고 부르는데, 이름 그대로 Decoder의 입력으로 주어지는 sentence 내부에서의 Attention을 계산한다. 이 때, 일반적인 pad masking뿐만 아니라 subsequent masking이 적용되기 떄문에 **Masked-Multi-Head Attention Layer**라고 부르기도 한다. 

두번째 layer는 Encoder에서 넘어온 context를 Key, Value로 사용한다는 점에서 Cross-Multi-Head Attention Layer라고 부른다. 즉, Encoder의 context는 Decoder 내 각 Decoder Block의 Cross-Multi-Head Attention Layer에서 사용되게 된다.

### 2) Sub-Layer1: Multi-head Attention(Self-Attention)

Encoder의 것과 완전히 동일한데 다만 <span style="color:red">**mask로 들어오는 인자가 일반적인 pad masking에 더해 subsequent masking까지 적용되어 있다**</span>는 점만이 차이일 뿐이다. 즉, 이 layer는 **Self-Attention을 수행**하는 layer이다. 즉, <u>Ground Truth sentence에 내부에서의 Attention을 계산</u>한다. 이는 다음 Multi-Head Attention Layer와 가장 큰 차이점이다.

<br>

### 3) Sub-Layer2: Multi-head Attention(Cross-Attention)

Decoder blcok내 이전 <span style = "color:red"><b>1)Multi-Head Self-Attention Layer에서 넘어온 출력을 입력으로 받는다.</b></span> 여기에 추가적으로 <span style = "color:green"><b>2)Encoder에서 도출된 context도 입력으로 받는다.</b></span> 두 입력의 용도는 완전히 다르다. Decoder Block 내부에서 전달된 입력1)은 <span style="color:red"><b>Query로써 사용</b></span>한다. 반면 Encoder에서 넘어온 context 2)는 <span style="color:green"><b>Key와 Value로써 사용</b></span>하게된다. 

요약하면 Decoer Block의 2번째 Sub-Layer는 서로 다른 두 문장의 Attention을 계산한다. Decoder에서 최종 목표는 <u><b>teaching forcing으로 넘어온 문장과 최대한으로 유사한 predicted sentence를 도출</b></u>하는 것이다. 따라서 Decoer Block 내 이전 Sub-Layer에서 넘어온 입력이 Query가 되고, 이에 상응하는 Encoder의 출력인 context가 Key, Value로 두게 된다. 만약에 영한 번역 모델이면, Encoder의 입력이 영어 문장이되고, Decoder의 입력(Teaching Forcing)과 출력은 한글 문장일 것이다. 따라서 Query가 한글, Key와 Value가 영어가 된다. 

```python
 class MultiHeadAttentionLayer(nn.Module):

        ...

    def forward(self, query, key, value, mask=None):
        
        ...
```



<br>

### 4) Sub-Layer3: Position-wise Feed Forward Neural Network(FFNN)

Encoder의 FFNN과 동일하다.

따라서 `query`, `key`, `value`를 굳이 각각 별개의 인자로 받는 이유가 cross-attention을 활용하기 위함이다.

```python
class Decoder(nn.Module):

    def __init__(self, decoder_block, n_layer):
        super(Decoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.n_layer)])


    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        for layer in self.layers:
            out = layer(out, encoder_out, tgt_mask, src_tgt_mask)
        return out
```

가장 주목할 부분은 `encoder_out`이다. Encoder에서 생성된 최종 출력은 모든 Decoder Block 내부의 Cross-Multi-Head Attention Layer에 `Key`, `Value`로써 주어진다. 두 번째로 주목할 부분은 인자로 주어지는 두 mask인 `tgt_mask`, `src_tgt_mask`이다. `tgt_mask`는 Decoder의 입력으로 주어지는 target sentence의 pad masking과 subsequent masking이다. 즉, 위에서 작성했던 `make_tgt_mask()`로 생성된 mask이다. 이는 Self-Multi-Head Attention Layer에서 사용된다. 

반면, `src_tgt_mask`는 Self-Multi-Head Attention Layer에서 넘어온 `query`, Encoder에서 넘어온 `key`, `value` 사이의 pad masking이다. 이를 구하는 `make_src_tgt_mask()`를 작성한다. 이 때를 위해 `make_pad_mask()`를 `query`와 `key`를 분리해서 인자로 받도록 한 것이다.

```python
def make_src_tgt_mask(self, src, tgt):
    pad_mask = self.make_pad_mask(tgt, src)
    return pad_mask

def make_pad_mask(self, query, key):

    ...

```

Decoder Block은 Encoder Block과 큰 차이가 없다. `forward()`에서 `self_attention`와 달리 `cross_attention`의 `key`, `value`는 `encoder_out`이라는 것, 각각 mask가 `tgt_mask`, `src_tgt_mask`라는 차이점이 있다.

```python
class DecoderBlock(nn.Module):

    def __init__(self, self_attention, cross_attention, position_ff):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(3)]


    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        out = self.residuals[0](out, lambda out: self.self_attention(query=out, key=out, value=out, mask=tgt_mask))
        out = self.residuals[1](out, lambda out: self.cross_attention(query=out, key=encoder_out, value=encoder_out, mask=src_tgt_mask))
        out = self.residuals[2](out, self.position_ff)
        return out
```

`Transformer`도 `src_tgt_mask`를 포함해 수정된다.

```python
class Transformer(nn.Module):

    ...

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        return out


    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)
        encoder_out = self.encode(src, src_mask)
        y = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        return y

    ...
```

## 5. Transformer Input(Positional Encoding)
사실 Transformer의 input으로 들어오는 문장의 shape는 <b>(n_batch $$\times$$ seq_len)</b> 인데, Encoder와 Decoder의 입력은 <b>(n_batch $$\times$$ seq_len $$\times \; d_{embed}$$)</b>의 shape를 가진 것으로 가정했다. 이는 Embedding 과정을 생략했기 때문이다. 사실 Transformer는 source / target sentence에 대한 각각의 Embedding이 포함된다. Transformer의 Embedding은 단순하게 Token Embedding과 Positional Encoding의 sequential로 구성된다.

```pyhon
class TransformerEmbedding(nn.Module):

    def __init__(self, token_embed, pos_embed):
        super(TransformerEmbedding, self).__init__()
        self.embedding = nn.Sequential(token_embed, pos_embed)


    def forward(self, x):
        out = self.embedding(x)
        return out
```

Token Embedding 역시 단순하다. vocabulary와 $$d_{embed}$$를 사용해 embedding을 생성해낸다. 주목할 점은 embedding에도 scaling을 적용한다는 점이다. `forward()`에서 $$\sqrt{d_{embed}}$$를 곱해주게 된다.

```python
class TokenEmbedding(nn.Module):

    def __init__(self, d_embed, vocab_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.d_embed = d_embed


    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_embed)
        return out
```

마지막은 Positional Emcoding이다.

```python
class PositionalEncoding(nn.Module):

    def __init__(self, d_embed, max_len=256, device=torch.device("cpu")):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(8000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0).to(device)


    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out
```

PositionalEncoding의 목적은 positional 정보(token index number 등)를 정규화시키기 위한 것이다. 단순하게 index number를 positionalEncoding으로 사용하게 될 경우, 만약 training data에서는 최대 문장의 길이가 30이었는데 test data에서 길이 50인 문장이 나오게 된다면 30~49의 index는 model이 학습한 적이 없는 정보가 된다. 

이는 제대로 된 성능을 기대하기 어려우므로, positonal 정보를 일정한 범위 안의 실수로 제약해두는 것이다. 여기서 sin 함수와 cos함수를 사용하는데, 짝수 index에는 sin함수를, 홀수 index에는 cos함수를 사용하게 된다. 이를 사용할 경우 항상 -1에서 1 사이의 값만이 positional 정보로 사용되게 된다.

구현 상에서 유의할 점은 생성자에서 만든 `encoding`을 `forward()` 내부에서 slicing해 사용하게 되는데, 이 `encoding`이 학습되지 않도록 `requires_grad=False` 을 부여해야 한다는 것이다. PositionalEncoding은 학습되는 **parameter가 아니기 때문**이다. 이렇게 생성해낸 `embedding`을 `Transformer`에 추가한다. `forward()` 내부에서 Encoder와 Decoder의 `forward()`를 호출할 때 각각 `src_embed(src)`, `tgt_embed(tgt)`와 같이 입력을 `TransformerEmbedding`으로 감싸 넘겨준다.

```python
class Transformer(nn.Module):

    def __init__(self, src_embed, tgt_embed, encoder, decoder):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder


    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)


    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)

    ...

```

## 6. Generator(Decoder의 출력 변환)

Decoder의 출력이 그대로 트랜스포머의 최종 출력이 되는 것은 아니다. Decoder의 출력의 shape는 <b>(n_batch $$\times$$ seq_len $$\times \; d_{embed}$$)인데, 우리가 원하는 출력은 target sentence인 <b>(n_batch $$\times$$ seq_len)</b>이기 때문이다. 즉, Embedding이 아닌 실제 target vocab에서의 token sequence를 원하는 것이다. 이를 위해 추가적인 FC layer를 거쳐간다. 이 layer를 대개 **Generator**라고 부른다.

Generator가 하는 일은 <span style = "color:red"><b>Decoder 출력의 마지막 dimension을 dembed에서 `len(vocab)`으로 변경하는 것</b></span>이다. 이를 통해 실제 vocabulary 내 token에 대응시킬 수 있는 shape가 된다. 이후 `softmax()`를 사용해 각 vocabulary에 대한 확률값으로 변환하게 되는데, 이 때 `log_softmax()`를 사용해 성능을 향상시킨다. `log_softmax()`에서는 `dim=-1`이 되는데, 마지막 dimension인 `len(vocab)`에 대한 확률값을 구해야 하기 때문이다.

```python
class Transformer(nn.Module):

    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    ...

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1)
        return out, decoder_out

    ...

```

## 7. Factory Method
Transformer를 생성하는 `build_model()`은 다음과 같이 작성할 수 있다. 각 module의 submodule을 생성자 내부에서 생성하지 않고, 외부에서 인자로 받는 이유는 더 자유롭게 모델을 변경해 응용할 수 있게 하기 위함이다.

```python
def build_model(src_vocab_size, tgt_vocab_size, device=torch.device("cpu"), max_len=256, d_embed=512, n_layer=6, d_model=512, h=8, d_ff=2048):
    import copy
    copy = copy.deepcopy

    src_token_embed = TokenEmbedding(
                                     d_embed = d_embed,
                                     vocab_size = src_vocab_size)
    tgt_token_embed = TokenEmbedding(
                                     d_embed = d_embed,
                                     vocab_size = tgt_vocab_size)
    pos_embed = PositionalEncoding(
                                   d_embed = d_embed,
                                   max_len = max_len,
                                   device = device)

    src_embed = TransformerEmbedding(
                                     token_embed = src_token_embed,
                                     pos_embed = copy(pos_embed))
    tgt_embed = TransformerEmbedding(
                                     token_embed = tgt_token_embed,
                                     pos_embed = copy(pos_embed))

    attention = MultiHeadAttentionLayer(
                                        d_model = d_model,
                                        h = h,
                                        qkv_fc = nn.Linear(d_embed, d_model),
                                        out_fc = nn.Linear(d_model, d_embed))
    position_ff = PositionWiseFeedForwardLayer(
                                               fc1 = nn.Linear(d_embed, d_ff),
                                               fc2 = nn.Linear(d_ff, d_embed))

    encoder_block = EncoderBlock(
                                 self_attention = copy(attention),
                                 position_ff = copy(position_ff))
    decoder_block = DecoderBlock(
                                 self_attention = copy(attention),
                                 cross_attention = copy(attention),
                                 position_ff = copy(position_ff))

    encoder = Encoder(
                      encoder_block = encoder_block,
                      n_layer = n_layer)
    decoder = Decoder(
                      decoder_block = decoder_block,
                      n_layer = n_layer)
    generator = nn.Linear(d_model, tgt_vocab_size)

    model = Transformer(
                        src_embed = src_embed,
                        tgt_embed = tgt_embed,
                        encoder = encoder,
                        decoder = decoder,
                        generator = generator).to(device)
    model.device = device

    return model

```


> masking을 생성하는 code는 일반적인 Transformer 구현의 code와 다소 상이한데, 본 포스팅에서 사용한 code가 memory를 더 많이 소비한다는 점에서 비효율적이기 때문이다.
> 다만, 본 포스팅의 masking code는 tensor 사이의 broadcasting을 최소화하고, 본래 의도한 tensor의 shape를 그대로 갖고 있기 때문에 학습하는 입장에서는 더 이해가 수월할 것이기에 이를 채택해 사용했다.


<br>

# Reference
[마스킹| 패딩 마스크(Padding Mask), 룩 어헤드 마스킹(Look-ahead masking)](https://velog.io/@cha-suyeon/%EB%A7%88%EC%8A%A4%ED%82%B9-%ED%8C%A8%EB%94%A9-%EB%A7%88%EC%8A%A4%ED%81%ACPadding-Mask-%EB%A3%A9-%EC%96%B4%ED%97%A4%EB%93%9C-%EB%A7%88%EC%8A%A4%ED%82%B9Look-ahead-masking)  
[pytorch로 구현하는 Transformer (Attention is All You Need)](https://cpm0722.github.io/pytorch-implementation/transformer)  
[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)  
[Paper](https://arxiv.org/abs/1706.03762)  
[나동빈 Youtube](https://www.youtube.com/watch?v=AA621UofTUA&t=2664s)  
[Github](https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice)    
[Blog: Transformer 논문 리뷰](https://wandukong.tistory.com/19)
