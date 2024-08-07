---
title: "[그래프 AI]Graph Attention Network(GAT) 구현하기"
categories: 
  - Graph
  
toc: true
toc_sticky: true

date: 2024-08-07
last_modified_at: 2024-08-07
---

글의 가독성을 위해 일부만 코드를 설명하며, 전체 코드는 [Github]()에 있다.

# GNN Stack Module 정의하기
**GNNStack 클래스는 여러 계층의 그래프 주의 신경망을 쌓아 올려서 복잡한 그래프 데이터의 특징을 추출하고 예측하는 구조**를 가지고 있다. 이 모델은 입력 차원(input_dim), 은닉층 차원(hidden_dim), 출력 차원(output_dim), 그리고 몇 가지 하이퍼파라미터를 포함하는 args를 입력으로 받는다.

GNNStack 클래스의 init 메서드에서는 먼저 그래프 신경망 계층을 생성하는 build_conv_model 메서드를 호출하여 지정된 모델 타입(GAT)에 맞는 그래프 신경망 계층을 만든다. 그런 다음, 주어진 계층 수(num_layers)에 따라 여러 그래프 신경망 계층을 쌓는다. post_mp 속성은 메시지 전달 후에 적용될 두 개의 선형 계층과 드롭아웃 계층으로 구성된 신경망이다.

forward 메서드는 모델의 순전파 과정을 정의한다. 입력 데이터(data)의 특징 행렬(x), 엣지 인덱스(edge_index), 배치(batch)를 받아서 각 계층을 거치면서 특징을 변환한다. 각 계층에서는 활성화 함수 ReLU와 드롭아웃을 적용하여 과적합을 방지한다. 마지막으로 post_mp 신경망을 통해 출력값을 얻는다. 만약 emb 매개변수가 True로 설정되어 있으면 임베딩 값을 반환하고, 그렇지 않으면 소프트맥스를 적용하여 예측 결과를 반환한다.

마지막으로 loss 메서드는 예측값(pred)과 실제 레이블(label)을 받아서 음의 로그 우도 손실(Negative Log-Likelihood Loss)을 계산한다.
```python
class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, emb=False):
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(args.heads * hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(args.heads * hidden_dim, hidden_dim), nn.Dropout(args.dropout), 
            nn.Linear(hidden_dim, output_dim))

        self.dropout = args.dropout
        self.num_layers = args.num_layers

        self.emb = emb

    def build_conv_model(self, model_type):
        if model_type == 'GAT':
            return GAT

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
          
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)

        x = self.post_mp(x)

        if self.emb == True:
            return x

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
```

GAT 클래스는 MessagePassing 클래스를 상속받아 메시지 전달 및 집계 과정을 정의한다. 이 모델은 입력 채널(in_channels), 출력 채널(out_channels), 주의 헤드(heads), 음의 기울기(negative_slope), 드롭아웃(dropout) 등의 하이퍼파라미터를 입력으로 받는다.

GAT 클래스의 init 메서드에서는 입력 및 출력 채널, 주의 헤드 수, 음의 기울기, 드롭아웃 비율을 초기화하고, 선형 변환 계층(self.lin_l, self.lin_r)과 주의 가중치(self.att_l, self.att_r)를 정의한다. 그런 다음, reset_parameters 메서드를 호출하여 가중치들을 Xavier 균등 분포로 초기화한다.

forward 메서드는 모델의 순전파 과정을 정의한다. 입력 특징 행렬(x)을 선형 변환한 후 주의 헤드 수와 출력 채널 수에 맞게 변형한다. 변형된 특징 행렬 x_l, x_r을 각각 주의 가중치(att_l, att_r)와 곱하여 주의 점수(alpha_l, alpha_r)를 계산한다. 그런 다음, propagate 메서드를 호출하여 메시지를 전달하고 집계한다. 최종적으로 결과를 다시 변형하여 반환한다.

message 메서드는 메시지 전달 과정을 정의한다. 이 메서드에서는 주의 점수(alpha_i, alpha_j)를 더한 후 Leaky ReLU 활성화 함수를 적용하여 주의 가중치(att_weight)를 계산한다. 주의 가중치는 소프트맥스 함수를 통해 정규화되며, 드롭아웃을 적용하여 과적합을 방지한다. 최종적으로 주의 가중치와 입력 특징(x_j)을 곱하여 메시지를 생성한다.

aggregate 메서드는 메시지 집계 과정을 정의한다. torch_scatter의 scatter 함수를 사용하여 주어진 인덱스(index)에 따라 메시지를 집계한다. 이 과정에서 'sum' 연산을 사용하여 메시지를 합산한다.

1. **선형 변환**:
   노드의 특징 벡터에 가중치 행렬 $$ \mathbf{W} $$를 적용하여 변환한다.

2. **셀프 어텐션 계산**:
   노드 $$i$$와 이웃 노드 $$j$$의 중요도를 계산하는 어텐션 계수를 구한다.

<center>$$e_{ij} = a(\mathbf{W_l}\overrightarrow{h_i}, \mathbf{W_r} \overrightarrow{h_j})$$</center>
   
3. **소프트맥스 정규화**:
   어텐션 계수를 소프트맥스 함수로 정규화하여 $\alpha_{ij} $를 계산한다.

<center>$$\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$$</center>

5. **특징 결합**:
   정규화된 어텐션 계수를 사용하여 이웃 노드들의 특징 벡터를 가중합하여 최종 출력 특징 벡터를 계산한다.

<center>$$h_i' = \sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W_r} \overrightarrow{h_j}$$</center>

6. **멀티 헤드 어텐션**:
   여러 개의 독립적인 어텐션 메커니즘(헤드)을 사용하여 각각의 출력을 결합(concatenate)하거나 평균(average)하여 최종 출력 특징 벡터를 얻는다.

<center>$$\overrightarrow{h_i}' = ||_{k=1}^K \Big(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^{(k)} \mathbf{W_r}^{(k)} \overrightarrow{h_j}\Big)$$</center>

```python
class GAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads = 2,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = None
        self.lin_r = None
        self.att_l = None
        self.att_r = None

        self.lin_l = nn.Linear(self.in_channels, self.out_channels * self.heads)
        self.lin_r = self.lin_l

        self.att_l = nn.Parameter(torch.zeros(self.heads, self.out_channels))
        self.att_r = nn.Parameter(torch.zeros(self.heads, self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size = None):

        H, C = self.heads, self.out_channels
        x_l = self.lin_l(x).reshape(-1, H, C)
        x_r = self.lin_r(x).reshape(-1, H, C)
        alpha_l = self.att_l * x_l
        alpha_r = self.att_r * x_r
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size)
        out = out.reshape(-1, H*C)

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = F.leaky_relu(alpha_i + alpha_j, negative_slope=self.negative_slope)
        if ptr:
            att_weight = F.softmax(alpha, ptr)
        else:
            att_weight = torch_geometric.utils.softmax(alpha, index)
        att_weight = F.dropout(att_weight, p=self.dropout)
        out = att_weight * x_j

        return out


    def aggregate(self, inputs, index, dim_size = None):
        out = torch_scatter.scatter(inputs, index, self.node_dim, dim_size=dim_size, reduce='sum')

        return out
```

