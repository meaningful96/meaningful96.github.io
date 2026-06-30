---
title: "[논문리뷰]Do Transformers Need Three Projections? Systematic Study of QKV Variants (ICML, 2026)"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2026-06-10
last_modified_at: 2026-06-10
---
*Ali Kayyam, Anusha Madan Gopal, and M Anthony Lewis*.  ICML 2026. **[Do Transformers Need Three Projections? Systematic Study of QKV Variants](https://arxiv.org/abs/2606.04032)**. arXiv:2606.04032

# 1. Problem Statement
트랜스포머의 입력은 self-attention과정에서 독립된 linear layer를 거쳐 Query, Key, Value로 분해 후 attention 연산을 수행한다. 이 논문에서는 이 구조가 실제로 모두 필요한지를 검증하는 architecture efficiency analysis를 진행하였다. 구체적으로는 표준 QKV attention을 유지하되 projection matrix를 공유하는 세 가지 방법을 제안 및 비교하고, projection sharing이 성능·파라미터·연산량·KV cache memory에 미치는 영향을 synthetic reasoning, vision, language modeling task에서 평가한다.

<br/>
<br/>

# 2. Limitations of Exising Works
**[표준 QKV 설계의 필요성 검증 부족]** 기존 Transformer는 Query ($$Q$$), Key ($$K$$), Value ($$V$$)를 각각 $$W_q$$, $$W_k$$, $$W_v$$로 독립 projection하는 구조를 사실상 기본값으로 사용해 왔지만, 세 projection 각각이 성능에 얼마나 기여하는지, 일부를 공유하거나 제거해도 되는지에 대한 체계적 검증은 부족하다.

[효율화 연구의 초점 편중] 기존 efficient Transformer 연구는 Performer, Linformer, Ring Attention, blockwise attention처럼 주로 self-attention의 $$O(n^2) $$score computation을 줄이는 방향에 집중했다. 반면 이 논문이 다루는 병목은 projection matrix redundancy와 autoregressive generation에서 누적되는 KV cache memory이며, 이는 attention complexity reduction과 다른 축의 효율화 문제이다.

<br/>
<br/>

# 3. Methodology
## 3.1. Background: Standard Self-Attention
표준 Self-attention은 입력 $$X \in \mathbb{R}^{n \times d}$$를 세 개의 **learnable projection layer**로 변환한 뒤, $$QK^\top$$로 token affinity를 계산하고, $$V$$를 가중합하는 구조이다. 단일 head $$h$$에 대해 다음과 같이 수식으로 표현 할 수 있다.

<center>$$A_h = \text{Softmax}(\alpha Q_hK_h^\top)V_h$$</center>

여기서 $$Q_h = XW_q, K_h = XW_k, V_h = XW_v$$이며, $$W_q, W_k, W_v \in \mathbb{R}^{d \times d_k}$$, $$\alpha=1 /\sqrt{d_k}, d_k=d /H$$이다. 입력은 sequence representation $$X$$, 출력은 head output $$A_h$$이며, multi-head attention에서는 $$A_1, \ldots, A_H$$를 병렬 계산한 뒤, concatenate하고 final linear projection을 적용한다.

## 3.2. Proposed Projection-Shared Attention Variants
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.10%5DAttentionSharing/figure1.png?raw=true">
</p>

이 논문의 핵심 방법은 attention mechanism을 유지하면서 projection sharing constraint만 바꾸는 것이다. Figure 1은 표준 Multi-Head Attention과 세 변형인 Q=K-V Attention, Q-K=V Attention, Q=K=V Attention을 비교하며, ($$X^{+}$$) 는 2D positional encoding을 attention score에 추가한 변형을 의미한다.

### 3.2.1. Variant 1: Q=K-V
**Variant 1: Q=K-V**는 query projection을 제거하고 $$Q=K$$로 묶는 방식이다. 입력 $$X$$는 $$K=XW_k, V=XW_v$$로 변환되며, 출력 attention은 다음과 같다.

<center>$$A = \text{Softmax}(\alpha KK^\top)V$$</center>

이 변형 방식은 Projection을 3개에서 2개로 줄인다는 장점이 있다. 하지만, $$KK^\top$$는 symmetric attention matrix이고 이는 causal dependency가 필요한 방향성을 약화시킨다. 따라서 저자들은 이를 보완하고자 $$()^+$$ 세팅에서 fixed 2D sinusoidal positional encoding $$P\in \mathbb{R}^{n\times n\times m}$$을 추가해 asymmetry를 주입한다.

<br/>

### 3.2.2. Variant 2: Q-K=V
**Variant 2: Q-K=V**는 key와 value의 projection을 공유해 $$V=K$$로 설정하는 방식이다. 입력 $$X$$는 $$Q=XW_q, K=XW_k$$로 변환되고, $$V$$는 별도로 생성하지 않고 $$K$$를 재사용한다. 

<center>$$\text{Softmax}(\alpha QK^\top)K$$</center>

이 변형 방식은 $$Q$$와 $$K$$가 독립이므로 $$QK^\top$$ attention map의 asymmetry를 유지한다. 동시에 autoregressive decoding에서 $$K$$만 caching하면 $$V$$로도 재사용할 수 있어 KV cache 비용을 약 50% 줄일 수 있다.

<br/>

### 3.2.3. Variant 3: Q=K=V
**Variant 3: Q=K=V**는 세 projection을 하나로 묶는 가장 강한 constraint setting이다. 입력 $$X$$는 단일 projection $$K$$로 변환되고, $$Q, K, V$$ 역할을 모두 같은 representation이 수행한다.

<center>$$A = \text{Softmax}(\alpha KK^\top)K$$</center>

이 방식은 projection parameter와 projection computational cost를 가장 크게 줄이지만, $$Q=K$$로 인한 symmetric attention과 $$K=V$$로 인한 representational bottleneck이 동시에 발생한다. 저자들은 $$()^+$$방식으로 이를 일부 완화한다.

<br/>

### 3.2.4. $$()^+$$ Positional Encoding
$$P \in \mathbb{R}^{n \times n \times m}$$인 fixed 2D sinusoidal positional encoding을 정의하고, 각 $$P_{i,j}$$가 query position $$i$$와 key position $$j$$의 relative interaction을 encode하도록 만든다. 여기서 raw attention score가 $$A =QK^\top \in \mathbb{R}^{n \times n}$$일 대, 이를 channel dimension으로 브로드캐스팅하고

<center>$$A^{'} = A + P$$</center>

를 계산하고, $$1\times 1$$ convolution 또는 channel-wise linear projection으로 $$A^{'} \in \mathbb{R}^{n\times n \times m}$$을 다시 $$\mathbb{R}^{n\times n}$$ attention matrix로 변환한다. 이 augmentation은 causal LM이 아니라 vision, synthetic task와 같은 non-causal setting에만 적용된다.

<br/>

## 3.3. Combining Projection Sharing with Head Sharing
이 논문에서 제안하는 두 번째 방법론은 Projection layer뿐만 아니라, Head sharing까지 하는 것이다. 즉 GQA (Grouped Query Attention)와 MQA (Multi-Query Attention)에도 sharing을 적용하는 것이다. GQA와 MQA는 KV cache를 줄이기 위해 여러 attention head가 Key/Value head를 공유하게 만든 attention 구조이다. 이 논문에서는 이를 head sharing이라고 부르고, 논문이 제안하는 projection sharing과는 다른 축의 효율화로 본다. 즉 GQA/MQA는 “head 수를 줄이는 방법”이고, 이 논문의 Q-K=V는 “projection matrix 자체를 공유하는 방법”이다. 논문도 GQA/MQA가 여러 query head가 key-value head를 공유해 memory를 줄인다고 설명한다.  

---
<details style="margin-left: 1.5em;">
  <summary>GQA</summary>
GQA (Grouped Query Attention) 여러 query head를 group으로 묶고 각 group이 하나의 shared K/V head를 사용하게 하는 방식이다. 예를 들어 16개의 query head가 있고, GQA-4를 쓴다면, query head는 16개 그대로 유지하지만, K/V head는 4개 group만 둔다. 즉 16개의 query head가 4개의 shared key-value group을 나눠 쓰는 구조이다.

<center>$$\text{GQA-}g \; \text{Cache size} \propto 2g$$M</center>

예를 들어, H=16, g=4이면 GQA-4의 cache는 2g=8단위이다. 따라서 cache reduction은 다음과 같다.

<center>$$1 - \frac{2g}{2H} = 1 - \frac{g}{H} = 1-\frac{4}{16} =  75\%$$</center>
</details> 

<details style="margin-left: 1.5em;">
  <summary>MQA</summary>  
MQA (Multi-Query Attention)는 GQA의 극단적인 형태이다. 모든 query head가 단 하나의 shared K head와 단 하나의 shared V head를 사용한다. 즉 head가 16개여도, K/V head는 1개뿐이다. 
<center>$$\text{MQA Cache size} \propto 2$$</center>

따라서 H=16일 때, MQA의 cache reduction은 다음과 같다.

<center>$$1 - \frac{2}{2H} = 1 - \frac{1}{H} = 1 - \frac{1}{16} = 93.75\%$$</center>
즉 MQA는 cache를 매우 크게 줄이지만, 모든 query head가 같은 K/V를 공유하기 때문에 표현력 측면에서 GQA에 비해 제한적이다.
</details>

---

GQA-g에서는 $$H$$개의 query head가 $$g < H$$개의 shared KV head를 사용하고, MQA는 하나의 KV head만 모든 query head가 공유하는 극단적 형태이다. 이 때 projection sharing은 head 개수를 줄이는 것이 아니라, 각 KV head 내부에서 $$K=V$$를 강제하는 것이므로 두 방법은 병렬적으로 결합된다. Q-GQA-g 방법은 $$H$$개의 query head, $$g$$개의 KV group에 대해 각 group 내부에서 $$K=V$$를 적용한 방법이다. 따라서 cache reduction은 다음과 같다.

<center>$$\text{Cache Reduction} = 1 - \frac{g}{2H}$$</center>

예를 들어, head가 16개이고, 4개의 KV group으로 구성할 때, GQA-4는 75%의 cache reduction을 제공하고, 여기에 $$K=V$$를 적용한 **Q-GQA-g**는 각 group별 캐쉬를 절반으로 줄여 87.5% cache reduction을 달성한다. MQA에 대해서는 96.9%의 cache reductiond을 달성한다.

<br/>
<br/>

# 4. Experiments
## 4.1. Computational and Memory Analysis
<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.10%5DAttentionSharing/figure2.png?raw=true">
</p>

Computational cost관점에서 모든 variant를 공유하는 Q=K=V 셋팅은, 기존 트랜스포머 방식 대비 1/3배로 줄어든다. 여기서 중요한 점은 **Q-K=V는 어텐션 방향성을 유지하면서 KV Cache를 줄이는 유일한 2-projection variant**이다. 

## 4.2. Performance on Synthetic Tasks
<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.10%5DAttentionSharing/figure3.png?raw=true">
</p>

Synthetic task에서는 Reverse, Sort, Sub, Swap, Copy 다섯 task를 사용해 projection sharing이 algorithmic reasoning에 미치는 영향을 본다. 결과적으로 Q=K-V와 Q-K=V가 QKV와 거의 같은 평균 성능을 보인 반면, Q=K=V는 지나치게 강한 constraint 때문에 크게 떨어진다. 중요한 점은  <span style="color:#9e0000">**Q=K-V나 Q=K=V도 QKV와 동등하거나 더 나은 성능을 보인다는 것**</span>이다.  (X)+는 특히 Swap에서 Q=K-V를 0.597에서 0.671로, Q=K=V를 0.446에서 0.576으로 끌어올려 symmetry 완화 효과를 보인다. 

## 4.3. Performance on Vision Tasks
<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.10%5DAttentionSharing/figure4.png?raw=true">
</p>

Vision task에서는 MNIST, FashionMNIST, CIFAR-10, CIFAR-100, TinyImageNet, set anomaly detection을 평가한다. Stnthetic task와 마찬가지로 Vision task에서도 symmetric attention이 language modeling만큼 치명적이지 않으며, <span style="color:#9e0000">**Q=K-V나 Q=K=V도 QKV와 동등하거나 더 나은 성능**</span>을 보인다.

## 4.4. KV Cache Memory Analysis
<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.10%5DAttentionSharing/figure5.png?raw=true">
</p>

가장 중요한 분석은 Q-K=V가 실질적인 inference memory bottleneck을 직접 줄인다는 점이다. Table 7에서 QKV와 Q=K-V는 token당 80KB, 32K context에서 2.62GB를 사용하지만, Q-K=V는 token당 40KB, 32K context에서 1.31GB만 사용한다. Q=K-V는 2-projection임에도 K와 V를 모두 저장해야 하므로 cache reduction이 0%이고, Q-K=V는 K만 저장해 50%를 줄인다. Combined variant는 Q-GQA-4 0.33GB(87.5%↓), Q-MQA 0.08GB(96.9%↓)까지 줄인다. 이 분석 때문에 **저자들은 Q-K=V를 “practical deployment advantage”가 있는 유일한 2-projection variant**로 본다.

<br/>
<br/>

# Conclusion
**Contribution**  
이 논문의 기여는 **projection sharing을 Transformer attention의 독립적 효율화 축으로 정식화하고, 실험적으로 Q-K=V가 가장 실용적임을 보인 것**이다. 
- 저자들은 synthetic reasoning, vision, language modeling을 포함한 12개 task에서 Q=K-V, Q-K=V, Q=K=V를 비교했고, 300M 및 1.2B LLM에서 scaling behavior를 검증했다.
- Q-K=V가 300M에서 +3.1% PPL degradation으로 50% cache reduction을 달성하고, 1.2B에서는 +2.48% degradation으로 같은 50% cache reduction을 유지한다.
- 마지막으로 저자들은 K=V가 작동하는 이유를 shared representational space와 attention directionality 관점에서 설명하고, QKV collapse가 linear attention에서 SSM-like recurrent formulation으로 이어진다는 이론적 연결도 제시한다.

**Limitations**  
- 이 논문의 한계는 scale, theory, context length, ablation coverage에 있다. 저자들이 명시한 가장 큰 scale은 1.2B parameters이며, Q-K=V의 degradation이 7B 이상에서 계속 줄어드는지는 확인되지 않았다.
- Q-K=V가 잘 작동하는 이유는 K-V similarity와 effective rank에 기반한 empirical explanation이며, formal theory는 아니다.
- Evalutation 시 sequence length도 최대 2048 tokens로 제한되어 length extrapolation은 분석하지 못했다.
