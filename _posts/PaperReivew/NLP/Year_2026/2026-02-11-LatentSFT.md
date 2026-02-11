---
title: "[논문리뷰]Latent Reasoning in LLMs as a Vocabulary-Space Superposition (arXiv, 2026)"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2026-02-11
last_modified_at: 2026-02-11
---

*Jingcheng Deng, Liang Pang, Zihao Wei, Shichen Xu, Zenghao Duan, Kun Xu, Yang Song, Huawei Shen, and Xueqi Cheng*. 2025. [Latent Reasoning in LLMs as a Vocabulary-Space Superposition](https://arxiv.org/abs/2510.15522). arXiv:2510.15522 

# 1. Problem Statement
이 논문이 다루는 핵심 문제는 **Large Language Model(LLM)의 추론 효율성과 성능 간의 근본적인 트레이드오프**이다. 특히 GSM8k, Math500, AIME24와 같은 **수학적 추론 태스크**에서 Chain-of-Thought(CoT) 기반 추론은 높은 정확도를 보이지만, 긴 자연어 추론 체인으로 인해 계산 비용과 추론 시간이 과도하게 증가한다는 문제가 있다.

기존 CoT 방식은 추론 과정을 자연어 토큰 단위로 명시적으로 생성하기 때문에, 추론 과정이 길어질수록 토큰 수가 선형적으로 증가하며 병렬 처리가 어렵다. 이에 대한 대안으로 최근 **Latent Reasoning**이 제안되었으며, 이는 자연어 토큰을 생성하지 않고 연속적인 잠재 공간에서 추론을 수행함으로써 토큰 수를 줄이려는 접근이다.

그러나 기존 latent reasoning 방법들은 추론 효율은 개선하지만, **성능이 크게 저하되는 문제가 반복적으로 관찰**되었다. 본 논문은 이러한 성능 저하의 원인을 **latent token의 정의가 비구조적이며, LLM이 학습한 vocabulary embedding 공간과 정렬되지 않았기 때문**이라고 분석한다.


<br/>
<br/>

# 2. Limitation of Existing Works
기존 latent reasoning 연구들은 공통적으로 **latent token의 표현 방식과 학습 안정성 문제**를 해결하지 못했다.

- **[Hidden State 기반 Latent Token 정의의 한계]**: COCONUT, CODI 등 기존 연구들은 LLM의 **마지막 레이어의 hidden state**를 latent token으로 직접 사용한다. 그러나 논문에서 보인 바와 같이, 마지막 레이어의 hidden state의 분포는 토큰 임베딩 분포와 통계적으로 크게 다르다(Figure 2, Figure 7). LLM은 학습 과정에서 토큰 임베딩 분포만을 입력으로 경험했기 때문에, hidden state를 직접 입력으로 사용할 경우 심각한 distribution shift가 발생하며 성능이 저하된다.
- **[Latent Space의 비구조성과 학습 목표 불일치]**: Hidden state는 고차원 연속 벡터이며 아무런 제약이 없다. 이로 인해 기존 연구들은 MSE나 코사인 유사도와 같은 목적 함수를 사용해 latent token을 학습한다. 이는 LLM의 pre-training objective인 next-token prediction(CE loss)과 근본적으로 불일치하며, latent reasoning 학습을 어렵게 만든다.
- **[토큰 임베딩 Space의 저차원 구조 미반영]**: 논문은 토큰 임베딩 matrix가 full-rank가 아니라 **low effective rank**를 가진다는 점을 관찰한다(Figure 3). 즉, LLM의 semantic space는 본질적으로 저차원 구조를 가진다. 기존 방법들은 latent token을 이 공간에 제한하지 않기 때문에 의미적 일관성이 쉽게 붕괴될 수 있다.

<br/>
<br/>

# 3. Methodology
## 3.0. Difference between Explicit Reasoning and Latent Reasoning
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.01.19%5DLatent-SFT/figure1.png?raw=true">
</p>

모델 설명에 앞서, 위의 그림은 명시적인 추론(explicit reasoning)과 잠재적 추론(latent reasoning)의 추론 메커니즘 차이를 직관적으로 보여준다. 

**[Explicit Reasoning]** (a)는 전통적인 CoT 기반 explicit reasoning을 나타낸다. 각 reasoning step에서 모델은 다음 토큰을 거의 **one-hot에 가까운 분포**로 생성한다.이는 explicit reasoning이 매 step마다 결정된 단어 하나로 사고를 진행함을 의미하며 다시 말해, 매 step마다 하나의 토큰이 선택되고, 그 토큰이 다음 step의 입력이 된다. 이러한 구조의 본질적 특징은 다음과 같다.

- 각 step은 단일 reasoning path만을 따름
- 토큰 분포의 entropy가 낮으며, 거의 deterministic함
- reasoning chain 길이는 reasoning step 수에 비례해 증가함
- 병렬적 reasoning path 탐색이 불가능함

**[Latent Reasoning Model]** (b)는 본 논문에서 제안하는 latent reasoning을 시각화한 것으로, 각 step에서 단일 토큰을 샘플링하지 않고 <span style="gold">**vocabulary 전체에 대한 확률 분포를 유지**</span>하는 방식을 보여준다. 이로 인해 초기 단계에서는 상위 확률 질량이 특정 토큰에 집중되지 않고 넓게 분포하며, 결과적으로 entropy가 높은 상태로 나타난다. 이는 모델이 하나의 확정된 reasoning 경로를 즉시 선택하지 않고, <span style="gold">**여러 가능한 reasoning continuation을 동시에 유지한 채 사고를 진행**</span>하고 있음을 의미한다.

이러한 구조의 특징은 다음과 같다.

- reasoning 과정 동안 <span style="color:gold">**entropy가 높은 분포**</span>를 유지함
- 하나의 step이 <span style="color:gold">**여러 reasoning path를 동시에 포함**</span>함
- reasoning은 token sampling 없이 latent space에서 진행됨
- reasoning chain 길이가 크게 단축됨

## 3.1. Preliminaries Experiments
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.01.19%5DLatent-SFT/figure2.png?raw=true">
</p>

**Observation 1. 마지막 레이어의 hidden state 분포와 토큰 임베딩 분포가 완전히 불일치한다.**  
> The distribution of last-layer hidden states in the LLM is entirely inconsistent with that of the token embeddings

Observation 1은 <span style="color:gold">**LLM의 마지막 레이어의 hidden states 분포가 토큰 임베딩 분포와 통계적으로 완전히 다르다**</span>는 것이다. 논문은 기존 latent reasoning 계열(COCONUT류)이 마지막 레이어 hidden state를 다음 step 입력(=latent token)으로 그대로 feed하는 설계를 따르는데, 이게 성능이 약한 이유가 <span style="color:gold">**입력 분포 자체가 학습 시 분포와 다르기 때문**이라고 주장한다.

- LLM은 pre-training 동안 **토큰 임베딩 분포를 입력으로 받는** 상황에 최적화되어 있다.
- 그런데 hidden state를 그대로 입력으로 다시 넣으면, 모델 관점에서 out-of-distribution 입력을 받는 셈이다.

Figure 2는 (예: LLaMA-3.2-1B-Instruct, GSM8k)에서 PCA로 차원을 줄여서 두 분포를 시각화하고, 단순 시각화에 그치지 않고 FID, MMD² 같은 분포 거리 지표, 그리고 랜덤 샘플링 기반 코사인 유사도와 같은 통계량까지 함께 제시해서 <span style="color:gold">**둘이 같은 공간의 변형 정도가 아니라, 분포적으로 별개**</span>라는 것을 강조한다.

<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.01.19%5DLatent-SFT/figure3.png?raw=true">
</p>

**Observation 2. 토큰 임베딩 matrix의 effective rank가 full이 아니다.**  
> *The Effective Rank of LLM  Token embedding Is Not Full*

Observation 2는 <span style="color:gold">**LLM의 토큰 임베딩 matrix는 nominal dimension은 크지만, 실제로는 low-dimensional subspace에 가깝다**</span>는 것이다. 논문은 여러 LLM의 embedding matrix에 대해 특이값의 스펙트럼을 보고, 특이값이 평평하게 유지되는 게 아니라 **빠르게 decay**한다는 점을 근거로 effective rank가 full이 아니라고 주장한다.

## 3.2. Model Overview
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.01.19%5DLatent-SFT/figure4.png?raw=true">
</p>

**Definition 1. Soft Embedding:**  마지막 토큰의 임베딩을 $$z \in \mathbb{R}^d$$라고 할 대, 이는 토큰 임베딩의 선형 결합(linear combination)으로 표현된다.

<center>$$z = \displaystyle\sum_{i=1}^V \alpha_ie_i, \quad \text{with } \alpha_i \in \mathbb R, e_i \in \mathbb R^d$$</center>

이 때, $$\{ e_i \}_{i=1}^V$$는 vocabulary의 임베딩 벡터들을 의미하고, $$\alpha_i$$는 의미적 혼합도를 결정하는 파라미터이다. 즉, 하나의 토큰이 아니라 <span style="color:gold">**여러 토큰의 확률적 중첩(superposition)**</span>이다. 이 정의 때문에 latent token은 항상 vocabulary 임베딩의 span안에 존재한다는 성질을 가진다.

Latent-SFT는 <span style="color:gold">**(1) 먼저 latent token의 정답 레이블을 만들고(Stage 1), (2) 그 레이블을 LLM이 스스로 생성하도록 학습(Stage 2)**</span>하는 방식이다. 

- **Stage 1: Generating Latent Tokens via Induction–Supervision Masking**: explicit reasoning chain을 짧은 구간으로 나누고, 각 구간을 대표하는 latent toke $$z_i$$를 만들되, 그 $$z_i$$가 **뒤 구간과 최종 Answer를 복원**할 수 있게 강한 마스킹으로 강제한다. 
    
- **Stage 2: Training the LLM to Autonomously Generate Latent Tokens**: Stage 1에서 만든 latent token(정확히는 그에 대응하는 분포 $$p_t = \alpha_t$$)을 **teacher label**로 삼아, LLM이 latent slot에서는 KL로, explicit slot에서는 CE로 학습한다.

## 3.3. Stage 1. Generating Latent Tokens via Induction-Supervision Masking
Stage 1은 encoder–decoder처럼 보이지만, 실제로는 **같은 LLM 구조를 공유**하면서 마스킹으로 역할을 분리한다 (Latent Token Encoder vs Large Language Model). 

**[Latent Token Induction Mask]**  
explicit reasoning chain을 $$N$$개의 subsegment로 나누고, 각 segment의 압축 정보를 담을 special token $$L_i$$를 삽입한다. Segment는 고정 길이 혹은 의미 단위로 입력 텍스트를 쪼개는 것으로, 논문에서는 고정 길이로의 분할이 더 낫다고 말한다.

- **입력:** $$\{Q, <think>, S_1, L_1, S_2, L_2, ..., S_N, L_N, </think>\}$$
- **출력:** Special token $$L_i$$의 임베딩

<p align="center">
<img width="350" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.01.19%5DLatent-SFT/figure5.png?raw=true">
</p>

이 때, 기존 LLM의 causal attention mask를 조작하고, 이를 **latent token induction mask**라고 한다. 이는 $$L_i$$가 <span style="color:gold">**자기 앞의 reasoning subsegment들에서만 정보를 모으도록 어텐션을 제한하는 역할**</span>을 한다. 이를 통해 semantic compactness 성질을 강제한다. 예를 들어, special token $$L_3$$의 입장에서 $$L_1$$과 $$L_2$$ 토큰은 보지않고, segment 1,2,3과 $$L_3$$자기 자신만 보고 토큰의 표현을 만들도록 유도하는 것이다.

<center>$$Soft \; Embedding: Z_i = \sum_{j=1}^V \alpha_j^ie_j, \quad e_j \in \text{Vocab.Matrix}$$</center>

Latent token induction mask를 적용해 latent token encoder가 각 $$L_i$$의 확률 분포를 뽑아내면, 이를 통해 vocabulary 임베딩의 선형 결합 형태롤 만든다.

[Latent Token Supervision Mask (LTSuM)]  
<p align="center">
<img width="350" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.01.19%5DLatent-SFT/figure6.png?raw=true">
</p>

각 latent token $$Z_i$$가 뒤에 남은 explict reasoning 구간들과 최종 answer을 복원하도록 강항 supervision을 건다. 즉 $$Z_i$$가 단순 압축이 아니라 <span style="color:gold">**정답을 내기 위한 충분한 의미를 담도록 semantic correctness를 강제**</span>한다.

- **입력:** $$\Pi_i = \{Q, <think>, Z_1, S_2, Z_2, \cdots, Z_i\}$$
- **복원 대상:** $$Y_i = \{ S_{i+1}\cdots, S_N, </think> \}$$

최종적으로 Supervised decoding loss는 다음과 같다.

<center>$$\mathcal L_{\text{sup}} = \frac{1}{N} \displaystyle\sum_{i=1}^N \frac{1}{\vert \mathcal J_i \vert} \displaystyle\sum_{t \in \mathcal J_i} \Big( -\log p_\theta(x_t \mid \Pi_i ; \text{LTSuM}) \Big)$$</center>

마스킹을 통해 의미를 해석하면, 어텐션을 할 때 오직 $$\Pi_i$$만 보게 만든다는 것이다. 특히 $$S_1, S_2$$와 같은 explicit reasoning 토큰을 보지 못하게 막고, 오직 잠재적인 임베딩만을 활용해 추론하게 만드는 것이다.

## 3.4. Training the LLM to Autonomously Generate Latent Tokens
Stage 2에서는 Stage 1의 latent token encoder를 버리고, LLM이 아래 형태의 시퀀스를 직접 생성하게 만든다.

<center>$$X = [Q, <think>, z_1, \cdots, z_N, </think>, \text{Answer}]$$</center>

여기서 latent slot과 explicit slot을 나누고, latent에는 KL, explicit에는 CE를 건다. 

- **Latent slot label:**  $$p_t = \alpha_t$$
- **Student prediction:** $$q_t = \text{softMax}(W^\top h_t)$$, W는 LM head

<center>$$\mathcal L_{\text{auto}} (\theta_{\text{llm}}) = \lambda \cdot \frac{1}{\vert S_{\text{lat}} \vert} \text{KL}(p_t \mid q_t) + \beta \cdot \frac{1}{\vert S_{\text{exp}} \vert} \displaystyle\sum_{t \in S_{\text{exp}}} (-\log q_t[y_t])$$</center>

- **입력:** Stage 1에서 만든 $$\alpha_t$$ (soft label) + 정답 토큰 $$y_t$$
- **출력:** LLM이 latent token 분포를 먼저 생성하고, 마지막에 Answer를 생성하는 정책

참고로, 학습의 안정성을 위해서는 Joint learning을 매 step마다 하지않는다. 먼저 KL loss를 한 스텝 업데이트하고, 그 다음으로는 NTP, 그 다음 스텝은 joint learning 방식으로 항상 같이 파라미터를 업데이트 하는 방식이 아니라, alternating training방식을 사용한다.

## 3.5. Inference
추론 시에는 Stage 2 모델만 사용한다. 

1. 입력 question를 받고 `<think>` 구간에서 **latent token(soft embedding)들을 생성**한다. 이때 생성되는 것은 본질적으로 $$\alpha_t$$ (vocabulary distribution)이며, 그로부터 $$z_t = W\alpha_t$$라는 soft embedding 입력이 구성되는 형태다.
2. `<think>` 가 끝나면, 모델은 Answer를 자연어 토큰으로 출력한다. 이게 Figure 1에서 말하는 superposition 상태에서 explicit answer로 넘어가는 구간이다(사용자가 이전에 물었던 collapse와 연결되는 지점)

<br/>
<br/>

# 4. Experiments
## 4.1. Main Results
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.01.19%5DLatent-SFT/figure7.png?raw=true">
</p>

**Low-Difficulty Tasks (Table 1)**  
GSM8k-Aug, GSM-Hard, SVAMP, MultiArith에서 Latent-SFT는 모든 latent baseline을 일관되게 상회한다. reasoning length(#L)를 크게 줄이면서도 Pass@1을 유지하거나 향상시킨다. 특히 Latent-SFT(2)는 GSM8k-Aug와 MultiArith에서 explicit CoT-SFT보다 높은 Pass@1을 달성하며, 평균 reasoning length를 약 25% 감소시킨다.

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.01.19%5DLatent-SFT/figure8.png?raw=true">
</p>

**High-Difficulty Tasks (Table 2)**  
Math500과 AIME24에서 Soft Embedding 기반 Latent-SFT는 Hidden State 기반 latent reasoning보다 명확히 우수하다. AIME24에서의 성능 하락은 training 시 reasoning chain 길이를 4k로 제한한 구조적 제약에서 기인한다.

## 4.2. Ablation Study
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.01.19%5DLatent-SFT/figure9.png?raw=true">
</p>

- **LTIM 제거:** LTIM을 제거하면 Pass@1이 평균 약 5% 하락한다. 이는 latent token이 이전 reasoning 정보를 점진적으로 압축해야 함을 보여준다.
- **LTSuM 제거:** LTSuM 제거 시 intermediate supervision이 사라져 성능이 크게 저하된다.
- **Soft Embedding → Hidden State 대체:** 동일한 프레임워크에서 Soft Embedding은 Hidden State보다 항상 우수하며, latent token 정의의 핵심적 중요성을 입증한다.

결론적으로 성능 gain이 가장 큰 것은 Soft Embedding을 사용 유무이다.

## 4.3. Multi-Path Superposition
<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.01.19%5DLatent-SFT/figure91.png?raw=true">
</p>

Figure 6에서의 Analysis 2는 latent reasoning이 단순히 하나의 reasoning path를 압축해 들고 있는 것이 아니라, <span style="color:gold">**여러 개의 서로 다른 reasoning 가설을 동시에 유지한 채로 추론을 진행한다는 점**</span>을 검증한다. 실험 결과에서 Effective Global Parallelism $$N_{eff}$$가 평균적으로 3~4 수준이라는 것은, 모델이 추론 과정 동안 “이렇게 풀 수도 있고, 저렇게 풀 수도 있다”는 복수의 가능성을 하나로 확정하지 않고 병렬적으로 유지하고 있음을 의미한다. 이는 확률 분포가 겉보기로만 넓게 퍼져 있는 상태가 아니라, 실제로 의미 있는 확률 질량이 여러 reasoning path에 나뉘어 배분되어 있다는 뜻이다. 

특히 Top-2 Score가 높게 유지된다는 점은 가장 유력한 reasoning path와 두 번째 후보가 끝까지 경쟁적인 상태로 남아 있음을 보여주며, latent reasoning이 조기에 하나의 해법으로 수렴하는 것이 아니라 여러 대안을 동시에 고려하다가 마지막에야 하나로 정리되는 구조임을 뒷받침한다. 따라서 이 분석은 latent reasoning이 본질적으로 <span style="color:gold">**multiple reasoning path의 superposition 상태에서 사고를 진행한 뒤, 최종 단계에서 하나의 해답으로 collapse하는 추론 방식**</span>임을 high-level에서 명확히 보여준다.

<br/>
<br/>

# 5. Conclusion
**Contribution**
- **[잠재 추론의 새로운 정의 제시]:** 잠재 토큰을 임의의 hidden state가 아니라 어휘 임베딩의 column space에 제한된 “vocabulary 확률의 선형결합(soft embedding)”으로 정의하여, 잠재 공간의 비구조성으로 인한 불일치 문제를 정면으로 다루는 관점을 제시함.

**Limitations**
- **[고난도 장문 추론에서 정확도 격차]:** 고난도 수학 추론처럼 추론 체인이 길어지는 설정에서 잠재 토큰이 의미를 안정적으로 보존하지 못해 오류가 누적되고, 그 결과 명시적 CoT 대비 정확도 격차가 크게 남는 한계가 여전히 존재함.
- **[방법 구성의 복잡성과 불투명성]:** 단계가 많고 학습 절차가 복합적이어서 핵심 아이디어와 각 구성요소의 기여가 직관적으로 분리되어 보이지 않는다는 지적이 제기되며, 더 강한 아블레이션과 단순화가 필요하다는 한계가 있음.
- **[학습·추론 컴퓨트 비용 보고의 불충분]:** 추론 길이 외에도 stage-1의 추가 학습 오버헤드, 학습 시간/자원, 효율-정확도 트레이드오프를 정량적으로 더 명확히 제시할 필요가 있음.
