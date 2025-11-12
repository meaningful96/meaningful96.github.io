---
title: "[논문리뷰]Following the Autoregressive Nature of LLM Embeddings via Compression and Alignment(EMNLP, 2025)"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2025-11-12
last_modified_at: 2025-11-12
---

*Jingcheng Deng, Zhongtao Jiang, Liang Pang, Zihao Wei, Liwei Chen, Kun Xu, Yang Song, Huawei Shen, and Xueqi Cheng*. 2025. [**Following the Autoregressive Nature of LLM Embeddings via Compression and Alignment**](https://aclanthology.org/2025.emnlp-main.639/). In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (Eds.). Association for Computational Linguistics, Suzhou, China, 12672–12688. https://doi.org/10.18653/v1/2025.emnlp-main.639

# 1. Problem Statement
이 논문은 LLM(deconder-only)을 이용해 고성능 텍스트 임베딩 모델을 효율적으로 학습하는 방법을 다룬다. 특히 RAG 벤치마크에서 효과적인 instruction 기반 텍스트 임베딩을 만드는 것을 목표로 한다. 특징적으로 기존 베이스라인들과 달리 (질문, 정답 문서, 오답 문서)의 triplet구조로 대조학습을 하지 않고, 그에 따라 방대한 양의 데이터를 요구하지 않기 때문에 GPU Memory 측면에서 latency가 베이스라인 대비 크게 감소하였다.

이를 위해 저자는 기존 LLM 기반 contrastive 학습이 **LLM의 autoregressive(AR) 생성 구조와 근본적으로 불일치**한다는 점을 문제로 지적하고, LLM의 "다음 토큰 분포를 예측하는" 본질을 그대로 따르면서 alignment·uniformity를 만족하는 임베딩을 학습하는 새로운 방법을 제안한다.

**기존 연구의 한계점**

- **[LLM 임베딩의 로컬(Local) 의미 집중 문제]** LLM의 unidirectional attention 구조 때문에, 마지막 토큰의 hidden state에는 **다음 토큰을 예측하기 위한 국소(local) 의미만 강하게 집중**되어 있고, 전체 입력 문장의 전역(global) 의미를 충분히 담지 못한다는 문제가 존재한다. 이 상태를 직접 대조 학습에 사용하는 경우, 로컬 의미에서 글로벌 의미로 전환하기 위해 추가적인 학습 비용과 시간이 요구된다.
- **[생성(Generative) 목적과 대조(Contrastive) 목적의 불일치]** LLM은 **다음 토큰 확률 분포**를 생성하도록 학습되어 있고, hidden state는 곧 이 분포를 산출하는 파라미터에 해당한다. 반면 전통적인 대조 학습(e.g.,  InfoNCE)은 **서로 다른 텍스트 임베딩 간의 cosine distance**를 직접 최적화한다. 즉, 하나는 token-level 확률 분포를 위한 파라미터 공간, 다른 하나는 sample-level embedding 공간에서의 거리 최소화로, **최적화 대상과 관점이 서로 다르기 때문에** LLM의 사전 학습 능력을 충분히 활용하지 못하고 추가적인 training cost를 야기하게 된다
- **[LLM 기반 contrastive 학습의 극단적인 자원 소모]** RepLLaMA, LLM2Vec, NV-Embed와 같은 기존 7B급 LLM 기반 임베딩 모델은 **수백만~수억 규모의 triplet 데이터**와, A100 80GB 수천 시간 단위의 학습 리소스를 요구된다. 이는 (i) LLM이 원래 수행하던 AR next-token prediction과 (ii) contrastive 학습 목적 간의 불일치로 인해, **동일한 성능을 내기 위해 훨씬 더 많은 데이터와 연산이 필요한 비효율성**으로 이어진다고 저자는 지적한다.
- **[Autoregressive 특성을 무시한 임베딩 학습]** 기존 방법은 대부분 "LLM의 마지막 hidden state를 pooling해서 embedding으로 쓰고 cosine contrastive loss로 정렬하는" 식으로, **LLM이 본질적으로 "조건부 분포를 생성하는 AR 모델"이라는 점을 거의 활용하지 않는다**. 즉, 원래의 pretraining objective(다음 토큰 분포)를 버리고 새로운 discriminative objective로 다시 학습하는 셈이라, LLM의 장점을 온전히 살리지 못하는 구조이다. 이 논문은 바로 이 점, 즉 "<span style="color:gold">**AR 특성을 따르면서도 alignment/uniformity를 만족하는 임베딩을 만들 수 없을까?**</span>"를 핵심 질문 제기한다.

<br/>
<br/>

# 2. Methodology
## 2.1. Overview
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.11.12%5DAutoRagEmbed/Auto_figure1.png?raw=true">
</p>

- **Information Compression (IC)** 단계에서는 LLM Encoder는 질문(또는 문서)과 적절한 지시문, 그리고 학습 가능한 compress tokens(일종의 gist token)를 입력으로 받아, compress token 위치의 임베딩을 산출한다.
- 이 임베딩만을 Frozen LLM Decoder에 조건으로 넣고 원본 텍스트의 로그우도를 최대화하도록 학습하여, 인코더가 전역 정보를 잘 압축한 벡터를 만들도록 유도한다.
- 이후 **Conditional Distribution Alignment (CDA)** 단계에서 질문용 임베딩과 문서-자기 임베딩을 만들고, 분포 기반 점수로 정답은 가깝게, 오답은 멀게 정렬한다.
  - 예를 들어 $$S_1(q, d^{+})$$은 "질문용 임베딩이 정답 문서를 생성할 확률"과 "문서-자기 임베딩이 자기 자신을 생성할 확률"의 로그비율 차이를 줄여 두 분포를 맞추는 점수이다.
  - CDA의 핵심은 코사인 같은 벡터 거리에서 벗어나, <span style="color:gold">**조건부 생성 분포의 일치/선호도를 직접 최적화**</span>하는 것이다.

## 2.2. Information Compression (IC): from Discriminative to Generative Embeddings
<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.11.12%5DAutoRagEmbed/Auto_figure2.png?raw=true">
</p>

Instruction Compressio의 목적은, 기존 "<span style="color:gold">**마지막 토큰의 hidden state가 가지는 로컬 의미 중심 임베딩(local semantic embedding)을 전역 의미(global semantic)를  포착하는 generative embedding으로 바꾸는 것**</span>"이다.

기존 decoder LLM에서는, embedding을 마지막 토큰 hidden state로 뽑으면 주로 "첫 출력 토큰을 위한 local semantics"만 담지하는 문제가 있다. average pooling, attention pooling 등 다양한 pooling 기법이 시도되었지만, 다음과 같은 문제가 존재한다.

- average pooling은 "convexity preservation" 이상을 보장하지 못한다.
- attention pooling은 모델 구조를 바꾸거나 추가 파라미터를 도입해, pretraining 구조를 해친다.

따라서, **입력을 압축한 embedding만을 decoder에 입력하고 원문을 복원하도록 강제**하는 방식으로, embedding이 전역 의미를 학습하도록 유도한다.

### 2.2.1. LLM Encoder
- **입력**: `Context + Instruction + Compress Tokens`
먼저 LLM 인코더는 `Context + Instruction` 를 입력으로 받아 **압축 토큰(compress token)** 위치의 hidden state들을 $$e_c$$를 생성한다. 이 때 이 LLM 인코더는 full fine-tunig을 진행하다. 이를 통해 압축 토큰은 입력으로 들어온 Context와 Instruction의 정보를 몇 개의 "압축 토큰"에 효과적으로 담을 수 있다 (입력 텍스트와 instruciton의 전역 정보를 압축 토큰에 담음).


### 2.2.2. LLM Decoder
- **입력**: `Compressed Token Embeddings`

LLM인코더를 통해 생성된 압축 토큰들은 freeze된 LLM 디코더에 입력된다. 디코더는 오직 이 $$e_c$$만을 조건으로 타깃 텍스트 $$d$$를 생성한다. 인코더는 이 텍스트 $$d$$의 likelihood를 최대화 하도록 학습하는 것이다.

<center>$$\begin{aligned}
\mathcal{L}_{IC}
&= \max_{e_{c_1}, \dots, e_{c_k}} P(d \mid e_{c_1}, \dots, e_{c_k} ; \theta_D) \\
&= \max_{\Theta_E} P(d \mid c_1, \dots, c_k, t_1, \dots, t_m, q_1, \dots, q_n ; \theta_E, \theta_D)
\end{aligned}$$</center>

- $$c_1,\dots,c_k$$: Compressed Token
- $$t_1,\dots,t_m$$: Instruction
- $$q_1,\dots,q_n$$: Context (=Query)

결론적으로 이 과정을 통해 $$e_c$$가 <span style="color:gold">**전역적인(global) 의미를 담은 generative embedding**</span>이 된다.

## 2.3. Conditional Distribution Alignment: from Data-Point to Distribution Perspective
<p align="center">
<img width="600" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.11.12%5DAutoRagEmbed/Auto_figure3.png?raw=true">
</p>

Information Compression 단계에서 얻은 임베딩은 **AR decoder를 통해 target을 재구성할 수 있는 generative embedding**이다. 다음 단계에서는 이 embedding이 **alignment와 uniformity를 만족**하도록 학습해야 한다. 이를 위해 **Conditional Distribution Alignment (CDA)** 방법을 제안한다. 

CDA는 "임베딩 벡터 간 거리" 대신 "임베딩이 유도하는 생성 분포 간의 차이"를 이용해서, positivie-negative를 정렬하는 대조 학습 방법이다. 즉, point-level similarity(코사인)에서 distribution-level similarity(조건부 확률 분포) 로 관점을 바꾸는 단계이다.

### 2.3.1. Auto Regressive (AR) 조건부 분포 기반 유사도 정의
<center>$$p(d \mid e_c) = \prod_{t=1}^T p(d_t \mid d_{<t}, e)$$</center>

디코더 $$L_D$$가 주어진 embedding $$ e_c$$에 대한 AR 조건부 분포 위와 같이 정의된다. 이 때, $$d$$는 디코더가 생성하는 문장 토큰 시퀀스이다. 

<center>$$S(q, d) = \frac{1}{T} \displaystyle\sum_{t=1}^T D\big(p(d_t \mid d_{<t}, e_q), p(d_t \mid d_{<t}, e_d)\big)$$</center>

- $$D(\cdot, \cdot)$$: 두 분포 간의 divergence 함수

핵심은 <span style="color:gold">**벡터 거리(point alignment)에서 조건부 분포 거리(distribution alignment)로 전환**</span>하는 것이다.

### 2.3.2. Instruction-Dependent라는 점을 활용한 실질적 정렬 전략
CDA에서는 instruction에 따라 다른 임베딩을 사용한다.

- $$I_{\text{next}}$$: 쿼리를 주면 관련 문서를 생성하는 instruction
- $$I_{\text{self}}$$ : 문서를 주면 자기 자신을 재구성하는 instruction

<center>$$I_{\text{next}} + q \quad \rightarrow \quad e_{q, I_{\text{next}}} \\ I_{\text{self}} + q \quad \rightarrow \quad e_{q, I_{\text{self}}}$$</center>

이를 이용해 두 가지 임베딩을 만들게 된다.

### 2.3.3. Training Objective
**Positive Alignment Score**  
<center>$$S_1(q, d^+) = -\sigma\Big( \beta \mid \log\frac{p_{\Theta_E(d^+ \mid e_{q, I_{\text{next}}})}}{p_{\Theta_E(d^+ \mid e_{d^+, I_{\text{self}}})}} \Big)$$</center>

- $$p_{\Theta_E(d^+ \mid e_{q, I_{\text{next}}})}$$: 질의 $$q$$ 임베딩으로 positive 문서 $$d^+$$를 생성할 확률.
- $$p_{\Theta_E(d^+ \mid e_{d^+, I_{\text{next}}})}$$: positive 문서 $$d^+$$ 임베딩(자기 자신)으로 문서 $$d^+$$를 생성할 확률. 이 값은 일종의 upper bound 역할을 함.

즉 positive alignment score는 같은 문서 $$d^+$$를 생성할 때,  질의 임베딩과 문서-자기 임베딩이 만들어내는 분포의 차이가 logit으로 사용되며, 이 값이 작을수록 두 분포가 비슷하다는 의미이다.

**Positive와 Negative 분리를 위한 Score**  

<center$$>S_2(d^+, d_i^-; q)= -\sigma\!\left(\beta \log\frac{p_{\Theta_E}\!\left(d^+ \mid e_{q,I_{\text{next}}}\right)}{p_{\text{ref}}\!\left(d^+ \mid e_{q,I_{\text{next}}}\right)}-\beta \log\frac{p_{\Theta_E}\!\left(d_i^- \mid e_{q,I_{\text{next}}}\right)}{p_{\text{ref}}\!\left(d_i^- \mid e_{q,I_{\text{next}}}\right)}\right)$$</center>

여기서 새로 등장한 것이 reference분포 $$p_{\text{ref}}$$이다. CDA 학습 이전 상태의 LLM의 분포를 reference로 이용하여 문장 길이, 빈도 등으로 인한 편향을 보정하기 위해 사용한다. 

Logit에서 첫번째 항은  "현재 모델이 reference에 비해 $$d^+$$를 얼마나 더 잘생성하는가"를 나타내고, 두 번째 항은 "현재 모델이 reference에 비해 negative $$d^-$$를 얼마나 잘 생성하는가"를 나타낸다. 이 둘의 차이를 크게 만들어 줌으로써, positive는 더 잘 생성하고, negative는 잘 못생성하게 만드는 것이다.

이는, DPO(Directional Preference Optimization)에서 "**우리가 선호하는 응답의 log prob를 reference 대비 더 크게 만들자**" 는 아이디어와 거의 동일한 구조이다.

<center>$$\mathcal{L}_{\mathrm{CDA}}
= \mathbb{E}\!\left[-\log \frac{e^{S_1(q, d^+)/\tau}}{e^{S_1(q, d^+)/\tau}+\sum_i e^{S_2(d^+, d_i^-; q)/\tau}}\right]$$</center>

최종적으로 Loss 함수는 위와 같이 정의된다. <span style="color:gold">**기본 형태는 InfoNCE와 동일하지만 logit자리가 임베딩간의 유사도가 아닌, 분포의 정렬로 정의된 스코어를 사용**</span>한다는 것이다.

<br/>
<br/>

# 3. Experiments
## 3.1. Main Results
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.11.12%5DAutoRagEmbed/Auto_figure4.png?raw=true">
</p>

논문에서는 STS에서 다음 세 그룹으로 나눠서 비교한다.

- **Training-free base LLM**
    - LLaMA2 / Mistral, last-token / mean pooling 등 (no contrastive training)
- **전통적 contrastive embedding**
    - LLM2Vec (unsupervised / supervised), LLaMA2-inbatch-M, 기타 single-task contrastive baselines
- **대규모 SOTA supervised models**
    - LLM2Vec(S), SFR-Embedding-2_R, NV-Embed, gte-Qwen2-7B-instruct 등
    - 대부분 7B급 모델 + multi-task + 수백만~수천만 triplets

AutoRegEmbed는 LLaMA2-7B를 백본으로, PWC + NLI 정도의 비교적 작은 데이터만으로 학습한 모델이다. 실험 결과 Base LLM에서 LaMA2를 그대로 써서 마지막 토큰/평균 pooling 하는 것과 비교하면, 약 +26pt 이상의 큰 차이를 보여준다. 또한 LLM2Vec이나 NV-Embed등의 LLM기반의 인코더에 비해서도 높은 성능을 달성하였다.

## 3.2. Ablation Study
<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.11.12%5DAutoRagEmbed/Auto_figure5.png?raw=true">
</p>

- **LLaMA2-L → w/o CDA (IC only)**
    - 56.91 → 73.90
    - **+16.99pt** 향상 → Information Compression만으로도 huge gain
- **w/o CDA → Full AutoRegEmbed (IC + CDA)**
    - 73.90 → 83.24
    - **+9.34pt**

IC가 가장 높은 성능 gain을 보여주었으며, <span style="color:gold">**IC는 기본 성능 바닥을 크게 끌어올리고**</span>, **CDA는 alignment를 distribution 레벨에서 정제하는 역할**을 통해 추가적인 gain을 준다.

# 4. Conclusion
**Limitations**  
- **[안전성과 편향 미해결]** AutoRegEmbed 자체는 입력 텍스트의 유해성·편향성·차별적 표현을 필터링하거나 감지하는 메커니즘을 포함하지 않으므로, 학습 데이터에 존재하는 편향과 유해 콘텐츠가 그대로 임베딩에 반영될 수 있다는 한계를 가진다.
- **[도메인 일반화 한계]** Retrieval 실험에서 AutoRegEmbed는 MS MARCO에서는 LLaMA2 기반 single-task contrastive를 능가하지만, NFcorpus·SCIDOCS에서는 multi-domain 대규모 학습 모델(SFR-Embedding-2_R 등)에 비해 낮은 성능을 보인다.
- **[Frozen Decoder 및 베이스 LLM 의존성]** 
Information Compression과 CDA 모두 Frozen Decoder가 제공하는 AR 분포를 기준으로 Encoder를 정렬하는 구조이다. 따라서 베이스 LLM(e.g., LLaMA2, Mistral)의 언어적·논리적 한계, 도메인 커버리지 한계가 그대로 임베딩 품질의 상한으로 작용할 수 있다. 저자가 직접 "한계"로 쓰지는 않지만, 설계상 베이스 LLM 품질에 강하게 종속되는 구조적 제약이 존재한다.

**Contribution**  
- **[Autoregressive 분포를 따르는 임베딩 학습 패러다임 제안]**  
기존 임베딩 학습이 hidden state 간 cosine 유사도만을 최적화하는 point-level contrastive 관점에 머문 것과 달리, 이 논문은 **LLM의 본질인 "다음 토큰 AR 분포"를 그대로 활용하는 distribution-level contrastive 패러다임**을 제안한다. 즉, 임베딩을 "벡터"가 아니라 "어떤 텍스트를 생성하는 확률 분포의 파라미터"로 보고, 이 분포 간 정렬을 학습 목표를 제시하였다.
- **[LLM 임베딩 설계 원칙을 명시적으로 정리]** 
좋은 임베딩이 가져야 할 조건으로 **(1) global semantics 반영, (2) alignment·uniformity, (3) LLM의 autoregressive 특성과의 일관성** 세 가지를 제시하고, 이를 만족하는 구체적 구조(Information Compression + Conditional Distribution Alignment)를 실제로 구현·검증한다.
