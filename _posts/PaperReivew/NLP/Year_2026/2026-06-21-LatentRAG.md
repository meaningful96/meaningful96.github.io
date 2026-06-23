---
title: "[논문리뷰]LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG (arXiv, 2026)"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2026-06-21
last_modified_at: 2026-06-21
---

*Yijia Zheng and Marcel Worring*. 2026. **LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG**. arXiv:2605.06285 [cs.CL]

# 1. Problem Statement
이 논문은 iterative reasoning과 retrieval의 장점은 유지하면서 명시적 thought/subquery 생성 비용을 제거하는 것이다. 이를 위해 LatentRAG는 reasoning과 subquery 생성을 LLM의 continuous hidden state인 latent token으로 수행하고, latent subquery를 dense retriever와 정렬하여 자연어 query 없이 직접 검색한다. 동시에 제한된 QA trajectory만으로 latent retrieval을 학습하고, 필요한 경우 latent token을 자연어로 복원하여 효율성과 투명성 사이의 선택을 가능하게 한다.

<br/>
<br/>

# 2. Limitations of Existing Works
기존 single-step RAG는 한 번의 검색만 수행하므로 복합 질문을 여러 단계로 분해해 필요한 정보를 순차적으로 수집하기 어렵다. Agentic RAG는 LLM이 매 단계에서 thought와 subquery를 생성하고 검색 결과를 반영하는 방식으로 이 문제를 완화하지만, 긴 중간 출력을 token-by-token으로 생성해야 하므로 naive RAG보다 약 16–22배 높은 지연을 유발한다.

- **[Single-step RAG의 복합 질문 처리 한계]** Traditional RAG는 원 질문으로 한 번만 검색한 뒤 답을 생성하는 구조를 사용한다. 따라서 질문이 여러 entity나 relation을 순차적으로 연결해야 하는 경우, 첫 검색 결과에 없는 후속 정보를 추가로 요청하거나 이전 검색 결과에 근거해 query를 수정하기 어렵다. 이 구조적 한계로 인해 HotpotQA나 2wiki와 같은 multi-hop QA에서 agentic RAG보다 낮은 성능을 보인다.
- **[명시적 thought 및 subquery 생성의 순차적 지연]** 기존 agentic RAG는 각 검색 단계에서 긴 natural-language thought와 subquery를 autoregressive decoding으로 생성한다. 각 출력 토큰이 이전 토큰에 의존하므로 여러 번의 순차적인 LLM forward pass가 필요하고 GPU 병렬화만으로 지연을 제거하기 어렵다. 논문의 측정에서는 Search-R1의 thought와 subquery 생성이 전체 지연의 약 90%를 차지하며, 전체 추론 시간은 naive RAG의 16–22배에 이른다.
- **[Discrete subquery interface]** 기존 방법은 LLM이 생성한 문자열 subquery를 외부 검색 시스템에 전달하는 구조를 사용한다. Text generation과 document retrieval 사이에 discrete sampling이 존재하므로 retrieval 결과에서 발생한 signal을 LLM의 subquery representation까지 직접 역전파하기 어렵다. 결과적으로 LLM과 retriever가 별도 구성요소로 학습되며, 최종 QA 목적에 맞게 두 구성요소의 representation을 공동 최적화하기 어렵다.

# 3. Methodology
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.21%5DLatentRAG/figure1.png?raw=true">
</p>
Figure 2의 핵심은 explicit agentic RAG의 반복 구조는 유지하되, 각 단계에서 생성하던 자연어 thought와 subquery를 고정된 수의 special token hidden state로 대체하는 것이다. Figure 2(a)의 기존 방식은 `Generation → Retrieval → Generation`을 반복하면서 매번 자연어 thought와 subquery를 순차 생성한다. Figure 2(b)의 LatentRAG는 같은 검색 순서를 따르지만 thought와 subquery 위치에 latent token만 배치하며, latent subquery hidden state로 검색을 수행한다.

Figure 2(c)의 전체 architecture는 세 구성요소로 나뉜다. **Generation**은 question, 이전 retrieved documents, latent thought/subquery token을 입력받아 다음 action과 latent representation을 생성한다. **Retrieval**은 latent subquery token을 projector와 dense retrieval model에 통과시켜 subquery embedding을 만들고 top-$$k$$ document를 반환한다. **Latent Decoding**은 학습 시 latent representation에 semantic supervision을 제공하고, 추론 시 선택적으로 이를 자연어 thought와 subquery로 복원한다.

## 3.0. Preliminaries
LLM 추론은 prefill과 decoding으로 구분된다. Prefill에서는 입력 sequence 전체를 병렬 처리하여 KV cache를 계산하지만, decoding에서는 이전 출력 token에 의존해 다음 token을 하나씩 생성한다. LatentRAG가 제거하려는 병목은 긴 thought와 subquery가 모두 후자의 autoregressive decoding으로 생성된다는 점이다. 기존 agentic RAG의 ttt번째 iteration 이전 interaction trajectory는 다음과 같이 정의된다.

<center>$$
I_t = (\tau_0, s_0, c_0,  \ldots, \tau_{t-1}, s_{t-1}, c_{t-1})
$$</center>

여기서 $$\tau_i$$는 $$i$$번째 turn에 생성된 자연어 thought, $$s_i$$는 $$i$$번째  subquery, $$c_i$$는 $$s_i$$로 검색한 top-$$k$$ 문서 컨텍스트 이다. Agent는 $$q$$와 $$I_t$$를 조건으로

<center>$$
(\tau_t, s_t) = g_\text{LLM} (q, I_t; \theta_\text{LLM})
$$</center>

를 생성하여 검색을 계속하거나, 충분한 정보를 확보했다고 판단하면

<center>$$
(\tau_t, a) = g_\text{LLM} (q, I_t; \theta_\text{LLM})
$$</center>

를 생성하여 종료한다. 즉 <span style="color:red">**종료 시점을 조절하는 Adaptive-RAG**</span>이다. LatentRAG는  text-based thought와 subquery를 latent computation slot으로 대체하고, latent representation을 retrieval model과 직접 연결한다. Methodology는 latent token generation, latent retrieval, latent decoding, joint training objective의 순서로 구성된다.

## 3.1. Generation with Latent Tokens

이 모듈의 목적은 긴 자연어 thought와 subquery를 생성하지 않고, 고정된 special token 위치의 마지막 레이어의 hidden state로 같은 기능을 수행하는 것이다. 각 iteration의 thought와 subquery는 다음 special token sequence로 대체된다.

<center>$$
\begin{aligned}
\tau_t^\ell &= (\langle \texttt{think1} \rangle, \ldots, \langle \texttt{thinkm} \rangle), \\
s_t^\ell &= (\langle \texttt{query1} \rangle, \ldots, \langle \texttt{queryn} \rangle).
\end{aligned}
$$</center>

논문의 기본 설정은 한 reasoning step당 $$m=4$$개의 thought token과 $$n=16$$개의 subquery token을 사용한다. 이 토큰들은 자연어를 직접 표현하는 출력 토큰이 아니라 LLM이 추가 계산을 수행할 수 있는 latent computation slot이다. 이를 통해, trajectory sequence는  $$\mathcal I_t$$는 latent token 기반의 $$\mathcal I_t^\ell$$로 바뀐다.

<center>$$
\mathcal I_t^\ell = (\tau_0^\ell, s_0^\ell, c_0, \ldots, \tau_{t-1}^\ell, s_{t-1}^\ell, c_{t-1}^\ell)
$$</center>

앞선 과정을 통해 <span style="color:red">**thought와 subquery만 latent token으로 바뀌었고, 문서의 컨텍스트는 여전히 자연어**</span>이다. 

### 3.1.1. Latent Thought 생성과 action 결정

질문 $$q$$와 $$\mathcal I_t^\ell$$뒤에 $$\tau_t^\ell$$을 추가하면, 모든 thought token이 prefill 과정에서 병렬 처리되며 각 위치의 마지막 레이어의 hidden state가 $$H_t^\tau$$가 된다. 마지막 latent thought 토큰을 기반으로 다음 action token을 생성한다.

<center>$$
\alpha_t = g_\text{LLM}(q, \mathcal I_t^\ell, \tau_t^\ell, \theta_\text{LLM})
$$</center>

여기서 action token $$\alpha_t$$는  `<query>` 혹은 `<answer>` 이라는 special token 중 하나를 출력하며, `query` 는 검색을 한 번 더 수행해야 함을, `<answer>`는 정보를 충분히 확보하여 최종 답을 생성해야 함을 나타낸다. 따라서 latent thought는 단순한 hidden representation이 아니라 검색 지속 여부를 결정하는 policy representation으로도 사용된다.

### 3.1.2. Latent Subquery 생성

만약 $$\alpha_t$$가 `<query>` 이면, $$(q, \mathcal I_t^\ell, \tau_t^\ell)$$ 뒤에 $$s_t^\ell$$을 추가하고, 한 번의 forward pass로 모든 subquery 토큰의 hidden state를 계산한다.

<center>$$
H_t^s = f_\text{LLM} (s_t^\ell; q, \mathcal I_t^\ell. \tau_t^\ell, \theta_\text{LLM})
$$</center>

기존 방식이 <span style="color:red">**$$n$$개 이상의 자연어 토큰을 순차적으로 생성하던 것과 달리, $$n$$개의 latent subquery 토큰의 hidden state를 병렬적으로 계산**</span>한다.

만약 $$\alpha_t$$가 `<answer>` 이면, special subquery 토큰 대신 `<answer>` 토큰을 배치하여 최종 정답 $$a$$를 autoregressive하게 생성한다. 즉, LatentRAG는 <span style="color:red">**thought와 subquery의 긴 decoding은 제거하지만 최종 답 자체는 일반 LLM과 동일하게 자연어 토큰으로 생성**</span>한다.

## 3.2. Latent Retrieval

이 모듈의 목적은 LLM output space에 존재하는 latent subquery token을 dense retriever가 사용할 수 있는 query embedding으로 변환하고, ground-truth intermediate document 없이 retrieval behavior를 학습하는 것이다.

LLM hidden state와 retriever input embedding은 서로 다른 representation space에 존재하므로, $$H_t^s$$를 직접 retriever에 입력할 수 없다. 따라서 lightweight projector $$\text{Proj}_\text{ret}$$을 사용한다. Projected latent subquery token은 trainable retrieval model에 입력되어 하나의 latent subquery embedding을 생성한다.

<center>$$
v_{s_t^\ell} = f_\text{ret}(\text{Proj}_\text{ret}(H_t^s); \theta_\text{ret})
$$</center>

여기서 $$\theta_{\mathrm{ret}}$$은 pretrained dense retriever에서 초기화되지만 fine-tuning 중 업데이트되는 parameter이다. $$v_{s_t^\ell}$$는 corpus document embedding과 cosine similarity를 계산하는 데 사용되며, 가장 유사한 top-$$k$$ document가 $$c_t$$로 반환된다.

각 subquery $$s_t$$와 candidate document $$d_i$$에 대해 reference embedding과 latent embedding이 유도하는 확률분포를 각각 계산한다.

<center>$$
\begin{aligned}
p_i(s_t)
&=
\frac{
\exp\left(\cos\left(v'_{s_t}, v_{d_i}\right) / \beta\right)
}{
\sum_{j=1}^{N_d}
\exp\left(\cos\left(v'_{s_t}, v_{d_j}\right) / \beta\right)
}, \\[1em]
q_i(s_t)
&=
\frac{
\exp\left(\cos\left(v_{s_t^\ell}, v_{d_i}\right) / \beta\right)
}{
\sum_{j=1}^{N_d}
\exp\left(\cos\left(v_{s_t^\ell}, v_{d_j}\right) / \beta\right)
}.
\end{aligned}
$$</center>

$$v_{d_i}$$는 candidate document embedding이고, $$p_i(s_t)$$는 자연어 subquery가 유도하는 reference distribution이며, $$q_i(s_t)$$는 latent subquery가 유도하는 learned distribution이다. $$N_d$$는 candidate document 수이고, candidate set은 batch 내 모든 subquery의 pseudo-relevant document를 합쳐 구성한다.

Latent subquery가 reference retriever와 유사한 document ranking distribution을 생성하도록 다음 KL divergence를 최소화한다.

<center>$$
L_{\mathrm{ret}}
=
\frac{1}{\lVert B_s \rVert}
\sum_{s_t \in B_s}
\sum_{i=1}^{N_d}
p_i(s_t)
\log
\frac{
p_i(s_t)
}{
q_i(s_t)
}
$$</center>

저자들은 standard InfoNCE가 noisy pseudo-positive를 정답으로 강하게 취급하고, agentic RAG의 작은 학습 데이터에서 충분한 negative structure를 학습하기 어렵다고 판단한다.

## 3.3. Latent Decoding

이 모듈의 목적은 latent thought와 subquery가 대응하는 자연어 의미를 보존하도록 학습하고, 필요할 때 agent의 중간 판단을 사람이 확인할 수 있게 만드는 것이다.

Latent thought와 latent subquery는 각각 별도의 projector인 $$\mathrm{Proj}_{\tau}$$와 $$\mathrm{Proj}_{s}$$를 통해 LLM input space로 변환된다. 두 projector는 retrieval projector와 마찬가지로 bidirectional self-attention과 position-wise FFN으로 구성된다.

Projected latent representation만을 조건으로 원래의 natural-language thought와 subquery를 복원한다.

<center>$$
\begin{aligned}
\tau_t
&=
g_{\mathrm{LLM}}
\left(
\mathrm{Proj}_{\tau}(H_t^{\tau});
\theta_{\mathrm{LLM}}
\right), \\[0.8em]
s_t
&=
g_{\mathrm{LLM}}
\left(
\mathrm{Proj}_{s}(H_t^{s});
\theta_{\mathrm{LLM}}
\right)
\end{aligned}
$$</center>

Thought decoding은 teacher trajectory의 $$\tau_t$$, subquery decoding은 $$s_t$$를 target으로 사용한다. 두 sequence 모두 standard cross-entropy로 학습되며, 각각의 loss를 $$L_{\mathrm{dec}}^{\tau}$$와 $$L_{\mathrm{dec}}^{s}$$로 정의한다.

<center>$$
L_{\mathrm{dec}}
=
L_{\mathrm{dec}}^{\tau}

L_{\mathrm{dec}}^{s}
$$</center>

이 objective는 투명성을 위한 decoder를 학습하는 동시에, $$H_t^{\tau}$$와 $$H_t^{s}$$가 teacher thought와 subquery의 semantic information을 유지하도록 하는 auxiliary supervision으로 작동한다.

추론 시 latent decoding은 선택 사항이다. 효율성이 우선이면 모든 reasoning과 retrieval을 latent space에서 수행하고 중간 text를 생성하지 않는다. 투명성이 필요하면 각 step에서 저장된 $$H_t^\tau$$와 $$H_t^s$$를 추론 종료 후 자연어로 복원한다. 각 sequence 내부의 token generation은 여전히 autoregressive하지만, 서로 다른 step의 thought와 subquery는 각각의 latent representation에만 의존하므로 하나의 batch로 병렬 decoding할 수 있다. 따라서 전체 decoding 시간은 모든 sequence 길이의 합이 아니라 가장 긴 sequence 길이에 크게 좌우된다.

<br/>
<br/>

# 4. Experiments
## 4.1. Main Results
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.21%5DLatentRAG/figure2.png?raw=true">
</p>

LatentRAG는 retriever 종류와 관계없이 explicit agentic RAG와 유사한 EM을 유지하면서 약 90%의 latency를 제거한다. 대표적으로 LatentRAG△는 AutoRefine△보다 평균 EM이 2.5% 높으면서 latency를 89.4% 줄였지만, e5-base-v2와 Harrier에서는 baseline보다 낮은 EM을 보여 성능 향상이 모든 retriever에서 일관되지는 않는다.

## 4.2. Ablation Study
<p align="center">
<img width="450" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.21%5DLatentRAG/figure3.png?raw=true">
</p>

LatentRAG는 retriever와 LLM 크기를 확장해도 낮은 latency를 유지하며, 작은 LLM에서 특히 큰 성능 이점을 보인다. Qwen2.5-3B 설정에서 LatentRAG♢는 Search-R1♢보다 평균 EM이 5.13점 높았으며, LLM 크기가 커질수록 두 방법의 성능 차이는 감소한다.

<br/>
<br/>

# 5. Conclusion
**Contributions**
- **[Latent-space Agentic RAG]** 자연어 thought와 subquery를 반복적으로 생성하는 기존 agentic RAG와 달리, reasoning과 retrieval query를 LLM hidden state에서 직접 표현하는 LatentRAG를 제안한다. 이를 통해 iterative retrieval 능력을 유지하면서 순차적 중간 생성 비용을 크게 줄인다.
- **[End-to-end Latent Retrieval Alignment]** Natural-language subquery가 만드는 document similarity distribution을 latent subquery가 모사하도록 하는 retrieval alignment objective를 제안한다. 이 설계는 명시적인 intermediate ground-truth document 없이 LLM, projector, dense retriever를 공동 학습할 수 있게 한다.
- **[Parallel Latent Decoding]** Latent thought와 subquery를 다시 자연어로 복원하는 선택적 decoding mechanism을 도입한다. 중간 단계들을 병렬적으로 복원할 수 있어 reasoning 과정의 투명성을 제공하면서도 explicit agentic RAG보다 낮은 지연을 유지한다.

**Limitations**
- **[Teacher trajectory 품질 의존성]** LatentRAG는 기존 agentic RAG가 생성한 성공 trajectory를 이용해 학습되므로 성능이 training data의 reasoning 및 retrieval 품질에 부분적으로 제한된다. Teacher의 불필요한 검색이나 잘못된 행동 패턴도 student가 함께 학습할 수 있다.
- **[직접적인 retrieval policy 학습 부재]** 저장된 trajectory에 대한 supervised fine-tuning은 retrieval environment와 직접 상호작용하며 최적 search policy를 탐색하는 과정을 제공하지 않는다. 저자들은 향후 reinforcement learning을 통해 exploration과 exploitation을 수행하는 방향을 제시한다.
