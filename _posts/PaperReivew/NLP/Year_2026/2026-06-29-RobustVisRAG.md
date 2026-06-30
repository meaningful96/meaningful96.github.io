---
title: "[논문리뷰]RobustVisRAG: Causaliy-Aware Vision-Based Retrieval-Augmented Generation under Visual Degradations (CVPR, 2026)"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2026-06-29
last_modified_at: 2026-06-29
---

*I-Hsiang Chen, Yu-Wei Liu, Tse-Yu Wu, Yu-Chien Chiang, Jen-Chien Yang, and Wei-Ting Chen*. 2026. [**RobustVisRAG: Causality-Aware Vision-Based Retrieval-Augmented Generation under Visual Degradations**](https://arxiv.org/abs/2602.22013)

# 1. Problem Statement

이 논문의 task는 visual document를 대상으로 한 **Vision-based Retrieval-Augmented Generation (VisRAG)** 이며, 핵심 문제는 blur, noise, low light, shadow 같은 **visual degradation**이 VLM visual encoder 내부에서 semantic factor와 degradation factor를 entangle시켜 retrieval과 generation을 동시에 불안정하게 만든다는 점이다. 기존 VisRAG는 문서 이미지를 직접 encoding하여 OCR parsing error와 layout loss를 줄일 수 있지만, 입력 이미지나 corpus 이미지가 degraded되면 visual representation 자체가 왜곡되어 관련 문서를 잘못 retrieve하거나, 올바른 문서를 retrieve하더라도 generation 단계에서 잘못된 답변을 생성할 수 있다.

논문은 이 문제를 해결하기 위해 **RobustVisRAG**를 제안한다. RobustVisRAG의 목표는 visual encoding 단계에서 task-relevant semantics와 degradation cues를 명시적으로 분리하여, degraded visual condition에서도 retrieval, generation, end-to-end VisRAG 성능을 안정적으로 유지하는 것이다. 이를 위해 논문은 causality-guided dual-path encoder와 Distortion-VisRAG dataset을 함께 제시한다.

<br/>
<br/>

# 2. Limitations of Existing Works
<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.29%5DRobustVisRAG/figure1.png?raw=true">
</p>

기존 연구의 핵심 한계는 visual document의 품질이 항상 충분히 좋다고 가정하거나, degradation을 단순한 입력 품질 문제로만 다루어 semantic representation과 distortion representation의 entanglement를 직접 해결하지 못한다는 점이다. 그 결과 degradation은 retrieval mismatch와 unstable generation이라는 VisRAG pipeline 전체의 error propagation으로 이어진다.

- **[OCR 기반 TextRAG의 degradation 취약성]** Text-based RAG는 document image를 먼저 OCR과 layout analysis로 text로 변환한 뒤 retrieval과 generation을 수행한다. 이 구조는 복잡한 문서의 table, chart, figure, layout cue를 손실하기 쉽고, blur, noise, compression 같은 degradation이 있을 때 recognition error와 parsing failure가 누적된다. 따라서 retrieved context가 불완전하거나 잘못되어 grounding quality가 낮아진다.
- **[기존 VisRAG의 clean-image 가정]** Vision-based RAG는 document image를 VLM으로 직접 encoding하여 OCR error를 줄이고 visual/spatial context를 보존하지만, current models는 대체로 ideal image quality를 가정한다. degraded image가 들어오면 pretrained visual encoder의 embedding space에서 semantics와 distortion factors가 intertwined되어 관련 문서 검색이 틀어지고, generation도 degraded visual signal에 의해 misled될 수 있다.
- **[Two-Stage restoration 전략의 downstream 불일치]** 직관적인 해결책은 degraded image를 image restoration model로 먼저 복원한 뒤 VisRAG에 넣는 것이다. 그러나 restoration method는 perceptual fidelity를 높일 수는 있어도 downstream semantic consistency를 항상 보장하지 않으며, 복원된 이미지가 retrieval이나 generation 성능 향상으로 일관되게 이어지지 않는다. 논문은 이러한 two-stage pipeline이 degraded VisRAG setting에서 제한적인 gain만 제공한다고 지적한다.
- **[Fine-tuning 및 adversarial robustness의 표현 분리 부재]** PEFT는 adaptation cost가 낮지만 degradation-corrupted embedding을 충분히 회복할 representational capacity가 제한적이고, full fine-tuning은 adaptability는 높지만 computation cost와 overfitting, catastrophic forgetting 위험이 크다. 또한 TeCoA, FARE 같은 adversarial fine-tuning 방식은 주로 small controllable pixel perturbation에 초점을 두므로 blur, low light, compression, shadow 같은 natural degradation으로 일반화되기 어렵다. 이들 방법은 공통적으로 semantic factor와 degradation factor를 causality-guided 방식으로 disentangle하지 않는다.

<br/>
<br/>

# 3. Methodology
<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.29%5DRobustVisRAG/figure2.png?raw=true">
</p>

RobustVisRAG는 표준 VisRAG의 vision encoder를 **causality-guided dual-path encoder**로 확장하여, degradation을 담는 non-causal path와 semantics를 담는 causal path를 같은 forward pass 안에서 분리 학습하는 구조이다. 논문 p.4의 Figure 2는 세 부분으로 구성된다. 
- Figure 2(a)는 semantics $$S$$와 degradation $$D$$가 image $$X$$를 함께 만들고, encoder representation $$Z$$를 거쳐 retrieval/generation output $$A$$
에 영향을 주는 structural causal model을 보여준다.
- Figure 2(b)는 degraded document image가 vanilla VisRAG의 retriever와 generator에 그대로 들어가 error를 유발하는 흐름을 나타낸다.
- Figure 2(c)는 RobustVisRAG가 patch token과 non-causal token을 함께 넣고, Transformer layer 안에서 Causal Path와 Non-Causal Path를 분리한 뒤, 각각 Causal Semantic Alignment와 Non-Causal Distortion Modeling으로 학습하는 구조를 보여준다.

## 3.1. Preliminaries
### 3.1.1. Vision-based RAG
**Vision-based RAG**는 텍스트 질문 $$q$$를 보고, visual corpus $$\mathcal V = \{ X_i \}_{i=1}^N$$안에서 관련 문서 이미지를 찾은 뒤, 그 이미지를 근거로 응답 $$Y$$를 생성하는 구조이다. 즉, TextRAG처럼 OCR로 문서를 텍스트로 바꾸는 것이 아니라, <span style="color:red">**문서 이미지를 VLM 인코더가 직접 임베딩해서 검색과 생성에 사용**</span>한다.

<center>$$R = \mathcal R \big( q, \mathcal E_r (\mathcal V) \big), \quad Y = \mathcal G \big( q, \mathcal E_g(R) \big)$$</center>

여기서 $$R$$은 top-$$k$$ 검색된 문서 이미지이고, 는 $$\mathcal{E}_r$$는 검색 인코더, $$\mathcal{E}_g$$는 생성 인코더, $$\mathcal{R}(\cdot)$$과 $$\mathcal{G}(\cdot)$$는 각각 검색과 생성 모듈이다. 
- **검색 인코더 (Retrieval Encoder, $$\mathcal{E}_r$$)**: document image corpus $$\mathcal{V}$$를 검색용 embedding으로 바꾸는 vision 인코더
- **생성 인코더 (Generation Encoder, $$\mathcal{E}_g$$)**: retrieve된 document image $$R$$을 answer generation에 사용할 visual feature로 바꾸는 generator-side vision 인코더

논문은 이후 두 인코더를 통합적으로 $$\mathcal{E}_\theta$$라고 부른다. Vision-based RAG 프레임워크에서 발생하는 핵심 문제는 문서 이미지가 <span style="color:red">**blur, noise, low light, shadow 등으로 degraded되면 인코더의 embedding space가 흔들리고, 이 오류가 검색에서 끝나지 않고 생성까지 전파**</span>된다는 점이다. 쉽게 말하면, VisRAG는 원래 “이미지 안의 내용”을 보고 문서를 찾아야 하는데, degraded image에서는 모델이 내용과 화질 문제를 함께 embedding한다. 그래서 질문과 관련된 문서가 아니라, **왜곡된 visual pattern에 의해 잘못된 문서를 검색**하거나, 맞는 문서를 검색해도 generator가 흐릿한 visual evidence에 의해 잘못된 답을 만들 수 있다.

## 3.1.2. Causal Formulation of Degradation in VisRAG
논문은 이 문제를 **semantic factor와 degradation factor가 latent representation 안에서 섞이는 문제**로 해석한다. 여기서 $$S$$는 문서의 실제 의미 정보, 즉 text, table, chart, layout 등 task에 필요한 semantic factor이고, $$D$$는 blur, shadow 같은 degradation factor이다. 관측되는 문서 이미지 $$X$$는 이 둘이 함께 작용해서 만들어진다.

$$
X=f(S,D,\varepsilon_X)
$$

그다음 VLM의 사전학습된 vision 인코더는 $$X$$를 latent representation $$Z$$로 바꾼다.

$$
Z=\mathcal{E}_\theta(X)
$$

그리고 검색과 생성의 출력은 query $$q$$와 representation $$Z$$를 사용해 결정된다.

$$
(R,Y)=g(q,Z)
$$

이 관계를 **causal graph (인과 그래프)**로 쓰면 다음과 같다.

$$
S \rightarrow X \leftarrow D,
\qquad
X \rightarrow Z,
\qquad
Z \rightarrow (R,Y)
$$

여기서 중요한 직관은 <span style="color:red">**$$S$$와 $$D$$가 모두 이미지 $$X$$를 만들지만, VisRAG의 답변 $$A \in \{R,Y\}$$는 원래 $$S$$에 의해 결정되어야 한다는 점**</span>이다. 즉, 문서에 적힌 내용은 답을 바꾸는 원인이지만, 이미지가 흐리거나 어두운 것은 답을 바꾸는 원인이 되면 안 된다. 그런데 기존 인코더는 $$Z$$ 안에 $$S$$와 $$D$$를 같이 담기 때문에, semantic 정보와 degradation 정보가 entangle된다.

따라서 논문은 representation을 다음처럼 두 부분으로 나누는 것을 목표로 한다.

<center>$$Z=[Z_{\text{sem}},Z_{\text{deg}}]$$</center>

여기서 $$Z_{\text{sem}}$$은 semantic information을 담고, $$Z_{\text{deg}}$$는 degradation information을 내포하는 임베딩이다. 이상적으로는 $$Z_{\text{sem}}$$이 $$S$$에는 의존하지만 $$D$$와는 독립적이어야 한다.

<center>$$Z_{\text{sem}} \not\!\perp\!\!\!\perp S,
\qquad
Z_{\text{sem}} \perp\!\!\!\perp D$$</center>

이 조건이 만족되면 모델의 예측은 degradation의 영향을 제거한 상태, 즉 <span style="color:red">**이미지가 clean한 기준 상태였다고 개입한 경우**</span>의 output에 가까워진다.

<center>$$P(A\mid do(D=d_0)),
\qquad
A\in\{R,Y\}$$</center>

따라서 이 논문의 causal / non-causal 구분은 정답 추론과 오답 추론의 구분이 아니다. <span style="color:red">**causal path는 답을 결정해야 하는 semantic information의 경로이고, non-causal path는 답을 결정하면 안 되는 degradation information을 따로 모으는 경로**</span>이다.

<br/>
<br/>

## 3.2. RobustVisRAG
RobustVisRAG는 기존 VisRAG의 vision 인코더를 **dual-path 인코더**로 바꾸어 semantic information과 degradation information을 분리한다. Figure 2(c)의 구조를 보면, 입력 문서 <span style="color:red">**이미지는 patch 토큰들로 나뉘고, 여기에 별도의 non-causal 토큰 하나가 추가**</span>된다. 이후 트랜스포머 레이어 안에서 patch 토큰들은 causal path를 따라 semantic representation $$Z_{\text{sem}}$$을 만들고, non-causal 토큰은 non-causal path를 따라 degradation representation $$Z_{\text{deg}}$$을 만든다.

전체 직관은 단순하다. 모델 안에 <span style="color:red">**문서 내용 담당 branch와 화질 문제 담당 branch를 따로 만들고, 학습 중에는 화질 문제 branch가 무엇이 degradation인지 알려주도록**</span> 한다. 그러면 semantic branch는 clean image에서 얻은 의미 표현과 비슷해지면서, degradation branch와는 멀어지도록 학습된다.

### 3.2.1. Non-Causal Path
**Non-Causal Path**의 목적은 <span style="color:red">**이미지 안에 있는 blur, noise, shadow 같은 degradation cue를 $$Z_{\text{deg}}$$에 모으는 것**</span>이다. 이를 위해 입력 단계에서 single non-causal 토큰 $$z_{nc}^{(0)}$$을 추가한다. 이 토큰은 모든 patch 토큰을 볼 수 있지만, patch 토큰들은 이 non-causal 토큰을 보지 못하도록 attention mask를 둔다.

<center>$$
z_{nc}^{(l+1)}
=
z_{nc}^{(l)}

\sum_{j=1}^{T}
\alpha_{nc\leftarrow j}^{(l)}v_j^{(l)}
$$</center>

여기서 $$\alpha_{nc\leftarrow j}^{(l)}$$는 non-causal 토큰이 $$j$$번째 patch 토큰을 볼 때의 attention weight이고, $$v_j^{(l)}$$는 해당 patch 토큰의 value projection이다. 이 식의 의미는 <span style="color:red">**non-causal 토큰이 이미지 전체 patch에서 degradation cue를 모은다는 것**</span>이다. 다만 방향이 **한쪽으로만 (uni-directional) 열려** 있기 때문에, degradation cue가 semantic patch 토큰으로 다시 흘러 들어가는 것을 막는다. 마지막 레이어의 출력을 통해 degradation representation은 다음과 같이 정의된다.

<center>$$Z_{\text{deg}}=z_{nc}^{(L)}$$</center>

즉 $$Z_{\text{deg}}$$는 최종 답변에 직접 쓰기 위한 표현이 아니라, **이 이미지가 어떤 방식으로 degraded되었는가**를 학습 중에 알려주는 보조 representation이다.

### 3.2.2. Non-Causal Distortion Modeling
**Non-Causal Distortion Modeling (NCDM)** $$Z_{\text{deg}}$$가 실제로 degradation 정보를 담도록 만드는 loss이다. 구조적으로 non-causal 토큰을 추가했다고 해서 그 토큰이 자동으로 degradation만 담는 것은 아니므로, 논문은 contrastive objective를 사용한다.

<center>$$
\mathcal{L}_{\text{NCDM}}
=
\max\Big(
0,\;
\|Z_{\text{deg}}^a-Z_{\text{deg}}^p\|_2^2
-
\|Z_{\text{deg}}^a-Z_{\text{deg}}^n\|_2^2
+
\delta
\Big)
$$</center>

여기서 anchor image $$X_a$$와 positive image $$X_p$$는 같은 degradation type을 가지고, negative image $$X_n$$은 다른 degradation type을 가진다. 이 loss는 같은 degradation을 가진 sample들의 $$Z_{\text{deg}}$$는 가깝게 만들고, 다른 degradation을 가진 sample들의 $$Z_{\text{deg}}$$는 멀게 만든다. 중요한 점은 NCDM이 <span style="color:red">**degradation class를 맞히는 classifier**</span>를 만드는 것이 아니라는 점이다. 목적은 $$Z_{\text{deg}}$$ 공간 안에 degradation-consistent structure를 만드는 것이다. 이렇게 해야 causal path가 학습될 때 "**이 부분은 semantic이 아니라 degradation임**" ****이라는 신호를 받을 수 있다.

### 3.2.3. Causal Path
**Causal Path**의 목적은 <span style="color:red">**degradation information을 배제하고, 문서의 의미 정보를 담는 $$Z_{\text{sem}}$$을 만드는 것**</span>이다. 이 path에서는 patch 토큰들끼리만 bidirectional attention을 수행하고, non-causal 토큰은 key/value set에서 제거된다. 즉, semantic 토큰들이 degradation 토큰을 직접 참고하지 못하게 한다.

<center>$$
x_i^{(l+1)}
=
x_i^{(l)}

\sum_{j=1}^{T}
\alpha_{i\leftrightarrow j}^{(l)}v_j^{(l)},
\qquad
i=1,\ldots,T
$$</center>

여기서 $$x_i^{(l)}$$는 $$l$$-번째 레이어의 $$i$$-번째 patch 토큰이고, $$\alpha_{i\leftrightarrow j}^{(l)}$$는 patch 토큰들 사이의 attention weight이다. 이 식은 patch 토큰들이 서로 정보를 교환하면서 문서의 semantic context를 구성한다는 의미이다. 마지막 레이어의 semantic representation 은 patch 토큰들의 hidden representation을 aggregation해서 얻는다.

<center>$$
Z_{\text{sem}}
=
\mathrm{Agg}\big(x_1^{(L)},\ldots,x_T^{(L)}\big)
$$</center>

여기서 $$\mathrm{Agg}(\cdot)$$는 인코더가 사용하는 semantic aggregation function이다. 직관적으로 $$Z_{\text{sem}}$$은 “**문서가 무엇을 말하는가**”를 나타내는 representation이고, $$Z_{\text{deg}}$$는 “**이미지가 어떻게 망가졌는가**”를 나타내는 representation이다.

### 3.2.4. Causal Semantic Alignment
**Causal Semantic Alignment (CSA)**는 <span style="color:red">**degraded image에서 얻은 $$Z_{\text{sem}}$$이 clean image에서 얻은 semantic representation과 비슷해지도록 만드는 loss**</span>이다. 동시에 $$Z_{\text{sem}}$$이 $$Z_{\text{deg}}$$와는 멀어지도록 하여 degradation leakage를 줄인다.

Clean/degraded image pair가 있을 때, clean image에서는 $$Z_{\text{sem}}^{\text{clean}}$$을 얻고, degraded image에서는 $$Z_{\text{sem}}^{\text{deg}}$$와 $$Z_{\text{deg}}^{\text{deg}}$$를 얻는다. 먼저 semantic consistency와 degradation independence를 함께 다루는 loss는 다음과 같다.

<center>$$
\mathcal{L}_{\text{SIL}}
=
\frac{1}{T}
\sum_{i=1}^{T}
\Big[
(1-\langle Z_{\text{sem},i}^{\text{deg}},Z_{\text{sem},i}^{\text{clean}}\rangle)

|\langle Z_{\text{sem},i}^{\text{deg}},Z_{\text{deg}}^{\text{deg}}\rangle|
\Big]
$$</center>

- 첫 번째 term은 degraded semantic representation이 clean semantic representation과 가까워지도록 한다. 쉽게 말해, **같은 문서라면 흐릿하게 찍혔든 깨끗하게 찍혔든 semantic representation은 같아야 한다**는 뜻이다.
- 두 번째 term은 $$Z_{\text{sem}}^{\text{deg}}$$와 $$Z_{\text{deg}}^{\text{deg}}$$의 similarity를 낮추어, semantic representation 안에 degradation 정보가 섞이지 않도록 한다.

여기에 token-level local structure를 맞추기 위해 fine-grained alignment term을 추가한다.

<center>$$
\mathcal{L}_{\text{FSAL}}
=
\frac{1}{T}
\sum_{i=1}^{T}
\|Z_{\text{sem},i}^{\text{deg}}-Z_{\text{sem},i}^{\text{clean}}\|_2^2
$$</center>

전체 CSA objective는 $$\mathcal{L}_{\text{CSA}}=\mathcal{L}_{\text{SIL}}+\lambda_{\text{FSAL}}\mathcal{L}_{\text{FSAL}}$$이다.

정리하면 CSA는 degraded image의 semantic representation을 clean image의 semantic representation에 맞추고, 동시에 degradation representation과는 분리한다. 그래서 $$Z_{\text{sem}}$$이 $$P(Z_{\text{sem}}\mid do(D=d_0))$$, 즉 degradation이 제거된 상태의 semantic representation에 가까워지도록 학습한다.

### 3.2.5. Overall Objective
<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.29%5DRobustVisRAG/figure3.png?raw=true">
</p>

정리하자면 이 논문에서는 VLM 전체를 학습하는 것이 아니라, generator 내부의 **LLM은 frozen**하고, retrieval stage와 generation stage에 쓰이는 **두 vision 인코더**를 학습한다. **RobustVisRAG**는 **검색 인코더**와 **생성 인코더** 모두에 같은 causality-guided framework를 적용한다. 
- **causality-guided framework:** 실제로 정답을 결정해야 하는 원인과 정답을 방해하는 요인을 causal graph로 구분하고, 그 구분을 모델 구조와 loss 설계에 반영한 framework

검색 인코더는 query와 positive/negative document image를 구분해야 하므로 standard contrastive retrieval loss를 함께 사용한다.

<center>$$
\mathcal{L}_{\text{Ret}}
=

\log

\frac{
\exp(\langle q,X^+\rangle/\tau)
}{
\exp(\langle q,X^+\rangle/\tau)

+

\sum_{x^-}\exp(\langle q,X^-\rangle/\tau)
}
$$</center

여기서 $$X^+$$는 query 와 관련된 positive visual document이고, $$\{X^-\}$$는 negative visual documents이다. Retrieval의 전체 objective는 다음과 같다.

<center>$$
\mathcal{L}_{\text{Retrieval}}
=
\mathcal{L}_{\text{Ret}}
+
\lambda_1\mathcal{L}_{\text{CSA}}
+
\lambda_2\mathcal{L}_{\text{NCDM}}
$$</center>

**Generation**에서는 language model은 frozen하고, visual 인코더만 causality-guided objectives로 fine-tuning한다.

<center>$$
\mathcal{L}_{\text{Generation}}
=
\mathcal{L}_{\text{CSA}}
+
\lambda_3\mathcal{L}_{\text{NCDM}}
$$</center>

즉 generation 쪽에서는 <span style="color:red">**답변 생성 능력 자체를 새로 학습시키기보다는, degraded visual input을 보더라도 generator에 전달되는 visual representation이 clean semantic representation처럼 유지되도록 인코더를 조정**</span>한다.

### 3.2.6. Inference
Inference에서는 $$Z_{\text{sem}}$$만 사용하고, non-causal branch에서 얻은 $$Z_{\text{deg}}$$는 버린다. Non-causal path는 학습 중에 semantic–degradation disentanglement를 유도하기 위한 장치이고, 최종 검색과 생성에는 degradation-invariant semantic representation만 필요하기 때문이다. 따라서 RobustVisRAG는 test time에는 표준 VisRAG와 거의 같은 방식으로 동작한다. 핵심 차이는 인코더가 이미 학습 과정에서 degradation cue를 분리하는 법을 배웠기 때문에, degraded image가 들어와도 $$Z_{\text{sem}}$$이 더 안정적으로 유지된다는 점이다.

## 3.3. Distortion-VisRAG Dataset

Distortion-VisRAG Dataset은 degraded visual condition에서 **VisRAG의 retrieval, generation, end-to-end robustness를 평가하기 위해 만든 benchmark**이다. 이 dataset은 scientific papers, charts, slides, infographics, forms, handwritten notes, reports 등 7개 document VQA domain을 포함하고, 총 **367,608개의 question–document pairs**로 구성된다.

- **Synthetic Degradation Dataset**은 기존 VisRAG의 source를 기반으로 만들며, blur, noise, brightness variation, color saturation change, resolution reduction 등 12개 degradation type을 5개 severity level로 적용한다. 질문과 정답은 그대로 두고 document image만 degrade하기 때문에, clean image와 degraded image에서 모델이 얼마나 흔들리는지 직접 비교할 수 있다.
- **Real Degradation Dataset**은 실제 촬영 환경에서 생기는 degradation에 대한 generalization을 평가하기 위한 test-only subset이다. 문서를 출력한 뒤 카메라로 촬영하면서 blur, low light, low resolution, shadow, paper damage 같은 5개 real degradation type을 만든다. 이 부분은 학습에는 사용하지 않고 test에만 사용하므로, RobustVisRAG가 synthetic distortion에만 맞춰진 것이 아니라 real-world degradation에도 일반화되는지 확인하는 역할을 한다.

# 4. Experiments
## 4.1. Retrieval Performance
<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.29%5DRobustVisRAG/figure4.png?raw=true">
</p>

RobustVisRAG는 retrieval setting에서 모든 dataset 조건의 최고 성능을 기록하며, 특히 degraded document image에서도 검색 embedding이 안정적으로 유지됨을 보여준다. Table 1에서 가장 중요한 real degradation 기준 최고값은 RobustVisRAG의 **MRR@10 63.82**이다. 이는 기존 VisRAG-Ret이나 VisRAG-Ret-FM (FARE)보다 높은 결과로, 단순 fine-tuning이나 adversarial robustness training보다 semantic–degradation disentanglement가 real-world distortion에 더 효과적임을 의미한다. 즉, 이 실험은 RobustVisRAG의 causal path가 degraded image에서도 query와 관련된 문서 의미를 더 잘 보존한다는 것을 보여준다.

## 4.2. Generation Performance
<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.29%5DRobustVisRAG/figure5.png?raw=true">
</p>

RobustVisRAG는 generation setting에서도 가장 높은 성능을 보이며, retrieval이 맞더라도 degraded visual evidence를 해석하는 능력이 중요하다는 점을 보여준다. Table 2에서 가장 큰 값은 real degradation의 Oracle setting에서 RobustVisRAG가 기록한 **Accuracy 69.03**이다. Oracle은 정답 문서가 주어진 상태이므로 retrieval error가 제거된 setting인데, 이 조건에서도 RobustVisRAG가 가장 높다는 것은 성능 향상이 retriever 개선만이 아니라 generation-side visual 인코더의 degradation-invariant semantic encoding에서도 발생한다는 의미이다. 즉, RobustVisRAG는 흐릿하거나 어두운 문서를 generator가 읽을 때도 clean semantic feature에 가까운 표현을 전달한다.

## 4.3. End-to-end Performance
<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.29%5DRobustVisRAG/figure6.png?raw=true">
</p>

End-to-end 결과는 RobustVisRAG의 retrieval 개선이 generation output까지 실제로 이어진다는 점을 보여준다. Table 3에서 real degradation의 end-to-end generation 기준 최고값은 RobustVisRAG의 **Top-1 Accuracy 55.39**이다. 이는 retrieval과 generation을 따로 잘하는 수준이 아니라, degraded image에서 잘못된 문서를 찾는 문제와 올바른 문서를 잘못 읽는 문제를 함께 줄였다는 뜻이다. 특히 Two-Stage restoration baseline이 제한적인 성능을 보인다는 점은, 이미지 품질을 먼저 복원하는 방식보다 representation 수준에서 semantic factor와 degradation factor를 분리하는 방식이 downstream VisRAG에는 더 직접적이라는 것을 의미한다.

## 4.4. Ablation Study
<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.29%5DRobustVisRAG/figure7.png?raw=true">
</p>

Ablation study는 RobustVisRAG의 성능 향상이 단일 요소가 아니라 unidirectional Non-Causal Path, NCDM, CSA가 함께 작동한 결과임을 보여준다. Table 4에서 real degradation generation 기준 최고값은 full RobustVisRAG의 **Top-1 Accuracy 55.39**이다. Unidirectional constraint를 제거하거나, $$\mathcal{L}_{\text{NCDM}}$$ **또는 $$\mathcal{L}_{\text{CSA}}$$를 제거하면 모두 성능이 낮아지므로, non-causal token을 단순히 추가하는 것만으로는 충분하지 않다. 특히 **가장 큰 성능 gain은 $$\mathcal{L}_{\text{CSA}}$$**이다.

<br/>
<br/>

# 5. Conclusion
**Contributions**
- 저자는 degraded visual document 환경에서 VisRAG의 retrieval–generation error propagation을 줄이기 위해 **causality-guided dual-path framework**를 제안한다. 핵심 차이는 visual encoder 내부에서 **semantic factor와 degradation factor를 분리하고, inference cost를 추가하지 않는다는 점**이다.
- 저자는 synthetic degradation과 real-world degradation을 모두 포함하는 **Distortion-VisRAG benchmark**를 구축한다. 이 dataset은 retrieval, generation, end-to-end robustness를 degraded visual condition에서 함께 평가할 수 있도록 설계되었다.

**Limitations**
- **[Visual Encoder 중심 Adaptation의 한계]** RobustVisRAG는 generator의 language model은 frozen하고, retrieval-side vision encoder와 generation-side vision encoder를 causality-guided objective로 fine-tuning한다. 이 구조는 inference cost를 늘리지 않는 장점이 있지만, 오류의 원인이 visual representation이 아니라 generator의 reasoning, evidence selection, multi-document synthesis 능력에 있을 경우에는 한계가 있다. 즉, degraded image robustness는 개선할 수 있지만, 복잡한 multi-hop reasoning이나 답변 생성 단계의 오류까지 직접적으로 학습하는 방식은 아니다.
