---
title: "[논문리뷰]LoRA-Mixer: Coordinate Modular LoRA Experts Through Serial Attention Routing(arXiv, 2025)"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2025-12-26
last_modified_at: 2025-12-26
---

*Wenbing Li, Zikai Song, Hang Zhou, Yunyao Zhang, Junqing Yu, and Wei Yang*. 2025. [LoRA-Mixer: Coordinate Modular LoRA Experts Through Serial Attention Routing](https://arxiv.org/abs/2507.00029). arXiv:2507.00029 [cs.LG] https://arxiv.org/abs/2507.00029

# 1. Problem Statement
본 논문이 풀고자 하는 핵심 문제는 여러 개의 LoRA모듈을 **모듈형 전문가(expert)로 보고, 입력(토큰/문장)의 의미에 따라 적절한 LoRA 전문가 조합을 동적으로 선택·융합**하여 멀티태스크/멀티도메인 성능을 높이는 것이다. 기존의 LoRA는 단일 태스크에서는 효율적이지만 태스크 특이성 때문에 멀티태스크 일반화에 한계가 있고, 여러 LoRA를 단순 합성하면 태스크별 서브스페이스 간 간섭(interference)로 시너지가 제한된다는 문제의식에서 출반한다. 결론적으로 LoRA-Mixer가 풀려고 하는 문제는 크게 세 가지이다.

1. 다양한 백본(Transformer뿐 아니라 SSM까지)에서 동작 가능한 구조-비종속(architecture-agnostic) LoRA-MoE 구성
2. 최소한의 추가 데이터로도 태스크 인지적(task-aware) 라우팅을 학습
3. LoRA 모듈의 재사용성과 파라미터 효율성을 유지

이를 위해 attension/SSM의 핵심 경로를 우회하는 병렬 브랜치가 아니라, attention 모듈의 projection layer 자체에 LoRA expert를 직렬로 삽입하고, 라우터로 동적 선택을 수행하는 방식을 제안한다.

<br/>
<br/>

# 2. Limitations of Existing Works
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.12.26%5DLoRA-Mixer/figure1.png?raw=true">
</p>

- **[Switch-Expert 방식의 학습/데이터 요구량 증가 및 LoRA 재사용성 저하]** 기존의 vanilla MoE 패러다임 계열은 attention/FFN 레이어 자체를 switch expert로 치환하는 방식이라, 모든 **expert를 함께 joint training**해야 하는 경향이 강하고 그 결과 훈련 데이터 요구량이 커지며, 사전 학습된 LoRA의 모듈 재사용·전이(transferability)가 제한된다.
- **[병렬 LoRA-branch 융합의 단순 결합 문제 및 핵심 메커니즘 미활용]** 다른 LoRA 사용 방식인 병렬 브랜치로 LoRA expert를 붙인 뒤 **출력만 fuse**하는 방식은 attention 또는 **state transition(=state space models(SSM)) 메커니즘을 "우회(branch)"**하게 되어, 결과적으로 출력 융합이 단순해지고 통합이 최적이 아니며, 태스크별 지식이 모델의 핵심 표현학습 경로에 충분히 반영되기 어렵다.
- **[라우팅 최적화의 "균등 사용 강제"로 인한 태스크 인지력 약화]** 라우팅을 위해 흔히 쓰이는 auxiliary load-balancing loss가 expert를 균등하게 사용하도록 강하게 유도하여, 입력 의미와 무관하게 expert 사용이 평준화되는 문제가 발생한다. 이는 세밀한 task-awareness를 떨어뜨리고, 효과적인 라우팅을 위해 더 많은 데이터가 필요해지는 병목으로 이어진다.

<br/>
<br/>

# 3. Methodology
## 3.1. Preliminaries
- **Key Point:** <span style="color:gold">**Projection Layer에 LoRA expert를 직렬로 삽입**</span>

LoRA-Mixer의 핵심 아이디어는 LoRA를 병렬 브랜치로 붙여서 나중에 합치는 방식이 아니라, Transformer의 **Attention 또는 SSM(State-Space Model) 블록이 실제로 사용하는 선형 projection layer에 LoRA expert 혼합을 직렬**로 끼워 넣는 것이다. $$E$$개의 LoRA expert와 라우터 $$\alpha(\mathbf{x})\in \mathbb R^E$$를 사용해 projection 행렬 $$W \in \mathbb{R}^{d_{out} \times d_{in}}$$을 업데이트한다. 각 expert는 다음과 같이 정의된다.

$$
\Delta W^{(e)} = A^{(e)}B^{(e)}
$$

이 때, $$A^{(e)} \in \mathbb{R}^{d_{out} \times r}$$, $$B^{(e)} \in \mathbb{R}^{r \times d_{in}}$$으로 정의된다. 최종적으로 projection 입력 토큰 표현 $$\mathbf{x}\in \mathbb{R}^{d_{in}}$$에 대해, 기본 projection 출력 $$W\mathbf{x}$$에 expert들의 low-rank 업데이트를 라우팅해 더한 출력이 최종 y가 된다.

$$
y = W\mathbf{x} + F_{route}\Big( \{\alpha_e(\mathbf{x}) \cdot \Delta W^{(e)}\mathbf{x} \}_{e=1}^E \Big)
$$

여기서 중요한 점은 이 **$$y$$가 그대로 다음 attention 모듈 또는 SSM 모듈로 전달되어, 모델의 핵심 표현학습 경로를 직접 바꾼다**는 것이다

- $$y$$가 곧바로 Q/K/V/O projection 이후의 어텐션 계산으로 들어가거나,
- SSM에서 selective scan 전후의 선형변환 출력으로 들어감

즉, LoRA-Mixer는 branch(=우회)로 결과를 합치는 것이 아니라, <span style="color:gold">**projection layer라는 가장 표현력이 큰 지점에서 출력을 바꿔서 이후의 attention/SSM 연산 전체에 영향을 주도록 설계**</span>되어 있으며, 이 때문에 아키텍처를 깨지 않으면서도 효과적으로 적용된다.


## 3.2. LoRA-Mixer for Compositing LoRAs
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.12.26%5DLoRA-Mixer/figure2.png?raw=true">
</p>

- **Key Point: 외부 LoRA 재사용 + Hard-routing 기반 joint training 지원**

LoRA-Mixer의 expert는 새로 학습해야만 하는 것이 아니라, **서로 다른 출처에서 준비된 LoRA 모듈을 그대로 expert로 가져와 조합할 수 있게 설계**되어 있다. 논문이 제시하는 acquirement 시나리오는 크게 3가지이며, 이 3가지를 지원하는 것이 LoRA-Mixer의 efficiency contribution중 하나이다.

- Public repositoy의 사전학습 LoRA를 그대로 Expert로 재사용
- 사용자 데이터로 도메인 특화  LoRA를 각각 독립적으로 학습한 뒤, 이를 expert로 가져와 조합
- Hard-routing 기반 joint training을 지원. 여러 LoRA expert를 한꺼번에 공동 최적화해야 하는데 데이터가 heterogeneous하고 라벨이 있는 경우를 타깃으로 함.

### 3.4. Specialization Balance Loss for Routing Optimization

- **Key Point:** <span style="color:gold">**라우팅 기반 최적화 - RSL(Route-Specialization Balance Loss) + Expert Preservation**</span>

논문에서는 기존 auxiliary loss가 과도한 균등 사용을 만들어 입력 인지 라우팅을 방해한다고 보고, 균형(balance)과 특화(specialization)를 동시에 목표로 하는 RSL을 제안한다.

**RSL (Route-Specialization Balnce Loss)**  
$$\bar p_i$$를 토큰들에 대한 평균 soft route score, $$\bar f_i$$를 top-$$k$$ route에서 expert $$i$$에 실제 할당된 토큰 비중을 정규환 값으로 정의할 때, RSL은 다음과 같이 정의된다.

<center>$$\mathcal L_{\text{RSL}} = \alpha \cdot \displaystyle\sum_{i=1}^K \bar p_i \cdot \bar f_i - \lambda \cdot \mathbb{E}_{x \sim \mathcal D} [\mathcal H (p(\mathbb{x}))]$$</center>

<center>$$\mathcal H (p(\mathbb{x})) = - \displaystyle\sum_i p_i(\mathbf x)\log p_i (\mathbf{x})$$</center>

이 때, $$\mathcal H (p(\mathbb{x}))$$는 엔트로피이며 $$\alpha$$와 $$\lambda$$는 각각 balance와 entropy regularizer의 영향력을 반영하는 하이퍼파라미터이다. RSL의 첫 항 $$\alpha \cdot \displaystyle\sum_{i=1}^K \bar p_i \cdot \bar f_i$$ 은 <span style="color:gold">**의도한 라우팅 확률과 실제 사용량의 일치를 유도해 전역 균형**</span>을 잡고, 둘째 항 $$\mathbb{E}_{x \sim \mathcal D} [\mathcal H (p(\mathbb{x}))]$$은 <span style="color:gold">**엔트로피 정규화를 통해 입력별로 더 선택적인 라우팅을 유도하여 무의미한 균등 활성화를 완화**</span>한다.

**Expert Preservation Regularization**  
학습된 expert 지식이 라우터 학습(또는 joint 과정)에서 오염되는 것을 막기 위해, 일부 expert 집합 $$C$$에 대해 초기 파라미터 $$\theta_i^{(0)}$$에서 멀어지는 것을 벌점으로 둔다.

<center>$$\mathcal{L}_{\text{preserve}} = \beta \cdot \displaystyle\sum_{i \in C}\Big\vert \Big\vert \theta_i - \theta_i^{(0)} \Big\vert \Big\vert^2 = \beta \cdot \displaystyle\sum_{i \in C}\Big\vert \Big\vert \Delta \theta_i \Big\vert \Big\vert^2$$</center>

최종적으로 학습 Loss는 다음과 같다.

<center>$$\mathcal L_{\text{total}} = \mathcal L_{\text{task}} + \alpha +\cdot \mathcal{L}_{\text{RSL}} + \beta \cdot \displaystyle\sum_{i \in C}\Big\vert \Big\vert \theta_i - \theta_i^{(0)} \Big\vert \Big\vert^2$$</center>

학습 시에는 라우터가 모든 expert에 대해 softmax 점수를 출력하고 **soft fusion**으로 완전 미분 가능하게 학습 안정성을 확보한다. 반면 추론 시에는 **Top-3 sparse routing**을 사용하여 계산량을 줄인다. 즉 "학습은 dense/soft로 안정화, 추론은 sparse/top-k로 효율화"라는 분리 전략을 취한다.

<br/>
<br/>

# 4. Experiments
## 4.1. Main Results (1)
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.12.26%5DLoRA-Mixer/figure3.png?raw=true">
</p>

**Table 1**은 LoRA나 MoE 계열 기법을 적용하기 전, 세 가지 베이스 모델(Falcon-Mamba-7B, Mistral-7B, LLaMA3-8B)의 순수 기본 성능을 7개 벤치마크(Medical, CoLA, SST2, GSM8K, ARC-E, ARC-C, HumanEval)에서 비교한다. 

전체적으로 **LLaMA3-8B가 대부분의 벤치마크에서 가장 강한 베이스라인**을 보이며(예: Medical 78.47, GSM8K 57.92, ARC-E 88.45, ARC-C 78.65, HumanEval 52.44), Falcon-Mamba-7B는 CoLA/SST2 같은 NLU 계열에서 준수하지만(HumanEval 29.29, GSM8K 52.54 등) 코딩/수학 추론에서 LLaMA3-8B보다 약하고, Mistral-7B는 전반적으로 세 모델 중 가장 낮은 수치를 보인다(예: Medical 66.32, GSM8K 40.83, HumanEval 27.95).

## 4.2. Main Results (2)
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.12.26%5DLoRA-Mixer/figure4.png?raw=true">
</p>

**Table 2**는 동일한 7개 벤치마크에서 LoRAHub/MoLE/MixLoRA/LoRA/LoRA-Mixer를 비교한 메인 결과로, **LoRA-Mixer가 세 베이스 모델 모두에서 전 태스크 최고 성능을 달성**한 것을 볼 수 있다. 

- Falcon-Mamba(SSM 계열)에서는 MixLoRA가 Transformer 전용이라 제외된 조건에서, LoRA-Mixer가 Medical 78.01, CoLA 85.91, SST2 95.76, GSM8K 57.87, ARC-E 86.87, ARC-C 77.19, HumanEval 35.37로 LoRAHub/MoLE/단일 LoRA를 모두 상회함.
- Mistral에서는 LoRA-Mixer가 특히 CoLA 82.17, SST2 95.16, HumanEval 36.76처럼 “언어 이해/코딩” 축에서 상승폭이 크고, LLaMA3에서도 Medical 81.55, GSM8K 65.53, HumanEval 57.32 등 전 지표에서 다른 방법(MoLE, MixLoRA, LoRA 포함)보다 높다.
- 단순히 여러 LoRA를 결합하는 것보다는 입력 의미에 맞춘 동적 expert조합 (LoRA-Mixer)방식이 일관되게 좋음을 알 수 있음.

## 4.3. Parameter Transferability
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.12.26%5DLoRA-Mixer/figure5.png?raw=true">
</p>

Table 5의 Parameter Transferability 실험은 **Mistral-7B에서 학습한 LoRA-Mixer의 expert 및 라우팅 파라미터를 LLaMA3-8B로 추가 파인튜닝 없이 그대로 이식했을 때 성능이 얼마나 유지/개선되는지**를 보는 것으로, 두 모델이 동일한 아키텍처라는 점을 근거로 직접 마이그레이션한 뒤 ARC-E, ARC-C, GSM8K에서 평가한다. 

결과적으로 LLaMA3-8B 베이스라인 대비 이식 모델(+Mistral)은 GSM8K에서 0-shot 57.92→59.13(+2%), 2-shot 75.88→76.26(+1%), 5-shot 78.64→81.43(+4%)로 일관되게 상승했고, ARC-C도 78.65→79.14(+1%)로 소폭 개선된 반면, ARC-E는 88.45→85.89(−3%)로 하락하여 태스크에 따라 전이 이득이 달라짐을 보여준다. 저자들은 이를 종합해 3개 중 2개 태스크에서 베이스라인보다 좋아졌다고 해석하며, 이러한 크로스모델 마이그레이션 결과가 **LoRA expert와 학습된 routing function이 특정 베이스 모델에 강하게 결합되지 않아서, 같은 아키텍처 내에서는 expert 공유가 가능함**을 뒷받침한다고 주장한다.

<br/>
<br/>

# 5. Conclusion
**Limitations**  
- Loss를 제안했음에도 불구하고 loss를 하나씩 제거했을 때 성능을 비교하는 ablation study가 없음.
- 추론시에는 3개의 expert를 라우팅해서 사용한다고 했는데, 이는 여전히 입력이 모호할 경우 유연한 expert 선택에 제약이 있음

**Contributions**  
- 외부 LoRA 재사용 + 소량 데이터로 라우터 학습이 가능한 형태로, LoRA 생태계(LoRAHub 등)를 모듈 조합 기반의 실용 파이프라인으로 확장
- projection layer에 LoRA expert를 **직렬 삽입**하고 입력 기반 라우팅으로 expert를 조합하는 새로운 구조를 제안
