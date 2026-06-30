---
title: "[논문리뷰]How Retrieved Context Shapes Internal Representations in RAG (ACL findings, 2026)"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2026-06-16
last_modified_at: 2026-06-16
---

*Samuel Yeh and Sharon Li*. 2026. **[How Retrieved Context Shapes Internal Representations in RAG](https://arxiv.org/abs/2602.20091v1)**. In Findings of the Association for Computational Linguistics: ACL 2026. Association for Computational Linguistics.

# 1. Problem Statement
이 논문의 task는 **RAG 내부표현 분석(representation-level analysis of RAG)**이다. 기존 QA 성능을 높이는 새로운 RAG 모델을 제안하는 논문이라기보다, retrieved context가 LLM의 hidden states를 어떻게 변화시키고, 그 변화가 정답 생성·오답·abstention으로 어떻게 이어지는지를 분석하는 연구이다.


<br/>
<br/>

# 2. Limitations of Exsiting Works

이 논문에서 지적하는 핵심 문제는 <span style="color:red">**retrieved context가 유용한 evidence인지, semantic하게 비슷하지만 misleading한 noise인지, 완전히 무관한 random context인지에 따라 LLM 내부에서 전혀 다른 representation dynamics가 발생하지만, 기존 output-level 평가만으로는 이를 구분할 수 없다는 점**</span>이다. 예를 들어 accuracy가 떨어졌을 때 그것이 evidence integration 실패인지, parametric knowledge 억제인지, uncertainty/refusal mode 진입인지 알기 어렵다. 따라서 이 논문은 RAG를 output behavior가 아니라 last prompt token hidden state의 layer-wise 변화로 분석한다.

- **[출력 중심 RAG 분석의 한계]** 기존 RAG 분석은 retrieval strategy, context composition, hallucination rate, accuracy 같은 최종 출력 중심 지표에 집중해 왔다. 그러나 output-level 관찰만으로는 모델이 retrieved evidence를 실제로 통합했는지, parametric knowledge를 우선했는지, 혹은 불확실성 때문에 abstain/refuse했는지를 분리해 설명할 수 없다. 이 논문은 이 한계를 해결하기 위해 hidden representation을 직접 분석 대상으로 삼는다.
- **[현실적 retrieval mixture 분석의 부족]** 실제 RAG 환경의 retrieved document set은 relevant evidence, semantically similar but misleading distracting document, irrelevant random document가 섞여 있다. 기존 연구는 noisy context가 generation quality를 떨어뜨린다는 결과를 보였지만, 이 heterogeneous context가 LLM 내부표현을 어떻게 바꾸는지는 충분히 분석하지 않았다.
- **[기존 hidden representation 연구의 RAG 특화성 부족]** LLM hidden state 연구는 uncertainty, hallucination, knowledge conflict detection 등에 internal state를 활용해 왔지만, 이는 hidden representation을 downstream detector의 feature로 쓰는 방식에 가깝다. 반면 이 논문은 hidden representation 자체를 분석 도구로 사용하여, relevant/distracting/random document가 내부 처리 과정에서 어떻게 분기되는지를 체계적으로 본다.

<br/>
<br/>


# 3. Methodology
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.16%5DInternalRAG/figure0.png?raw=true">
</p>

## 3.1. Definition
이 논문은 RAG를 query $$q$$, retriever $$r$$, retrieved set $$S_q$$, instruction prompt $$I$$, LLM $$p_\theta$$로 정의한다. Retriever는 데이터베이스 $$D$$에서 $$N$$개의 문서 $$S_q = \{ d_1, \ldots, d_N\} \in D$$를 가져온다.

<center>$$\hat y \sim p_\theta(Y \mid I, q, S_q)$$</center>

검색 이후 위의 식처럼 지시문 ($$I$$), 쿼리 ($$q$$), 검색된 문서들 ($$S_q$$)가 LLM의 입력되고, 정답을 생성하게 된다. 이 때, 검색된 문서는 세 가지 문서 분포의 혼합에서 샘플링된 것으로 본다.

<center>$$d_i \sim \alpha_1 P_{\text{rel}} + \alpha_2 P_{\text{dist}} + \alpha_3 P_{\text{rand}} \\ \alpha_1, \alpha_2, \alpha_3 > 0, \quad \alpha_1+\alpha_2+\alpha_3 = 1$$</center>

$$P_{\text{rel}}$$은 ground truth에 해당하는 문서 분포이고, $$P_{\text{dist}}$$는 쿼리와 유사도는 높지만 정답 문서는 아닌, 즉 hard negative에 해당하는 distractor 문서 분포이다. 마지막으로 $$P_{\text{rand}}$$는 쿼리와 유사도가 낮아 쉽게 구분되는, 즉 answer derivation에 도움이 되지 않는 easy negative 문서들의 분포이다.

논문에서는 분석을 위해 사용하는 representation을 다음과 같이 정의한다. $$h^{q, S_q} \in \mathbb{R}^{L \times D}$$는 쿼리와 검색된 문서들이 주어졌을 때, 모든 $$L$$개의 트랜스포머 레이어에서 마지막 토큰의 hidden states를 모은 것이다. 논문은 특히 마지막 레이어의 마지막 토큰의 representation인 $$h_{-1}^{q, S_q}$$를 PCA와 코사인 유사도 분석에 집중한다.

## 3.2. Analysis Settings
### 3.2.1. Datasets and LLM backbones
- **Dataset**: Trivia QA, NQ, PopQA, StrategyQA
- **LLM**: Gemma3-27B, Llama4-17B, Qwen3-Next-80B

Retrieval database는 약 1.4T tokens 규모의 MassiveDS이고, **Contriever**를 사용해 각 쿼리마다 상위 20개의 문서를 검색한다.

### 3.2.2. Representation Analysis setting
분석을 하기 전, 쿼리의 난이도를 easy와 hard 두 개로 분류한다. LLM에게 쿼리 $$q$$만 입력으로 주고 retrieval 없이 문제를 맞히면 easy, 틀리면 hard로 레이블링한다. 

Multiple-document setting의 목적은 현실적인 RAG처럼 여러 문서가 동시에 주어질 때 relevant evidence가 noise 사이에서 어떻게 표현되는지를 보는 것이다. 입력 prompt는 네 개 document를 포함한다. 조건은 one relevant + three distracting, one relevant + three random, relevant-only baseline이다. 문서 순서는 positional bias를 줄이기 위해 random shuffle한다. 출력은 multi-context 조건별 hidden representation과 QA accuracy이다.

### 3.2.3. ### Effect of Relevant Documents on Uncertainty
Relevant 문서가 모델 confidence를 높이는지 보기 위해 length-normalized log-likelihood를 uncertainty score로 사용한다. 검색 없이 생성한 답변의 스코어는 아래와 같다.

<center>$$s_{\text{nodoc}} = \frac{1}{\vert \hat y \vert} \displaystyle\sum_{i=1}^{\vert \hat y \vert} \log \big( p_\theta(y_t \mid I, q, y_{<t}) \big)$$</center>

Relevant 문서를 포함해 생성한 답변의 스코어는 아래와 같다.

<center>$$s_{\text{rel}} = \frac{1}{\vert \hat y \vert} \displaystyle\sum_{i=1}^{\vert \hat y \vert} \log \big( p_\theta(y_t \mid I, q, S_q^{\text{rel}}, y_{<t}) \big)$$</center>

저자들은 $$s_\text{rel} > s_\text{nodoc}$$ 라는 가설을 세운다. 즉 relevant한 문서가 있으면 같은 정답을 생성하더라도 토큰의 log-likeligood가 높아져 confidence가 증가해야 한다.

### 3.2.4. Representations Separability Analyses
PCA visualization을 정량적으로 보강하기 위해 linear probe separability를 측정한다. Linear probe의 weight와 bias를  $$w, b$$, test representation을 $$\{ x_i\}_{i=1}^N$$라 할 때, decision boundary까지의 평균 거리는 다음과 같이 정의한다.

<center>$$d = \frac{1}{N} \displaystyle\sum_{i=1}^N \Big\vert \frac{w^\top x_i+b}{||w||_2} \Big\vert$$</center>

<br/>
<br/>

# 4. Experiments
## 4.1. Effect of Context Relevancy
이 섹션의 Research question은 relevant, distracting, random document가 query-only 상태 대비 LLM 내부표현을 어떻게 이동시키는가이다. 여기서 representation drift의 기준은 No Doc, 즉 검색된 문서를 넣지 않은 query-only baseline이다. 같은 쿼리 $$q$$에 대해 no-document 프롬프트의 마지막 레이어의 마지막 토큰 rerpesentation과, relevant/distracting/random 문서를 넣은 프롬프트의 representation을 비교한다. 예를 들어, “random document가 큰 drift를 유발한다”는 말은 random condition의 representation cluster가 No Doc cluster에서 가장 멀리 떨어져 있다는 뜻이다.

### Observation 1: Random Documents Induce Large Representation Drift
Figure 2는 <span style="color:red">**semantically dissimilar한 random document가 가장 큰 representation drift를 유발한다**</span>는 결과를 보여준다. Relevant와 distracting document는 대체로 no-document baseline 근처에 위치하지만, random document는 query-only representation에서 크게 떨어진 cluster를 형성한다. 직관적으로는 random document가 유용한 정보를 담고 있지 않으므로 representation 변화도 작을 것 같지만, 실제 결과는 반대이다. 논문은 이를 LLM이 uninformative context를 내부적으로 감지하고, 답변 생성을 진행하기보다 refusal/abstention mode로 이동한 결과로 해석한다.

논문은 처음에 직관적 기대를 제시한다. Relevant document는 모델의 parametric knowledge에 없는 정보를 제공할 수 있으므로, 특히 hard query에서 query-only representation을 크게 바꿀 것으로 예상할 수 있다. 반대로 random document는 query와 semantic similarity가 낮고 유용한 정보가 없으므로 representation 변화가 작을 것으로 예상할 수 있다. 그러나 Figure 2의 결과는 이 직관과 반대로 나타난다.

### Observation 1: Random Documents Induce Large Representation Drift
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.16%5DInternalRAG/figure1.png?raw=true">
</p>

Figure 2는 <span style="color:red">**random document가 No Doc baseline으로부터 가장 크게 벗어나는 representation drift를 유발한다는 것**</span>을 보여준다. PCA 공간에서 relevant와 distracting document는 대체로 No Doc cluster 근처에 위치하지만, random document는 No Doc에서 떨어진 별도 cluster를 형성한다. 이 판단의 기준은 No Doc baseline이며, Figure 2 자체는 PCA visualization이므로 시각적 관찰에 해당한다. 

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.16%5DInternalRAG/figure2.png?raw=true">
</p>

Figure 3은 Figure 2에서 관찰된 drift가 output behavior와 연결됨을 보여준다.  쿼리와 검색된 문서를 포함하여 만든 representation과의 유사도를 구하고, 응답을 correct, incorrect, abstain으로 분류한다. Cosine similarity가 낮다는 것은 No Doc 기준 representation에서 더 멀리 이동했다는 뜻이며, 이 경우 abstain response가 증가한다. 따라서 random context의 large drift는 단순한 embedding 변화가 아니라, 모델이 uninformative context를 감지하고 refusal/abstention mode로 전환되는 현상과 연결된다.

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.16%5DInternalRAG/figure3.png?raw=true">
</p>

Figure 4와 Table 1은 이 현상이 instruction tuning에 의해 크게 증폭됨을 보여준다. Gemma3-27B Trivia QA 기준으로 instruction-tuned model은 random context에서 easy query도 97.6%, hard query도 98.1% abstain한다. 반면 base model은 random context easy query에서 89.4% correct를 유지하고 abstain은 3.8%에 그친다. 따라서 instruction tuning은 unsupported answer를 줄이는 데는 도움이 되지만, irrelevant context가 들어오면 모델이 본래 parametric knowledge로 맞힐 수 있는 easy query까지 과도하게 abstain하게 만들 수 있다.

### Observation 2: Relevant Documents Largely Preserve Internal Representations
Figure 2는 **relevant document가 No Doc representation에서 크게 벗어나지 않는다는 것**도 보여준다. Relevant document condition은 No Doc condition과 가까운 위치에 남는다. Easy query에서는 이 결과가 자연스럽다. 모델이 retrieval 없이도 답을 알고 있으므로, relevant document는 representation을 새로운 영역으로 이동시키기보다 기존 parametric knowledge를 확인하는 confirmation signal로 작동한다. 

그러나 hard query에서는 이 결과가 RAG의 한계를 드러낸다. Table 1에서 Gemma3-27B base model은 relevant document가 제공되어도 hard query에서 35.6% incorrect를 보인다. 즉 relevant evidence가 있더라도 final representation이 충분히 evidence-driven하게 이동하지 못한다. 논문은 이를 parametric knowledge가 부족한 상황에서 retrieved evidence가 모델 내부표현을 충분히 주도하지 못하는 문제로 해석한다.

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.16%5DInternalRAG/figure4.png?raw=true">
</p>

논문은 이 해석을 보강하기 위해 불확실성 (Uncertainty) 스코어를 추가로 도입한다. Relevant document가 representation을 크게 이동시키지는 않지만, easy query에서는 generated response의 log-likelihood를 높여 confidence를 증가시킨다. 따라서 relevant document의 주된 효과는 hard query에서 새로운 정보를 강하게 주입하는 것이라기보다, easy query에서 기존 지식과 일치하는 evidence를 제공해 모델의 확신을 높이는 데 가깝다.

### Observation 3: A single relevant document stabilizes representations in multi-document settings
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.16%5DInternalRAG/figure5.png?raw=true">
</p>

Figure 5는 <span style="color:red">**multi-document setting에서 relevant document 하나가 representation anchor 역할을 한다**</span>는 것을 보여준다. 논문은 one relevant + three distracting, one relevant + three random 조건을 relevant-only baseline과 비교한다. 그 결과, relevant document가 포함된 multi-document representation은 relevant-only representation과 가깝게 유지된다. 즉 distracting 또는 random document가 함께 들어가도 reliable evidence 하나가 내부표현을 안정화한다.

<p align="center">
<img width="450" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.16%5DInternalRAG/figure6.png?raw=true">
</p>

Table 2는 이 representation stability가 accuracy에도 반영됨을 보여준다. Relevant document가 포함된 조건은 distracting-only 또는 random-only보다 훨씬 높은 accuracy를 유지한다. 예를 들어 Gemma3-27B Trivia QA easy에서 relevant-only는 90.4%, +distracting은 82.6%, +random은 87.7%인 반면, distracting-only는 8.4%, random-only는 1.7%이다. 따라서 LLM은 reliable grounding이 하나라도 존재하면 additional noise를 어느 정도 suppress할 수 있다.

## 4.2 Effect of Layer-wise Process
이 섹션에서 말하는 research question은 **retrieved context의 영향이 transformer layer를 지나며 어떻게 형성되는가**이다. 이 섹션에서는 여러 layer의 last prompt token representation을 PCA로 시각화하여, random/relevant/distracting context가 어느 layer에서 분리되고, later layer에서 어떻게 조정되는지 분석한다.

<p align="center">
<img width="450" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.06.16%5DInternalRAG/figure7.png?raw=true">
</p>

Figure 6은 <span style="color:red">**random document가 relevant/distracting document보다 먼저 분리된다는 것**</span>을 보여준다. Llama4-17B Trivia QA 기준으로 12번째 레이어 (L12)에서는 relevant, distracting, random document representation이 많이 겹친다. 그러나 L23 이후 random document representation이 별도 cluster로 분리되기 시작한다. 이는 query와 context 사이의 coarse semantic mismatch를 LLM이 비교적 early layer에서 감지할 수 있음을 의미한다.

반면 relevant와 distracting document는 더 깊은 layer까지 강하게 얽혀 있다. Distracting document는 query와 semantic similarity는 높지만 answer를 지지하지 않는 hard negative이므로, random document처럼 단순한 semantic mismatch로 구분하기 어렵다. 논문은 relevant와 distracting document를 구분하려면 higher-level semantic reasoning과 parametric knowledge integration이 필요하며, current LLM은 final layer에서도 이 둘을 완전히 분리하지 못한다고 해석한다.

### Observation 5: Later layers close the gap between no-document and relevant-document representations
Figure 6의 또 다른 결과는 <span style="color:red">**later layer에서 relevant-document representation이 No Doc representation 쪽으로 다시 가까워진다는 것**</span>이다. Early/middle layer에서는 document가 있는 condition과 No Doc condition이 멀리 떨어져 있지만, L35 이후 relevant context representation이 No Doc representation에 가까워지고 L46에서는 상당히 근접한다.

이 현상은 later layers가 retrieved evidence를 parametric knowledge와 reconcile하면서, 최종 출력 직전에는 parametric knowledge의 영향이 다시 강해진다는 뜻이다. Easy query에서는 이 과정이 noise를 줄이고 generation을 안정화하는 데 도움이 된다. 그러나 hard query에서는 retrieved evidence가 더 강하게 반영되어야 하는데도 final representation이 parametric knowledge 쪽으로 돌아가므로, RAG 효과가 제한될 수 있다.

<br/>
<br/>

# 5. Conclusion
- **Observation 1: Random documents induce large representation drift.** Random document는 No Doc, 즉 query-only baseline을 기준으로 가장 큰 representation drift를 유발한다. 모델은 무관한 context를 단순히 무시하기보다, 내부적으로 답변 근거가 부족하다고 판단해 abstention/refusal mode로 이동하는 경향을 보인다. (Figure 2, 3, 4; Table 1)
- **Observation 2: Relevant documents largely preserve internal representations.** Relevant document는 representation을 크게 이동시키기보다 No Doc representation 근처에 머문다. 이는 easy query에서는 retrieved evidence가 새로운 정보를 주입하기보다 parametric knowledge를 확인하고 confidence를 높이는 confirmation signal로 작동하지만, hard query에서는 외부 evidence가 final representation을 충분히 바꾸지 못하는 한계로 이어진다. (Figure 2; Table 1, 5)
- **Observation 3: A single relevant document stabilizes multi-document RAG.** Multi-document setting에서는 relevant document 하나가 representation anchor 역할을 한다. Relevant evidence가 포함되면 distracting 또는 random document가 함께 들어가도 내부표현이 안정적으로 유지되고, 성능도 distracting-only/random-only 조건보다 훨씬 높게 유지된다. (Figure 5; Table 2, 3)
- **Observation 4: Random context is detected earlier than distracting context.** LLM은 완전히 무관한 random context는 비교적 이른 layer에서 감지하지만, query와 semantic similarity가 높은 distracting context는 relevant context와 late layer까지도 잘 분리하지 못한다. 즉 current LLM은 coarse noise에는 강하지만, hard negative retrieval noise에는 여전히 취약하다. (Figure 6; Figure 10-12; Table 9)
- **Observation 5: Later layers favor parametric knowledge.** Retrieved evidence의 영향은 intermediate layer에서 나타나지만, later layer로 갈수록 relevant-document representation이 다시 No Doc representation에 가까워진다. 이는 final generation 직전에는 external evidence보다 parametric knowledge의 영향이 강해지며, hard query에서 RAG 효과가 제한되는 원인이 된다. (Figure 6; Figure 13-14; Table 10)

RAG를 사용할 때는 **relevant document recall을 우선 확보하되, hard negative filtering과 evidence 활용 학습을 함께 강화해야 한다**. 이 논문은 relevant document 하나만 있어도 noisy context를 어느 정도 억제할 수 있음을 보이므로, 지나치게 aggressive한 filtering보다는 reliable evidence가 포함될 가능성을 높이는 retrieval breadth가 중요하다. 그러나 hard query에서는 단순히 문서를 많이 넣는 것만으로 부족하며, semantically similar distracting document를 구분하는 reranking/filtering이 필요하다. 

학습 측면에서는 later layer에서 retrieval signal이 parametric knowledge에 의해 약해지지 않도록 retrieval-aware decoding, evidence-fusion module, retrieval-sensitive representation subspace, 또는 external evidence 쪽으로 representation shift를 유도하는 objective가 필요하다. 또한 instruction tuning은 irrelevant context에서 abstention을 과도하게 강화할 수 있으므로, “context에 근거는 없지만 parametric knowledge로 답할 수 있는 경우”와 “정말 답하면 안 되는 경우”를 구분하도록 학습해야 한다.
