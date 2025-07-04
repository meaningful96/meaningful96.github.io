---
title: "[논문리뷰]EfficientRAG: Efficient Retriever for Multi-Hop Question Answering"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2025-07-04
last_modified_at: 2025-07-04
---

*Ziyuan Zhuang, Zhiyang Zhang, Sitao Cheng, Fangkai Yang, Jia Liu, Shujian Huang, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, and Qi Zhang*. “[**EfficientRAG: Effective Retriever for Multi-Hop Question Answering**](https://aclanthology.org/2024.emnlp-main.199/).” In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 3392–3411, Miami, Florida, USA, November 2024

# Problem Statement
이 논문에서는 기존의 Standard RAG와 Single-step Retrieval의 한계점을 네 가지로 분류한다.

**[Single-step retrieval의 비효율성]** 기존 RAG는 주로 1회성 단순 retrieval에 의존하여 multi-hop reasoning이 필요한 복잡 질문에 비효율적이다. 전통적인 one-round RAG 방법들은 단순한 질문에는 효과적이지만, 여러 단계의 추론이 필요한 multi-hop 질문에서는 첫 번째 검색으로 얻은 정보만으로는 답변이 불가능하다.

**[Multi-step retrieval의 비용 문제]** 매 retrieval 라운드마다 LLM 호출이 필요하여 지연시간 (latency)과 비용 (cost)이 증가한다. 또한 복잡한 프롬프트 설계 및 few-shot 예시가 필요하여 도메인 확장성이 저하된다.

**[Noisy chunks로 인한 정확도 저하]** Multi-hop QA에서는 필요없는 정보 (noisy chunks)로 인해 정확도가 저하된다. 무관한 청크들의 존재가 LLM 생성기에게 지속적인 도전을 제기한다.

**[LLM의 noisy context 취약성]** LLM도 noisy context에 취약해, 중간 reasoning 단계에서 오류가 발생할 가능성이 있다. LLM 기반 시스템들이 노이즈가 있는 지식 입력으로 부분 답변을 생성해야 하는 중간 단계에서 실패하는 경우가 많다.

<br/>
<br/>

# Methodology
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.07.04%5DEfficientRAG/1_Overview.png?raw=true">
</p>

- **EfficientRAG**는 RAG 프레임워크 내에서 작동하는 **경량(소형) 모델 기반 Query 재작성 및 필터링 시스템**이다.
- 주요 구성: **Labeler & Tagger**, **Filter**, 기존 Retriever, Generator(LLM)

**EfficientRAG**는 전통적인 RAG 시스템에 통합되는 plug-and-play 방식의 효율적인 retrieval 프레임워크다. 주어진 쿼리에 대해 retriever가 데이터베이스에서 관련 청크들을 검색하면, EfficientRAG의 두 가지 경량 컴포넌트인 Labeler & Tagger와 Filter가 작동한다.

## Labler & Tagger
- **입력**: (Query, Retrieved Chunk)
- **출력**:
    - Token 단위로 "useful"/"useless" 여부 라벨
    - Chunk 전체에 대해 <Continue> 또는 <Terminate> 태그 지정
- **세부 설명**:
    - Labeler는 chunk 내부의 중요 단어를 식별하여 다음 Query 생성에 활용
    - Tagger는 해당 chunk가 다음 hop 추론에 필요한지 (<Continue>) 아닌지 (<Terminate>) 판단
    - <Continue>로 분류된 경우, 해당 chunk 정보와 Query가 다음 hop을 위한 재작성 입력으로 사용됨

**Labeler**는 검색된 문서 내에서 쿼리에 부분적으로 답변할 수 있는 유용한 정보를 나타내는 토큰 시퀀스에 주석을 단다. **Tagger**는 검색된 청크가 도움이 되는지 무관한지를 나타내는 태그를 부여한다. 만약 태그가 더 많은 정보가 필요함을 나타내는 <Continue>라면, 해당 청크를 후보 풀에 추가하여 최종 LLM 기반 생성기에 전달한다. 반면 문서가 무용하거나 무관하다고 라벨링되면 해당 쿼리로부터의 후속 브랜치 검색을 중단한다.

## Filter
- **입력**: (현재 Query, Labeler가 식별한 중요 Tokens)
- **출력**: 다음 Retrieval을 위한 재작성 Query
- **설명**:
    - Query 내 미해결 부분을, 이전 Retrieval에서 얻은 중요 정보로 치환하여 새로운 Query 구성
    - LLM 호출 없이도 iterative Retrieval 가능하게 함

**Filter**는 라벨링된 토큰들과 현재 쿼리를 받아 다음 라운드 검색을 위한 새로운 쿼리를 구성한다. 이는 쿼리의 미지의 부분을 라벨링된 토큰들 (유용한 정보)로 대체함으로써 수행된다. 이 과정을 통해 초기 쿼리 범위를 넘어서는 정보를 검색하기 위한 새로운 쿼리를 효율적으로 생성한다.

## Synthetic Data Construction
- **LLM 기반으로 Labeler & Filter 학습 데이터 자동 생성**
- **단계:**
    1. Multi-hop 질문을 Single-hop으로 분해
    2. 각 Chunk의 중요 Token을 라벨링
    3. 다음 hop Query 생성 (불필요 부분 제거, 새로운 정보 반영)
    4. Hard Negative 샘플링: 비슷하지만 관련 없는 chunk는 <Terminate>로 분류

합성 데이터 구축은 네 단계로 구성된다. Multi-hop question decomposition에서는 LLM을 사용하여 multi-hop 질문을 여러 개의 single-hop 질문으로 분해하고 의존성을 파싱한다. Token Labeling에서는 각 sub-question과 해당 청크에 대해 **SpaCy 툴킷**을 사용하여 중요한 단어들을 이진 라벨로 주석한다. Next-hop question filtering에서는 single-hop 질문과 의존적 질문들의 라벨링된 토큰들을 기반으로 다음 홉 질문을 생성한다. Negative Sampling에서는 각 필터링된 next-hop 질문에 대해 유사하지만 관련 없는 청크를 hard negative로 검색한다.

<br/>
<br/>

# Experiments
##  Main Results
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.07.04%5DEfficientRAG/2_Table2.png?raw=true">
</p>

Table2는 Retrieval 성능 비교 실험 결과이다. EfficientRAG는 HotpotQA에서 81.84, 2WikiMQA에서 84.08의 높은 Recall@K를 달성했다. 특히 주목할 점은 EfficientRAG의 경우 다른 베이스라인과 비교했을 때 **검색한 청크 수가 매우 적다**는 것이다 (HotpotQA: 6.41개, 2WikiMQA: 3.69개). 다른 방법들이 14-35개의 청크를 검색하는 것에 비해 현저히 적은 수로도 비교 가능한 성능을 보였다. 하지만 MuSiQue 데이터셋에서는 49.51로 상대적으로 낮은 성능을 보였는데, 이는 검색된 청크 수가 적고 데이터셋의 복잡성이 높기 때문으로 분석된다.

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.07.04%5DEfficientRAG/3_Table3.png?raw=true">
</p>

Table 3는 End-to-end QA 성능 비교 실험 결과이다. EfficientRAG는 HotpotQA와 2WikiMQA에서 두 번째로 높은 정확도를 달성했다 (HotpotQA: 57.86%, 2WikiMQA: 53.41%). MuSiQue에서도 낮은 recall에도 불구하고 20.00%의 정확도를 보였다. LLM 기반 시스템들이 노이즈가 있는 지식 입력으로 부분 답변을 생성해야 하는 중간 단계에서 실패하는 반면, 더 도움이 되는 지식과 적은 무관한 청크가 RAG 시스템의 핵심 요소임을 보여준다.

<br/>
<br/>

## Ablation Study
### Inference Efficiency
<p align="center">
<img width="450" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.07.04%5DEfficientRAG/4_Table4.png?raw=true">
</p>

추론 효율성 실험에서 MuSiQue 데이터셋의 200개 샘플에 대해 EfficientRAG는 1.00번의 LLM 호출과 2.73번의 반복으로 3.62초의 지연시간을 보였다. 이는 Iter-RetGen iter3 (9.68초)와 SelfASK (27.47초)에 비해 각각 60%-80% 개선된 시간 효율성을 보여준다. GPU 사용률은 유사한 수준 (65.55%)을 유지하면서도 직접 검색 방법과 동등한 속도를 달성했다.

### Replacing Backbone of Generator
<p align="center">
<img width="450" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.07.04%5DEfficientRAG/5_Table5.png?raw=true">
</p>

다양한 생성기와의 성능 실험에서 GPT-3.5를 생성기로 사용했을 때 2WikiMQA에서 EfficientRAG는 61.88%의 최고 정확도를 달성했다. 이는 더 강력한 생성기가 EfficientRAG의 성능을 더욱 향상시킬 수 있음을 보여준다.

## Analysis
<p align="center">
<img width="450" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.07.04%5DEfficientRAG/6_Table6.png?raw=true">
</p>

전이가능성 실험에서 EfficientRAG는 도메인별 지식에 크게 의존하지 않는 강건한 적응성을 보여준다. HotpotQA로 학습하고 2WikiMQA로 테스트했을 때 56.59%의 정확도를 보였으며, 이는 원본 데이터셋으로 학습한 모델 (53.41%)보다도 높은 성능이다. 이는 EfficientRAG가 추가적인 downstream 학습 없이도 다양한 태스크 시나리오에 적응할 수 있는 유연성을 가지고 있음을 보여준다.

쿼리 분해 효과 실험에서 검색 효율성에 대한 실험에서 EfficientRAG Decompose는 약 20개의 청크로 LLM Decompose가 200개 청크로 달성하는 것과 비교 가능한 recall을 보였다. 이는 EfficientRAG의 분해 방식이 훨씬 효율적임을 입증한다.

<br/>
<br/>

# Conclusion
**Contribution**  
- EfficientRAG는 다중 LLM 호출 없이 새로운 쿼리를 반복적으로 생성할 수 있는 Multi-hop QA 전용 효율적 Retriever를 제안하였다.
- 기존 Iterative RAG 대비 매우 적은 수의 Chunk만으로 높은 Recall을 달성하며, 특히 HotpotQA(81.84@6.41개), 2WikiMQA(84.08@3.69개)에서 기존 SOTA 수준의 성능을 보였다.
- 기존 LLM 기반 Iterative RAG 대비 약 60~80%의 시간 효율을 개선하였으며, LLM 호출 횟수를 단 1회로 최소화하였다.
- 추가 학습 없이도 다양한 데이터셋에 적용 가능한 높은 Transferability를 보여주었다.

**Limitations**
- Labeler와 Filter 모두 별도의 사전 학습 과정이 필요하며, 해당 데이터 구축을 위해 여전히 LLM 기반 Synthetic Data가 필요하므로 완전한 LLM 독립 구조는 아니다.
- MuSiQue처럼 질문 복잡도가 높거나 추론 단계가 많은 데이터셋에서는 적은 수의 Chunk Retrieval로 충분한 정답 정보를 확보하기 어렵다.


