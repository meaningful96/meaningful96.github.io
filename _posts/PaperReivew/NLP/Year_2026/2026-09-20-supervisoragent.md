---
title: "[논문리뷰]Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems (ICLR, 2026)"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2026-09-20
last_modified_at: 2026-09-20
---

*Fulin Lin, Ruishan Fang, Hongwei Wang, Tao Lin, et al*. 2026. [**Stop wasting your tokens: Towards efficient runtime multi-agent systems**](https://arxiv.org/abs/2510.26585). In International Conferenceon Learning Representations, Vol. 2026. 113497–113528.

# 1. Problem Statement
<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.09.20%5DSupervisorAgent/figure1.png?raw=true">
</p>

이 논문은 **runtime Multi-Agent Systems (MAS) supervision 및 efficiency optimization** 문제를 다룬다. 기존 MAS의 architecture나 base agent를 변경하지 않고, execution 중 발생하는 error propagation, inefficient behavior, excessive observation을 실시간으로 감지하고 개입하는 **Supervised Multi-Agent System (SMAS)**을 구축하는 것이 목적이다.

- **입력**: SUPERVISORAGENT의 context window $$W=(N,Q_g,Q_l,T_l,S)$$,
    - 여기서 N은 agent name, $$Q_g$$와 $$Q_l$$은 global/local task, $$T_l$$은 local trace, S는 latest interaction summary이다.
    - Inefficient behavior 진단에서는 global trace $$T_g$$까지 포함한 $$W_{\mathrm{ext}}=W\cup\{T_g\}$$를 사용한다.
- **출력**: intervention context $$c$$와 이에 대응하는 supervision action $$a\in A(c)$$ 이다. Action은 `approve`, `provide_guidance`, `correct_observation`, `run_verification` 중 하나이다.
- **최종 목표**: proactive runtime supervision을 통해 MAS의 **robustness와 efficiency를 동시에 개선**하고, 특히 task success를 유지하면서 token consumption을 줄이는 것이다.

<br/>
<br/>

# 2. Limitations of Existing Works
- **[Reactive Supervision]** 기존 robustness 연구는 주로 task failure가 발생한 뒤 원인을 추적하는 **post-hoc failure attribution**에 집중한다. 따라서 잘못된 information이 downstream agent나 memory로 전파되기 전에 runtime에서 차단하거나 수정하지 못한다. 저자들은 MAS의 긴 interaction chain에서 하나의 misinformation이 이후 reasoning 전체를 오염시킬 수 있다고 지적한다.
- **[Long Observations]** 기존 agent는 verbose web page나 tool output과 같은 긴 observation을 그대로 context에 포함할 수 있다. 이는 token cost를 증가시키고 critical information을 noise 속에 묻어 agent가 task execution에서 이탈하도록 만들 수 있다.
- **[Inefficient Execution]** Agent는 repetitive action loop에 빠지거나 solution에 필요 이상의 복잡한 path를 선택할 수 있다. 명시적인 error가 아니더라도 이러한 sub-optimal behavior는 token과 computational resource를 지속적으로 낭비한다.

<br/>
<br/>

# 3. Methodology
<p align="center">
<img width="800" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.09.20%5DSupervisorAgent/figure2.png?raw=true">
</p>

SUPERVISORAGENT의 핵심 pipeline을 보여주는 Figure이다. (a) **Adaptive Filter**, (b) **Context Window**, (c) **Action Space**를 중심으로 구성되며, (d)-(e)는 baseline MAS와 SMAS의 GAIA 실행 사례, (f)는 `monitor → filter → context construction → supervision action`의 전체 workflow를 보여준다.

SUPERVISORAGENT는 기존 MAS interaction을 intercept하여 

- **What to Supervise**
- **When to Supervise**
- **How to Supervise**

세 단계로 runtime supervision을 수행한다. 

**(i) Agent-Agent, (ii) Agent-Tool, (iii) Agent-Memory interaction**을 모니터링하고, LLM-free Adaptive Filter가 intervention이 필요한 스텝만 선택한 뒤, 컨텍스트 윈도우를 기반으로 SUPERVISORAGENT가 적절한 action을 결정한다. 

## 3.1. What to Supervise: High-Risk Interaction Points
SupervisorAgent는 error와 inefficiency가 유입될 가능성이 높은 세 interaction channel을 supervision 대상으로 정의한다.

- **Agent-Agent Interactions**: agent 출력이 다른 agent의 입력으로 전달되면서 hallucinated 또는 erroneous information이 전파될 수 있다.
- **Agent-Tool Interactions**: external tool/API에서 factually incorrect, irrelevant, outdated data가 유입될 수 있다.
- **Agent-Memory Interactions**: short-/long-term memory에서 stale 또는 flawed information이 검색될 수 있다.

실제 구현에서는 각 interaction이 `ActionStep` object로 표현되며 `model_output`, `tool_calls`, `observations`, `error` 등을 포함한다. SUPERVISORAGENT는 callback으로 모든 `ActionStep`을 intercept한다.

## 3.2. When to Supervise: The Adaptive Filter
<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.09.20%5DSupervisorAgent/figure3.png?raw=true">
</p>

SupervisorAgent는 모든 interaction마다 개입하지 않고, **LLM-free Adaptive Filter**를 통해 supervision이 필요한 high-risk interaction만 선택한다. 모든 step에서 LLM을 호출하면 supervision 자체가 큰 token/latency overhead를 만들 수 있기 때문에, Figure 2(a)의 Adaptive Filter가 먼저 lightweight heuristic으로 현재 interaction을 검사한다.

Filter는 크게 세 가지 상황을 감지한다.

- **Error occurrence**: tool use나 code execution에서 explicit error가 발생한 경우 즉시 **Proactive Error Correction**을 trigger한다.
- **Inefficient behavior**: 동일하거나 비효율적인 action이 반복되거나 sub-task execution이 지나치게 길어지는 경우 **Guidance for Inefficiency**를 trigger한다.
- **Excessive observation length**: web page, raw HTML, tool output 등 observation이 지나치게 길어진 경우 **Adaptive Observation Purification**을 trigger한다.

어느 조건에도 해당하지 않으면 SUPERVISORAGENT를 호출하지 않고 현재 interaction을 그대로 통과시킨다. 따라서 Adaptive Filter의 핵심 역할은 **“어디에 개입할지”를 판단하는 것이 아니라, expensive LLM-based supervision이 필요한 시점만 선별**하는 것이다. 

실제 Smolagent 구현에서는 sub-agent가 final report를 반환하는 경우도 별도로 감지하여 해당 report를 manager agent에게 전달하기 전에 purification/synthesis를 수행한다.

## 3.3. How to Supervise: Memory-Augmented, Multi-Level Intervention
<p align="center">
<img width="400" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/PaperReview(2026)/%5B2026.09.20%5DSupervisorAgent/figure4.png?raw=true">
</p>

!image.png

Adaptive Filter가 high-risk interaction을 감지하면, SUPERVISORAGENT는 현재 step만 보는 것이 아니라 **task와 execution history를 함께 포함하는 context window**를 구성한 뒤 intervention을 결정한다. Figure 2(b)-(c)는 각각 이러한 Context Window와 Action Space를 보여준다.

Context Window에는 다음 정보가 포함된다.

- 현재 supervision 대상 **agent name**
- 전체 시스템이 해결해야 하는 **global task $$Q_g$$**
- 해당 agent가 수행 중인 **local task $$Q_l$$**
- agent의 최근 **local action trace $$T_l$$**
- 현재 interaction의 **summary $$S$$**

특히 inefficient behavior를 판단할 때는 모든 agent interaction을 포함하는 **global trace $$T_g$$**까지 추가한다. 이를 통해 SUPERVISORAGENT는 특정 agent의 반복 행동만 보는 것이 아니라, 전체 MAS 관점에서 현재 행동을 계속하는 것이 합리적인지 판단한다.

이 context를 기반으로 SUPERVISORAGENT는 issue의 종류와 severity에 따라 네 가지 action 중 하나를 선택한다.

- **`approve`**: 반복적인 행동이라도 실제로 task completion을 향해 진행 중이라면 개입하지 않음
- **`provide_guidance`**: observation 자체는 유지하면서 agent가 더 효율적인 path를 선택하도록 concise guidance를 추가
- **`correct_observation`**: observation에 오류가 있거나 excessive information이 포함된 경우 기존 observation을 refined version으로 교체
- **`run_verification`**: 현재 context만으로 오류를 해결하기 어려운 경우 별도의 verification sub-agent를 호출하여 external fact-checking이나 debugging을 수행

실제 intervention 과정은 high-level하게 **Context Aggregation → LLM-based Decision → Action Execution**으로 이루어진다. SUPERVISORAGENT의 backbone LLM은 aggregated context를 분석하여 적절한 action을 선택하며, framework는 그 결과에 따라 observation을 수정하거나 guidance 또는 verification result를 downstream agent에게 전달한다.

## 3.4. Adaptive Observation Purification

**Adaptive Observation Purification**은 SUPERVISORAGENT의 주요 efficiency mechanism으로, 지나치게 긴 tool output이나 sub-agent report가 downstream agent의 컨텍스트를 불필요하게 차지하지 않도록 observation을 정제한다. 이 stage의 목적은 단순 summarization이 아니라 **token count를 줄이면서 agent execution에 필요한 information과 structure를 최대한 보존하는 것**이다.

논문은 purification을 두 가지 상황에서 사용한다.

- **첫째, excessive tool observation을 처리한다.** Raw HTML이나 긴 text observation이 threshold를 넘으면 `correct_observation`을 통해 압축된 observation으로 대체한다. 이 compressor는 **Context-Agnostic**하게 설계되어 agent의 global goal이나 previous action을 사용하지 않고, potentially useful information을 폭넓게 보존하도록 한다. HTML에서는 styling/script-related noise를 제거하되 text와 structural information을 유지하고, plain text에서는 named entity, number, date, ID 등의 factual information을 유지하면서 redundant prose를 축약한다.
- **둘째, sub-agent의 verbose final report를 manager agent에게 전달하기 전에 synthesis한다.** 이 경우에는 generic compression과 달리 manager의 **local task**, 전체 **global task**, 그리고 sub-agent의 full report를 함께 사용한다. SUPERVISORAGENT는 manager가 요청한 정보가 손실되지 않도록 report를 읽고, 필요한 evidence와 answer를 하나의 cleaner observation으로 재구성한다.

즉, 논문에는 실질적으로 **task-agnostic observation compression**과 **task-aware sub-agent result synthesis**라는 두 형태의 purification이 존재한다.
