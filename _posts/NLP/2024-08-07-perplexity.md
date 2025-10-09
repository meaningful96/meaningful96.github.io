---
title: "[NLP]언어 모델을 위한 평가지표 3. 혼잡도(Perplexity)"
categories: 
  - NLP
  
toc: true
toc_sticky: true

date: 2024-08-07
last_modified_at: 2024-08-07
---

# Perplexity란?
<span style="color:gold">**Peplexity(혼잡도, PPL)**</span>는 언어 모델의 성능을 평가할 때 사용하는 지표이다. 즉, **언어 모델이 다음 단어나 문장을 얼마나 잘 예측**하는지를 정량화한 값이다.

- **낮은 Perplexity ($$\downarrow$$)**: 모델이 예측을 더 잘함 (더 정확한 언어 모델, $$\uparrow$$)
- **높은 Perplexity ($$\uparrow$$)**: 모델이 예측을 잘 못함 (성능이 낮은 언어 모델, $$\downarrow$$) 

<center>$$\text{Perplexity}(W) = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(w_i \mid w_{<i})\right)= \left(\prod_{i=1}^{N}\frac{1}{P(w_i \mid w_{<i})}\right)^{\!1/N} = P(w_1,\dots,w_N)^{-\frac{1}{N}}$$</center>

**[시퀀스 전체의 PPL]** Perplexity는 문장의 길이로 정규화된 문장 확률의 역수이다. 문장의 길이를 $$W$$, 길이가 $$N$$이라고 하였을 때, PPL의 수식은 위와 같다. 이는 **시퀀스 전체에 대한 PPL**로, 문장 전체의 평균 예측 난이도를 기하평균의 역수로서, <span style="color:gold">**문장 전체를 평균적으로 얼마나 어렵계 예측**</span>했는지를 측정한다. 음의 log-likelihood의 산술평균을 지수화한 값이며, 크로스 엔트로피의 지수화와 동일합니다. 값이 낮을수록 모델이 문장 전체를 잘 예측했다는 뜻이다.

<center>$$\text{Perplexity}_N (W) \exp\!\big(-\log P(w_N \mid w_{<N})\big)= \frac{1}{P(w_N \mid w_{<N})}$$</center>

**[특정 시점 한 토큰에 대한 PPL]** 이 식은 오직 $$N$$번째 토큰 하나에 대한 PPL로, <span style="color:gold">**해당 스텝에서 모델이 정답 토큰을 얼마나 어렵게 맞췄는지**</span>를 나타낸다. 확률값이 높을수록 PPL은 1에 가까워지고, 확률이 낮을수록 크게 증가한다. 

 # PPL Example
<p align="center">
<img width="600" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Deep_Learning/NLP/%5B2025.09.30%5Dppl.png?raw=true">
</p>

**[입력 Prompt]**  
Naval carrier aviation had moved to the center of strategy, making reconnaissance, code-breaking, and carrier-air group operations decisive factors in the course of battle. In one engagement in particular, the fleet quickly fixed the position of the enemy carriers and delivered a crippling series of dive-bomber attacks; the victory became a symbol of how the fusion of intelligence and strike power could reverse the tide of war. In the aftermath, both sides raced to replace lost carriers and to expand pilot training pipelines, while superiority in sea-based air power reshaped convoy defense and enabled campaigns to reclaim occupied territory. In the Pacific theater, the turning-point engagement described here is

**[Candidate]**  
Midway, Guadalcanal, Coral, Leyte, Tarawa, Okinawa

이 글은 Midway 해전에 대한 예시로, 정답은 **Midway**이다. 위의 그림은 각 후보 토큰에 대한 확률값을 나타낸다.
