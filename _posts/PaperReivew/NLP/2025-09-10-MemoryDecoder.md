---
title: "[논문리뷰]Memory Decoder: Memory for Large Language Models"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2025-09-10
last_modified_at: 2025-09-10
---

*Jiaqi Cao, Jiarui Wang, Rubin Wei, Qipeng Guo, Kai Chen, Bowen Zhou, and Zhouhan Lin*. 2025. [Memory Decoder: A Pretrained, Plug-and-Play Memory for Large Language Models](https://arxiv.org/abs/2508.09874). arXiv:2508.09874 [cs.CL] https://arxiv.org/abs/2508.09874

# Problem Statement
<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.10%5DMemoryDecoder/figure1.png?raw=true">
</p>
<center>

<span style="font-size:110%">**Domain Adaptive PreTraining (DAPT)의 근본적인 한계**</span>  
DAPT는 특정 도메인의 데이터로 LLM을 continuous pretraining하는 방법이다. 하지만, LLM의 모든 파라미터를 full fine-tuning해야 하므로 계산 비용이 매우 높고, 모델 크기가 커질수록 매우 비효율적이다. 또한, 여러 모델을 동일한 도메인에 적응시키려면 각 모델마다 별도로 학습해야 하는 자원 비효율성이 존재한다. 가장 중요한 한계점은, <span style="color:gold">**학습 과정에서 모델이 가진 기존의 일반화 능력을 잃어버리는 catastrophic forgetting 현상이 발생**</span>한다.

<span style="font-size:110%">**Non-parametric Retrieval의 근본적인 한계**</span>  
기존 모델의 파라미터를 수정하지 않는 장점이 있지만, 추론을 위해서 모든 토큰들에 대해서 각각의 임베딩을 미리 계산해두어야하고 이를 대규모 KV 데이터스토어에 저장해두어야한다. 이 데이터스토어의 **storage memory는 매우 크므로 공간 비효율적**이며, 추론 시마다 대규모 DB에서 k-nearest neighbors를 검색해야 하므로 상당한 **추론 지연(inference latency)을 유발**한다.


<br/>
<br/>

# Methodology



<br/>
<br/>

# Experiments



<br/>
<br/>

# Conclusion
