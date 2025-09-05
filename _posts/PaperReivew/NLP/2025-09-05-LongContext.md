---
title: "[논문리뷰]Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG"

categories: 
  - NR
  
toc: true
toc_sticky: true

date: 2025-09-05
last_modified_at: 2025-09-05
---
*Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O. Arik*. 2025. [Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG](https://arxiv.org/abs/2410.05983). In Proceedings of the International Conference on Learning Representations (ICLR 2025). International Conference on Learning Representations.

# Problem Statement
<p align="center">
<img width="500" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.02%5DLongContext/figure2.png?raw=true">
</p>
<center><span style="font-size:80%">*[Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)*</span></center>

<span style="font-size:110%">**Lost-in-the-middle**</span>    
**Lost-in-the-middle**는 LLM이 긴 입력을 처리할 때 시퀀스의 <span style="color:red">**앞과 뒤 정보는 잘 활용하지만 중간에 위치한 정보는 무시**</span>하는 현상이다. 위의 그림은 20개의 검색 문서를 입력하고 정답 문서를 1~20번째 임의 위치에 배치했을 때의 성능 변화를 보여준다. 정답 문서가 7~16번째에 있을 경우 정답률이 크게 떨어지며, 심지어 검색 문서를 주지 않은 closed-book 설정보다도 낮아진다. 검색 문서 수가 많아질수록 핵심 문서가 중간에 위치할 가능성이 높기 때문에, 이 문제를 완화하는 것이 중요하다.

<span style="font-size:110%">**Double-edge Sword Effect of Retrieval**</span>  
이는 lost-in-the-middle 문제와 연결된다. 더 강력한 retrieval은 높은 recall과 precision을 제공하지만, 동시에 의미적으로 유사하나 실제로 정답을 포함하지 않거나 추론에 도움이 되지 않는 **distractor** 문서들을 더 많이 검색하게 된다. 즉, retrieval 성능이 향상될수록 정답 문서와 구분하기 어려운 <span style="color:red">**hard negative**</span>가 증가하며, 이는 LLM의 추론을 방해한다.

<br/>
<br/>

# Chanllenges of Long Context LLMs in RAG
<p align="center">
<img width="500" alt="1" src="https://github.com/meaningful96/Blogging/blob/main/Paper_Review/%5B2025.09.02%5DLongContext/figure2.png?raw=true">
</p>

<br/>
<br/>

# Methodology


<br/>
<br/>

# Conclusion
