---
title: Reward Shaping

categories: 
  - Etc
tags:
  - [ML,DL,DeepLearning]

toc: true
toc_sticky: true

date: 2022-12-27
last_modified_at: 2022-12-27 
---
## Reward Shaping이란?
### 1) 보상 형성의 개념
<span style = "color:aqua">**Reward Shaping(보상 형성)**</span>의 기본 아이디어는 알고리즘이 진행되는 중간중간에 일종의 보상을 주어 더 빨리 수렴하게 만드는 것이다.
즉, 매 iteration마다, 더 빠르게 Converging 될 수 있도록 일종의 비용 감소 수단을 추가하는 것이다.(Cost-Decrease Method)
```
The basic idea is to give small intermediate rewards to the algorithm that help it converge more quickly.
```

> Reward Shaping은 주로 Reinforcement Learning에서 많이 쓰이는 방법이다.

**Domain Knowledge** 라는 information이 그 핵심이다. 이 정보를 이용하면 알고리즘 더 빠르게 학습할 수 있도록 보조할수 있으며 동시에 Optimality를 보장한다.
```
can modify our reinforcement learning algorithm slightly to give the algorithm some information to help, while also guaranteeing optimality.
This information is known as domain knowledge — that is, stuff about the domain that the human modeller knows about while constructing the model to be solved.
```

### 2) Q-Value, Q-function
Q-fucntion의 메인 아이디어는 feature와 그 feature들의 weight를 Linear Combination 하는것이다.
he key idea is to approximate the Q-function using a linear combination of features and their weights.

<span style = "font-size:120%">**Process**</span>  
- 매 State마다, feature가 어떤 representation을 결정하는지 고려해야한다.
- 학습이 일어나는 동안, 업데이트를 State보다 feature의 weight에의해 업데이트가 진행되게 수행한다.
- $$Q(s,a)$$를 feature들과 weight들이 합으로 추정한다. 
- $$Q(s,a)$$에서 s는 state, a는 applied action이다.

<span style = "font-size:120%">**Linear Q-function Representation**</span>    
Linear Q-learning에서, feature들과 weight들을 저장한다. **state는 저장하지 않는다.** <span style = "color:aqua">각각의 action마다
각각의 feature나 weight가 학습에 얼마나 중요한지를 알아야 한다.</span>

<center>$$f(s,a) =
\begin{pmatrix}
f_1(s,a) \\
f_2(s,a) \\
\dots  & \\
f_{n \times|A|}(s,a) 
\end{pmatrix}$$</center>

이걸 표현하기 위해선, 두가지 벡터가 필요하다.  
Feature Vector, $$f(s,a)$$는 $$n \times |A|$$ different fuction의 벡터이다.   
$$n$$은 state feature의 수이고, $$|A|$$은 action의 수이다. 각각의 함수는 state-action pair(s,a)의 값(value)을 추출한다.  
함수 $$f_i(s,a)$$는 state-action pair (s,a)에서 i번째 feature를 추출한다.    
weight vector $$w$$ of size $$n \times |A|$$, 각각의 feature-action 쌍에 대해 하나의 weight이다.  
$$w_i^a$$는 action $$a$$에 대한 i feature의 가중치이다.  

<span style = "font-size:120%">**Defining State-Action Featrues**</span>   
종종 각 state마다 feature를 정의하는게 state-action 쌍을 정의하는것보다 쉽다. feature란 $$f_i(s)$$form의 $$n$$함수의 벡터이다.

어쨋든, 많은 application에서 feature의 가중치는 action과 연관(related)되어 있다.

However, for most applications, the weight of a feature is related to the action.
The weight of being one step away from the end in Freeway is different if we go Up to if we go Right.
<center>$$ f_{i,k}(s,a)=
\begin{cases}
f_i(s),\;if\;a = a_k\\
0,\;otherwise\;1\leq i \leq n, 1\leq k \leq |A|
\end{cases}$$</center>
