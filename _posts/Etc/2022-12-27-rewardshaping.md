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
## Q-function, Q-Value란?

### 1) Q-Value, Q-function
Q-fucntion의 메인 아이디어는 feature와 그 feature들의 weight를 Linear Combination 하는것이다.
he key idea is to approximate the Q-function using a linear combination of features and their weights.

<span style = "font-size:120%">**Process**</span>  
- 매 State마다, feature가 어떤 representation을 결정하는지 고려해야한다.
- 학습이 일어나는 동안, 업데이트를 State보다 feature의 weight에의해 업데이트가 진행되게 수행한다.
- $$Q(s,a)$$를 feature들과 weight들이 합으로 추정한다. 
- $$Q(s,a)$$에서 s는 state, a는 applied action이다.

### 2) Linear Q-function Representation 
Linear Q-learning에서, feature들과 weight들을 저장한다. **state는 저장하지 않는다.** <span style = "color:aqua">각각의 action마다
각각의 feature나 weight가 학습에 얼마나 중요한지를 알아야 한다.</span>

<center>$$f(s,a) =
\begin{pmatrix}
f_1(s,a) \\
f_2(s,a) \\
\dots  & \\
f_{n \times|A|}(s,a) 
\end{pmatrix}$$</center>

- **이걸 표현하기 위해선, 두가지 벡터가 필요하다.**  
  1) Feature Vector, $$f(s,a)$$는 $$n \times \vert A \vert$$ different fuction의 벡터이다.   
  2) $$n$$은 state feature의 수이고, $$\vert A \vert$$은 action의 수이다. 각각의 함수는 state-action pair(s,a)의 값(value)을 추출한다.  
  3) 함수 $$f_i(s,a)$$는 state-action pair (s,a)에서 i번째 feature를 추출한다.    
  4) weight vector $$w$$ of size $$n \times \vert A \vert$$, 각각의 feature-action 쌍에 대해 하나의 weight이다.  
  5) $$w_i^a$$는 action $$a$$에 대한 i feature의 가중치이다.  

### 3) Defining State-Action Featrues
종종 각 state마다 feature를 정의하는게 state-action 쌍을 정의하는것보다 쉽다. feature란 $$f_i(s)$$form의 $$n$$함수의 벡터이다.

어쨋든, 많은 application에서 feature의 가중치는 action과 연관(related)되어 있다.

<center>$$ f_{i,k}(s,a)=
\begin{cases}
f_i(s),\;if\;a = a_k\\
0,\;otherwise\;1\leq i \leq n, 1\leq k \leq |A|
\end{cases}$$</center>


이는 $$\vert A \vert$$ 다른 weight vector들을 효과적으로 야기한다.  
<center>$$f(s,a_1) =
\begin{pmatrix}
f_{1,a_1}(s,a) \\
f_{2,a_2}(s,a) \\
0\\
0\\
0\\
0\\
\vdots
\end{pmatrix}\; f(s,a_2) =
\begin{pmatrix}
0\\
0\\
f_{1,a_1}(s,a) \\
f_{2,a_2}(s,a) \\
0\\
0\\
\vdots
\end{pmatrix}\;  f(s,a_3) =
\begin{pmatrix}
0\\
0\\
0\\
0\\
f_{1,a_1}(s,a) \\
f_{2,a_2}(s,a) \\
\vdots
\end{pmatrix}\; \dots$$</center>

### 4) Q-Values from linear Q-functions
feature vector인 $$f$$와 weight vector인 $$w$$가 주어지면, state의 Q-value는 간단히 feature와 weight들의 linear combination으로 표현이된다.  
<center>
$$\begin{aligned} 
Q(s,a)\quad &=\quad f_1(s,a) \cdot w_1^a + f_2(s,a) \cdot w_2^a + \dots + f_n(s,a) \cdot w_n^a\\  
&= \quad  \displaystyle\sum_{i = 0}^nf_i(s,a)w_i^a
\end{aligned}$$</center>  

### 5) Linear Q-function Update  
Q-function approximation을 강화학습에서 사용하기 위해서는 원래의 알고리즘에서 두 가지 과정을 바꿔야 한다.

- Initialization
- update

<span style = "font-size:120%">**Initialization**</span>  
모든 가중치를 **0**으로 initialization 해야한다. 아마 특정한 가중치를 두고 업데이트 해보면 알 것이다.

<span style = "font-size:120%">**Update**</span>  
Q-table 값들 대신에 가중치를 업데이트해야한다.
- Update Rule
  - For each state-action featrue $$i$$  
    $$w_i^a \; \leftarrow \; w_i^a + \alpha \cdot \delta \cdot f_i(s,a) $$

$$\delta$$는 우리가 사용하는 알고리즘에 dependent하다. 결론적으로 이 식은 Linear하기에, Convex하다.
따라서 가중치는 결국 Converging된다.

```
Note that this has the effect of updating Q-values to states that have never been visited!
```

## 2. Reward Shaping이란?

### 1) 보상 형성(Reward Shaping)의 개념
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

보상이 희박한 경우, 우리는 우리를 솔루션에 더 가깝게 만든다고 생각하는 보상 행동을 위해 보상 기능을 수정 또는 증강할 수 있다.

### 2) Shaped Reward
TD learning(Temporal-Difference learning)에서 첫 번째 step으로 Q-function을 보상(reward)를 받아 업데이트 한다.  
<center>$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \underset{\alpha^\prime }{\operatorname{max}} Q(s^\prime , a^\prime) - Q(s,a)]$$</center>

중요한 것은, **reward shaping에 대한 접근법은, reward function이나 받은 reward 인 $$r$$을 수정하는 것이 아닌,** <span style = "color:aqua">**몇몇 action에 대해 추가적인 shaped reward를 주는 것**</span>이다.  
<span style = "font-size:120%"><center>$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \underset{=addtional \ shaped \ reward}{\operatorname{F(s, s^\prime)}} +\gamma \underset{\alpha^\prime }{\operatorname{max}} Q(s^\prime , a^\prime) - Q(s,a)]$$</center></span>


## Reference
[Reward shaping](https://gibberblot.github.io/rl-notes/single-agent/reward-shaping.html)


