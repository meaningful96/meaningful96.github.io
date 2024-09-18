---
title: "[그래프 AI]GNN 학습과 데이터 분할(Split)"
categories: 
  - Graph
  
toc: true
toc_sticky: true

date: 2024-09-18
last_modified_at: 2024-09-18
---

# Training GNN

## Supervised Learning vs Unsupervised Learining
<p align="center">
<img width="700" alt="1" src="https://github.com/user-attachments/assets/77f424e2-3c49-4e56-8f92-369cefa7bca4">
</p>

**Supervised Learning**은 정답 레이블이 있는 데이터를 이용해 학습하는 방식이며, **Unsupervised Learning**은 레이블 없이 데이터의 패턴이나 구조를 학습하는 방식이다.

Supervised Learning에서는 Ground Truth가 레이블로부터 오고, Unsupervised Learning에서는 Unsupervised Signal로부터 온다. **Supervised Learning**에서는 <span style="color:red">**학습을 위해 제공되는 정답 레이블(Label)이 존재**</span>하며, 이 레이블은 외부 데이터를 통해 얻는다. 모델은 예측 값과 실제 레이블 간의 차이를 기반으로 손실함수를 계산하여 학습을 진행하며, 이때 데이터셋에 존재하는 정답 레이블이 Ground Truth가 된다.

반면, Unsupervised Learning에서는 명확한 레이블이 주어지지 않고, 대신 외부에서 제공된 Unsupervised Signal을 기반으로 학습이 이루어진다. 이 신호는 데이터 간의 패턴이나 구조를 발견하는 데 사용되며, 학습의 목적은 데이터의 관계를 파악하는 것이다. 따라서, **Unsupervised Learning**에서 Ground Truth는 레이블이 아닌 <span style="color:navy">**패턴이나 구조적 정보**</span>로부터 나온다.

**GNN(Graph Neural Network)**에서도 예측을 수행하려면 Ground Truth가 필요하다. Supervised GNN의 경우 Ground Truth는 노드나 엣지의 레이블로부터 오고, Unsupervised GNN의 경우 Ground Truth는 그래프 내에서 발견되는 구조적 패턴이나 노드 간 유사성과 같은 Unsupervised Signal에서 비롯된다. 이처럼 Supervised와 Unsupervised 두 방식 모두 모델 학습에 중요한 정보를 제공한다.
