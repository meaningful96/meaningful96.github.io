---
title: "[그래프 AI]GraphSAGE(Inductive Representation Learning on Large Graphs)"
categories: 
  - Graph
  
toc: true
toc_sticky: true

date: 2024-08-18
last_modified_at: 2024-08-18
---
# GraphSAGE
## 1. GraphSAGE 배경
GCN이나 GAT는 반지도학습(Semi-Supervised Learning) 방식이다. 반면 비지도 학습(Unsupervised Learning) 방식은 채택한 모델은 GraphSAGE이다. Labeling이 되지 않은 데이터를 이용하거나, 큰 사이즈의 graph를 학습해야 하는 모델들은 일반적으로 비지도 학습에 속한다. 
