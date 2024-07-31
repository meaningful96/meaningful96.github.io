---
title: "[딥러닝]Skip-Connection이란?"

categories: 
  - DeepLearning

toc: true
toc_sticky: true

date: 2023-03-10
last_modified_at: 2023-03-10 
---

# Skip Connection의 정의
Neural Network 학습 시 모델이 층이 깊어질수록(Dense Layer) 성능이 좋아지는 경향성을 보인다. 하지만 무작정 층을 쌓는 것은 비효율적이다. 그 이유는, 계산해야하하는 파라미터수가 기하급수적으로 증가하기 때문이다. 또한, 연속적이고 많 미분 계산으로 인해 **Vanishing gradient(Exploding Gradient)** 현상이 발생한다.

 이 **Vanishing Gradient**현상을 해결하기 위해 사용되는 방법 중 하나가 바로 <span style = "color:green">**Skip Connection**</span>이다. Skip Connection은 이전 Layer의 정보를 직접적으로 Direct하게 이용하기 위해 이전 층의 <span style = "color:green">**입력(정보)를 연결**</span>한다는 개념이다.

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/231574827-9f2360b4-5ab2-4ca2-bc12-f41057b28744.png">
</p>

이러한 Skip Connection이 있는 Block을 Residual Block이라고 하며 이러한 학습 방식을 이용하는 것이 Residual Learning이라고도 한다. Transformer를 보면 Residual Connection이 들어간 것을 볼 수 있는데, Skip Connection과 그 의미와 용도가 동일하다.

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/231575960-351708ed-64b0-4007-b40f-668ec6ec34cb.png">
</p>

Skip Connection을 적용하면 레이어를 더 깊게 쌓을 수 있다. plain은 레이어를 쌓을수록 error가 줄어들지만 더욱 깊게 쌓을수록 다시 error가 커지는 것을 볼수있다. 반면에 skip connection을 적용한 ResNet모델의 error는 이상적으로 줄어드는 것을 확인할 수 있다.

