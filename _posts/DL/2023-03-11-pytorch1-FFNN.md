---
title: (Pytorch) - 1. Feedforward Neural Network(FFNN)

categories: 
  - DeepLearning
tags:
  - [DL, Pytorch]

toc: true
toc_sticky: true

date: 2023-03-11
last_modified_at: 2023-03-11 
---


## 1. 순방향 신경망

### 1) 정의

<p align="center">
<img width="500" alt="1" src="https://user-images.githubusercontent.com/111734605/224471534-c2d3daa4-a8ed-43d3-8b14-3a3b2cfd6120.png">
</p>
**순방향 신경망(Feedforward Neural Network)**은 **다층 퍼셉트론**의 다른 이름으로 인공 신경망 모델 중 가장 기본이 되는 모델이다. 
순방향 신경망은 Universal Approximation Theorem을 통해 n차원 공간의 연속 함수를 근사할 수 있다는 것이 증명되었다. 
쉽게 말해 <span style = "color:aqua">히든 레이어 하나로 어떤 함수든 다 표현이 가능</span>하다라는 것이다.

### 2) Universal Approximation Theorem

**Universal Approximation Theorem**이란 <u>한 개의 히든 레이어를 가진 Neural Network를 이용해 어떠한 함수든 근사시킬 수 있다</u>는 이론이다.(단, Activation function은 반드시 Nonlinear function이여야 한다.) 
하지만, 단점은 노드가 매우 많아진다.

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/224473388-5743ceda-aa18-4cf9-8424-33540ccc9b28.png">
</p>

Polynomial하지 않은 어떠한 연속함수를 이용해서 Weight와 bias를 조정하면 Perceptron처럼 Threshold가 있는 함수가 만들어진다는 것이 동치이다. 

### 3) 다층 퍼셉트론

다층 퍼셉트론(Multi-layer Perceptron)은 이러한 Hidden layer가 여러개인 순방향 신경망이다. 순방향 신경망 모델은 데이터가 **순방향(feedforward)** 연결만을 갖는 구조로 되어 있으며, 퍼셉트론의 연산과 같은 기본 뉴런 연산으로 실행된다.(즉, 역방향 Edge는 없다.)

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/224473695-b5b23d9a-02c5-4576-a621-4f745a243688.png">
</p>

순방향 신경망의 계층은 **입력 계층(Input Layer), 은닉 계층(Hidden Layer), 출력 계층(Output Layer)**로 구분된다. 

- 입력 계층: 데이터를 입력 받음
- 은닉 계층: 데이터의 특징을 추출, 특징을 기반으로 추론
- 출력 계층: 추론 결과를 외부에 출력

## 2. FC-Layer(Fully Connected Layer)

### 1) 정의

순방향 신경망은 모든 계층이 완전히 연결된 구성을 의미한다.

<p align="center">
<img width="400" alt="1" src="https://user-images.githubusercontent.com/111734605/224473882-65e13318-5c5a-4420-938a-6a04f80d4c2e.png">
</p>
각 layer에 속한 뉴런이 이전 layer의 모든 뉴런과 연결된 구조를 의미한다. 각 뉴런은 이전 layer에서 출력한 데이터를 동일하게 전달 받기 때문에 <span style = "color:aqua">같은 입력 데이터에서 뉴런마다 서로 다른 특징을 추출</span>한다. 이러한 사실때문에 데이터에 특징(feature)이 많을 수록 더 많은 노드가 늘어나야 그 특징들을 모두 추출할 수 있다.


각 뉴런에서 추출된 특징은 계층 단위로 출력되어 다음 layer에 전달된다. 
-  Input Data ➜ Input layer(Previous layer): 데이터를 입력받음
- Input layer ➜ Hidden layer: 데이터에서 특징이 추출되고 변환됨
- Hidden layer ➜ Output layer: 가장 추상화된 특징을 이용하여 예측

### 2) 특징을 추출하는 뉴런 구조

<p align="center">
<img width="500" alt="1" src="https://user-images.githubusercontent.com/111734605/224495619-547b326a-e53b-4f1a-8fa9-a573bac41431.png">
</p>

뉴런은 특징 추출을 위해 Weighted sum과 Activiation function을 이용한다. Activation function은 원하는 형태로의 feature 추출을 위해 데이터를 Nonlinear하게 변환하는 도구이다. 뉴런의 input이 들어오면 이 input이 가중치(weight)와 곱해지고 bias가 더해진다. 이렇게 하나의 weighted sum 형태의 식이 나오면 두 번째로 이 식에 activation function을 먹인다.

이 때, 특징을 추출할 때 영향력이 크면 Weight값이 크고, 영향력이 작으면 Weight값이 작게 설정된다.

### 3) 순방향 신경망(FFNN) 설계 항목

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/224495829-7664695c-7c9f-4de4-bae6-18638216965d.png">
</p>

FFNN을 설계하기 위해서는 다음과 같이 4가지의 Parameter를 설정해줘야 한다.

1. 모델의 입력 형태(Size)
2. 출력 형태(Size)
3. 활성 함수의 종류
4. 네트워크 크기

데이터와 신경망 모델의 종류가 결정되면 입력과 출력의 형태는 어느 정도 결정된다. 하지만, 모델의 크기나 activiation function의 종류는 모델 최적화가 최대로 되도록 탐색해야 하며 모델 검증 단계에서 Hyperparameter 탐색을 통해 최적의 모델을 찾는 과정이 필요하다.

## 3. Python으로 구현하기

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

'''
STEP 1: LOADING DATASET
'''

train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

'''
STEP 2: Dataset 만들기
'''

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

'''
STEP 3: 모델 정의
'''
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity
        self.tanh = nn.Tanh()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        # Linear function
        out = self.fc1(x)
        # Non-linearity
        out = self.tanh(out)
        # Linear function (readout)
        out = self.fc2(out)
        return out
'''
STEP 4: 모델 크기 조절
'''
input_dim = 28*28 # image 크기
hidden_dim = 100 # 아무 숫자나 가능
output_dim = 10 # 0 ~ 9 중 하나 고르기

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

'''
STEP 5: Loss 설정
'''
criterion = nn.CrossEntropyLoss()

'''
STEP 6: Optimizer 설정
'''
learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
STEP 7: 모델 학습
'''
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Load images with gradient accumulation capabilities
        images = images.view(-1, 28*28).requires_grad_()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                # Load images with gradient accumulation capabilities
                images = images.view(-1, 28*28).requires_grad_()

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
```

```
Iteration: 500. Loss: 0.28263184428215027. Accuracy: 91.04000091552734
Iteration: 1000. Loss: 0.24772046506404877. Accuracy: 92.58000183105469
Iteration: 1500. Loss: 0.21657991409301758. Accuracy: 93.54000091552734
Iteration: 2000. Loss: 0.2514685392379761. Accuracy: 93.81999969482422
Iteration: 2500. Loss: 0.13824449479579926. Accuracy: 94.5999984741211
Iteration: 3000. Loss: 0.22502899169921875. Accuracy: 94.8499984741211
```



## Reference

[[DL] 순방향 신경망(feedforward neural network)]("https://velog.io/@cha-suyeon/DL-%EC%88%9C%EB%B0%A9%ED%96%A5-%EC%8B%A0%EA%B2%BD%EB%A7%9Dfeedforward-neural-network#fc-layerfully-connected-layer")  

