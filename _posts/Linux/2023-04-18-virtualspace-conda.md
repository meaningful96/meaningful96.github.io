---
title: "[Linux]Conda로 가상환경 만들기"

categories: 
  - Linux

toc: true
toc_sticky: true

date: 2023-04-18
last_modified_at: 2023-04-18
---

- miniconda 또는 anconda를 설치한다.
- Conda 명령어를 이용한 List 확인

```bash
conda env list
```

- 가상환경 활성화

```bash
conda activate base
conda activate 가상환경이름
```

- 가상환경 삭제

```bash
conda activate base
conda env remove -n "환경이름"
```

- 가상환경 추가

```bash
conda create -n "환경이름" python=3.6
```

- 예시

```bash
conda create -n py27 python=2.7 keras==2.0.8 tensorflow==1.2.1 -c conda-forge

conda create -n tc18 pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda create -n tc18 -c conda-forge pytorch==1.3.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch 

conda create -n tf112 python=3.6
conda install -c anaconda tensorflow-gpu=1.12 keras==2.2.4
```

- 파일로 가상환경 추가

```bash
conda env create -f keggutils_env.yml
```

- yml 파일 예시
```bash
name: Relphormer
channels:
  - conda-forge
  - pytorch
  - defaults
dependencies:
  - numpy=1.21.5
  - pytorch==1.13.1
  - torchvision=0.14.1
  - torchaudio=0.13.1
  - pytorch-cuda=11.6
  - pytorch-lightning=1.9.4
  - transformers=4.26.1
  - faiss-gpu=1.7.2
  - tensorboard=2.12.0
  - tensorboardx=2.6
```
