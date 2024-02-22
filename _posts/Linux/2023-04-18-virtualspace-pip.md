---
title: "[Linux]pipenv로 가상환경 만들기"

categories: 
  - Linux

  
toc: true
toc_sticky: true

date: 2023-04-18
last_modified_at: 2023-04-18
---

# pipenv로 가상환경 만들기

- 설치
```bash
pip install pipenv
```

- 가상환경 생성
먼저 가상환경으로 사용하려는 폴더를 만들어야 한다.
```bash
mkdir "폴더이름"
```
그리고 폴더로 이동을 해준다.
```bash
cd "폴더이름"
```
가상환경 생성한다.
```bash
pipenv --python 3.8
```

- 가상환경 실행
```bash
pipenv shell
```
일반적으로 pipenv환경에서는 패키지 설치시 pipenv install로 설치를 한다.
```bash
pipenv install pandas
```

- 가상환경 비활성화
```bash
exit
```

- 가상환경 삭제
```bash
pipenv --rm
```
