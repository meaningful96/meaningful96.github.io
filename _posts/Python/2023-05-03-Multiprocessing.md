---
title: "[Python]Multiprocessing을 이용한 데이터 병렬 처리"

categories:
  - py

toc: true
toc_sticky: true

date:  2023-05-03
last_modified_at: 2023-05-03
---

# Multiprocessing

- 대용량 데이터를 효율적으로 처리하기 위해서는 병렬 처리를 활용하는 것이 좋다.
- Pytorch같은 Framework는 함수 내부에서 병렬 처리를 지원
- 하지만 데이터 가공 모듈인 numpy나 pandas같은 경우는 별도의 병렬처리가 가능하도록 코딩해야 한다.
- Pool은 병렬 연산을 지원하는 함수이다.

```python
def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))

out: [1,4,9]
```
