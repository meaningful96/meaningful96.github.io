---
title: "[Python]Scipy - LIL matrix, isspars 함수란?"

categories:
  - py


toc: true
toc_sticky: true

date:  2023-05-06
last_modified_at: 2023-05-06
---
# scipy.sparse의 lil_matrix()함수

## SciPy 모듈

SciPy는 파이썬을 기반으로 하여 과학, 분석, 그리고 엔지니어링을 위한 과학(계산)적 컴퓨팅 영역의 여러 기본적인 작업을 위한 라이브러리이다.
Scipy는 기본적으로 Numpy, Matplotlib, Pandas, Sympy등 과 함께 동작한다. NumPy와 Scipy를 함께 사용하면 확장 애드온을 포함한 MATLAB을 완벽하게 대체할 수 있다.
SciPy는 NumPy위에서 구동되는 라이브러리 정도로 이해해도 무방하다. SciPy는 기본적으로 NumPy의 ndarray를 기본 자료형으로 사용한다. 
일부 패키지는 중복되지만 SciPy가 보다 풍부한 기능을 제공한다. SciPy는 다른 과학 컴퓨팅 영역을 다루는 하위 패키지로 구성다.

- `scipy.cluster`: 계층적, K-평균 군집화 등 클러스터링 알고리즘을 제공
- `scipy.constants`: 물리 상수를 제공
- `scipy.fftpack`: FFT(Fast Fourier Transform) 알고리즘을 제공
- `scipy.integrate`: 수치적 적분을 위한 여러 함수를 제공
- `scipy.interpolate`: 데이터 보간을 위한 함수를 제공
- `scipy.linalg`: 선형 대수 연산을 위한 함수를 제공
- `scipy.ndimage`: 다차원 이미지 처리를 위한 함수를 제공
- `scipy.optimize`: 최적화 문제를 위한 함수를 제공
- `scipy.signal`: 시그널 처리를 위한 함수를 제공
- `scipy.sparse`: 희소 행렬 처리를 위한 함수를 제공
- `scipy.spatial`: 공간 데이터 처리를 위한 함수를 제공
- `scipy.special`: 특수 함수를 제공
- `scipy.stats`: 통계 분석을 위한 함수를 제공

## Scipy.sparse: 희소 행렬(Sparse Matrix)처리
행렬의 값이 대부분 non-zero인 행렬을 밀집행렬(Dense Matrix)라고 한다. 반면, 행렬의 요소들 대부분이<span style = "color:red">**zero('0')인 행렬을 희소 행렬(Sparse Matrix)**</span>라고 한다. 
LIL은 row-based format으로 non-zero 요소들을 리스트나 튜플과 연결해 저장하는 행렬이다. 각가의 튜플은 두 가지 값을 포함한다.
- column Index
- corresponding value of the non-zeros element

```python
import scipy.sparse as sp # 라이브러리 호출

matrix = s.lil_matrix((3,3)) # 3x3행렬 생성, 이 상태로 출력하면 아무것도 출력이 안됨.

matrix[0 ,1] = 2   # 0행 1열에 2라는 요소 삽입
matrix[1, 2] = 3
matrix[2, 0] = 4
print(matrix)
```

**[Output]**
```python
[[0. 2. 0.]
 [0. 0. 3.]
 [4. 0. 0.]]
```

## issparse()함수
sp.issmatrix 함수는 행렬을 input으로 받아 input행렬이 sparse matrix인지 아닌지를 <span style = "color:red">**Boolean 값**</span>으로 리턴한다. 만약 Sparse matrix가 맞다면 True, 아니면 False를 리턴한다.

```python
import numpy as np
from scipy import sparse

# create a sparse matrix using the COO format
data = np.array([1, 2, 3])
row = np.array([0, 1, 2])
col = np.array([0, 1, 2])
sparse_matrix = sparse.coo_matrix((data, (row, col)))

# check if the matrix is sparse
is_sparse = sparse.issparse(sparse_matrix)
print(is_sparse)  # Output: True

# create a dense matrix
dense_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# check if the matrix is sparse
is_sparse = sparse.issparse(dense_matrix)
print(is_sparse)  # Output: False
```
