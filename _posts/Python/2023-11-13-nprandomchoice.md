---
title: "[Python]np.random.choice 란?"

categories:
  - py


toc: true
toc_sticky: true

date:  2023-11-13
last_modified_at: 2023-11-13
---

# Definition




- `np.random.choice(a, size, replace, p)`는 총 4개의 인자를 입력으로 받을 수 있다.
  - `a`: 입력으로 들어가는 리스트 또는 어레이
  - `size`: 내가 추출한 sample 개수
  - `replace`: 한 번 뽑힌 sample이 다음 번에도 뽑힐 수 있는 경우인 복원 추출인 경우 `replace = True`이고 중복을 허용하지 않는 비복원 추출의 경우  `replace = False`로 입력을 준다.
  - `p`: p는 입력 리스트 또는 어레이와 직접적으로 연결되어 있는데, 리스트나 어레이의 각 요소가 뽑힐 확률을 입력으로 받는다. 만약 모든 것이 같다면 uniform하다.

```python
"""
Created on meaningful96

DL Project
"""

import numpy as np

"""
# numpy.random.choice(a, size=None, replace=True, p=None)

# a: Input list or Input array which to make the random selection
# size: The number of elements we want to choose
# replace: This is a boolean parameter that determines whether the sampling is done with replacement. 
#          If replace is set to True, it means that the same element can be chosen more than once. 
#          If set to False, each element can only be chosen once. In your code, it is not explicitly specified, 
#          so it defaults to True.
#          -> replace = True: Duplication OK!!
           -> replace = False: Duplication NOPE!!

# p: This parameter is an array of probabilities associated with each element in the input array a. 
#    It specifies the probabilities of selecting each element
"""

# Example dictionary representing state probabilities
q_1_dict = {'state1': 0.3, 'state2': 0.4, 'state3': 0.2, 'state4': 0.1}

# Example value for sample_num
sample_num = 0.3

# Example list of states (cur_state)
cur_state = [1, 2, 3]

# Check if sample_num is less than 0.5
if sample_num < 0.5:
    # Use np.random.choice to select elements based on the probabilities in q_1_dict
    y_list = np.random.choice(list(q_1_dict.keys()), len(cur_state), p=list(q_1_dict.values()))

    # Display the result
    print("Randomly selected states:", y_list)
else:
    print("Sample_num is greater than or equal to 0.5.")
```
