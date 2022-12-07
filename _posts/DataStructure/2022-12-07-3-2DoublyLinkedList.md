---
title: Chapter 3.2 Linked List(연결 리스트)- Singly Linked List & Doubly Linked List

categories:
  - DataStructure
tags:
  - [DS, DataStructure, Stack]

toc: true
toc_sticky: true

date: 2022-12-07
last_modified_at: 2022-12-07 
---

## 1. 단순 연결 리스트(한방향 연결 리스트) review
### 1) Singly Linked List의 개념

<p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/111734605/206080361-547ab3bb-7c88-4f49-afc4-ef7d7e4fa5da.png">
</p>

연결리스트는 파이썬의 리스트와는 다르게 key값과 link값으로 이루어져있다.

리스트는 각 인덱스가 메모리가 할당되어 각 원소의 주소값을 가지게된다. 그래서 상수시간 O(1) 내에 연산이 가능하다.

하지만 연결리스트는 각 메모리가 원소의 주소(key값) + 다음값이 저장된 주소(link값) 이 저장되어 있다.
key값과 link값이 함께 저장되어 있는 것을 Node라고 부른다.
### 2) 한방향 연결 리스트의 장단점
#### 단점
- 연결리스트는 배열처럼 인덱스로 접근이 불가능하다. 
- 따라서 우리가 만약에 연결리스트의 3번 인덱스 값을 찾고싶다면, head노드(노드의 맨 첫번째 값)의 링크부터 따라서 들어가야하므로
- head->1->2->3으로 총 3번의 계산이 필요하다. 
- 만약 100번 인덱스 값을 찾고 싶다면, 100번의 계산이 필요하므로 연결리스트의 시간복잡도는 O(N)이 된다.

#### 장점
- 일반적인 리스트가 insert를 하게되면 새로운 값을 맨 앞에 넣는다는 가정 뒤의 모든 값을 밀어내야하므로 시간복잡도는 O(N) 
- **연결리스트**는 head node를 새로 만들어 link값을 이전 head node의 주소를 가리키면 되므로 시간 복잡도는 O(1)이다.

### 2) 연산 및 Python 코드
#### (1) 여기서 **객채란?**
- 맨처음 한방향 연결리스트의 객체를 하나 만든다는 것은 비어있는 리스트, 노드가 아무 것도 없는 + 사이즈가 0인 Head를 만드는 것이다.
```python
class SinglyLinkedList:
  def __init__(self):
    self.head = None
    self.size = 0
```
#### (2) pushfront
1. 현재 head node 앞에 새로운 node을 삽입해야하므로, 새로운 노드가 head node가 된다.
2. 새로운 node의 next값은 이전의 head node가 된다.
3. 연결리스트의 전체 사이즈가 1 커진다. 

```python
def pushfront(self, key):
    new_node = Node(key) #key값 가진 새로운 노드 생성
    new_node.next = self.head #새로운 노드의 next는 원래 head값인 self.head임
    self.head = new_node #self.head 는 새로운 노등니 new_node가 된다.
    self.size += 1
```

#### (3) pushback
1. 빈 리스트에 pushback은 새로운 노드는 head node가 된다.
2. next가 none인 노드가 tail node가 된다. tail node 다음 값으로 주어진 node를 생성해 넣어야한다.
3. 이를 찾기 위해서는 head node로 먼저 접근해 .next가 none인 노드를 찾는 과정이 필요하다.

```python
def pushback(self, key):
    new_node = Node(key)
    if self.size == 0: #현재 연결리스트가 비어있는 경우
        self.head = new_node #연결리스트의 head node는 new_node가 된다.
    
    else:
        tail = self.head #tail node찾기 위해 tail에 self.head를 지정
        while tail.next != None: #tail.next 가 none이 아니라면
            tail = tail.next #tail에 tail.next(self.head.next)를 지정하고 다시 while문 반복
        tail.next = new_node #while문 끝나면 tail에 연결리스트의 마지막값이 들어와있으므로 다음값에 새로운 노드 new_node 연결
    self.size += 1
```

#### (4) popfront
1. 빈 리스트의 경우, popfront 불가능
2. popfront 이후 self.head는 self.head.next가 되어야한다.
3. popfront는 원래 self.head를 리턴해야한다.

```python
def popfront(self):
        if self.size == 0:
            return None
        else:
            x = self.head #|node(4)를 x 변수에 복사
            key = x.key#|key 는 node(4)의 key인 4
            self.head = x.next #|node(4).next = node(5)이므로 self.head 는 node(5)가 된다.
            del x #|node(4) 객체 자체를 메모리에서 제거
            self.size -= 1
            return key
```

#### (5) popback
1. popback은 맨 뒤 tail의 key값을 리턴해야한다.
2. tail의 이전값을 prev값으로 설정하는데, prev.next는 None이된다(.next가 none이라는건 리스트의 마지막 노드라는 의미)
3. 리스트의 노드가 없을때, 1개일때, 2개 이상일때로 나뉜다.
4. 노드가 없을때는 연산을 실행하지 않는다.
5. 노드가 1개일때, self.head이자 tail의 key값을 리턴하면된다, 그리고 리스트가 비워질 것이므로 self.head를 none으로 설정한다.
6. 노드가 2개 이상이라면, tail은 삭제될 것이므로 prev.next = none이 되어야한다.

```python
def popback(self):
    if self.size == 0: #노드가 0개인 경우
        return None

    else:
        prev, tail = None, self.head #prev에 tail을 지정해나가면서 tail.next가 none인걸 찾는다.
        while tail.next != None: #노드가 1개인 경우
            prev = tail
            tail = tail.next

        if prev == None:  #while 연산이 끝났는데, prev가 none인건 노드가 1개일때밖에 없다.
            self.head == None #pop하면 리스트가 비워지므로 self.head를 None으로 만들어 
        else:
            prev.next = None #노드가 2개이상인 경우, prev 노드가 마지막노드가 되므로 prev의 next(링크)를 지운다.

        key = tail.key
        del tail
        self.size -= 1
        return key
```

#### (6) search
1. 찾고자하는 key값을 가진 node를 리턴해야한다. 없으면 none을 리턴한다.
2. 1가지 방법은 while루프를 부르는 방법이있고,
3. 다른 방법은 for 루프를 돌리는거다.

```python
def search(self, key):
    x = self.head #연결리스트의 헤드노드를 x에 할당
    while x != None: #x가 none 아닌동안, 즉 연결리스트의 헤드부터 끝까지 
        if x.key = key: #만약 x의 키값이 찾고자하는 키값이라면
            return x #x 노드를 리턴한다.
        x = x.next #찾고자하는 키값이 아니라면 x에 다음 노드를 할당한다.
    return x (or None)#x가 none이되면 즉, 연결리스트의 끝값 +1이되면 x는 none이되는데, (x.next가 none이므로) none을 리턴하게된다.
```

#### (7) remove
삭제연산의 경우 3가지로 나뉜다.

1. 연결리스트의 길이가 0인 경우
2. 지우고자하는 값이 headnode인 경우->popfront
3. 연결리스트의 길이가 0이 아니면서 지우고자하는 값이 headnode가 아니라면 연결리스트에서 중간, 뒤에 있는 값을 의미한다.
4. 3의 경우, popback처럼 prev찾아야하며, prev.next를 tail.next에 연결하면 지우고자하는 키값을 가진 노드는 연결리스트에서 배제된다.
5. 배제된 노드는 삭제하고(tail), popfront와 달리 리스트의 사이즈를 줄이는 명령이 없으므로 별도로 self..size -= 1을 실행해줘야한다.

```python
def remove(self, key):
        if self.size == 0: #연결리스트 길이 0인 경우
            return None

        elif self.head.key == key: #리스트의 헤드노드와 찾고자하는 노드의 키가 같은경우
            self.popfront() #popfront로 다음 노드를 헤드노드로 변경, 본래 헤드노드삭제

        else:
            prev, tail = None, self.head #초기값설정
            while tail.next != None: #tail.next가 none이 아닌동안
                prev = tail #prev와 tail을 한칸씩 옮겨가기
                tail = tail.next
                if tail.key == key: #만약 tail.key와 찾고자하는 키가 같다면
                    prev.next = tail.next #remove할 node의 링크를 지워버리면된다.
                    del tail #tail node를 리스트 상에서 삭제
                    self.size -= 1 #한개 삭제되었으므로 리스트 길이 1개 줄인다.
                    break
```

#### (6) reverse
1. 각 노드의 .next를 이전노드로 설정해야한다.
2. prev, tail로 작서앟자.
3. while문 탈출하고나선 리스트의 head노드를 변경해주자.

```python
def reverse(self):
    a, b = None, self.head

    while b:
        if b:
            c = b.next #c에 b의 다음노드를 복사
            b.next = a #b의 링크를 반대로 설정
        a = b
        b = c
```

#### (7) generator
```python
def __iter__(self): #
        v = self.head #
        while v != None: #
            yield v
            v = v.next
```

위 함수를 generator라고 한다(yield가 있는 함수). 연결 리스트의 인스턴스에게 for문을 사용할 수 있게 해준다. 원래 연결리스트에클래스에 인스턴스는 iterator가 없다면

```python
for i in linkedlist:
    print(i)
```

를 할 경우, 에러가 나게 된다. 연결리스트에서는 일반적인 배열처럼 iter를 할 원소가 없기 때문이다. 이를 가능하게 해주는게 스페셜메서드로 
__iter__(self)이다. 그리고 이 메서드처럼 yield가 있는 함수를 generator라고 부른다.

1. 파이썬에서 인스턴스에 for문을 돌린다고 인지를 하게되면
2. 인스턴스에 선언된 클래스에서 __iter__(self)를 자동으로 호출한다.
3. __iter__는 yeild로 v를 리턴한다.
4. while 문에서 v는 자동으로 다음 값이 할당된다.
5. yeild된 값은 for문의 원소로 리턴이 가능하게된다.
6. 다시 for문이 호출되고, 다음 v가 리턴되게된다.
7. 만약 while문이 종료되고 __iter__(self)에서 yeild가 명령되지 않으면 자동으로 StopIterator Errormessage가 출력되면서 for문이 종료된다.

## 2. 양방향 연결 리스트(Doubly Linked List)
### 1) 양방향 연결 리스트란?
- 양방향 연결 리스트는 한방향 연결 리스트의 단점을 보완한다.
- 한방향 연결 리스트의 경우, Search연산에서 시간 복잡도가 O(N)이다.
- 양방향 연결리스트는 각 노드가 next node와 prev의 노드정보를 담는다.
- 따라서 한방향 연결리스트에 비해 시간복잡도가 크게 줄어든다.
- head node와 tail  node는 항상 None인 Dummy노드이다.
- 
<p align="center">
<img width="400" alt="1" src="https://user-images.githubusercontent.com/111734605/206078507-2415a107-d398-4239-9070-4d97558e2374.png">
</p>

- 위의 그림은 한방향 연결리스트이다. 이때, tail 노드만 안다고 prev 노드는 알 수 없다. Why?  
- 만약 tail 노드를 지우고 싶다.  
  - tail 노드에서 prev 노드로 가는 링크 없다.  
  - Head부터 따라가야 함  
  - O(N) <span style = "color:aqua">**Bad!!**</span>  
 
 ### 2) 양방향 연결 리스트의 장단점  
 #### (1) Pros  
 - List에 Node가 주어져 양방향으로 navigate가능  
 - prev Node 주소가 없어도 삭제 가능!!  
 #### (2) Cons  
 - Extra pointer 필요, 더 많은 공간필요  
 - 삽입, 삭제 시간 조금 더 필요함!!  
 
<p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/111734605/205992424-f7cb851d-e9f1-4256-ba65-2a01a0a197ce.png">
</p>

### 3) 원형 양방향 연결 리스트(Circular Doubly Linked List)  
#### (1) 원형 연결리스트의 빈 리스트  
**Dummy Node**  
- 원형 연결 리스트(양방향 연결 리스트)의 시작을 알리는 마커
- key == None
- Dummy node는 head역할도 함: Dummy node = Head node
- **key, next, prev의 세가지 정보를 담는다.**
- **양방향연결리스트의 기본 원칙은 빈 리스트라도 dummy node(None)이 있어야 한다는 것이다.
- 양방향 연결 리스트는 원형이므로 어느게 Head인지 tail인지 모른다.
- 이걸 구별해 주려고 Dummy node를 쓴다.

<p align="center">
<img width="500" alt="1" src="https://user-images.githubusercontent.com/111734605/206087216-07b81548-8bac-41ff-80a3-3b112c2757eb.png">
</p>

```python
class Node: #노드클래스는 리스트의 헤드노드가 Nond이라는 dummy node를 염두에두고 설정한다.
    def __init__(self, key = None): #노드인스턴스 생성함수, 자동호출, key값 미설정시 None설정
        self.key = key #키값은 선언되는 키값이된다.
        self.next = self #미설정시, 자기자신
        self.prev = self #미설정시, 자기자신

    def __str__(self):
        return str(self.key)
```
#### (2) 원형 연결리스트의 클래스 정의

1. __init__의 경우, self.head를 next와 prev가 노드 자기자신이고, key=None인 더미노드를 생성해야한다.
2. 제너레이터는 유의해야할 점이 양방향연결리스트는 리스트의 시작과 끝이 연결되어있다는 것이다. 따라서 v가 self.head가 아닐때까지 반복해야한다.
3. __str__은 __init__에서 만들어진 데이터를 출력한다.
4. __len__은 node의 개수를 반환한다.

```python
class Doublylinkedlist:
    def __init__(self):
        self.head = Node()
        self.size = 0

    def __iter__(self): #제너레이터
        v = self.head.next
        while v != self.head: #양방향리스트는 연결되어있으므로 v가 None이 되면 yield되면 안된다.
            yield v
            v = v.next
        
    def __str__(self):
        return "->".join(str(v) for v in self)
        
    def __len__(self):
        return self.size
```

#### (3) <span style = "color:aqua">Splice 연산</span>**(매우 중요!!)**  
- def splice(self, a, b, x)
- 세 개의 노드 a, b, x(key값이 아니라 노드이다!!)
- 총 6번의 연산이 필요하다
- Case 1) a,b 앞뒤로 원래 리스트의 연결을 바꾸는 cut연산
- Case 2) a,b를 붙인 paste 리스트에서 앞뒤로 링크를 수정

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/206088012-c50375ac-9605-4c95-a6bd-865407c89494.png">
</p>

조건 1. a는 slicing의 시작 노드, b는 마지막 노드이다. **(a,b)**  
  - ap = a.prev, bn = b.next  
조건 2. a와 b사이에 head 노드는 없다.  
조건 3. a와 b사이에 x노드가 없어야 함.

> <span style = "color:aqua">**Spclice 연산**</span>  
  > **a와 b사이에 있는 (a,b 포함) Cut 해서 어딘가에 있는 x노드와 x.next 노드 사이에 집어넣음!!**

```python
def splice(self, a, b, x):
    #경우(1) cut #| ap a .. .. b bn -> ap bn
    ap = a.prev
    bn = b.next

    ap.next = bn
    bn.prev = ap

    #경우(2) paste #| x xn -> x a .. .. .. b xn
    xn = x.next #원래 x의 다음 노드를 xn이라고 할당
    x.next = a
    a.prev = x

    xn.prev = b
    b.next = xn
```

#### (4) Insert연산 & 이동연산  
splice(a,b,x)란    
a의 앞부터 b의 뒤 까지의 노드들을 떼어네어 x뒤에 붙이는 연산이다.

1. splice 연산을 활용해 이동, 삽입연산을 간단하게 만들 수 있다.
2. splice연산의 경우, (a,a,x)라면 단순히 a를 x앞뒤로 옮기는것에 불과하다.
3. 하지만 a가 Node(key)로 바뀌면, 키값을 가진 노드를 생성하므로, 새로운 노드를 생성해 x 앞뒤에 붙일 수 있게 된다.
4. pushfront,back의 경우, self.head가 더미노드로, 앞뒤로 리스트의 마지막값, 첫값을 가진다는 것을 이용한다.

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/206089194-43bbf681-77df-4520-9bc2-cb061170548f.png">
</p>

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/206089658-80b74c5d-9256-4d07-b3b0-6b5da7dbc45a.png">
</p>

```python
def moveAfter(self,a,x): #| ap a an | xp x xn 
    # -> |ap an| xp x a xn
    self.splice(a,a,x) #와 동일.
        

def moveBefore(self,a,x): #a를 x뒤로 붙이기
    self.splice(a,a,x.prev) #와 동일

def insertAfter(self, x, key): #새로운 노드를 생성해 x뒤에
    self.moveAfter(self, Node(key), x) #키로 노드를 생성하면 초기에prev,next가 자기자신이다.

def insertBefore(self, x, key):
    self.moveBefore(self, Node(key), x)

def pushfront(self, key):
    self.insertAfter(self, self.head, key) #self.head는 더미노드.self.head의 다음값은 리스트의 처음 값


def pushback(self, key):
    self.insertBefore(self, self.head, key) #self.head는 더미노드.self.head의 이전값은 리스트의 마지막값
```  
만약 특정 노드 앞이나 뒤로 insertAfter나 moveAfter 등의 메서드를 사용하고 싶다면, 노드 x를 리턴받아야한다. 노드를 리턴받는 메서드를 search나 first, last로 만들 수 있다.

#### (5) 삭제 연산  

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/206089907-bcfb42da-e72b-4ea7-abef-d9cdf0527040.png">
</p>

1. 지우고자 하는 노드가 Head 노드이거나 None이면 제거하지 않는다.
2. 노드 x의 앞, 뒤로 링크를 수정한다.
3. del로 노드 x의 메모리를 삭제한다.

```python
def remove(self,x): #x노드를 삭제.
    if x == None or x == self.head:
        pass
    else:
        x.prev.next = x.next #노드 x의 이전 노드의 링크는 x.next
        x.next.prev = x.prev #노드 x의 다음 노드의 prev링크는 x.prev
        del x
        
def popFront(self):
    if self.isEmpty():
        return None

    else:
        key = self.head.next.key #헤드 노드의 다음 키값이 pop해야할 키값이다.
        self.remove(self.head.next) #remove는 앞선노드와 뒷노드를 매개변수를 제외하고 연결해준다.
        return key   

def popback(self):
    if self.isEmpty():
        return None

    else:
        key = self.head.prev.key
        self.remove(self.head.prev)
        return key
```

#### (6) 탐색 연산
```python
def search(self, key):
  v = self.head # dummy node
  while v.next != self.head:
    if v.key == key:
      return v
    v = v.next
  return None
```  
강의에서는 while문으로 돌린 search의 경우(위의 코드), 리스트의 길이가 1인 경우 none-key-none으로 연결되어있어 while문의 return에 걸리지 않아 버그가 발생한다.  
그래서 self.head를 yeild하지 않는 제너레이터함수가 선언되어 있으므로 for문을 사용해 key값을 찾아낸다. search의 시간복잡도는 while문은 쓰면 어차피 O(N)이 되기  
떄문에, for문을 사용해도 상관 없을 거라고 예상된다.
[참조]("https://koreanddinghwan.github.io/datastructurepy/doublylinkedlist/")
  1. 키값을 매개변수로 입력받고 인스턴스 내에서 제너레이터로 각 노드를 리턴받는다.
  2. 노드의 키값이 입력받은 키값과 일치한다면 제너레이터로 리턴된 노드를 리턴한다.
  3. for ~ else문으로 for문이 끝났는데, if문에서 아무런 실행이 없다면 None을 리턴한다.

```python
def search(self,key):
    for i in self:
        if i.key == key:
            return i
    else:
        return None
```        

#### (7) isEmpty  
Head 노드는 Dummy 노드이므로 사이즈 계산에 포함되지 않는다.  
```python
def isEmpty(self):
        if self.size != 0:
            return False
        else:
            return True
```

#### (8) first, last  
self.head는 리스트 상 첫 값이자 마지막값인 것을 이용한다.  

```python
def first(self):
        ch = self.head
        return ch.next #노드리턴

def last(self):
    ch = self.head
    return ch.prev #노드리턴
```

#### (9) join

```python
def join(self, list):
        if self.isEmpty():
            self = list
        elif list.isEmpty():
            self = self
        else:
            self.head.prev.next = list.head.next #self 리스트의 마지막값의 링크는 추가하고자하는 list의 head노드 다음 값이다.
            list.head.next.prev = self.head.prev #추가하고자하는 리스트의 첫값의 prev링크는 self리스트의 마지막값
            list.head.prev.next = self.head #추가하고자하는 리스트의 마지막값의 다음값은 self리스트의 헤드값이되어 서로 원형 연결한다.
            self.head.prev = list.head.prev #self.head의 prev링크는 list의 마지막값이되어야한다.
```

## Reference
[신천수 교수님 강의자료]("https://www.youtube.com/c/ChanSuShin/featured")  
강의 내용의 모든 저작권은 신천수 교수님께 있습니다.
