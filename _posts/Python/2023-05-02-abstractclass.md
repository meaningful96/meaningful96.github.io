---
title: "[Python]추상 클래스(Abstract Method)"

categories:
  - py


toc: true
toc_sticky: true

date:  2023-05-02
last_modified_at: 2023-05-02 
---

# 추상 클래스(Abstract Class)
## 1. 추상 클래스의 정의

논문 구현을 위해 코드를 살펴보다 보면 `from abc import abstractmethod`라는 라인을 보는 경우가 많다. 이건 **추상 클래스**를 호출하기 위해 abc 모듈로부터 abstractmethod라는 
패키지를 호출 한 것이다. 추상 클래스는 <span style = "color:red">**메서드의 목록만 가진 클래스이며 상속받는 클래스에서 메서드 구현을 강제하기 위해 사용**</span>한다. 즉 추상 클래스 내부에서는 `self`를 사용하지 않고 함수를 정의한다.

추상 클래스는 현재 클래스에서는 행동(함수, Method, Instance)를 지정하기 애매하거나 너무 다양할 때 사용한다. 예를 들어, `Person`이라면 `eat`나 `sleep`를 할 수 있다. 그런데 사람마다
 직접 밥을 어떻게 먹고, 잠을 어떻게 자고하는 것이 다르기 때문에 하나의 클래스로 정의하는 것은 불가능하다. 즉, <u>사람마다 구체적인 행동 수칙이 다르기 때문</u>이다. 그렇기 때문에 
`Person`의 `eat, sleep`등을 추상메서드로 두고 자식 클래스에서 구현하도록 하는 것이다.

```python
from abc import *

## 추상 클래스 정의(Person)
class Person(metaclass=ABCMeta):
    @abstractmethod
    def eat(self):
        pass

    @abstractmethod
    def sleep(self):
        pass

class James(Person):
    def eat(self):
        print("chop chop")

    def sleep(self):
        print("coa coa")

## 자식 클래스 정의(추상 클래스 Person을 상속받는다.)
class Dean(Person):
    def eat(self):
        print("yam yam")

    def sleep(self):
        print("zzzz")
        
james = James()
dean = Dean()

james.eat() # chop chop
james.sleep() # coa coa
dean.eat() # yam yam
dean.sleep() # zzzz        
```
다음과 같이 `Person`의 `eat, sleep`을 James와 Dean이 상속받아 오버라이딩하면 된다. 만약 오버라이딩을 하지 않으면 에러가 발생하게 된다.

```python
class Dean(Person):
    def eat(self):
        print("yam yam")
```
```python
TypeError: Can't instantiate abstract class Dean with abstract method sleep
```
지금의 경우는 `Dean`의 클래스에서 `slee` 오버라이드 부분을 지웠을 때 발생하는 에러이다. 즉, <u>추상 클래스의 추상 메서드를 자식 클래스에서 오버라이드하지 않았다는 의미</u>이다.  
이처럼 <span style = "color:red">**추상 클래스는 자식 클래스가 반드시 구현해야 하는 메서드를 정해줄 수 있다**</span>는 특징이 있다.

## 2. 추상 클래스의 특징
### 1) 추상 클래스는 인스턴스화 할 수 없다.

Python에서든 Java에서든 추상 클래스는 인스턴스화 할 수 없다. 그 이유는 사실 정의를 생각하면 간단하다. <u>인스턴스화해서 쓰려고 만든 것이 아니기 때문이다.</u> 그래서 추상 메서드의  
본문 부분이 모두 `pass`인 것이다.

```python
from abc import *

class Person(metaclass=ABCMeta):
    @abstractmethod
    def eat(self):
        pass

    @abstractmethod
    def sleep(self):
        pass

person = Person() # TypeError: Can't instantiate abstract class Person with abstract methods eat, sleep
```
위의 경우는 추상 클래스를 인스턴스화 하려고 한 시도고 이는 에러가 발생하게 된다. 따라서, 추상 클래스는 오직 <span style = "color:red">**상속에만 사용**</span>한다.

### 2) 추상 클래스도 Class이다.
추상 클래스도 클래스이다. 때문에 <span stlye = "color:red">**클래스는 상태(속성, 변수)와 행동(메서드)를 가진다**</span> 는 말이 성립된다. 
단지, 추상 메서드를 사용할 수 있을 뿐이고, 인스턴스화 하지 못할 뿐이다.

```python
from abc import *

class Person(metaclass=ABCMeta):
    heart = "두근두근"
    mind = ": love, sad, happy, angry"
    countofheart = 1
    
    def readme(self):
        return Person.heart + Person.mind + str(Person.countofheart)
    
    @abstractmethod
    def eat(self):
        pass
    
    @abstractmethod
    def sleep(self):
        pass
    
class James(Person):
    def eat(self):
        print(Person.heart, Person.mind, Person.countofheart)
        print("chop chop")
            
    def sleep(self):
        print(self.readme())
        print("coa coa")
            

james = James()
james.eat()
james.sleep()
```
```python
## 출력
두근두근 : love, sad, happy, angry 1
chop chop
두근두근: love, sad, happy, angry1
coa coa
```
이처럼 `Person`은 추상 클래스 임에도 클래스 변수 `heart, mind, countofhear`를 가질 수 있다. `readme`메서드는 일반 메서드이므로 추상 클래스에서도 일반 메서드를 만들 수 있다는 것을 
확인할 수 있다. 그리고, 상속받은 자식에서도 추상 클래스의 일반 메서드(`readme`)를 호출할 수 있다는 것을 확인할 수 있다. 물론 멤버 변수도 가질 수 있고, 자식에서 호출도 가능하다.





