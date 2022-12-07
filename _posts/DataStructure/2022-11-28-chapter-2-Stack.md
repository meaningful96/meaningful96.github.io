---
title: Chapter 2. Stack(스택)

categories:
  - DataStructure
tags:
  - [DS, DataStructure, Stack]

toc: true
toc_sticky: true

date: 2022-11-28
last_modified_at: 2022-11-28 
---

## 1. Stack
### 1) Stack이란?
Stack은 한 쪽 끝에서만 자료를 넣거나 뺼 수 있는, <span style = "color: aqua">**아래에서부터 저장이 되고 최근에 들어온 값부터 제거가 되는**</span> 선형으로
나열된 자료구조이다. 비유하자면 '프링글스 통'을 예로 들 수 있다. 과자를 먹 을때는 제일 위에서부터 꺼내 먹지만, 과자가 공장에서
생산되어 만들어질 때는 가장 아래쪽부터 쌓인다.

이러한 일반적인 후입선출 구조를 LIFO라고 한다.  
- <span style= "color:aqua">**LIFO : Last-In-First-Out**</span>

### 2) Stack의 연산, Method
- push()
  스택에 원소를 추가한다.
- pop()
  스택 가장 **위**의 원소를 삭제하고 그 원소를 return한다.
- peek()
  스택 가장 위에 있는 원소를 return한다. (삭제하지는 않는다.)
- empty()
  스택이 비어있다면 1(True), 아니면 0(False)를 반환한다.
- len()
  원소의 개수를 return
  
이 메서드들을 이용해서 Python 코드를 짜면 다음과 같다.
```python
class Stack:
    #리스트를 이용한 스택 구현
    def __init__(self):
        self.top = []
    #스택 크기 반환
    def __len__(self) -> bool :
        return len(self.top)
    
    #구현함수
    #스택에 원소 삽입
    def push(self, item):
        self.top.append(item)
    #스택 가장 위에 있는 원소를 삭제하고 반환   
    def pop(self):
        if not self.isEmpty():
            return self.top.pop(-1)
        else:
            print("Stack underflow")
            exit()
    #스택 가장 위에 있는 원소를 반환
    def peek(self):
        if not self.isEmpty():
            return self.top[-1]
        else:
            print("underflow")
            exit()
    #스택이 비어있는 지를 bool값으로 반환
    def isEmpty(self) -> bool :
        return len(self.top)==0
```

### 3) Stack을 이용한 예제 

<p align="center">
<img width="911" alt="image" src="https://user-images.githubusercontent.com/111734605/204209749-d1bca502-827c-4494-85f0-aa9e1ab2be57.png">
</p>

<p align="center">
<img width="911" alt="image" src="https://user-images.githubusercontent.com/111734605/204210246-00c9734c-da36-4844-978d-badfe0a1313c.png">
</p>

<p align="center">
<img width="404" alt="image" src="https://user-images.githubusercontent.com/111734605/204210392-cb6aa294-5fef-4b82-8f6e-cb1f9878797f.png">
</p>

## 2. Stack 구현하기
### 1) Python
```python
class stack:
    def __init___(self):
        self.items = []
    
    def push(self, val):
        self.items.append(val)
    
    def pop(self):
        try:
            return self.items.pop() # pop을 할 아이템이 없으면
        except IndexError:
            print("Stack is empty") # IndexError 발생        
    
    def peek(self):
        try:
            return self.items[-1]
        except IndexError:
            print("Stack is empty")
            
    
    def __len__(self): # len()로 호출하면 stack의 item 수 반환
        return len(self.items)
    
S = Stack()
S.push(1)      # 1 
S.push(10)     # 1, 10
S.push(-3)     # 1, 10, -3
S.push(4)      # 1, 10, -3, 4
print(S.items) # [1,10,-3,4]
S.pop()        # 1, 10, -3
S.pop()        # 1, 10    
```

### 2) C++
```cpp
#include <iostream>
#define MAX 6

using namespace std;
int STACK[MAX], TOP;

// stack initialization
void initStack()
{
	TOP = -1;
}

//Check it is empty or not
int isEmpty()
{
	if(TOP == -1)
		return 1;
	else
		return 0;
}

//Check stack os full or not
int isFull()
{
	if(TOP == MAX -1)
		return 1;
	else 
		return 0;
}

void push(int num)
{
	if(isFull())
	{
		cout<<"STACK is FULL.\n";
		return;
	}
	++TOP;
	STACK[TOP] = num;
	cout<<num<<"has been iserted.\n";
}

void display()
{
	int i;
	if(isEmpty())
	{
		cout<<"STACK is Empty: \n";
		return;
	}
	cout<<"STACK Elements: \n";
	for(i = TOP; i>=0; i--)
	{
		cout<<STACK[i]<<"\n";
	}
	cout<<"\n";
}
//Pop - to remove item
void pop()
{
	int temp;
	if(isEmpty())
	{
		cout<<"STACK is Empty.\n";
		return;
	}
	temp = STACK[TOP];
	TOP--;
	cout<<temp<<"has been deleted.\n";
}
int main()
{
	int num;
	initStack();
	char ch;
	do
	{
		int a;
		cout<<"Choose \n1.push\n"<<"2.pop\n"<<"3.display\n";
		cout<<"Please enter your choice: ";
		cin>>a;
		switch(a)
		{
			case 1:
				cout<<"Enter an Integer Number: ";
				cin>>num;
				push(num);
			break;
			case 2:
				pop();
				break;
			case 3:
				display();
				break;
			default:
				cout<<"An Invalid Choice!!!\n";
		}
	cout<<"Do you want to continue ?";
	cin>>ch;
		
	}
	while(ch == 'Y' || ch == 'y');
	return 0;
}
```

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/111734605/204218000-31e17c82-59eb-498a-a22a-f52f3611a820.png">
</p>


### 3) C++ 
```cpp
#include <iostream>
#include <stack>
using namespace std;
int main(void) {

	stack<int> st;
	stack<int> st2;

	st.push(1);
	st.push(2);
	st.push(3);

	st2.push(10);
	st2.push(20);
	st2.push(30);

	swap(st, st2);

	while (!st.empty()) {
		cout << st.top() << endl;
		st.pop();
	}

	return 0;
}
```
