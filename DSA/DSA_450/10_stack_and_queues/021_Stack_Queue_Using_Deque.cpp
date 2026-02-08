/*
Problem: Implement Stack and Queue using Deque
URL: https://www.geeksforgeeks.org/implement-stack-queue-using-deque/

Problem Statement:
Implement both stack (LIFO) and queue (FIFO) using a doubly-linked-list-based deque.
Create a Deque class, then Stack and Queue classes that use it.

Sample Input/Output:
Input: Stack push(1), push(2), push(3), pop()
Output: 3

Input: Queue enqueue(1), enqueue(2), enqueue(3), dequeue()
Output: 1
*/

#include <bits/stdc++.h>
using namespace std;

class Deque {
private:
    deque<int> dq;

public:
    void Push_Front(int x) {
        dq.push_front(x);
    }

    void Push_Back(int x) {
        dq.push_back(x);
    }

    int Pop_Front() {
        if (dq.empty()) return -1;
        int val = dq.front();
        dq.pop_front();
        return val;
    }

    int Pop_Back() {
        if (dq.empty()) return -1;
        int val = dq.back();
        dq.pop_back();
        return val;
    }

    bool Is_Empty() {
        return dq.empty();
    }

    int Front() {
        return dq.empty() ? -1 : dq.front();
    }

    int Back() {
        return dq.empty() ? -1 : dq.back();
    }
};

class Stack {
private:
    Deque dq;

public:
    void Push(int x) {
        dq.Push_Back(x);
    }

    int Pop() {
        return dq.Pop_Back();
    }

    int Top() {
        return dq.Back();
    }

    bool Is_Empty() {
        return dq.Is_Empty();
    }
};

class Queue {
private:
    Deque dq;

public:
    void Enqueue(int x) {
        dq.Push_Back(x);
    }

    int Dequeue() {
        return dq.Pop_Front();
    }

    int Front() {
        return dq.Front();
    }

    bool Is_Empty() {
        return dq.Is_Empty();
    }
};

class Solution {
public:
    void Test_Stack_Queue_Using_Deque() {
        Stack stack;
        stack.Push(1);
        stack.Push(2);
        stack.Push(3);
        cout << "Stack Pop: " << stack.Pop() << endl;
        cout << "Stack Top: " << stack.Top() << endl;
        cout << "Stack Pop: " << stack.Pop() << endl;

        Queue queue;
        queue.Enqueue(1);
        queue.Enqueue(2);
        queue.Enqueue(3);
        cout << "Queue Dequeue: " << queue.Dequeue() << endl;
        cout << "Queue Front: " << queue.Front() << endl;
        cout << "Queue Dequeue: " << queue.Dequeue() << endl;
    }
};

void Test_Stack_Queue_Using_Deque() {
    Solution solution;
    solution.Test_Stack_Queue_Using_Deque();
}

int main() {
    Test_Stack_Queue_Using_Deque();
    return 0;
}
