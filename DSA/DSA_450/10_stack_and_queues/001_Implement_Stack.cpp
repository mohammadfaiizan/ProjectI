/*
Problem: Implement Stack from Scratch
URL: https://www.geeksforgeeks.org/stack-data-structure-introduction-program/

Problem Statement:
Implement a stack data structure using array with operations: push, pop, top, isEmpty, size, isFull.

Sample Input/Output:
Input: push(10), push(20), top(), pop()
Output: top() returns 20, pop() removes 20
*/

#include <bits/stdc++.h>
using namespace std;

class MyStack_Array {
private:
    int* arr;
    int capacity;
    int topIndex;

public:
    MyStack_Array(int cap = 100) {
        capacity = cap;
        arr = new int[capacity];
        topIndex = -1;
    }

    ~MyStack_Array() {
        delete[] arr;
    }

    void push(int x) {
        if (isFull()) {
            cout << "Stack Overflow" << endl;
            return;
        }
        arr[++topIndex] = x;
    }

    int pop() {
        if (isEmpty()) {
            cout << "Stack Underflow" << endl;
            return -1;
        }
        return arr[topIndex--];
    }

    int top() {
        if (isEmpty()) {
            cout << "Stack is Empty" << endl;
            return -1;
        }
        return arr[topIndex];
    }

    bool isEmpty() {
        return topIndex == -1;
    }

    bool isFull() {
        return topIndex == capacity - 1;
    }

    int size() {
        return topIndex + 1;
    }
};

class Node {
public:
    int data;
    Node* next;
    Node(int val) : data(val), next(nullptr) {}
};

class MyStack_LinkedList {
private:
    Node* head;
    int stackSize;

public:
    MyStack_LinkedList() {
        head = nullptr;
        stackSize = 0;
    }

    void push(int x) {
        Node* newNode = new Node(x);
        newNode->next = head;
        head = newNode;
        stackSize++;
    }

    int pop() {
        if (isEmpty()) {
            cout << "Stack Underflow" << endl;
            return -1;
        }
        Node* temp = head;
        int val = temp->data;
        head = head->next;
        delete temp;
        stackSize--;
        return val;
    }

    int top() {
        if (isEmpty()) {
            cout << "Stack is Empty" << endl;
            return -1;
        }
        return head->data;
    }

    bool isEmpty() {
        return head == nullptr;
    }

    int size() {
        return stackSize;
    }
};

class Solution {
public:
    void Test_Array_Stack() {
        MyStack_Array stack(5);
        cout << "Array Stack Tests:" << endl;
        cout << "isEmpty: " << stack.isEmpty() << endl;
        stack.push(10);
        stack.push(20);
        stack.push(30);
        cout << "Size: " << stack.size() << endl;
        cout << "Top: " << stack.top() << endl;
        cout << "Pop: " << stack.pop() << endl;
        cout << "Top: " << stack.top() << endl;
        cout << "Size: " << stack.size() << endl;
    }

    void Test_LinkedList_Stack() {
        MyStack_LinkedList stack;
        cout << "\nLinked List Stack Tests:" << endl;
        cout << "isEmpty: " << stack.isEmpty() << endl;
        stack.push(10);
        stack.push(20);
        stack.push(30);
        cout << "Size: " << stack.size() << endl;
        cout << "Top: " << stack.top() << endl;
        cout << "Pop: " << stack.pop() << endl;
        cout << "Top: " << stack.top() << endl;
        cout << "Size: " << stack.size() << endl;
    }
};

void Test_Implement_Stack() {
    Solution solution;
    solution.Test_Array_Stack();
    solution.Test_LinkedList_Stack();
}

int main() {
    Test_Implement_Stack();
    return 0;
}
