/*
Problem: Implement Queue from Scratch
URL: https://www.geeksforgeeks.org/queue-set-1introduction-and-array-implementation/

Problem Statement:
Implement a queue data structure using array with operations: enqueue, dequeue, front, rear, isEmpty, isFull.

Sample Input/Output:
Input: enqueue(10), enqueue(20), front(), dequeue()
Output: front() returns 10, dequeue() removes 10
*/

#include <bits/stdc++.h>
using namespace std;

class MyQueue_Array {
private:
    int* arr;
    int capacity;
    int frontIndex;
    int rearIndex;
    int queueSize;

public:
    MyQueue_Array(int cap = 100) {
        capacity = cap;
        arr = new int[capacity];
        frontIndex = 0;
        rearIndex = -1;
        queueSize = 0;
    }

    ~MyQueue_Array() {
        delete[] arr;
    }

    void enqueue(int x) {
        if (isFull()) {
            cout << "Queue Overflow" << endl;
            return;
        }
        rearIndex = (rearIndex + 1) % capacity;
        arr[rearIndex] = x;
        queueSize++;
    }

    int dequeue() {
        if (isEmpty()) {
            cout << "Queue Underflow" << endl;
            return -1;
        }
        int val = arr[frontIndex];
        frontIndex = (frontIndex + 1) % capacity;
        queueSize--;
        return val;
    }

    int front() {
        if (isEmpty()) {
            cout << "Queue is Empty" << endl;
            return -1;
        }
        return arr[frontIndex];
    }

    int rear() {
        if (isEmpty()) {
            cout << "Queue is Empty" << endl;
            return -1;
        }
        return arr[rearIndex];
    }

    bool isEmpty() {
        return queueSize == 0;
    }

    bool isFull() {
        return queueSize == capacity;
    }

    int size() {
        return queueSize;
    }
};

class QueueNode {
public:
    int data;
    QueueNode* next;
    QueueNode(int val) : data(val), next(nullptr) {}
};

class MyQueue_LinkedList {
private:
    QueueNode* frontPtr;
    QueueNode* rearPtr;
    int queueSize;

public:
    MyQueue_LinkedList() {
        frontPtr = nullptr;
        rearPtr = nullptr;
        queueSize = 0;
    }

    void enqueue(int x) {
        QueueNode* newNode = new QueueNode(x);
        if (rearPtr == nullptr) {
            frontPtr = rearPtr = newNode;
        } else {
            rearPtr->next = newNode;
            rearPtr = newNode;
        }
        queueSize++;
    }

    int dequeue() {
        if (isEmpty()) {
            cout << "Queue Underflow" << endl;
            return -1;
        }
        QueueNode* temp = frontPtr;
        int val = temp->data;
        frontPtr = frontPtr->next;
        if (frontPtr == nullptr) {
            rearPtr = nullptr;
        }
        delete temp;
        queueSize--;
        return val;
    }

    int front() {
        if (isEmpty()) {
            cout << "Queue is Empty" << endl;
            return -1;
        }
        return frontPtr->data;
    }

    int rear() {
        if (isEmpty()) {
            cout << "Queue is Empty" << endl;
            return -1;
        }
        return rearPtr->data;
    }

    bool isEmpty() {
        return frontPtr == nullptr;
    }

    int size() {
        return queueSize;
    }
};

class Solution {
public:
    void Test_Array_Queue() {
        MyQueue_Array queue(5);
        cout << "Array Queue Tests:" << endl;
        cout << "isEmpty: " << queue.isEmpty() << endl;
        queue.enqueue(10);
        queue.enqueue(20);
        queue.enqueue(30);
        cout << "Size: " << queue.size() << endl;
        cout << "Front: " << queue.front() << endl;
        cout << "Rear: " << queue.rear() << endl;
        cout << "Dequeue: " << queue.dequeue() << endl;
        cout << "Front: " << queue.front() << endl;
        cout << "Size: " << queue.size() << endl;
    }

    void Test_LinkedList_Queue() {
        MyQueue_LinkedList queue;
        cout << "\nLinked List Queue Tests:" << endl;
        cout << "isEmpty: " << queue.isEmpty() << endl;
        queue.enqueue(10);
        queue.enqueue(20);
        queue.enqueue(30);
        cout << "Size: " << queue.size() << endl;
        cout << "Front: " << queue.front() << endl;
        cout << "Rear: " << queue.rear() << endl;
        cout << "Dequeue: " << queue.dequeue() << endl;
        cout << "Front: " << queue.front() << endl;
        cout << "Size: " << queue.size() << endl;
    }
};

void Test_Implement_Queue() {
    Solution solution;
    solution.Test_Array_Queue();
    solution.Test_LinkedList_Queue();
}

int main() {
    Test_Implement_Queue();
    return 0;
}
