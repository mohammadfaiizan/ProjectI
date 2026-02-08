/*
Problem: Implement Circular Queue
URL: https://www.geeksforgeeks.org/circular-queue-set-1-introduction-array-implementation/

Problem Statement:
Implement circular queue with enqueue, dequeue, display using array with front/rear wrapping.

Sample Input/Output:
Input: enqueue(1), enqueue(2), enqueue(3), dequeue()
Output: 1
*/

#include <bits/stdc++.h>
using namespace std;

class CircularQueue {
private:
    int* arr;
    int front;
    int rear;
    int size;
    int capacity;

public:
    CircularQueue(int cap) : capacity(cap), front(-1), rear(-1), size(0) {
        arr = new int[capacity];
    }

    ~CircularQueue() {
        delete[] arr;
    }

    bool Is_Full() {
        return size == capacity;
    }

    bool Is_Empty() {
        return size == 0;
    }

    void Enqueue(int x) {
        if (Is_Full()) {
            cout << "Queue is full" << endl;
            return;
        }
        if (Is_Empty()) {
            front = rear = 0;
        } else {
            rear = (rear + 1) % capacity;
        }
        arr[rear] = x;
        size++;
    }

    int Dequeue() {
        if (Is_Empty()) {
            cout << "Queue is empty" << endl;
            return -1;
        }
        int val = arr[front];
        if (front == rear) {
            front = rear = -1;
        } else {
            front = (front + 1) % capacity;
        }
        size--;
        return val;
    }

    int Front_Element() {
        if (Is_Empty()) return -1;
        return arr[front];
    }

    int Rear_Element() {
        if (Is_Empty()) return -1;
        return arr[rear];
    }

    void Display() {
        if (Is_Empty()) {
            cout << "Queue is empty" << endl;
            return;
        }
        int i = front;
        cout << "Queue: ";
        while (true) {
            cout << arr[i] << " ";
            if (i == rear) break;
            i = (i + 1) % capacity;
        }
        cout << endl;
    }
};

class Solution {
public:
    void Test_Circular_Queue() {
        CircularQueue cq(5);
        
        cq.Enqueue(1);
        cq.Enqueue(2);
        cq.Enqueue(3);
        cq.Display();
        
        cout << "Dequeue: " << cq.Dequeue() << endl;
        cout << "Front: " << cq.Front_Element() << endl;
        cout << "Rear: " << cq.Rear_Element() << endl;
        
        cq.Enqueue(4);
        cq.Enqueue(5);
        cq.Enqueue(6);
        cq.Display();
        
        cout << "Dequeue: " << cq.Dequeue() << endl;
        cout << "Dequeue: " << cq.Dequeue() << endl;
        cq.Display();
    }
};

void Test_Circular_Queue() {
    Solution solution;
    solution.Test_Circular_Queue();
}

int main() {
    Test_Circular_Queue();
    return 0;
}
