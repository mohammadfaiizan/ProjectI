/*
Problem: Implement Queue using Stacks
URL: https://practice.geeksforgeeks.org/problems/queue-using-two-stacks/1

Problem Statement:
Implement a queue using two stacks.

Sample Input/Output:
Input: enqueue(1), enqueue(2), enqueue(3), dequeue()
Output: 1
*/

#include <bits/stdc++.h>
using namespace std;

class Queue_Using_Stacks_Costly_Enqueue {
private:
    stack<int> s1, s2;

public:
    void Enqueue(int x) {
        while (!s1.empty()) {
            s2.push(s1.top());
            s1.pop();
        }
        s1.push(x);
        while (!s2.empty()) {
            s1.push(s2.top());
            s2.pop();
        }
    }

    int Dequeue() {
        if (s1.empty()) return -1;
        int val = s1.top();
        s1.pop();
        return val;
    }

    int Front() {
        return s1.empty() ? -1 : s1.top();
    }

    bool Is_Empty() {
        return s1.empty();
    }
};

class Queue_Using_Stacks_Costly_Dequeue {
private:
    stack<int> s1, s2;

public:
    void Enqueue(int x) {
        s1.push(x);
    }

    int Dequeue() {
        if (s1.empty() && s2.empty()) return -1;
        if (s2.empty()) {
            while (!s1.empty()) {
                s2.push(s1.top());
                s1.pop();
            }
        }
        int val = s2.top();
        s2.pop();
        return val;
    }

    int Front() {
        if (s2.empty()) {
            while (!s1.empty()) {
                s2.push(s1.top());
                s1.pop();
            }
        }
        return s2.empty() ? -1 : s2.top();
    }

    bool Is_Empty() {
        return s1.empty() && s2.empty();
    }
};

class Queue_Using_Stacks_Recursion {
private:
    stack<int> s;

public:
    void Enqueue(int x) {
        s.push(x);
    }

    int Dequeue() {
        if (s.empty()) return -1;
        int x = s.top();
        s.pop();
        if (s.empty()) return x;
        int item = Dequeue();
        s.push(x);
        return item;
    }

    int Front() {
        if (s.empty()) return -1;
        int x = s.top();
        s.pop();
        if (s.empty()) {
            s.push(x);
            return x;
        }
        int item = Front();
        s.push(x);
        return item;
    }

    bool Is_Empty() {
        return s.empty();
    }
};

class Solution {
public:
    void Test_Queue_Using_Stacks() {
        Queue_Using_Stacks_Costly_Enqueue q1;
        q1.Enqueue(1);
        q1.Enqueue(2);
        q1.Enqueue(3);
        cout << "Costly Enqueue - Dequeue: " << q1.Dequeue() << endl;
        cout << "Costly Enqueue - Dequeue: " << q1.Dequeue() << endl;

        Queue_Using_Stacks_Costly_Dequeue q2;
        q2.Enqueue(1);
        q2.Enqueue(2);
        q2.Enqueue(3);
        cout << "Costly Dequeue - Dequeue: " << q2.Dequeue() << endl;
        cout << "Costly Dequeue - Dequeue: " << q2.Dequeue() << endl;

        Queue_Using_Stacks_Recursion q3;
        q3.Enqueue(1);
        q3.Enqueue(2);
        q3.Enqueue(3);
        cout << "Recursion - Dequeue: " << q3.Dequeue() << endl;
        cout << "Recursion - Dequeue: " << q3.Dequeue() << endl;
    }
};

void Test_Queue_Using_Stacks() {
    Solution solution;
    solution.Test_Queue_Using_Stacks();
}

int main() {
    Test_Queue_Using_Stacks();
    return 0;
}
