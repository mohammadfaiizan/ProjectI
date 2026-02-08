/*
Problem: Implement N Queues in a Single Array
URL: https://www.geeksforgeeks.org/efficiently-implement-k-queues-single-array/

Problem Statement:
Efficiently implement k queues in a single array using front[], rear[], next[] arrays and free list.

Sample Input/Output:
Input: enqueue(1, 0), enqueue(2, 0), enqueue(3, 1), dequeue(0)
Output: 1
*/

#include <bits/stdc++.h>
using namespace std;

class KQueues {
private:
    int n, k;
    vector<int> arr;
    vector<int> front;
    vector<int> rear;
    vector<int> next;
    int free;

public:
    KQueues(int k, int n) : k(k), n(n), free(0) {
        arr.resize(n);
        front.resize(k, -1);
        rear.resize(k, -1);
        next.resize(n);
        
        for (int i = 0; i < n - 1; i++) {
            next[i] = i + 1;
        }
        next[n - 1] = -1;
    }

    bool Is_Full() {
        return free == -1;
    }

    bool Is_Empty(int qn) {
        return front[qn] == -1;
    }

    void Enqueue(int item, int qn) {
        if (Is_Full()) {
            cout << "Queue Overflow" << endl;
            return;
        }

        int i = free;
        free = next[i];

        if (Is_Empty(qn)) {
            front[qn] = i;
        } else {
            next[rear[qn]] = i;
        }

        next[i] = -1;
        rear[qn] = i;
        arr[i] = item;
    }

    int Dequeue(int qn) {
        if (Is_Empty(qn)) {
            cout << "Queue Underflow" << endl;
            return -1;
        }

        int i = front[qn];
        front[qn] = next[i];
        next[i] = free;
        free = i;
        return arr[i];
    }
};

class Solution {
public:
    void Test_N_Queues_In_Array() {
        KQueues kq(3, 10);
        
        kq.Enqueue(15, 2);
        kq.Enqueue(45, 2);
        kq.Enqueue(17, 1);
        kq.Enqueue(49, 1);
        kq.Enqueue(39, 1);
        kq.Enqueue(11, 0);
        kq.Enqueue(9, 0);
        kq.Enqueue(7, 0);

        cout << "Dequeued from queue 2: " << kq.Dequeue(2) << endl;
        cout << "Dequeued from queue 1: " << kq.Dequeue(1) << endl;
        cout << "Dequeued from queue 0: " << kq.Dequeue(0) << endl;
    }
};

void Test_N_Queues_In_Array() {
    Solution solution;
    solution.Test_N_Queues_In_Array();
}

int main() {
    Test_N_Queues_In_Array();
    return 0;
}
