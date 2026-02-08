/*
Problem: Implement N Stacks in a Single Array
URL: https://www.geeksforgeeks.org/efficiently-implement-k-stacks-single-array/

Problem Statement:
Implement N stacks in a single array efficiently using arrays: arr[] for data, top[] for stack tops, next[] for free-list and next-in-stack chain.

Sample Input/Output:
Input: push(0, 10), push(1, 20), push(0, 30), pop(0), pop(1)
Output: pop(0) returns 30, pop(1) returns 20
*/

#include <bits/stdc++.h>
using namespace std;

class NStacks {
private:
    int* arr;
    int* top;
    int* next;
    int n;
    int k;
    int free;

public:
    NStacks(int numStacks, int size) {
        n = size;
        k = numStacks;
        arr = new int[n];
        top = new int[k];
        next = new int[n];
        
        for (int i = 0; i < k; i++) {
            top[i] = -1;
        }
        
        free = 0;
        for (int i = 0; i < n - 1; i++) {
            next[i] = i + 1;
        }
        next[n - 1] = -1;
    }

    ~NStacks() {
        delete[] arr;
        delete[] top;
        delete[] next;
    }

    bool isFull() {
        return free == -1;
    }

    bool isEmpty(int sn) {
        return top[sn] == -1;
    }

    void push(int sn, int x) {
        if (isFull()) {
            cout << "Stack Overflow" << endl;
            return;
        }
        
        int i = free;
        free = next[i];
        next[i] = top[sn];
        top[sn] = i;
        arr[i] = x;
    }

    int pop(int sn) {
        if (isEmpty(sn)) {
            cout << "Stack Underflow" << endl;
            return -1;
        }
        
        int i = top[sn];
        top[sn] = next[i];
        next[i] = free;
        free = i;
        return arr[i];
    }

    int peek(int sn) {
        if (isEmpty(sn)) {
            cout << "Stack is Empty" << endl;
            return -1;
        }
        return arr[top[sn]];
    }
};

class Solution {
public:
    void Test_N_Stacks() {
        NStacks ns(3, 10);
        cout << "N Stacks in Array Tests:" << endl;
        
        ns.push(0, 10);
        ns.push(0, 20);
        ns.push(1, 100);
        ns.push(1, 200);
        ns.push(2, 1000);
        ns.push(2, 2000);
        
        cout << "Stack 0 top: " << ns.peek(0) << endl;
        cout << "Stack 1 top: " << ns.peek(1) << endl;
        cout << "Stack 2 top: " << ns.peek(2) << endl;
        
        cout << "Pop Stack 0: " << ns.pop(0) << endl;
        cout << "Pop Stack 1: " << ns.pop(1) << endl;
        cout << "Pop Stack 2: " << ns.pop(2) << endl;
        
        cout << "Stack 0 top: " << ns.peek(0) << endl;
        cout << "Stack 1 top: " << ns.peek(1) << endl;
        cout << "Stack 2 top: " << ns.peek(2) << endl;
        
        cout << "isEmpty Stack 0: " << ns.isEmpty(0) << endl;
    }
};

void Test_N_Stacks_In_Array() {
    Solution solution;
    solution.Test_N_Stacks();
}

int main() {
    Test_N_Stacks_In_Array();
    return 0;
}
