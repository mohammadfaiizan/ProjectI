/*
Problem: Implement Two Stacks in an Array
URL: https://practice.geeksforgeeks.org/problems/implement-two-stacks-in-an-array/1

Problem Statement:
Use a single array to implement two stacks efficiently. One stack grows from left to right, the other from right to left.

Sample Input/Output:
Input: push1(10), push2(20), push1(30), pop1(), pop2()
Output: pop1() returns 30, pop2() returns 20
*/

#include <bits/stdc++.h>
using namespace std;

class TwoStacks {
private:
    int* arr;
    int capacity;
    int top1;
    int top2;

public:
    TwoStacks(int cap = 100) {
        capacity = cap;
        arr = new int[capacity];
        top1 = -1;
        top2 = capacity;
    }

    ~TwoStacks() {
        delete[] arr;
    }

    void push1(int x) {
        if (top1 >= top2 - 1) {
            cout << "Stack Overflow" << endl;
            return;
        }
        arr[++top1] = x;
    }

    void push2(int x) {
        if (top1 >= top2 - 1) {
            cout << "Stack Overflow" << endl;
            return;
        }
        arr[--top2] = x;
    }

    int pop1() {
        if (top1 < 0) {
            cout << "Stack1 Underflow" << endl;
            return -1;
        }
        return arr[top1--];
    }

    int pop2() {
        if (top2 >= capacity) {
            cout << "Stack2 Underflow" << endl;
            return -1;
        }
        return arr[top2++];
    }

    int peek1() {
        if (top1 < 0) {
            cout << "Stack1 is Empty" << endl;
            return -1;
        }
        return arr[top1];
    }

    int peek2() {
        if (top2 >= capacity) {
            cout << "Stack2 is Empty" << endl;
            return -1;
        }
        return arr[top2];
    }

    bool isEmpty1() {
        return top1 < 0;
    }

    bool isEmpty2() {
        return top2 >= capacity;
    }
};

class Solution {
public:
    void Test_Two_Stacks() {
        TwoStacks ts(10);
        cout << "Two Stacks in Array Tests:" << endl;
        
        ts.push1(10);
        ts.push1(20);
        ts.push1(30);
        ts.push2(100);
        ts.push2(200);
        ts.push2(300);
        
        cout << "Stack1 top: " << ts.peek1() << endl;
        cout << "Stack2 top: " << ts.peek2() << endl;
        
        cout << "Pop1: " << ts.pop1() << endl;
        cout << "Pop2: " << ts.pop2() << endl;
        
        cout << "Stack1 top: " << ts.peek1() << endl;
        cout << "Stack2 top: " << ts.peek2() << endl;
        
        cout << "isEmpty1: " << ts.isEmpty1() << endl;
        cout << "isEmpty2: " << ts.isEmpty2() << endl;
    }
};

void Test_Two_Stacks_In_Array() {
    Solution solution;
    solution.Test_Two_Stacks();
}

int main() {
    Test_Two_Stacks_In_Array();
    return 0;
}
