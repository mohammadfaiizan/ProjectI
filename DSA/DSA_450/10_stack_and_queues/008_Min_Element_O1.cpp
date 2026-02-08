/*
Problem: Get Minimum Element from Stack in O(1) Time and O(1) Space
URL: https://practice.geeksforgeeks.org/problems/special-stack/1

Problem Statement:
Design a special stack with getMin in O(1) time and O(1) extra space using the 2*val - min_ele encoding trick.

Sample Input/Output:
Input: push(10), push(20), push(5), getMin(), pop(), getMin()
Output: getMin() returns 5, after pop() getMin() returns 10
*/

#include <bits/stdc++.h>
using namespace std;

class MinStack_O1Space {
private:
    stack<int> st;
    int minEle;

public:
    void push(int x) {
        if (st.empty()) {
            st.push(x);
            minEle = x;
        } else {
            if (x >= minEle) {
                st.push(x);
            } else {
                st.push(2 * x - minEle);
                minEle = x;
            }
        }
    }

    int pop() {
        if (st.empty()) {
            cout << "Stack Underflow" << endl;
            return -1;
        }
        int top = st.top();
        st.pop();
        if (top < minEle) {
            int actualTop = minEle;
            minEle = 2 * minEle - top;
            return actualTop;
        }
        return top;
    }

    int top() {
        if (st.empty()) {
            cout << "Stack is Empty" << endl;
            return -1;
        }
        int top = st.top();
        if (top < minEle) {
            return minEle;
        }
        return top;
    }

    int getMin() {
        if (st.empty()) {
            cout << "Stack is Empty" << endl;
            return -1;
        }
        return minEle;
    }

    bool isEmpty() {
        return st.empty();
    }
};

class MinStack_Auxiliary {
private:
    stack<int> st;
    stack<int> minSt;

public:
    void push(int x) {
        st.push(x);
        if (minSt.empty() || x <= minSt.top()) {
            minSt.push(x);
        }
    }

    int pop() {
        if (st.empty()) {
            cout << "Stack Underflow" << endl;
            return -1;
        }
        int top = st.top();
        st.pop();
        if (top == minSt.top()) {
            minSt.pop();
        }
        return top;
    }

    int top() {
        if (st.empty()) {
            cout << "Stack is Empty" << endl;
            return -1;
        }
        return st.top();
    }

    int getMin() {
        if (minSt.empty()) {
            cout << "Stack is Empty" << endl;
            return -1;
        }
        return minSt.top();
    }

    bool isEmpty() {
        return st.empty();
    }
};

class Solution {
public:
    void Test_O1Space() {
        MinStack_O1Space ms;
        cout << "O(1) Space MinStack Tests:" << endl;
        
        ms.push(10);
        ms.push(20);
        ms.push(5);
        ms.push(15);
        
        cout << "Top: " << ms.top() << endl;
        cout << "Min: " << ms.getMin() << endl;
        
        cout << "Pop: " << ms.pop() << endl;
        cout << "Min: " << ms.getMin() << endl;
        
        cout << "Pop: " << ms.pop() << endl;
        cout << "Min: " << ms.getMin() << endl;
    }

    void Test_Auxiliary() {
        MinStack_Auxiliary ms;
        cout << "\nAuxiliary Stack MinStack Tests:" << endl;
        
        ms.push(10);
        ms.push(20);
        ms.push(5);
        ms.push(15);
        
        cout << "Top: " << ms.top() << endl;
        cout << "Min: " << ms.getMin() << endl;
        
        cout << "Pop: " << ms.pop() << endl;
        cout << "Min: " << ms.getMin() << endl;
        
        cout << "Pop: " << ms.pop() << endl;
        cout << "Min: " << ms.getMin() << endl;
    }
};

void Test_Min_Element_O1() {
    Solution solution;
    solution.Test_O1Space();
    solution.Test_Auxiliary();
}

int main() {
    Test_Min_Element_O1();
    return 0;
}
