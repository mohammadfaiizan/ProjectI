/*
Problem: Sort a Stack using Recursion
URL: https://practice.geeksforgeeks.org/problems/sort-a-stack/1

Problem Statement:
Sort a stack in ascending order (top is largest) using recursion or a temporary stack.

Sample Input/Output:
Input: stack [34,3,31,98,92,23]
Output: sorted stack
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Sorted_Insert(stack<int>& st, int x) {
        if (st.empty() || st.top() <= x) {
            st.push(x);
            return;
        }
        int top = st.top();
        st.pop();
        Sorted_Insert(st, x);
        st.push(top);
    }

    void Sort_Stack_Recursion(stack<int>& st) {
        if (st.empty()) return;
        int top = st.top();
        st.pop();
        Sort_Stack_Recursion(st);
        Sorted_Insert(st, top);
    }

    void Sort_Stack_Iterative(stack<int>& st) {
        stack<int> temp;
        while (!st.empty()) {
            int x = st.top();
            st.pop();
            while (!temp.empty() && temp.top() > x) {
                st.push(temp.top());
                temp.pop();
            }
            temp.push(x);
        }
        while (!temp.empty()) {
            st.push(temp.top());
            temp.pop();
        }
    }
};

void Test_Sort_Stack() {
    Solution solution;
    
    cout << "=== Recursion Approach ===" << endl;
    stack<int> st1;
    st1.push(34);
    st1.push(3);
    st1.push(31);
    st1.push(98);
    st1.push(92);
    st1.push(23);
    
    cout << "Before sort: ";
    stack<int> temp1 = st1;
    vector<int> v1;
    while (!temp1.empty()) {
        v1.push_back(temp1.top());
        temp1.pop();
    }
    reverse(v1.begin(), v1.end());
    for (int x : v1) cout << x << " ";
    cout << endl;
    
    solution.Sort_Stack_Recursion(st1);
    
    cout << "After sort: ";
    vector<int> v2;
    while (!st1.empty()) {
        v2.push_back(st1.top());
        st1.pop();
    }
    reverse(v2.begin(), v2.end());
    for (int x : v2) cout << x << " ";
    cout << endl;
    
    cout << "\n=== Iterative Approach ===" << endl;
    stack<int> st2;
    st2.push(34);
    st2.push(3);
    st2.push(31);
    st2.push(98);
    st2.push(92);
    st2.push(23);
    
    cout << "Before sort: ";
    stack<int> temp2 = st2;
    vector<int> v3;
    while (!temp2.empty()) {
        v3.push_back(temp2.top());
        temp2.pop();
    }
    reverse(v3.begin(), v3.end());
    for (int x : v3) cout << x << " ";
    cout << endl;
    
    solution.Sort_Stack_Iterative(st2);
    
    cout << "After sort: ";
    vector<int> v4;
    while (!st2.empty()) {
        v4.push_back(st2.top());
        st2.pop();
    }
    reverse(v4.begin(), v4.end());
    for (int x : v4) cout << x << " ";
    cout << endl;
}

int main() {
    Test_Sort_Stack();
    return 0;
}
