/*
Problem: Reverse a Stack using Recursion
URL: https://www.geeksforgeeks.org/reverse-a-stack-using-recursion/

Problem Statement:
Reverse a stack using recursion only (no extra data structure). Uses insert_at_bottom helper.

Sample Input/Output:
Input: stack [1,2,3,4,5]
Output: [5,4,3,2,1]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Insert_At_Bottom(stack<int>& st, int x) {
        if (st.empty()) {
            st.push(x);
            return;
        }
        int top = st.top();
        st.pop();
        Insert_At_Bottom(st, x);
        st.push(top);
    }

    void Reverse_Stack_Recursion(stack<int>& st) {
        if (st.empty()) return;
        int top = st.top();
        st.pop();
        Reverse_Stack_Recursion(st);
        Insert_At_Bottom(st, top);
    }
};

void Test_Reverse_Stack() {
    Solution solution;
    
    stack<int> st;
    st.push(1);
    st.push(2);
    st.push(3);
    st.push(4);
    st.push(5);
    
    cout << "Before reverse: ";
    stack<int> temp = st;
    vector<int> v;
    while (!temp.empty()) {
        v.push_back(temp.top());
        temp.pop();
    }
    reverse(v.begin(), v.end());
    for (int x : v) cout << x << " ";
    cout << endl;
    
    solution.Reverse_Stack_Recursion(st);
    
    cout << "After reverse: ";
    vector<int> v2;
    while (!st.empty()) {
        v2.push_back(st.top());
        st.pop();
    }
    reverse(v2.begin(), v2.end());
    for (int x : v2) cout << x << " ";
    cout << endl;
    
    stack<int> st2;
    st2.push(10);
    st2.push(20);
    st2.push(30);
    
    cout << "\nBefore reverse: ";
    stack<int> temp2 = st2;
    vector<int> v3;
    while (!temp2.empty()) {
        v3.push_back(temp2.top());
        temp2.pop();
    }
    reverse(v3.begin(), v3.end());
    for (int x : v3) cout << x << " ";
    cout << endl;
    
    solution.Reverse_Stack_Recursion(st2);
    
    cout << "After reverse: ";
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
    Test_Reverse_Stack();
    return 0;
}
