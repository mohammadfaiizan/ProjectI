/*
Problem: Insert Element at the Bottom of a Stack
URL: https://stackoverflow.com/questions/45130465/inserting-at-the-end-of-stack

Problem Statement:
Insert an element at the bottom of a stack without using any other data structure.

Sample Input/Output:
Input: stack [1,2,3,4] insert 0 at bottom
Output: [0,1,2,3,4]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Insert_At_Bottom_Recursion(stack<int>& st, int x) {
        if (st.empty()) {
            st.push(x);
            return;
        }
        int top = st.top();
        st.pop();
        Insert_At_Bottom_Recursion(st, x);
        st.push(top);
    }

    void Insert_At_Bottom_Temp_Stack(stack<int>& st, int x) {
        stack<int> temp;
        while (!st.empty()) {
            temp.push(st.top());
            st.pop();
        }
        st.push(x);
        while (!temp.empty()) {
            st.push(temp.top());
            temp.pop();
        }
    }
};

void Test_Insert_At_Bottom() {
    Solution solution;
    
    cout << "=== Recursion Approach ===" << endl;
    stack<int> st1;
    st1.push(1);
    st1.push(2);
    st1.push(3);
    st1.push(4);
    cout << "Before: ";
    stack<int> temp1 = st1;
    vector<int> v1;
    while (!temp1.empty()) {
        v1.push_back(temp1.top());
        temp1.pop();
    }
    reverse(v1.begin(), v1.end());
    for (int x : v1) cout << x << " ";
    cout << endl;
    
    solution.Insert_At_Bottom_Recursion(st1, 0);
    cout << "After inserting 0: ";
    vector<int> v2;
    while (!st1.empty()) {
        v2.push_back(st1.top());
        st1.pop();
    }
    reverse(v2.begin(), v2.end());
    for (int x : v2) cout << x << " ";
    cout << endl;
    
    cout << "\n=== Temp Stack Approach ===" << endl;
    stack<int> st2;
    st2.push(1);
    st2.push(2);
    st2.push(3);
    st2.push(4);
    cout << "Before: ";
    stack<int> temp2 = st2;
    vector<int> v3;
    while (!temp2.empty()) {
        v3.push_back(temp2.top());
        temp2.pop();
    }
    reverse(v3.begin(), v3.end());
    for (int x : v3) cout << x << " ";
    cout << endl;
    
    solution.Insert_At_Bottom_Temp_Stack(st2, 0);
    cout << "After inserting 0: ";
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
    Test_Insert_At_Bottom();
    return 0;
}
