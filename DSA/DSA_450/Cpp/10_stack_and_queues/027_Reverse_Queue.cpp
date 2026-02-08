/*
Problem: Reverse a Queue
URL: https://practice.geeksforgeeks.org/problems/queue-reversal/1

Problem Statement:
Reverse all elements in a queue.

Sample Input/Output:
Input: queue [1,2,3,4,5]
Output: [5,4,3,2,1]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    queue<int> Reverse_Queue_Recursion(queue<int>& q) {
        if (q.empty()) return q;
        int front = q.front();
        q.pop();
        queue<int> reversed = Reverse_Queue_Recursion(q);
        reversed.push(front);
        return reversed;
    }

    queue<int> Reverse_Queue_Stack(queue<int>& q) {
        stack<int> st;
        while (!q.empty()) {
            st.push(q.front());
            q.pop();
        }
        while (!st.empty()) {
            q.push(st.top());
            st.pop();
        }
        return q;
    }
};

void Test_Reverse_Queue_Recursion() {
    Solution solution;
    queue<int> q1;
    q1.push(1);
    q1.push(2);
    q1.push(3);
    q1.push(4);
    q1.push(5);
    
    queue<int> reversed1 = solution.Reverse_Queue_Recursion(q1);
    cout << "Recursion - Reversed Queue: ";
    while (!reversed1.empty()) {
        cout << reversed1.front() << " ";
        reversed1.pop();
    }
    cout << endl;
}

void Test_Reverse_Queue_Stack() {
    Solution solution;
    queue<int> q2;
    q2.push(1);
    q2.push(2);
    q2.push(3);
    q2.push(4);
    q2.push(5);
    
    queue<int> reversed2 = solution.Reverse_Queue_Stack(q2);
    cout << "Stack - Reversed Queue: ";
    while (!reversed2.empty()) {
        cout << reversed2.front() << " ";
        reversed2.pop();
    }
    cout << endl;
}

int main() {
    Test_Reverse_Queue_Recursion();
    Test_Reverse_Queue_Stack();
    return 0;
}
