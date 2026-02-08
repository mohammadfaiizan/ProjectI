/*
Problem: Interleave First Half of Queue with Second Half
URL: https://www.geeksforgeeks.org/interleave-first-half-queue-second-half/

Problem Statement:
Given a queue of even size, interleave first half with second half.

Sample Input/Output:
Input: [11,12,13,14,15,16,17,18,19,20]
Output: [11,16,12,17,13,18,14,19,15,20]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    queue<int> Interleave_Queue_Stack(queue<int>& q) {
        int n = q.size();
        int half = n / 2;
        stack<int> st;
        
        for (int i = 0; i < half; i++) {
            st.push(q.front());
            q.pop();
        }
        
        while (!st.empty()) {
            q.push(st.top());
            st.pop();
        }
        
        for (int i = 0; i < half; i++) {
            q.push(q.front());
            q.pop();
        }
        
        for (int i = 0; i < half; i++) {
            st.push(q.front());
            q.pop();
        }
        
        while (!st.empty()) {
            q.push(st.top());
            st.pop();
            q.push(q.front());
            q.pop();
        }
        
        return q;
    }

    queue<int> Interleave_Queue_Auxiliary(queue<int>& q) {
        int n = q.size();
        int half = n / 2;
        queue<int> aux;
        
        for (int i = 0; i < half; i++) {
            aux.push(q.front());
            q.pop();
        }
        
        while (!aux.empty()) {
            q.push(aux.front());
            aux.pop();
            q.push(q.front());
            q.pop();
        }
        
        return q;
    }
};

void Test_Interleave_Queue_Stack() {
    Solution solution;
    
    queue<int> q1;
    q1.push(11);
    q1.push(12);
    q1.push(13);
    q1.push(14);
    q1.push(15);
    q1.push(16);
    q1.push(17);
    q1.push(18);
    q1.push(19);
    q1.push(20);
    
    queue<int> result1 = solution.Interleave_Queue_Stack(q1);
    cout << "Stack - Interleaved Queue: ";
    while (!result1.empty()) {
        cout << result1.front() << " ";
        result1.pop();
    }
    cout << endl;
}

void Test_Interleave_Queue_Auxiliary() {
    Solution solution;
    
    queue<int> q2;
    q2.push(1);
    q2.push(2);
    q2.push(3);
    q2.push(4);
    q2.push(5);
    q2.push(6);
    
    queue<int> result2 = solution.Interleave_Queue_Auxiliary(q2);
    cout << "Auxiliary - Interleaved Queue: ";
    while (!result2.empty()) {
        cout << result2.front() << " ";
        result2.pop();
    }
    cout << endl;
}

int main() {
    Test_Interleave_Queue_Stack();
    Test_Interleave_Queue_Auxiliary();
    return 0;
}
