/*
Problem: Reverse First K Elements of Queue
URL: https://practice.geeksforgeeks.org/problems/reverse-first-k-elements-of-queue/1

Problem Statement:
Given an integer K and a queue of integers, reverse the order of the first K elements
of the queue, leaving the other elements in the same relative order.

Sample Input/Output:
Input: queue = [1, 2, 3, 4, 5], k = 3
Output: [3, 2, 1, 4, 5]

Input: queue = [4, 3, 2, 1], k = 4
Output: [1, 2, 3, 4]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    queue<int> Reverse_First_K_Stack(queue<int> q, int k) {
        /*
        Stack-based approach - Push first K to stack, pop back, rotate remaining
        Time Complexity: O(n)
        Space Complexity: O(k)
        */
        if (k <= 0 || k > (int)q.size()) return q;
        stack<int> st;
        for (int i = 0; i < k; i++) {
            st.push(q.front());
            q.pop();
        }
        while (!st.empty()) {
            q.push(st.top());
            st.pop();
        }
        int remaining = q.size() - k;
        for (int i = 0; i < remaining; i++) {
            q.push(q.front());
            q.pop();
        }
        return q;
    }

    queue<int> Reverse_First_K_Recursive(queue<int>& q, int k) {
        /*
        Recursive approach - Recursively dequeue first K, then enqueue in reverse
        Time Complexity: O(n)
        Space Complexity: O(k) recursion stack
        */
        if (k <= 0) return q;
        int val = q.front();
        q.pop();
        Reverse_First_K_Recursive(q, k - 1);
        q.push(val);
        int remaining = q.size() - 1;
        for (int i = 0; i < remaining; i++) {
            q.push(q.front());
            q.pop();
        }
        return q;
    }
};

void Test_Reverse_First_K() {
    Solution solution;

    auto Print_Queue = [](queue<int> q) {
        cout << "[";
        bool first = true;
        while (!q.empty()) {
            if (!first) cout << ", ";
            cout << q.front();
            q.pop();
            first = false;
        }
        cout << "]" << endl;
    };

    queue<int> q1;
    for (int x : {1, 2, 3, 4, 5}) q1.push(x);
    cout << "Original: ";
    Print_Queue(q1);
    cout << "Reverse first 3 (Stack): ";
    Print_Queue(solution.Reverse_First_K_Stack(q1, 3));

    cout << string(50, '-') << endl;

    queue<int> q2;
    for (int x : {4, 3, 2, 1}) q2.push(x);
    cout << "Original: ";
    Print_Queue(q2);
    cout << "Reverse first 4 (Stack): ";
    Print_Queue(solution.Reverse_First_K_Stack(q2, 4));

    cout << string(50, '-') << endl;

    queue<int> q3;
    for (int x : {10, 20, 30, 40, 50, 60}) q3.push(x);
    cout << "Original: ";
    Print_Queue(q3);
    cout << "Reverse first 2 (Stack): ";
    Print_Queue(solution.Reverse_First_K_Stack(q3, 2));

    cout << string(50, '-') << endl;

    queue<int> q4;
    q4.push(1);
    cout << "Original: ";
    Print_Queue(q4);
    cout << "Reverse first 1 (Stack): ";
    Print_Queue(solution.Reverse_First_K_Stack(q4, 1));
}

int main() {
    Test_Reverse_First_K();
    return 0;
}
