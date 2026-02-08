/*
Problem: Implement Stack using Queues
URL: https://practice.geeksforgeeks.org/problems/stack-using-two-queues/1

Problem Statement:
Implement stack using two queues.

Sample Input/Output:
Input: push 1,2,3; pop -> 3,2,1
Output: Stack operations work correctly
*/

#include <bits/stdc++.h>
using namespace std;

class Stack_Using_Queue_Costly_Pop {
private:
    queue<int> q1, q2;

public:
    void Push(int x) {
        q1.push(x);
    }

    int Pop() {
        if (q1.empty()) return -1;
        while (q1.size() > 1) {
            q2.push(q1.front());
            q1.pop();
        }
        int top = q1.front();
        q1.pop();
        swap(q1, q2);
        return top;
    }

    int Top() {
        if (q1.empty()) return -1;
        while (q1.size() > 1) {
            q2.push(q1.front());
            q1.pop();
        }
        int top = q1.front();
        q2.push(q1.front());
        q1.pop();
        swap(q1, q2);
        return top;
    }

    bool Empty() {
        return q1.empty();
    }
};

class Stack_Using_Queue_Costly_Push {
private:
    queue<int> q1, q2;

public:
    void Push(int x) {
        q2.push(x);
        while (!q1.empty()) {
            q2.push(q1.front());
            q1.pop();
        }
        swap(q1, q2);
    }

    int Pop() {
        if (q1.empty()) return -1;
        int top = q1.front();
        q1.pop();
        return top;
    }

    int Top() {
        if (q1.empty()) return -1;
        return q1.front();
    }

    bool Empty() {
        return q1.empty();
    }
};

class Solution {
public:
    void Test_Costly_Pop() {
        Stack_Using_Queue_Costly_Pop st;
        st.Push(1);
        st.Push(2);
        st.Push(3);
        cout << "Pop: " << st.Pop() << endl;
        cout << "Pop: " << st.Pop() << endl;
        cout << "Top: " << st.Top() << endl;
        cout << "Pop: " << st.Pop() << endl;
        cout << "Empty: " << (st.Empty() ? "true" : "false") << endl;
    }

    void Test_Costly_Push() {
        Stack_Using_Queue_Costly_Push st;
        st.Push(1);
        st.Push(2);
        st.Push(3);
        cout << "Pop: " << st.Pop() << endl;
        cout << "Pop: " << st.Pop() << endl;
        cout << "Top: " << st.Top() << endl;
        cout << "Pop: " << st.Pop() << endl;
        cout << "Empty: " << (st.Empty() ? "true" : "false") << endl;
    }
};

void Test_Stack_Using_Queue() {
    Solution solution;
    
    cout << "=== Costly Pop Approach ===" << endl;
    solution.Test_Costly_Pop();
    
    cout << "\n=== Costly Push Approach ===" << endl;
    solution.Test_Costly_Push();
}

int main() {
    Test_Stack_Using_Queue();
    return 0;
}
