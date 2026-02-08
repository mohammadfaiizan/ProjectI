/*
Problem: Find Middle Element of a Stack
URL: https://www.geeksforgeeks.org/design-a-stack-with-find-middle-operation/

Problem Statement:
Design a stack that supports findMiddle and deleteMiddle operations in O(1) time using a doubly linked list with a mid pointer.

Sample Input/Output:
Input: push(10), push(20), push(30), findMiddle(), deleteMiddle()
Output: findMiddle() returns 20, deleteMiddle() removes 20
*/

#include <bits/stdc++.h>
using namespace std;

class DLLNode {
public:
    int data;
    DLLNode* prev;
    DLLNode* next;
    DLLNode(int val) : data(val), prev(nullptr), next(nullptr) {}
};

class MiddleStack {
private:
    DLLNode* head;
    DLLNode* mid;
    int count;

public:
    MiddleStack() {
        head = nullptr;
        mid = nullptr;
        count = 0;
    }

    void push(int x) {
        DLLNode* newNode = new DLLNode(x);
        newNode->next = head;
        
        if (head != nullptr) {
            head->prev = newNode;
        }
        
        head = newNode;
        count++;
        
        if (count == 1) {
            mid = newNode;
        } else if (count % 2 == 0) {
            mid = mid->prev;
        }
    }

    int pop() {
        if (head == nullptr) {
            cout << "Stack Underflow" << endl;
            return -1;
        }
        
        DLLNode* temp = head;
        int val = temp->data;
        head = head->next;
        
        if (head != nullptr) {
            head->prev = nullptr;
        }
        
        count--;
        
        if (count == 0) {
            mid = nullptr;
        } else if (count % 2 == 1) {
            mid = mid->next;
        }
        
        delete temp;
        return val;
    }

    int findMiddle() {
        if (mid == nullptr) {
            cout << "Stack is Empty" << endl;
            return -1;
        }
        return mid->data;
    }

    int deleteMiddle() {
        if (mid == nullptr) {
            cout << "Stack is Empty" << endl;
            return -1;
        }
        
        DLLNode* temp = mid;
        int val = temp->data;
        
        if (temp->prev != nullptr) {
            temp->prev->next = temp->next;
        }
        if (temp->next != nullptr) {
            temp->next->prev = temp->prev;
        }
        
        if (head == mid) {
            head = mid->next;
        }
        
        count--;
        
        if (count == 0) {
            mid = nullptr;
            head = nullptr;
        } else if (count % 2 == 0) {
            mid = mid->prev;
        } else {
            mid = mid->next;
        }
        
        delete temp;
        return val;
    }

    int top() {
        if (head == nullptr) {
            cout << "Stack is Empty" << endl;
            return -1;
        }
        return head->data;
    }

    bool isEmpty() {
        return head == nullptr;
    }
};

class Solution {
public:
    void Test_Middle_Stack() {
        MiddleStack ms;
        cout << "Middle Stack Tests:" << endl;
        
        ms.push(10);
        ms.push(20);
        ms.push(30);
        ms.push(40);
        ms.push(50);
        
        cout << "Top: " << ms.top() << endl;
        cout << "Middle: " << ms.findMiddle() << endl;
        
        cout << "Delete Middle: " << ms.deleteMiddle() << endl;
        cout << "Top: " << ms.top() << endl;
        cout << "Middle: " << ms.findMiddle() << endl;
        
        cout << "Pop: " << ms.pop() << endl;
        cout << "Middle: " << ms.findMiddle() << endl;
        
        cout << "isEmpty: " << ms.isEmpty() << endl;
    }
};

void Test_Middle_Element_Stack() {
    Solution solution;
    solution.Test_Middle_Stack();
}

int main() {
    Test_Middle_Element_Stack();
    return 0;
}
