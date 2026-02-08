/*
Problem: Can We Reverse a Linked List in Less Than O(n)?
URL: https://www.geeksforgeeks.org/can-we-reverse-a-linked-list-in-less-than-on/

Problem Statement:
A singly linked list cannot be reversed in less than O(n). However, a doubly linked list with head and tail pointers can be reversed in O(1) by swapping head and tail pointers (traversal direction changes via prev/next interpretation). This file demonstrates this concept.

Sample Input/Output:
Input: List: 1 <-> 2 <-> 3 <-> 4 <-> 5 (head=1, tail=5)
Output: List: 5 <-> 4 <-> 3 <-> 2 <-> 1 (head=5, tail=1)
Explanation: By swapping head and tail, we effectively reverse the list in O(1)
*/

#include <bits/stdc++.h>
using namespace std;

struct DLLNode {
    int data;
    DLLNode* next;
    DLLNode* prev;
    DLLNode(int x) : data(x), next(NULL), prev(NULL) {}
};

struct DLL {
    DLLNode* head;
    DLLNode* tail;
    DLL() : head(NULL), tail(NULL) {}
};

DLL* Create_DLL_With_Tail(vector<int>& arr) {
    DLL* dll = new DLL();
    if (arr.empty()) return dll;
    
    dll->head = new DLLNode(arr[0]);
    DLLNode* curr = dll->head;
    for (int i = 1; i < arr.size(); i++) {
        curr->next = new DLLNode(arr[i]);
        curr->next->prev = curr;
        curr = curr->next;
    }
    dll->tail = curr;
    return dll;
}

void Print_DLL_Forward(DLLNode* head) {
    DLLNode* curr = head;
    while (curr) {
        cout << curr->data << " ";
        curr = curr->next;
    }
    cout << endl;
}

void Print_DLL_Backward(DLLNode* tail) {
    DLLNode* curr = tail;
    while (curr) {
        cout << curr->data << " ";
        curr = curr->prev;
    }
    cout << endl;
}

class Solution {
public:
    void Reverse_DLL_O1(DLL* dll) {
        /*
        DLL O(1) reversal by swapping head/tail
        Time Complexity: O(1)
        Space Complexity: O(1)
        */
        if (!dll || !dll->head) return;
        DLLNode* temp = dll->head;
        dll->head = dll->tail;
        dll->tail = temp;
    }
    
    DLLNode* Reverse_DLL_Standard(DLLNode* head) {
        /*
        Standard DLL reversal O(n)
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (!head || !head->next) return head;
        
        DLLNode* curr = head;
        DLLNode* temp = NULL;
        
        while (curr) {
            temp = curr->prev;
            curr->prev = curr->next;
            curr->next = temp;
            curr = curr->prev;
        }
        
        if (temp) head = temp->prev;
        return head;
    }
};

void Test_Reverse_DLL_O1() {
    Solution solution;
    
    vector<int> arr1 = {1, 2, 3, 4, 5};
    DLL* dll1 = Create_DLL_With_Tail(arr1);
    cout << "Original (forward): ";
    Print_DLL_Forward(dll1->head);
    cout << "Original (backward): ";
    Print_DLL_Backward(dll1->tail);
    
    solution.Reverse_DLL_O1(dll1);
    cout << "After O(1) reversal (using head as start): ";
    Print_DLL_Backward(dll1->head);
    cout << "After O(1) reversal (using tail as start): ";
    Print_DLL_Forward(dll1->tail);
    
    vector<int> arr2 = {10, 20, 30};
    DLLNode* head2 = Create_DLL_With_Tail(arr2)->head;
    cout << "Original: ";
    Print_DLL_Forward(head2);
    head2 = solution.Reverse_DLL_Standard(head2);
    cout << "After O(n) standard reversal: ";
    Print_DLL_Forward(head2);
    
    vector<int> arr3 = {5};
    DLL* dll3 = Create_DLL_With_Tail(arr3);
    cout << "Original: ";
    Print_DLL_Forward(dll3->head);
    solution.Reverse_DLL_O1(dll3);
    cout << "After O(1) reversal: ";
    Print_DLL_Backward(dll3->head);
}

int main() {
    Test_Reverse_DLL_O1();
    return 0;
}
