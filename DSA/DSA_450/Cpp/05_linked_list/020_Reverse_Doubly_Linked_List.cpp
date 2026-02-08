/*
Problem: Reverse a Doubly Linked List
URL: https://practice.geeksforgeeks.org/problems/reverse-a-doubly-linked-list/1

Problem Statement:
Given a doubly linked list, reverse it.

Sample Input/Output:
Input: 1 <-> 2 <-> 3 <-> 4 <-> 5
Output: 5 <-> 4 <-> 3 <-> 2 <-> 1
Explanation: All pointers are reversed
*/

#include <bits/stdc++.h>
using namespace std;

struct DLLNode {
    int data;
    DLLNode* next;
    DLLNode* prev;
    DLLNode(int x) : data(x), next(NULL), prev(NULL) {}
};

DLLNode* Create_DLL(vector<int>& arr) {
    if (arr.empty()) return NULL;
    DLLNode* head = new DLLNode(arr[0]);
    DLLNode* curr = head;
    for (int i = 1; i < arr.size(); i++) {
        curr->next = new DLLNode(arr[i]);
        curr->next->prev = curr;
        curr = curr->next;
    }
    return head;
}

void Print_DLL(DLLNode* head) {
    DLLNode* curr = head;
    while (curr) {
        cout << curr->data << " ";
        curr = curr->next;
    }
    cout << endl;
}

class Solution {
public:
    DLLNode* Reverse_DLL_Iterative(DLLNode* head) {
        /*
        Iterative pointer swap
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
    
    DLLNode* Reverse_DLL_Stack(DLLNode* head) {
        /*
        Stack-based reversal
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (!head || !head->next) return head;
        
        stack<int> st;
        DLLNode* curr = head;
        while (curr) {
            st.push(curr->data);
            curr = curr->next;
        }
        
        curr = head;
        while (!st.empty()) {
            curr->data = st.top();
            st.pop();
            curr = curr->next;
        }
        
        return head;
    }
};

void Test_Reverse_Doubly_Linked_List() {
    Solution solution;
    
    vector<int> arr1 = {1, 2, 3, 4, 5};
    DLLNode* head1 = Create_DLL(arr1);
    cout << "Original: ";
    Print_DLL(head1);
    head1 = solution.Reverse_DLL_Iterative(head1);
    cout << "Reversed (Iterative): ";
    Print_DLL(head1);
    
    vector<int> arr2 = {10, 20};
    DLLNode* head2 = Create_DLL(arr2);
    cout << "Original: ";
    Print_DLL(head2);
    head2 = solution.Reverse_DLL_Stack(head2);
    cout << "Reversed (Stack): ";
    Print_DLL(head2);
    
    vector<int> arr3 = {5};
    DLLNode* head3 = Create_DLL(arr3);
    cout << "Original: ";
    Print_DLL(head3);
    head3 = solution.Reverse_DLL_Iterative(head3);
    cout << "Reversed: ";
    Print_DLL(head3);
}

int main() {
    Test_Reverse_Doubly_Linked_List();
    return 0;
}
