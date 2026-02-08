/*
Problem: Reverse a Linked List
URL: https://www.geeksforgeeks.org/reverse-a-linked-list/

Problem Statement:
Given a linked list, reverse it.

Sample Input/Output:
Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL
Explanation: The linked list is reversed completely.
*/

#include <bits/stdc++.h>
using namespace std;

struct ListNode {
    int data;
    ListNode* next;
    ListNode(int x) : data(x), next(NULL) {}
};

ListNode* Create_List(vector<int>& arr) {
    if (arr.empty()) return NULL;
    ListNode* head = new ListNode(arr[0]);
    ListNode* curr = head;
    for (int i = 1; i < arr.size(); i++) {
        curr->next = new ListNode(arr[i]);
        curr = curr->next;
    }
    return head;
}

void Print_List(ListNode* head) {
    while (head) {
        cout << head->data;
        if (head->next) cout << "->";
        head = head->next;
    }
    cout << "->NULL" << endl;
}

class Solution {
public:
    ListNode* Reverse_Iterative(ListNode* head) {
        /*
        Iterative approach using three pointers
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        ListNode* prev = NULL;
        ListNode* curr = head;
        ListNode* next = NULL;
        
        while (curr) {
            next = curr->next;
            curr->next = prev;
            prev = curr;
            curr = next;
        }
        
        return prev;
    }
    
    ListNode* Reverse_Recursive(ListNode* head) {
        /*
        Recursive approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (!head || !head->next) return head;
        
        ListNode* rest = Reverse_Recursive(head->next);
        head->next->next = head;
        head->next = NULL;
        
        return rest;
    }
    
    ListNode* Reverse_Stack_Based(ListNode* head) {
        /*
        Stack-based approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (!head || !head->next) return head;
        
        stack<ListNode*> st;
        ListNode* curr = head;
        
        while (curr) {
            st.push(curr);
            curr = curr->next;
        }
        
        ListNode* newHead = st.top();
        st.pop();
        curr = newHead;
        
        while (!st.empty()) {
            curr->next = st.top();
            st.pop();
            curr = curr->next;
        }
        
        curr->next = NULL;
        return newHead;
    }
};

void Test_Reverse_Linked_List() {
    Solution solution;
    
    vector<int> arr1 = {1, 2, 3, 4, 5};
    ListNode* head1 = Create_List(arr1);
    cout << "Original: ";
    Print_List(head1);
    head1 = solution.Reverse_Iterative(head1);
    cout << "Reversed (Iterative): ";
    Print_List(head1);
    
    vector<int> arr2 = {1, 2};
    ListNode* head2 = Create_List(arr2);
    cout << "\nOriginal: ";
    Print_List(head2);
    head2 = solution.Reverse_Recursive(head2);
    cout << "Reversed (Recursive): ";
    Print_List(head2);
    
    vector<int> arr3 = {1};
    ListNode* head3 = Create_List(arr3);
    cout << "\nOriginal: ";
    Print_List(head3);
    head3 = solution.Reverse_Stack_Based(head3);
    cout << "Reversed (Stack): ";
    Print_List(head3);
}

int main() {
    Test_Reverse_Linked_List();
    return 0;
}
