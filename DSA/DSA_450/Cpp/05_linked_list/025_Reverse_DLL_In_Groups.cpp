/*
Problem: Reverse a Doubly Linked List in Groups of Given Size
URL: https://www.geeksforgeeks.org/reverse-doubly-linked-list-groups-given-size/

Problem Statement:
Given a doubly linked list, reverse it in groups of given size K.

Sample Input/Output:
Input: List: 1 <-> 2 <-> 3 <-> 4 <-> 5 <-> 6, K = 3
Output: List: 3 <-> 2 <-> 1 <-> 6 <-> 5 <-> 4
Explanation: Reversed in groups of 3
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
    DLLNode* Reverse_DLL_Groups_Recursive(DLLNode* head, int k) {
        /*
        Recursive group reversal
        Time Complexity: O(n)
        Space Complexity: O(n/k)
        */
        if (!head) return NULL;
        
        DLLNode* curr = head;
        DLLNode* next = NULL;
        DLLNode* new_head = NULL;
        int count = 0;
        
        while (curr && count < k) {
            next = curr->next;
            curr->next = new_head;
            if (new_head) new_head->prev = curr;
            new_head = curr;
            new_head->prev = NULL;
            curr = next;
            count++;
        }
        
        if (next) {
            head->next = Reverse_DLL_Groups_Recursive(next, k);
            if (head->next) head->next->prev = head;
        }
        
        return new_head;
    }
    
    DLLNode* Reverse_DLL_Groups_Iterative(DLLNode* head, int k) {
        /*
        Iterative approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (!head || k == 1) return head;
        
        DLLNode* dummy = new DLLNode(0);
        dummy->next = head;
        head->prev = dummy;
        
        DLLNode* group_prev = dummy;
        
        while (group_prev->next) {
            DLLNode* group_start = group_prev->next;
            DLLNode* group_end = group_start;
            int count = 1;
            
            while (group_end->next && count < k) {
                group_end = group_end->next;
                count++;
            }
            
            DLLNode* group_next = group_end->next;
            
            DLLNode* prev_node = group_prev;
            DLLNode* curr = group_start;
            
            while (curr != group_next) {
                DLLNode* next_temp = curr->next;
                curr->next = prev_node;
                curr->prev = next_temp;
                prev_node = curr;
                curr = next_temp;
            }
            
            group_prev->next = group_end;
            if (group_next) {
                group_next->prev = group_start;
            }
            group_start->next = group_next;
            group_prev = group_start;
        }
        
        DLLNode* result = dummy->next;
        result->prev = NULL;
        delete dummy;
        return result;
    }
};

void Test_Reverse_DLL_In_Groups() {
    Solution solution;
    
    vector<int> arr1 = {1, 2, 3, 4, 5, 6};
    DLLNode* head1 = Create_DLL(arr1);
    cout << "Original: ";
    Print_DLL(head1);
    head1 = solution.Reverse_DLL_Groups_Recursive(head1, 3);
    cout << "Reversed in groups of 3 (Recursive): ";
    Print_DLL(head1);
    
    vector<int> arr2 = {1, 2, 3, 4, 5, 6, 7, 8};
    DLLNode* head2 = Create_DLL(arr2);
    cout << "Original: ";
    Print_DLL(head2);
    head2 = solution.Reverse_DLL_Groups_Iterative(head2, 3);
    cout << "Reversed in groups of 3 (Iterative): ";
    Print_DLL(head2);
    
    vector<int> arr3 = {1, 2, 3, 4};
    DLLNode* head3 = Create_DLL(arr3);
    cout << "Original: ";
    Print_DLL(head3);
    head3 = solution.Reverse_DLL_Groups_Recursive(head3, 2);
    cout << "Reversed in groups of 2: ";
    Print_DLL(head3);
}

int main() {
    Test_Reverse_DLL_In_Groups();
    return 0;
}
