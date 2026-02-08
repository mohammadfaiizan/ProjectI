/*
Problem: Sort a K-Sorted Doubly Linked List
URL: https://www.geeksforgeeks.org/sort-k-sorted-doubly-linked-list/

Problem Statement:
Given a K-sorted doubly linked list where each node is at most K positions away from its correct position, sort the list.

Sample Input/Output:
Input: List: 3 <-> 6 <-> 2 <-> 12 <-> 56 <-> 8, K = 2
Output: List: 2 <-> 3 <-> 6 <-> 8 <-> 12 <-> 56
Explanation: Each element is at most 2 positions away from its sorted position
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
    DLLNode* Sort_K_Sorted_Insertion_Sort(DLLNode* head, int k) {
        /*
        Insertion sort with swaps
        Time Complexity: O(n*k)
        Space Complexity: O(1)
        */
        if (!head || !head->next) return head;
        
        DLLNode* curr = head->next;
        while (curr) {
            DLLNode* temp = curr;
            DLLNode* prev = curr->prev;
            
            int count = 0;
            while (prev && prev->data > temp->data && count < k) {
                int swap_data = prev->data;
                prev->data = temp->data;
                temp->data = swap_data;
                temp = prev;
                prev = prev->prev;
                count++;
            }
            
            curr = curr->next;
        }
        
        return head;
    }
    
    DLLNode* Sort_K_Sorted_Min_Heap(DLLNode* head, int k) {
        /*
        Min heap approach
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        */
        if (!head) return NULL;
        
        priority_queue<pair<int, DLLNode*>, vector<pair<int, DLLNode*>>, greater<pair<int, DLLNode*>>> pq;
        DLLNode* curr = head;
        
        for (int i = 0; i <= k && curr; i++) {
            pq.push({curr->data, curr});
            curr = curr->next;
        }
        
        DLLNode* new_head = NULL;
        DLLNode* tail = NULL;
        
        while (!pq.empty()) {
            DLLNode* node = pq.top().second;
            pq.pop();
            
            if (!new_head) {
                new_head = node;
                tail = node;
                new_head->prev = NULL;
            } else {
                tail->next = node;
                node->prev = tail;
                tail = node;
            }
            
            if (curr) {
                pq.push({curr->data, curr});
                curr = curr->next;
            }
        }
        
        if (tail) tail->next = NULL;
        return new_head;
    }
};

void Test_Sort_K_Sorted_DLL() {
    Solution solution;
    
    vector<int> arr1 = {3, 6, 2, 12, 56, 8};
    DLLNode* head1 = Create_DLL(arr1);
    cout << "Original: ";
    Print_DLL(head1);
    head1 = solution.Sort_K_Sorted_Insertion_Sort(head1, 2);
    cout << "Sorted (Insertion Sort): ";
    Print_DLL(head1);
    
    vector<int> arr2 = {3, 6, 2, 12, 56, 8};
    DLLNode* head2 = Create_DLL(arr2);
    cout << "Original: ";
    Print_DLL(head2);
    head2 = solution.Sort_K_Sorted_Min_Heap(head2, 2);
    cout << "Sorted (Min Heap): ";
    Print_DLL(head2);
    
    vector<int> arr3 = {10, 9, 8, 7, 4, 70, 60, 50};
    DLLNode* head3 = Create_DLL(arr3);
    cout << "Original: ";
    Print_DLL(head3);
    head3 = solution.Sort_K_Sorted_Min_Heap(head3, 4);
    cout << "Sorted (K=4): ";
    Print_DLL(head3);
}

int main() {
    Test_Sort_K_Sorted_DLL();
    return 0;
}
