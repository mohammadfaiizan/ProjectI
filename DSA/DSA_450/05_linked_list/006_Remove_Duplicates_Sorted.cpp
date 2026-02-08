/*
Problem: Remove Duplicates from Sorted Linked List
URL: https://practice.geeksforgeeks.org/problems/remove-duplicate-element-from-sorted-linked-list/1

Problem Statement:
Remove duplicate nodes from a sorted linked list.

Sample Input/Output:
Input: 1->1->2->3->3->4->NULL
Output: 1->2->3->4->NULL
Explanation: Duplicate nodes are removed, keeping only one occurrence.
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
    ListNode* Remove_Duplicates_Iterative(ListNode* head) {
        /*
        Iterative approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (!head || !head->next) return head;
        
        ListNode* curr = head;
        
        while (curr && curr->next) {
            if (curr->data == curr->next->data) {
                ListNode* temp = curr->next;
                curr->next = curr->next->next;
                delete temp;
            } else {
                curr = curr->next;
            }
        }
        
        return head;
    }
    
    ListNode* Remove_Duplicates_Recursive(ListNode* head) {
        /*
        Recursive approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (!head || !head->next) return head;
        
        head->next = Remove_Duplicates_Recursive(head->next);
        
        if (head->data == head->next->data) {
            ListNode* temp = head->next;
            delete head;
            return temp;
        }
        
        return head;
    }
};

void Test_Remove_Duplicates_Sorted() {
    Solution solution;
    
    vector<int> arr1 = {1, 1, 2, 3, 3, 4};
    ListNode* head1 = Create_List(arr1);
    cout << "Original: ";
    Print_List(head1);
    head1 = solution.Remove_Duplicates_Iterative(head1);
    cout << "After removal (Iterative): ";
    Print_List(head1);
    
    vector<int> arr2 = {1, 1, 1};
    ListNode* head2 = Create_List(arr2);
    cout << "\nOriginal: ";
    Print_List(head2);
    head2 = solution.Remove_Duplicates_Recursive(head2);
    cout << "After removal (Recursive): ";
    Print_List(head2);
    
    vector<int> arr3 = {1, 2, 3, 4, 5};
    ListNode* head3 = Create_List(arr3);
    cout << "\nOriginal: ";
    Print_List(head3);
    head3 = solution.Remove_Duplicates_Iterative(head3);
    cout << "After removal (No duplicates): ";
    Print_List(head3);
}

int main() {
    Test_Remove_Duplicates_Sorted();
    return 0;
}
