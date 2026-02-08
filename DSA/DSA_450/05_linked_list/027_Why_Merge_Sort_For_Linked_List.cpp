/*
Problem: Why Quick Sort is Preferred for Arrays and Merge Sort for Linked Lists
URL: https://www.geeksforgeeks.org/why-quick-sort-preferred-for-arrays-and-merge-sort-for-linked-lists/

Problem Statement:
Quick Sort is preferred for arrays due to cache locality and in-place partitioning. Merge Sort is preferred for linked lists since merge can be done without extra space and there's no random access penalty. This file demonstrates both algorithms on a linked list to compare.

Sample Input/Output:
Input: List: 3 -> 1 -> 4 -> 2 -> 5
Output (Merge Sort): 1 -> 2 -> 3 -> 4 -> 5
Output (Quick Sort): 1 -> 2 -> 3 -> 4 -> 5
Explanation: Both algorithms sort the list, but Merge Sort is more efficient for linked lists
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
    ListNode* curr = head;
    while (curr) {
        cout << curr->data << " ";
        curr = curr->next;
    }
    cout << endl;
}

ListNode* Get_Middle(ListNode* head) {
    if (!head) return NULL;
    ListNode* slow = head;
    ListNode* fast = head->next;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow;
}

ListNode* Merge_Two_Lists(ListNode* left, ListNode* right) {
    ListNode* dummy = new ListNode(0);
    ListNode* curr = dummy;
    
    while (left && right) {
        if (left->data <= right->data) {
            curr->next = left;
            left = left->next;
        } else {
            curr->next = right;
            right = right->next;
        }
        curr = curr->next;
    }
    
    if (left) curr->next = left;
    if (right) curr->next = right;
    
    ListNode* result = dummy->next;
    delete dummy;
    return result;
}

class Solution {
public:
    ListNode* Merge_Sort_Linked_List(ListNode* head) {
        /*
        Merge Sort on linked list
        Time Complexity: O(n log n)
        Space Complexity: O(log n)
        */
        if (!head || !head->next) return head;
        
        ListNode* mid = Get_Middle(head);
        ListNode* right = mid->next;
        mid->next = NULL;
        
        ListNode* left_sorted = Merge_Sort_Linked_List(head);
        ListNode* right_sorted = Merge_Sort_Linked_List(right);
        
        return Merge_Two_Lists(left_sorted, right_sorted);
    }
    
    ListNode* Quick_Sort_Linked_List(ListNode* head) {
        /*
        Quick Sort on linked list
        Time Complexity: O(n log n) average
        Space Complexity: O(log n)
        */
        if (!head || !head->next) return head;
        
        ListNode* pivot = head;
        ListNode* smaller = NULL;
        ListNode* equal = NULL;
        ListNode* larger = NULL;
        
        ListNode* curr = head;
        while (curr) {
            ListNode* next = curr->next;
            if (curr->data < pivot->data) {
                curr->next = smaller;
                smaller = curr;
            } else if (curr->data == pivot->data) {
                curr->next = equal;
                equal = curr;
            } else {
                curr->next = larger;
                larger = curr;
            }
            curr = next;
        }
        
        smaller = Quick_Sort_Linked_List(smaller);
        larger = Quick_Sort_Linked_List(larger);
        
        ListNode* result = NULL;
        ListNode* tail = NULL;
        
        if (smaller) {
            result = smaller;
            tail = smaller;
            while (tail->next) tail = tail->next;
        }
        
        if (equal) {
            if (!result) {
                result = equal;
                tail = equal;
            } else {
                tail->next = equal;
            }
            while (tail->next) tail = tail->next;
        }
        
        if (larger) {
            if (!result) {
                result = larger;
            } else {
                tail->next = larger;
            }
        }
        
        return result;
    }
};

void Test_Why_Merge_Sort_For_Linked_List() {
    Solution solution;
    
    vector<int> arr1 = {3, 1, 4, 2, 5};
    ListNode* head1 = Create_List(arr1);
    cout << "Original: ";
    Print_List(head1);
    
    ListNode* head1_merge = Create_List(arr1);
    head1_merge = solution.Merge_Sort_Linked_List(head1_merge);
    cout << "Merge Sort: ";
    Print_List(head1_merge);
    
    ListNode* head1_quick = Create_List(arr1);
    head1_quick = solution.Quick_Sort_Linked_List(head1_quick);
    cout << "Quick Sort: ";
    Print_List(head1_quick);
    
    vector<int> arr2 = {10, 5, 8, 3, 1, 9, 2};
    ListNode* head2 = Create_List(arr2);
    cout << "Original: ";
    Print_List(head2);
    
    ListNode* head2_merge = Create_List(arr2);
    head2_merge = solution.Merge_Sort_Linked_List(head2_merge);
    cout << "Merge Sort: ";
    Print_List(head2_merge);
    
    ListNode* head2_quick = Create_List(arr2);
    head2_quick = solution.Quick_Sort_Linked_List(head2_quick);
    cout << "Quick Sort: ";
    Print_List(head2_quick);
}

int main() {
    Test_Why_Merge_Sort_For_Linked_List();
    return 0;
}
