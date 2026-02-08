/*
Problem: Merge Sort for Linked List
URL: https://practice.geeksforgeeks.org/problems/sort-a-linked-list/1

Problem Statement:
Given Pointer/Reference to the head of the linked list, the task is to Sort the given linked list using Merge Sort.

Sample Input/Output:
Input: N = 5, value[] = {3,5,2,4,1}
Output: 1->2->3->4->5
Explanation: After sorting the given linked list, the resultant will be 1->2->3->4->5.
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
    cout << endl;
}

class Solution {
public:
    ListNode* Merge_Sort_Recursive(ListNode* head) {
        /*
        Merge Sort recursive with split
        Time Complexity: O(n log n)
        Space Complexity: O(log n)
        */
        if (!head || !head->next) return head;
        
        ListNode* mid = Get_Middle(head);
        ListNode* nextToMid = mid->next;
        mid->next = NULL;
        
        ListNode* left = Merge_Sort_Recursive(head);
        ListNode* right = Merge_Sort_Recursive(nextToMid);
        
        return Merge_Two_Sorted_Lists(left, right);
    }
    
    ListNode* Merge_Sort_Iterative(ListNode* head) {
        /*
        Iterative merge (bottom-up)
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        if (!head || !head->next) return head;
        
        int length = Get_Length(head);
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        
        for (int size = 1; size < length; size *= 2) {
            ListNode* prev = dummy;
            ListNode* curr = dummy->next;
            
            while (curr) {
                ListNode* left = curr;
                ListNode* right = Split_List(left, size);
                curr = Split_List(right, size);
                
                prev->next = Merge_Two_Sorted_Lists(left, right);
                while (prev->next) {
                    prev = prev->next;
                }
            }
        }
        
        head = dummy->next;
        delete dummy;
        return head;
    }
    
private:
    ListNode* Get_Middle(ListNode* head) {
        ListNode* slow = head;
        ListNode* fast = head->next;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        return slow;
    }
    
    ListNode* Merge_Two_Sorted_Lists(ListNode* list1, ListNode* list2) {
        ListNode* dummy = new ListNode(0);
        ListNode* tail = dummy;
        
        while (list1 && list2) {
            if (list1->data <= list2->data) {
                tail->next = list1;
                list1 = list1->next;
            } else {
                tail->next = list2;
                list2 = list2->next;
            }
            tail = tail->next;
        }
        
        tail->next = list1 ? list1 : list2;
        ListNode* result = dummy->next;
        delete dummy;
        return result;
    }
    
    int Get_Length(ListNode* head) {
        int len = 0;
        while (head) {
            len++;
            head = head->next;
        }
        return len;
    }
    
    ListNode* Split_List(ListNode* head, int n) {
        for (int i = 1; head && i < n; i++) {
            head = head->next;
        }
        if (!head) return NULL;
        ListNode* next = head->next;
        head->next = NULL;
        return next;
    }
};

void Test_Merge_Sort_Linked_List() {
    Solution solution;
    
    vector<int> arr = {3, 5, 2, 4, 1};
    ListNode* head = Create_List(arr);
    ListNode* result1 = solution.Merge_Sort_Recursive(head);
    cout << "Test 1 - Recursive: ";
    Print_List(result1);
    
    arr = {3, 5, 2, 4, 1};
    head = Create_List(arr);
    ListNode* result2 = solution.Merge_Sort_Iterative(head);
    cout << "Test 1 - Iterative: ";
    Print_List(result2);
    
    vector<int> arr2 = {4, 2, 1, 3};
    head = Create_List(arr2);
    result1 = solution.Merge_Sort_Recursive(head);
    cout << "Test 2 - Recursive: ";
    Print_List(result1);
}

int main() {
    Test_Merge_Sort_Linked_List();
    return 0;
}
