/*
Problem: Intersection of Two Sorted Linked Lists
URL: https://practice.geeksforgeeks.org/problems/intersection-of-two-sorted-linked-lists/1

Problem Statement:
Given two lists sorted in increasing order, create a new list representing the intersection of the two lists. The new list should be made with its own memory — the original lists should not be changed.

Sample Input/Output:
Input: First linked list: 1->2->3->4->6, Second linked list: 2->4->6->8
Output: 2->4->6
Explanation: Nodes 2, 4 and 6 are common in both lists.
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
    ListNode* Intersection_Two_Pointer(ListNode* head1, ListNode* head2) {
        /*
        Two pointer on sorted lists
        Time Complexity: O(m + n)
        Space Complexity: O(min(m, n))
        */
        ListNode* dummy = new ListNode(0);
        ListNode* tail = dummy;
        
        while (head1 && head2) {
            if (head1->data == head2->data) {
                tail->next = new ListNode(head1->data);
                tail = tail->next;
                head1 = head1->next;
                head2 = head2->next;
            } else if (head1->data < head2->data) {
                head1 = head1->next;
            } else {
                head2 = head2->next;
            }
        }
        
        ListNode* result = dummy->next;
        delete dummy;
        return result;
    }
    
    ListNode* Intersection_Recursive(ListNode* head1, ListNode* head2) {
        /*
        Recursive approach
        Time Complexity: O(m + n)
        Space Complexity: O(min(m, n))
        */
        if (!head1 || !head2) return NULL;
        
        if (head1->data < head2->data) {
            return Intersection_Recursive(head1->next, head2);
        }
        
        if (head1->data > head2->data) {
            return Intersection_Recursive(head1, head2->next);
        }
        
        ListNode* node = new ListNode(head1->data);
        node->next = Intersection_Recursive(head1->next, head2->next);
        return node;
    }
};

void Test_Intersection_Two_Sorted_Lists() {
    Solution solution;
    
    vector<int> arr1 = {1, 2, 3, 4, 6};
    vector<int> arr2 = {2, 4, 6, 8};
    ListNode* head1 = Create_List(arr1);
    ListNode* head2 = Create_List(arr2);
    ListNode* result1 = solution.Intersection_Two_Pointer(head1, head2);
    cout << "Test 1 - Two Pointer: ";
    Print_List(result1);
    
    head1 = Create_List(arr1);
    head2 = Create_List(arr2);
    ListNode* result2 = solution.Intersection_Recursive(head1, head2);
    cout << "Test 1 - Recursive: ";
    Print_List(result2);
    
    vector<int> arr3 = {1, 3, 5};
    vector<int> arr4 = {2, 4, 6};
    head1 = Create_List(arr3);
    head2 = Create_List(arr4);
    result1 = solution.Intersection_Two_Pointer(head1, head2);
    cout << "Test 2 - Two Pointer: ";
    if (result1) Print_List(result1);
    else cout << "NULL" << endl;
}

int main() {
    Test_Intersection_Two_Sorted_Lists();
    return 0;
}
