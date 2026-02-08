/*
Problem: Middle of the Linked List
URL: https://leetcode.com/problems/middle-of-the-linked-list/

Problem Statement:
Given the head of a singly linked list, return the middle node of the linked list. If there are two middle nodes, return the second middle node.

Sample Input/Output:
Input: head = [1,2,3,4,5]
Output: [3,4,5]
Explanation: The middle node of the list is node 3.
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
    ListNode* Middle_Node_Slow_Fast(ListNode* head) {
        /*
        Slow-Fast pointer (Tortoise and Hare)
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        ListNode* slow = head;
        ListNode* fast = head;
        
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        
        return slow;
    }
    
    ListNode* Middle_Node_Count_Based(ListNode* head) {
        /*
        Count-based approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int count = 0;
        ListNode* temp = head;
        while (temp) {
            count++;
            temp = temp->next;
        }
        
        int mid = count / 2;
        temp = head;
        while (mid--) {
            temp = temp->next;
        }
        
        return temp;
    }
};

void Test_Middle_Of_Linked_List() {
    Solution solution;
    
    vector<int> arr = {1, 2, 3, 4, 5};
    ListNode* head = Create_List(arr);
    ListNode* result1 = solution.Middle_Node_Slow_Fast(head);
    cout << "Test 1 - Slow-Fast: ";
    Print_List(result1);
    
    head = Create_List(arr);
    ListNode* result2 = solution.Middle_Node_Count_Based(head);
    cout << "Test 1 - Count-Based: ";
    Print_List(result2);
    
    vector<int> arr2 = {1, 2, 3, 4, 5, 6};
    head = Create_List(arr2);
    result1 = solution.Middle_Node_Slow_Fast(head);
    cout << "Test 2 - Slow-Fast: ";
    Print_List(result1);
}

int main() {
    Test_Middle_Of_Linked_List();
    return 0;
}
