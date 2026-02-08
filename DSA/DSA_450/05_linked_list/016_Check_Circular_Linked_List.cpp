/*
Problem: Check if Linked List is Circular
URL: https://practice.geeksforgeeks.org/problems/circular-linked-list/1

Problem Statement:
Given a singly linked list, find if the linked list is circular or not. A linked list is called circular if it not NULL terminated and all nodes are connected in the form of a cycle.

Sample Input/Output:
Input: LinkedList: 1->2->3->4->5->1 (5 is connected to 1)
Output: 1
Explanation: The given linked list is circular.
*/

#include <bits/stdc++.h>
using namespace std;

struct ListNode {
    int data;
    ListNode* next;
    ListNode(int x) : data(x), next(NULL) {}
};

ListNode* Create_Circular_List(vector<int>& arr) {
    if (arr.empty()) return NULL;
    ListNode* head = new ListNode(arr[0]);
    ListNode* curr = head;
    for (int i = 1; i < arr.size(); i++) {
        curr->next = new ListNode(arr[i]);
        curr = curr->next;
    }
    curr->next = head;
    return head;
}

ListNode* Create_Linear_List(vector<int>& arr) {
    if (arr.empty()) return NULL;
    ListNode* head = new ListNode(arr[0]);
    ListNode* curr = head;
    for (int i = 1; i < arr.size(); i++) {
        curr->next = new ListNode(arr[i]);
        curr = curr->next;
    }
    return head;
}

void Print_Circular_List(ListNode* head, int maxNodes = 10) {
    if (!head) {
        cout << "NULL" << endl;
        return;
    }
    ListNode* curr = head;
    int count = 0;
    do {
        cout << curr->data;
        if (curr->next != head && count < maxNodes - 1) cout << "->";
        curr = curr->next;
        count++;
    } while (curr != head && count < maxNodes);
    if (count >= maxNodes) cout << "...";
    cout << endl;
}

class Solution {
public:
    bool Is_Circular_Traverse(ListNode* head) {
        /*
        Traverse and check if last points to head
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (!head) return false;
        
        ListNode* curr = head->next;
        while (curr && curr != head) {
            curr = curr->next;
        }
        
        return curr == head;
    }
    
    bool Is_Circular_Floyd(ListNode* head) {
        /*
        Floyd's approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (!head) return false;
        
        ListNode* slow = head;
        ListNode* fast = head;
        
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) {
                return slow == head || fast == head;
            }
        }
        
        return false;
    }
};

void Test_Check_Circular_Linked_List() {
    Solution solution;
    
    vector<int> arr = {1, 2, 3, 4, 5};
    ListNode* circularHead = Create_Circular_List(arr);
    bool result1 = solution.Is_Circular_Traverse(circularHead);
    cout << "Test 1 - Traverse (Circular): " << result1 << endl;
    
    bool result2 = solution.Is_Circular_Floyd(circularHead);
    cout << "Test 1 - Floyd (Circular): " << result2 << endl;
    
    ListNode* linearHead = Create_Linear_List(arr);
    result1 = solution.Is_Circular_Traverse(linearHead);
    cout << "Test 2 - Traverse (Linear): " << result1 << endl;
    
    result2 = solution.Is_Circular_Floyd(linearHead);
    cout << "Test 2 - Floyd (Linear): " << result2 << endl;
}

int main() {
    Test_Check_Circular_Linked_List();
    return 0;
}
