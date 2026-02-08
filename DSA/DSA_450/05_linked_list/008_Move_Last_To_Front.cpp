/*
Problem: Move Last Element to Front of Linked List
URL: https://www.geeksforgeeks.org/move-last-element-to-front-of-a-given-linked-list/

Problem Statement:
Move the last element of the linked list to the front.

Sample Input/Output:
Input: 1->2->3->4->5->NULL
Output: 5->1->2->3->4->NULL
Explanation: Last node (5) is moved to the front.
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
    ListNode* Move_Last_To_Front_Traverse(ListNode* head) {
        /*
        Traverse to end approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (!head || !head->next) return head;
        
        ListNode* curr = head;
        ListNode* prev = NULL;
        
        while (curr->next) {
            prev = curr;
            curr = curr->next;
        }
        
        prev->next = NULL;
        curr->next = head;
        head = curr;
        
        return head;
    }
    
    ListNode* Move_Last_To_Front_Two_Pointer(ListNode* head) {
        /*
        Two pointer approach (fast pointer reaching end first)
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (!head || !head->next) return head;
        
        ListNode* slow = head;
        ListNode* fast = head->next;
        
        while (fast->next) {
            slow = slow->next;
            fast = fast->next;
        }
        
        fast->next = head;
        head = fast;
        slow->next = NULL;
        
        return head;
    }
};

void Test_Move_Last_To_Front() {
    Solution solution;
    
    vector<int> arr1 = {1, 2, 3, 4, 5};
    ListNode* head1 = Create_List(arr1);
    cout << "Original: ";
    Print_List(head1);
    head1 = solution.Move_Last_To_Front_Traverse(head1);
    cout << "After moving last to front (Traverse): ";
    Print_List(head1);
    
    vector<int> arr2 = {1, 2, 3, 4, 5};
    ListNode* head2 = Create_List(arr2);
    cout << "\nOriginal: ";
    Print_List(head2);
    head2 = solution.Move_Last_To_Front_Two_Pointer(head2);
    cout << "After moving last to front (Two pointer): ";
    Print_List(head2);
    
    vector<int> arr3 = {1, 2};
    ListNode* head3 = Create_List(arr3);
    cout << "\nOriginal: ";
    Print_List(head3);
    head3 = solution.Move_Last_To_Front_Traverse(head3);
    cout << "After moving last to front: ";
    Print_List(head3);
}

int main() {
    Test_Move_Last_To_Front();
    return 0;
}
