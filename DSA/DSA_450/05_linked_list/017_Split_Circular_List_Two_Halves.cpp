/*
Problem: Split a Circular Linked List into Two Halves
URL: https://practice.geeksforgeeks.org/problems/split-a-circular-linked-list-into-two-halves/1

Problem Statement:
Given a Cirular Linked List of size N, split it into two halves circular lists. If there are odd number of nodes in the given circular linked list then out of the resulting two halved lists, first list should have one node more than the second list. The resultant lists should also be circular lists and not linear lists.

Sample Input/Output:
Input: Circular LinkedList: 1->5->7
Output: 1->5 and 7->1
Explanation: Your function will split the given circular linked list into two circular linked lists, one having 1->5 and another having 7->1.
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

void Print_Circular_List(ListNode* head) {
    if (!head) {
        cout << "NULL" << endl;
        return;
    }
    ListNode* curr = head;
    do {
        cout << curr->data;
        if (curr->next != head) cout << "->";
        curr = curr->next;
    } while (curr != head);
    cout << endl;
}

class Solution {
public:
    void Split_Circular_List_Size_Based(ListNode* head, ListNode** head1, ListNode** head2) {
        /*
        Size-based approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (!head) {
            *head1 = NULL;
            *head2 = NULL;
            return;
        }
        
        int count = 1;
        ListNode* curr = head->next;
        while (curr != head) {
            count++;
            curr = curr->next;
        }
        
        int mid = (count + 1) / 2;
        curr = head;
        for (int i = 1; i < mid; i++) {
            curr = curr->next;
        }
        
        *head1 = head;
        *head2 = curr->next;
        curr->next = *head1;
        
        ListNode* tail = *head2;
        while (tail->next != head) {
            tail = tail->next;
        }
        tail->next = *head2;
    }
    
    void Split_Circular_List_Slow_Fast(ListNode* head, ListNode** head1, ListNode** head2) {
        /*
        Slow-Fast pointer approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (!head) {
            *head1 = NULL;
            *head2 = NULL;
            return;
        }
        
        ListNode* slow = head;
        ListNode* fast = head;
        
        while (fast->next != head && fast->next->next != head) {
            slow = slow->next;
            fast = fast->next->next;
        }
        
        if (fast->next->next == head) {
            fast = fast->next;
        }
        
        *head1 = head;
        if (head->next != head) {
            *head2 = slow->next;
        } else {
            *head2 = head;
        }
        
        fast->next = slow->next;
        slow->next = head;
    }
};

void Test_Split_Circular_List_Two_Halves() {
    Solution solution;
    
    vector<int> arr = {1, 5, 7};
    ListNode* head = Create_Circular_List(arr);
    ListNode* head1 = NULL, *head2 = NULL;
    solution.Split_Circular_List_Size_Based(head, &head1, &head2);
    cout << "Test 1 - Size-Based:" << endl;
    cout << "First half: ";
    Print_Circular_List(head1);
    cout << "Second half: ";
    Print_Circular_List(head2);
    
    arr = {1, 2, 3, 4, 5};
    head = Create_Circular_List(arr);
    head1 = NULL;
    head2 = NULL;
    solution.Split_Circular_List_Slow_Fast(head, &head1, &head2);
    cout << "Test 2 - Slow-Fast:" << endl;
    cout << "First half: ";
    Print_Circular_List(head1);
    cout << "Second half: ";
    Print_Circular_List(head2);
}

int main() {
    Test_Split_Circular_List_Two_Halves();
    return 0;
}
