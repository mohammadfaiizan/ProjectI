/*
Problem: Reverse a Linked List in Groups of Given Size
URL: https://practice.geeksforgeeks.org/problems/reverse-a-linked-list-in-groups-of-given-size/1

Problem Statement:
Given a linked list, reverse every k nodes.

Sample Input/Output:
Input: 1->2->3->4->5->NULL, k=3
Output: 3->2->1->4->5->NULL
Explanation: First 3 nodes reversed, then next group.
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
    ListNode* Reverse_Groups_Recursive(ListNode* head, int k) {
        /*
        Recursive group reversal approach
        Time Complexity: O(n)
        Space Complexity: O(n/k)
        */
        if (!head) return NULL;
        
        ListNode* curr = head;
        ListNode* prev = NULL;
        ListNode* next = NULL;
        int count = 0;
        
        while (curr && count < k) {
            next = curr->next;
            curr->next = prev;
            prev = curr;
            curr = next;
            count++;
        }
        
        if (next) {
            head->next = Reverse_Groups_Recursive(next, k);
        }
        
        return prev;
    }
    
    ListNode* Reverse_Groups_Stack_Based(ListNode* head, int k) {
        /*
        Stack-based approach
        Time Complexity: O(n)
        Space Complexity: O(k)
        */
        if (!head) return NULL;
        
        stack<ListNode*> st;
        ListNode* curr = head;
        ListNode* prev = NULL;
        
        while (curr) {
            int count = 0;
            while (curr && count < k) {
                st.push(curr);
                curr = curr->next;
                count++;
            }
            
            while (!st.empty()) {
                if (!prev) {
                    prev = st.top();
                    head = prev;
                } else {
                    prev->next = st.top();
                    prev = prev->next;
                }
                st.pop();
            }
        }
        
        if (prev) prev->next = NULL;
        return head;
    }
};

void Test_Reverse_Linked_List_In_Groups() {
    Solution solution;
    
    vector<int> arr1 = {1, 2, 3, 4, 5};
    ListNode* head1 = Create_List(arr1);
    cout << "Original: ";
    Print_List(head1);
    head1 = solution.Reverse_Groups_Recursive(head1, 3);
    cout << "Reversed in groups of 3 (Recursive): ";
    Print_List(head1);
    
    vector<int> arr2 = {1, 2, 3, 4, 5, 6, 7, 8};
    ListNode* head2 = Create_List(arr2);
    cout << "\nOriginal: ";
    Print_List(head2);
    head2 = solution.Reverse_Groups_Stack_Based(head2, 4);
    cout << "Reversed in groups of 4 (Stack): ";
    Print_List(head2);
    
    vector<int> arr3 = {1, 2};
    ListNode* head3 = Create_List(arr3);
    cout << "\nOriginal: ";
    Print_List(head3);
    head3 = solution.Reverse_Groups_Recursive(head3, 2);
    cout << "Reversed in groups of 2: ";
    Print_List(head3);
}

int main() {
    Test_Reverse_Linked_List_In_Groups();
    return 0;
}
