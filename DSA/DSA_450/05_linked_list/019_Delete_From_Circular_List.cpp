/*
Problem: Deletion from a Circular Linked List
URL: https://www.geeksforgeeks.org/deletion-circular-linked-list/

Problem Statement:
Given a circular linked list and a key, delete the node with the given key.

Sample Input/Output:
Input: List: 1->2->3->4->5 (circular), key = 3
Output: List: 1->2->4->5 (circular)
Explanation: Node with value 3 is removed from the circular list
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
    if (!head) return;
    ListNode* curr = head;
    do {
        cout << curr->data << " ";
        curr = curr->next;
    } while (curr != head);
    cout << endl;
}

class Solution {
public:
    ListNode* Delete_From_Circular_List_Search_Delete(ListNode* head, int key) {
        /*
        Search and delete with edge cases (head, middle, not found)
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (!head) return NULL;
        
        if (head->data == key) {
            if (head->next == head) {
                delete head;
                return NULL;
            }
            ListNode* last = head;
            while (last->next != head) {
                last = last->next;
            }
            last->next = head->next;
            ListNode* new_head = head->next;
            delete head;
            return new_head;
        }
        
        ListNode* curr = head;
        while (curr->next != head) {
            if (curr->next->data == key) {
                ListNode* to_delete = curr->next;
                curr->next = to_delete->next;
                delete to_delete;
                return head;
            }
            curr = curr->next;
        }
        
        return head;
    }
};

void Test_Delete_From_Circular_List() {
    Solution solution;
    
    vector<int> arr1 = {1, 2, 3, 4, 5};
    ListNode* head1 = Create_Circular_List(arr1);
    cout << "Original: ";
    Print_Circular_List(head1);
    head1 = solution.Delete_From_Circular_List_Search_Delete(head1, 3);
    cout << "After deleting 3: ";
    Print_Circular_List(head1);
    
    vector<int> arr2 = {10};
    ListNode* head2 = Create_Circular_List(arr2);
    cout << "Original: ";
    Print_Circular_List(head2);
    head2 = solution.Delete_From_Circular_List_Search_Delete(head2, 10);
    cout << "After deleting 10: ";
    if (head2) Print_Circular_List(head2);
    else cout << "Empty list" << endl;
    
    vector<int> arr3 = {5, 10, 15};
    ListNode* head3 = Create_Circular_List(arr3);
    cout << "Original: ";
    Print_Circular_List(head3);
    head3 = solution.Delete_From_Circular_List_Search_Delete(head3, 5);
    cout << "After deleting head (5): ";
    Print_Circular_List(head3);
}

int main() {
    Test_Delete_From_Circular_List();
    return 0;
}
