/*
Problem: Remove Duplicates from Unsorted Linked List
URL: https://practice.geeksforgeeks.org/problems/remove-duplicates-from-an-unsorted-linked-list/1

Problem Statement:
Remove duplicate nodes from an unsorted linked list.

Sample Input/Output:
Input: 5->2->2->4->NULL
Output: 5->2->4->NULL
Explanation: Duplicate node with value 2 is removed.
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
    ListNode* Remove_Duplicates_Hashing(ListNode* head) {
        /*
        Hashing approach using unordered_set
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (!head || !head->next) return head;
        
        unordered_set<int> seen;
        ListNode* curr = head;
        seen.insert(curr->data);
        
        while (curr && curr->next) {
            if (seen.find(curr->next->data) != seen.end()) {
                ListNode* temp = curr->next;
                curr->next = curr->next->next;
                delete temp;
            } else {
                seen.insert(curr->next->data);
                curr = curr->next;
            }
        }
        
        return head;
    }
    
    ListNode* Remove_Duplicates_Two_Loops(ListNode* head) {
        /*
        Two loops brute force approach
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        if (!head || !head->next) return head;
        
        ListNode* curr = head;
        
        while (curr && curr->next) {
            ListNode* runner = curr;
            
            while (runner->next) {
                if (runner->next->data == curr->data) {
                    ListNode* temp = runner->next;
                    runner->next = runner->next->next;
                    delete temp;
                } else {
                    runner = runner->next;
                }
            }
            
            curr = curr->next;
        }
        
        return head;
    }
};

void Test_Remove_Duplicates_Unsorted() {
    Solution solution;
    
    vector<int> arr1 = {5, 2, 2, 4};
    ListNode* head1 = Create_List(arr1);
    cout << "Original: ";
    Print_List(head1);
    head1 = solution.Remove_Duplicates_Hashing(head1);
    cout << "After removal (Hashing): ";
    Print_List(head1);
    
    vector<int> arr2 = {2, 2, 2, 2};
    ListNode* head2 = Create_List(arr2);
    cout << "\nOriginal: ";
    Print_List(head2);
    head2 = solution.Remove_Duplicates_Two_Loops(head2);
    cout << "After removal (Two loops): ";
    Print_List(head2);
    
    vector<int> arr3 = {1, 2, 3, 4, 5};
    ListNode* head3 = Create_List(arr3);
    cout << "\nOriginal: ";
    Print_List(head3);
    head3 = solution.Remove_Duplicates_Hashing(head3);
    cout << "After removal (No duplicates): ";
    Print_List(head3);
}

int main() {
    Test_Remove_Duplicates_Unsorted();
    return 0;
}
