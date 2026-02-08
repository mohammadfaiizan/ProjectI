/*
Problem: Quick Sort on Linked List
URL: https://practice.geeksforgeeks.org/problems/quick-sort-on-linked-list/1

Problem Statement:
Sort the given Linked List using quicksort. which takes O(n^2) time in worst case and O(nLogn) in average and best cases, otherwise you may get TLE.

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
    ListNode* Quick_Sort(ListNode* head) {
        /*
        Quick Sort with last element as pivot
        Time Complexity: O(n log n) average, O(n^2) worst
        Space Complexity: O(log n) stack
        */
        if (!head || !head->next) return head;
        
        ListNode* tail = Get_Tail(head);
        return Quick_Sort_Helper(head, tail);
    }
    
private:
    ListNode* Quick_Sort_Helper(ListNode* head, ListNode* tail) {
        if (!head || head == tail || !head->next) return head;
        
        ListNode* pivot = Partition(head, tail);
        
        if (head != pivot) {
            ListNode* temp = head;
            while (temp->next != pivot) {
                temp = temp->next;
            }
            temp->next = NULL;
            head = Quick_Sort_Helper(head, temp);
            temp = Get_Tail(head);
            temp->next = pivot;
        }
        
        pivot->next = Quick_Sort_Helper(pivot->next, tail);
        return head;
    }
    
    ListNode* Partition(ListNode* head, ListNode* tail) {
        ListNode* pivot = tail;
        ListNode* prev = NULL;
        ListNode* curr = head;
        ListNode* end = tail;
        
        while (curr != pivot) {
            if (curr->data < pivot->data) {
                if (!prev) {
                    prev = curr;
                } else {
                    prev = prev->next;
                }
                swap(prev->data, curr->data);
            }
            curr = curr->next;
        }
        
        if (!prev) {
            prev = head;
        } else {
            prev = prev->next;
        }
        swap(prev->data, pivot->data);
        return prev;
    }
    
    ListNode* Get_Tail(ListNode* head) {
        while (head && head->next) {
            head = head->next;
        }
        return head;
    }
};

void Test_Quick_Sort_Linked_List() {
    Solution solution;
    
    vector<int> arr = {3, 5, 2, 4, 1};
    ListNode* head = Create_List(arr);
    ListNode* result = solution.Quick_Sort(head);
    cout << "Test 1: ";
    Print_List(result);
    
    vector<int> arr2 = {4, 2, 1, 3};
    head = Create_List(arr2);
    result = solution.Quick_Sort(head);
    cout << "Test 2: ";
    Print_List(result);
    
    vector<int> arr3 = {5, 1, 4, 2, 3};
    head = Create_List(arr3);
    result = solution.Quick_Sort(head);
    cout << "Test 3: ";
    Print_List(result);
}

int main() {
    Test_Quick_Sort_Linked_List();
    return 0;
}
