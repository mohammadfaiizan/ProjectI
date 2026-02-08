/*
Problem: Sort a Linked List of 0s, 1s, and 2s
URL: https://practice.geeksforgeeks.org/problems/given-a-linked-list-of-0s-1s-and-2s-sort-it/1

Problem Statement:
Given a linked list of 0s, 1s and 2s, sort it.

Sample Input/Output:
Input: 1 -> 1 -> 2 -> 0 -> 2 -> 0 -> 1 -> NULL
Output: 0 -> 0 -> 1 -> 1 -> 1 -> 2 -> 2 -> NULL
Explanation: All 0s come first, then 1s, then 2s.
*/

#include <bits/stdc++.h>
using namespace std;

struct ListNode {
    int data;
    ListNode* next;
    ListNode(int x) : data(x), next(NULL) {}
};

ListNode* Create_List(vector<int> arr) {
    if (arr.empty()) return NULL;
    ListNode* head = new ListNode(arr[0]);
    ListNode* current = head;
    for (int i = 1; i < arr.size(); i++) {
        current->next = new ListNode(arr[i]);
        current = current->next;
    }
    return head;
}

void Print_List(ListNode* head) {
    while (head) {
        cout << head->data;
        if (head->next) cout << " -> ";
        head = head->next;
    }
    cout << " -> NULL" << endl;
}

class Solution {
public:
    ListNode* Sort_012_Count_Based(ListNode* head) {
        /*
        Count occurrences then rebuild list
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int count[3] = {0};
        ListNode* current = head;
        while (current) {
            count[current->data]++;
            current = current->next;
        }
        current = head;
        for (int i = 0; i < 3; i++) {
            while (count[i]--) {
                current->data = i;
                current = current->next;
            }
        }
        return head;
    }

    ListNode* Sort_012_Three_Dummy_Nodes(ListNode* head) {
        /*
        Separate into three lists then merge
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        ListNode* zero_head = new ListNode(0);
        ListNode* one_head = new ListNode(0);
        ListNode* two_head = new ListNode(0);
        ListNode* zero = zero_head;
        ListNode* one = one_head;
        ListNode* two = two_head;
        ListNode* current = head;
        while (current) {
            if (current->data == 0) {
                zero->next = current;
                zero = zero->next;
            } else if (current->data == 1) {
                one->next = current;
                one = one->next;
            } else {
                two->next = current;
                two = two->next;
            }
            current = current->next;
        }
        zero->next = one_head->next ? one_head->next : two_head->next;
        one->next = two_head->next;
        two->next = NULL;
        ListNode* result = zero_head->next;
        delete zero_head;
        delete one_head;
        delete two_head;
        return result;
    }
};

void Test_Sort_012_Linked_List() {
    Solution solution;
    
    vector<int> test1 = {1, 1, 2, 0, 2, 0, 1};
    ListNode* list1 = Create_List(test1);
    ListNode* result1 = solution.Sort_012_Count_Based(list1);
    cout << "Test 1 Count Based: ";
    Print_List(result1);
    
    vector<int> test2 = {2, 1, 2, 1, 1, 2, 0, 2, 0};
    ListNode* list2 = Create_List(test2);
    ListNode* result2 = solution.Sort_012_Three_Dummy_Nodes(list2);
    cout << "Test 2 Three Dummy: ";
    Print_List(result2);
    
    vector<int> test3 = {2, 2, 1, 1, 0};
    ListNode* list3 = Create_List(test3);
    ListNode* result3 = solution.Sort_012_Count_Based(list3);
    cout << "Test 3 Mixed: ";
    Print_List(result3);
}

int main() {
    Test_Sort_012_Linked_List();
    return 0;
}
