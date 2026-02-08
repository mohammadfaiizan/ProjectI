/*
Problem: Delete Nodes Having Greater Value on Right Side
URL: https://practice.geeksforgeeks.org/problems/delete-nodes-having-greater-value-on-right/1

Problem Statement:
Given a singly linked list, remove all the nodes which have a greater value on their right side. The rightmost node is always kept.

Sample Input/Output:
Input: 12 -> 15 -> 10 -> 11 -> 5 -> 6 -> 2 -> 3 -> NULL
Output: 15 -> 11 -> 6 -> 3 -> NULL
Explanation: Nodes 12, 10, 5, 2 are deleted as they have greater values on right.
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
    ListNode* Delete_Nodes_Greater_Right_Reverse_Filter_Reverse(ListNode* head) {
        /*
        Reverse list, filter nodes, reverse back
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (head == NULL || head->next == NULL) return head;
        head = Reverse_List(head);
        ListNode* current = head;
        int max_so_far = current->data;
        while (current && current->next) {
            if (current->next->data < max_so_far) {
                ListNode* temp = current->next;
                current->next = current->next->next;
                delete temp;
            } else {
                max_so_far = max(max_so_far, current->next->data);
                current = current->next;
            }
        }
        return Reverse_List(head);
    }

    ListNode* Delete_Nodes_Greater_Right_Recursive(ListNode* head) {
        /*
        Recursive approach processing from right to left
        Time Complexity: O(n)
        Space Complexity: O(n) for recursion stack
        */
        if (head == NULL || head->next == NULL) return head;
        head->next = Delete_Nodes_Greater_Right_Recursive(head->next);
        if (head->next && head->data < head->next->data) {
            ListNode* temp = head;
            head = head->next;
            delete temp;
        }
        return head;
    }

private:
    ListNode* Reverse_List(ListNode* head) {
        ListNode* prev = NULL;
        ListNode* current = head;
        while (current) {
            ListNode* next = current->next;
            current->next = prev;
            prev = current;
            current = next;
        }
        return prev;
    }
};

void Test_Delete_Nodes_Greater_Right() {
    Solution solution;
    
    ListNode* list1 = Create_List({12, 15, 10, 11, 5, 6, 2, 3});
    ListNode* result1 = solution.Delete_Nodes_Greater_Right_Reverse_Filter_Reverse(list1);
    cout << "Test 1 Reverse Filter: ";
    Print_List(result1);
    
    ListNode* list2 = Create_List({10, 20, 30, 40, 50});
    ListNode* result2 = solution.Delete_Nodes_Greater_Right_Recursive(list2);
    cout << "Test 2 Recursive: ";
    Print_List(result2);
    
    ListNode* list3 = Create_List({5, 2, 13, 3, 8});
    ListNode* result3 = solution.Delete_Nodes_Greater_Right_Reverse_Filter_Reverse(list3);
    cout << "Test 3 Mixed: ";
    Print_List(result3);
}

int main() {
    Test_Delete_Nodes_Greater_Right();
    return 0;
}
