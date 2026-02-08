/*
Problem: Segregate Even and Odd Nodes in a Linked List
URL: https://practice.geeksforgeeks.org/problems/segregate-even-and-odd-nodes-in-a-linked-list5035/1

Problem Statement:
Given a linked list, segregate even and odd nodes such that all even nodes come before all odd nodes.

Sample Input/Output:
Input: 17 -> 15 -> 8 -> 12 -> 10 -> 5 -> 4 -> NULL
Output: 8 -> 12 -> 10 -> 4 -> 17 -> 15 -> 5 -> NULL
Explanation: All even nodes (8, 12, 10, 4) come before odd nodes (17, 15, 5).
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
    ListNode* Segregate_Even_Odd_Separate_Merge(ListNode* head) {
        /*
        Separate into even and odd lists then merge
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (head == NULL || head->next == NULL) return head;
        ListNode* even_head = new ListNode(0);
        ListNode* odd_head = new ListNode(0);
        ListNode* even = even_head;
        ListNode* odd = odd_head;
        ListNode* current = head;
        while (current) {
            if (current->data % 2 == 0) {
                even->next = current;
                even = even->next;
            } else {
                odd->next = current;
                odd = odd->next;
            }
            current = current->next;
        }
        even->next = odd_head->next;
        odd->next = NULL;
        ListNode* result = even_head->next;
        delete even_head;
        delete odd_head;
        return result;
    }

    ListNode* Segregate_Even_Odd_Move_Odd_To_End(ListNode* head) {
        /*
        Move odd nodes to end while keeping even nodes at front
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (head == NULL || head->next == NULL) return head;
        ListNode* last = head;
        int count = 0;
        while (last->next) {
            last = last->next;
            count++;
        }
        ListNode* current = head;
        ListNode* prev = NULL;
        int moved = 0;
        while (current && moved <= count) {
            if (current->data % 2 == 1) {
                if (prev == NULL) {
                    head = current->next;
                    last->next = current;
                    last = last->next;
                    last->next = NULL;
                    current = head;
                } else {
                    prev->next = current->next;
                    last->next = current;
                    last = last->next;
                    last->next = NULL;
                    current = prev->next;
                }
                moved++;
            } else {
                prev = current;
                current = current->next;
            }
        }
        return head;
    }
};

void Test_Segregate_Even_Odd_Nodes() {
    Solution solution;
    
    ListNode* list1 = Create_List({17, 15, 8, 12, 10, 5, 4});
    ListNode* result1 = solution.Segregate_Even_Odd_Separate_Merge(list1);
    cout << "Test 1 Separate Merge: ";
    Print_List(result1);
    
    ListNode* list2 = Create_List({1, 3, 5, 7});
    ListNode* result2 = solution.Segregate_Even_Odd_Move_Odd_To_End(list2);
    cout << "Test 2 Move Odd: ";
    Print_List(result2);
    
    ListNode* list3 = Create_List({2, 4, 6, 8});
    ListNode* result3 = solution.Segregate_Even_Odd_Separate_Merge(list3);
    cout << "Test 3 All Even: ";
    Print_List(result3);
}

int main() {
    Test_Segregate_Even_Odd_Nodes();
    return 0;
}
