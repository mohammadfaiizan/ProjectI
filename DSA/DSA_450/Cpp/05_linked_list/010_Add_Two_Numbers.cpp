/*
Problem: Add Two Numbers Represented by Linked Lists
URL: https://practice.geeksforgeeks.org/problems/add-two-numbers-represented-by-linked-lists/1

Problem Statement:
Given two numbers represented by two linked lists, write a function that returns the sum list. The sum list is linked list representation of the addition of two input numbers.

Sample Input/Output:
Input: First List: 5->6->3, Second List: 8->4->2
Output: Resultant list: 1->4->0->5
Explanation: 563 + 842 = 1405
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

ListNode* Reverse_List(ListNode* head) {
    ListNode* prev = NULL;
    ListNode* curr = head;
    while (curr) {
        ListNode* next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
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
    ListNode* Add_Two_Numbers_Reverse(ListNode* first, ListNode* second) {
        /*
        Reverse both lists, add with carry, reverse result
        Time Complexity: O(m + n)
        Space Complexity: O(max(m, n))
        */
        first = Reverse_List(first);
        second = Reverse_List(second);
        
        ListNode* dummy = new ListNode(0);
        ListNode* curr = dummy;
        int carry = 0;
        
        while (first || second || carry) {
            int sum = carry;
            if (first) {
                sum += first->data;
                first = first->next;
            }
            if (second) {
                sum += second->data;
                second = second->next;
            }
            carry = sum / 10;
            curr->next = new ListNode(sum % 10);
            curr = curr->next;
        }
        
        ListNode* result = Reverse_List(dummy->next);
        delete dummy;
        return result;
    }
    
    ListNode* Add_Two_Numbers_Recursive(ListNode* first, ListNode* second) {
        /*
        Recursive approach for same-size lists
        Time Complexity: O(m + n)
        Space Complexity: O(max(m, n))
        */
        int len1 = 0, len2 = 0;
        ListNode* temp1 = first, *temp2 = second;
        while (temp1) { len1++; temp1 = temp1->next; }
        while (temp2) { len2++; temp2 = temp2->next; }
        
        if (len1 < len2) {
            while (len1 < len2) {
                ListNode* node = new ListNode(0);
                node->next = first;
                first = node;
                len1++;
            }
        } else if (len2 < len1) {
            while (len2 < len1) {
                ListNode* node = new ListNode(0);
                node->next = second;
                second = node;
                len2++;
            }
        }
        
        int carry = 0;
        ListNode* result = Add_Recursive_Helper(first, second, carry);
        if (carry > 0) {
            ListNode* node = new ListNode(carry);
            node->next = result;
            result = node;
        }
        return result;
    }
    
private:
    ListNode* Add_Recursive_Helper(ListNode* first, ListNode* second, int& carry) {
        if (!first && !second) return NULL;
        
        ListNode* next = Add_Recursive_Helper(first->next, second->next, carry);
        int sum = first->data + second->data + carry;
        carry = sum / 10;
        ListNode* node = new ListNode(sum % 10);
        node->next = next;
        return node;
    }
};

void Test_Add_Two_Numbers() {
    Solution solution;
    
    vector<int> arr1 = {5, 6, 3};
    vector<int> arr2 = {8, 4, 2};
    ListNode* first = Create_List(arr1);
    ListNode* second = Create_List(arr2);
    ListNode* result1 = solution.Add_Two_Numbers_Reverse(first, second);
    cout << "Test 1 - Reverse Approach: ";
    Print_List(result1);
    
    first = Create_List(arr1);
    second = Create_List(arr2);
    ListNode* result2 = solution.Add_Two_Numbers_Recursive(first, second);
    cout << "Test 1 - Recursive Approach: ";
    Print_List(result2);
    
    vector<int> arr3 = {9, 9, 9};
    vector<int> arr4 = {1};
    first = Create_List(arr3);
    second = Create_List(arr4);
    result1 = solution.Add_Two_Numbers_Reverse(first, second);
    cout << "Test 2 - Reverse Approach: ";
    Print_List(result1);
}

int main() {
    Test_Add_Two_Numbers();
    return 0;
}
