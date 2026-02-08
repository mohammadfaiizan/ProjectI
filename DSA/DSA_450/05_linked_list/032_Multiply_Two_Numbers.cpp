/*
Problem: Multiply Two Numbers Represented by Linked Lists
URL: https://practice.geeksforgeeks.org/problems/multiply-two-linked-lists/1

Problem Statement:
Given two numbers represented by linked lists, multiply them and return the result as a number.

Sample Input/Output:
Input: First List: 3->2->1 (represents 123)
       Second List: 2->1 (represents 12)
Output: 1476
Explanation: 123 * 12 = 1476
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
    long long Multiply_Two_Numbers_Modular(ListNode* first, ListNode* second) {
        /*
        Convert to numbers using modular arithmetic to avoid overflow
        Time Complexity: O(m + n) where m and n are lengths
        Space Complexity: O(1)
        */
        const long long MOD = 1000000007;
        long long num1 = 0;
        long long num2 = 0;
        ListNode* current = first;
        while (current) {
            num1 = (num1 * 10 + current->data) % MOD;
            current = current->next;
        }
        current = second;
        while (current) {
            num2 = (num2 * 10 + current->data) % MOD;
            current = current->next;
        }
        return (num1 * num2) % MOD;
    }

    long long Multiply_Two_Numbers_Build_Then_Multiply(ListNode* first, ListNode* second) {
        /*
        Build numbers then multiply directly
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        */
        long long num1 = 0;
        long long num2 = 0;
        ListNode* current = first;
        while (current) {
            num1 = num1 * 10 + current->data;
            current = current->next;
        }
        current = second;
        while (current) {
            num2 = num2 * 10 + current->data;
            current = current->next;
        }
        return num1 * num2;
    }
};

void Test_Multiply_Two_Numbers() {
    Solution solution;
    
    ListNode* list1 = Create_List({3, 2, 1});
    ListNode* list2 = Create_List({2, 1});
    long long result1 = solution.Multiply_Two_Numbers_Modular(list1, list2);
    cout << "Test 1 Modular: " << result1 << endl;
    
    ListNode* list3 = Create_List({9, 9, 9});
    ListNode* list4 = Create_List({1, 1});
    long long result2 = solution.Multiply_Two_Numbers_Build_Then_Multiply(list3, list4);
    cout << "Test 2 Build Multiply: " << result2 << endl;
    
    ListNode* list5 = Create_List({1});
    ListNode* list6 = Create_List({5});
    long long result3 = solution.Multiply_Two_Numbers_Modular(list5, list6);
    cout << "Test 3 Single Digit: " << result3 << endl;
}

int main() {
    Test_Multiply_Two_Numbers();
    return 0;
}
