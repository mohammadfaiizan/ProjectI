/*
Problem: Add 1 to a Number Represented as Linked List
URL: https://practice.geeksforgeeks.org/problems/add-1-to-a-number-represented-as-linked-list/1

Problem Statement:
Add 1 to a number represented as a linked list.

Sample Input/Output:
Input: 1->9->9->9->NULL
Output: 2->0->0->0->NULL
Explanation: 1999 + 1 = 2000
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

ListNode* Reverse_List(ListNode* head) {
    ListNode* prev = NULL;
    ListNode* curr = head;
    ListNode* next = NULL;
    
    while (curr) {
        next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    
    return prev;
}

class Solution {
public:
    ListNode* Add_One_Reverse_Add_Reverse(ListNode* head) {
        /*
        Reverse, Add, Reverse approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (!head) return new ListNode(1);
        
        head = Reverse_List(head);
        ListNode* curr = head;
        int carry = 1;
        
        while (curr && carry) {
            int sum = curr->data + carry;
            curr->data = sum % 10;
            carry = sum / 10;
            
            if (!curr->next && carry) {
                curr->next = new ListNode(carry);
                carry = 0;
            }
            
            curr = curr->next;
        }
        
        head = Reverse_List(head);
        return head;
    }
    
    int Add_One_Recursive_Helper(ListNode* head) {
        if (!head) return 1;
        
        int carry = Add_One_Recursive_Helper(head->next);
        int sum = head->data + carry;
        head->data = sum % 10;
        return sum / 10;
    }
    
    ListNode* Add_One_Recursive_Carry(ListNode* head) {
        /*
        Recursive carry approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (!head) return new ListNode(1);
        
        int carry = Add_One_Recursive_Helper(head);
        
        if (carry) {
            ListNode* newHead = new ListNode(carry);
            newHead->next = head;
            return newHead;
        }
        
        return head;
    }
};

void Test_Add_One_To_Number() {
    Solution solution;
    
    vector<int> arr1 = {1, 9, 9, 9};
    ListNode* head1 = Create_List(arr1);
    cout << "Original: ";
    Print_List(head1);
    head1 = solution.Add_One_Reverse_Add_Reverse(head1);
    cout << "After adding 1 (Reverse-Add-Reverse): ";
    Print_List(head1);
    
    vector<int> arr2 = {9, 9, 9};
    ListNode* head2 = Create_List(arr2);
    cout << "\nOriginal: ";
    Print_List(head2);
    head2 = solution.Add_One_Recursive_Carry(head2);
    cout << "After adding 1 (Recursive): ";
    Print_List(head2);
    
    vector<int> arr3 = {1, 2, 3};
    ListNode* head3 = Create_List(arr3);
    cout << "\nOriginal: ";
    Print_List(head3);
    head3 = solution.Add_One_Reverse_Add_Reverse(head3);
    cout << "After adding 1: ";
    Print_List(head3);
}

int main() {
    Test_Add_One_To_Number();
    return 0;
}
