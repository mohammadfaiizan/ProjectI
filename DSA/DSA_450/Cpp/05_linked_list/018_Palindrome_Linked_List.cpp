/*
Problem: Check if Linked List is Palindrome
URL: https://practice.geeksforgeeks.org/problems/check-if-linked-list-is-pallindrome/1

Problem Statement:
Given a singly linked list of size N of integers. The task is to check if the given linked list is palindrome or not.

Sample Input/Output:
Input: N = 3, value[] = {1,2,1}
Output: 1
Explanation: The given linked list is 1->2->1, which is a palindrome and hence, the output is 1.
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
    bool Is_Palindrome_Reverse_Second_Half(ListNode* head) {
        /*
        Reverse second half
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (!head || !head->next) return true;
        
        ListNode* slow = head;
        ListNode* fast = head;
        
        while (fast->next && fast->next->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        
        ListNode* secondHalf = slow->next;
        slow->next = NULL;
        secondHalf = Reverse_List(secondHalf);
        
        ListNode* first = head;
        ListNode* second = secondHalf;
        bool result = true;
        
        while (first && second) {
            if (first->data != second->data) {
                result = false;
                break;
            }
            first = first->next;
            second = second->next;
        }
        
        secondHalf = Reverse_List(secondHalf);
        slow->next = secondHalf;
        
        return result;
    }
    
    bool Is_Palindrome_Stack(ListNode* head) {
        /*
        Stack-based approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (!head || !head->next) return true;
        
        stack<int> st;
        ListNode* curr = head;
        while (curr) {
            st.push(curr->data);
            curr = curr->next;
        }
        
        curr = head;
        while (curr) {
            if (curr->data != st.top()) {
                return false;
            }
            st.pop();
            curr = curr->next;
        }
        
        return true;
    }
    
    bool Is_Palindrome_Recursive(ListNode* head) {
        /*
        Recursive approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        ListNode* front = head;
        return Is_Palindrome_Recursive_Helper(head, front);
    }
    
private:
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
    
    bool Is_Palindrome_Recursive_Helper(ListNode* curr, ListNode*& front) {
        if (!curr) return true;
        
        bool result = Is_Palindrome_Recursive_Helper(curr->next, front);
        if (!result) return false;
        
        if (curr->data != front->data) return false;
        front = front->next;
        return true;
    }
};

void Test_Palindrome_Linked_List() {
    Solution solution;
    
    vector<int> arr = {1, 2, 1};
    ListNode* head = Create_List(arr);
    bool result1 = solution.Is_Palindrome_Reverse_Second_Half(head);
    cout << "Test 1 - Reverse Second Half: " << result1 << endl;
    
    head = Create_List(arr);
    bool result2 = solution.Is_Palindrome_Stack(head);
    cout << "Test 1 - Stack: " << result2 << endl;
    
    head = Create_List(arr);
    bool result3 = solution.Is_Palindrome_Recursive(head);
    cout << "Test 1 - Recursive: " << result3 << endl;
    
    vector<int> arr2 = {1, 2, 3};
    head = Create_List(arr2);
    result1 = solution.Is_Palindrome_Reverse_Second_Half(head);
    cout << "Test 2 - Reverse Second Half: " << result1 << endl;
}

int main() {
    Test_Palindrome_Linked_List();
    return 0;
}
