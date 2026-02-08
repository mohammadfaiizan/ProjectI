/*
Problem: Count Triplets in Sorted Doubly Linked List whose Sum equals X
URL: https://www.geeksforgeeks.org/count-triplets-sorted-doubly-linked-list-whose-sum-equal-given-value-x/

Problem Statement:
Given a sorted doubly linked list and a target value X, count all triplets whose sum equals X.

Sample Input/Output:
Input: List: 1 <-> 2 <-> 4 <-> 5 <-> 6 <-> 8 <-> 9, X = 17
Output: 2
Explanation: Triplets: (4, 5, 8) = 17, (2, 6, 9) = 17
*/

#include <bits/stdc++.h>
using namespace std;

struct DLLNode {
    int data;
    DLLNode* next;
    DLLNode* prev;
    DLLNode(int x) : data(x), next(NULL), prev(NULL) {}
};

DLLNode* Create_DLL(vector<int>& arr) {
    if (arr.empty()) return NULL;
    DLLNode* head = new DLLNode(arr[0]);
    DLLNode* curr = head;
    for (int i = 1; i < arr.size(); i++) {
        curr->next = new DLLNode(arr[i]);
        curr->next->prev = curr;
        curr = curr->next;
    }
    return head;
}

void Print_DLL(DLLNode* head) {
    DLLNode* curr = head;
    while (curr) {
        cout << curr->data << " ";
        curr = curr->next;
    }
    cout << endl;
}

class Solution {
public:
    int Count_Triplets_Brute_Force(DLLNode* head, int target) {
        /*
        Brute force approach
        Time Complexity: O(n^3)
        Space Complexity: O(1)
        */
        int count = 0;
        DLLNode* first = head;
        while (first) {
            DLLNode* second = first->next;
            while (second) {
                DLLNode* third = second->next;
                while (third) {
                    if (first->data + second->data + third->data == target) {
                        count++;
                    }
                    third = third->next;
                }
                second = second->next;
            }
            first = first->next;
        }
        return count;
    }
    
    int Count_Triplets_Hashing(DLLNode* head, int target) {
        /*
        Hashing approach
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        int count = 0;
        DLLNode* first = head;
        while (first) {
            DLLNode* second = first->next;
            unordered_set<int> seen;
            while (second) {
                int complement = target - first->data - second->data;
                if (seen.find(complement) != seen.end()) {
                    count++;
                }
                seen.insert(second->data);
                second = second->next;
            }
            first = first->next;
        }
        return count;
    }
    
    int Count_Triplets_Two_Pointer(DLLNode* head, int target) {
        /*
        Two pointer approach
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        int count = 0;
        DLLNode* first = head;
        
        while (first) {
            DLLNode* second = first->next;
            DLLNode* third = head;
            while (third->next) third = third->next;
            
            while (second && third && second != third && third->next != second) {
                int sum = first->data + second->data + third->data;
                if (sum == target) {
                    count++;
                    second = second->next;
                    third = third->prev;
                } else if (sum < target) {
                    second = second->next;
                } else {
                    third = third->prev;
                }
            }
            first = first->next;
        }
        
        return count;
    }
};

void Test_Count_Triplets_DLL() {
    Solution solution;
    
    vector<int> arr1 = {1, 2, 4, 5, 6, 8, 9};
    DLLNode* head1 = Create_DLL(arr1);
    cout << "List: ";
    Print_DLL(head1);
    int count1 = solution.Count_Triplets_Brute_Force(head1, 17);
    cout << "Triplets with sum 17 (Brute Force): " << count1 << endl;
    
    int count2 = solution.Count_Triplets_Hashing(head1, 17);
    cout << "Triplets with sum 17 (Hashing): " << count2 << endl;
    
    int count3 = solution.Count_Triplets_Two_Pointer(head1, 17);
    cout << "Triplets with sum 17 (Two Pointer): " << count3 << endl;
    
    vector<int> arr2 = {1, 2, 3, 4, 5};
    DLLNode* head2 = Create_DLL(arr2);
    cout << "List: ";
    Print_DLL(head2);
    int count4 = solution.Count_Triplets_Two_Pointer(head2, 6);
    cout << "Triplets with sum 6: " << count4 << endl;
}

int main() {
    Test_Count_Triplets_DLL();
    return 0;
}
