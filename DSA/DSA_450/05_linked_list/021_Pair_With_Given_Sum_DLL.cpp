/*
Problem: Find Pairs with Given Sum in Sorted Doubly Linked List
URL: https://www.geeksforgeeks.org/find-pairs-given-sum-doubly-linked-list/

Problem Statement:
Given a sorted doubly linked list and a target sum, find all pairs of nodes whose sum equals the target.

Sample Input/Output:
Input: List: 1 <-> 2 <-> 4 <-> 5 <-> 6 <-> 8 <-> 9, sum = 7
Output: (1, 6), (2, 5)
Explanation: Pairs that sum to 7
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
    vector<pair<int, int>> Find_Pairs_Two_Pointer(DLLNode* head, int target) {
        /*
        Two pointer approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        vector<pair<int, int>> result;
        if (!head || !head->next) return result;
        
        DLLNode* left = head;
        DLLNode* right = head;
        while (right->next) right = right->next;
        
        while (left != right && right->next != left) {
            int sum = left->data + right->data;
            if (sum == target) {
                result.push_back({left->data, right->data});
                left = left->next;
                right = right->prev;
            } else if (sum < target) {
                left = left->next;
            } else {
                right = right->prev;
            }
        }
        
        return result;
    }
    
    vector<pair<int, int>> Find_Pairs_Hashing(DLLNode* head, int target) {
        /*
        Hashing approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<pair<int, int>> result;
        if (!head) return result;
        
        unordered_set<int> seen;
        DLLNode* curr = head;
        
        while (curr) {
            int complement = target - curr->data;
            if (seen.find(complement) != seen.end()) {
                result.push_back({complement, curr->data});
            }
            seen.insert(curr->data);
            curr = curr->next;
        }
        
        return result;
    }
};

void Test_Pair_With_Given_Sum_DLL() {
    Solution solution;
    
    vector<int> arr1 = {1, 2, 4, 5, 6, 8, 9};
    DLLNode* head1 = Create_DLL(arr1);
    cout << "List: ";
    Print_DLL(head1);
    vector<pair<int, int>> pairs1 = solution.Find_Pairs_Two_Pointer(head1, 7);
    cout << "Pairs with sum 7 (Two Pointer): ";
    for (auto p : pairs1) {
        cout << "(" << p.first << ", " << p.second << ") ";
    }
    cout << endl;
    
    vector<pair<int, int>> pairs2 = solution.Find_Pairs_Hashing(head1, 7);
    cout << "Pairs with sum 7 (Hashing): ";
    for (auto p : pairs2) {
        cout << "(" << p.first << ", " << p.second << ") ";
    }
    cout << endl;
    
    vector<int> arr3 = {1, 3, 5, 7};
    DLLNode* head3 = Create_DLL(arr3);
    cout << "List: ";
    Print_DLL(head3);
    vector<pair<int, int>> pairs3 = solution.Find_Pairs_Two_Pointer(head3, 8);
    cout << "Pairs with sum 8: ";
    for (auto p : pairs3) {
        cout << "(" << p.first << ", " << p.second << ") ";
    }
    cout << endl;
}

int main() {
    Test_Pair_With_Given_Sum_DLL();
    return 0;
}
