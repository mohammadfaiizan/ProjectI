/*
Problem: Detect Loop in Linked List
URL: https://practice.geeksforgeeks.org/problems/detect-loop-in-linked-list/1

Problem Statement:
Detect if there is a loop in the linked list.

Sample Input/Output:
Input: 1->2->3->4->5->2 (loop at node 2)
Output: true
Explanation: Loop exists in the linked list.
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

void Create_Loop(ListNode* head, int pos) {
    if (pos < 0) return;
    ListNode* loopNode = NULL;
    ListNode* curr = head;
    int index = 0;
    
    while (curr->next) {
        if (index == pos) {
            loopNode = curr;
        }
        curr = curr->next;
        index++;
    }
    
    if (loopNode) {
        curr->next = loopNode;
    }
}

class Solution {
public:
    bool Detect_Loop_Hashing(ListNode* head) {
        /*
        Hashing approach using unordered_set
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        unordered_set<ListNode*> visited;
        ListNode* curr = head;
        
        while (curr) {
            if (visited.find(curr) != visited.end()) {
                return true;
            }
            visited.insert(curr);
            curr = curr->next;
        }
        
        return false;
    }
    
    bool Detect_Loop_Floyd(ListNode* head) {
        /*
        Floyd's Cycle Detection Algorithm
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (!head || !head->next) return false;
        
        ListNode* slow = head;
        ListNode* fast = head;
        
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            
            if (slow == fast) {
                return true;
            }
        }
        
        return false;
    }
    
    bool Detect_Loop_Temp_Node(ListNode* head) {
        /*
        Temp node marking approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (!head) return false;
        
        ListNode* temp = new ListNode(0);
        ListNode* curr = head;
        
        while (curr) {
            if (curr->next == temp) {
                return true;
            }
            
            ListNode* next = curr->next;
            curr->next = temp;
            curr = next;
        }
        
        return false;
    }
};

void Test_Detect_Loop() {
    Solution solution;
    
    vector<int> arr1 = {1, 2, 3, 4, 5};
    ListNode* head1 = Create_List(arr1);
    Create_Loop(head1, 1);
    cout << "Test 1 (Loop exists): " << solution.Detect_Loop_Hashing(head1) << endl;
    cout << "Test 1 (Floyd): " << solution.Detect_Loop_Floyd(head1) << endl;
    
    vector<int> arr2 = {1, 2, 3, 4, 5};
    ListNode* head2 = Create_List(arr2);
    cout << "\nTest 2 (No loop): " << solution.Detect_Loop_Hashing(head2) << endl;
    cout << "Test 2 (Floyd): " << solution.Detect_Loop_Floyd(head2) << endl;
    
    vector<int> arr3 = {1};
    ListNode* head3 = Create_List(arr3);
    cout << "\nTest 3 (Single node, no loop): " << solution.Detect_Loop_Temp_Node(head3) << endl;
}

int main() {
    Test_Detect_Loop();
    return 0;
}
