/*
Problem: Find First Node of Loop in a Linked List
URL: https://www.geeksforgeeks.org/find-first-node-of-loop-in-a-linked-list/

Problem Statement:
Find the first node of the loop in a linked list.

Sample Input/Output:
Input: 1->2->3->4->5->2 (loop at node 2)
Output: 2
Explanation: Node with value 2 is the first node of the loop.
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
    ListNode* First_Node_Floyd(ListNode* head) {
        /*
        Floyd's Cycle Detection Algorithm
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (!head || !head->next) return NULL;
        
        ListNode* slow = head;
        ListNode* fast = head;
        
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            
            if (slow == fast) {
                break;
            }
        }
        
        if (slow != fast) return NULL;
        
        slow = head;
        while (slow != fast) {
            slow = slow->next;
            fast = fast->next;
        }
        
        return slow;
    }
    
    ListNode* First_Node_Hashing(ListNode* head) {
        /*
        Hashing approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        unordered_set<ListNode*> visited;
        ListNode* curr = head;
        
        while (curr) {
            if (visited.find(curr) != visited.end()) {
                return curr;
            }
            visited.insert(curr);
            curr = curr->next;
        }
        
        return NULL;
    }
};

void Test_First_Node_In_Loop() {
    Solution solution;
    
    vector<int> arr1 = {1, 2, 3, 4, 5};
    ListNode* head1 = Create_List(arr1);
    Create_Loop(head1, 1);
    ListNode* result1 = solution.First_Node_Floyd(head1);
    cout << "Test 1 (Floyd): First node value = " << (result1 ? result1->data : -1) << endl;
    
    vector<int> arr2 = {1, 2, 3, 4, 5};
    ListNode* head2 = Create_List(arr2);
    Create_Loop(head2, 1);
    ListNode* result2 = solution.First_Node_Hashing(head2);
    cout << "Test 1 (Hashing): First node value = " << (result2 ? result2->data : -1) << endl;
    
    vector<int> arr3 = {1, 2, 3};
    ListNode* head3 = Create_List(arr3);
    Create_Loop(head3, 0);
    ListNode* result3 = solution.First_Node_Floyd(head3);
    cout << "\nTest 2 (Loop at head): First node value = " << (result3 ? result3->data : -1) << endl;
    
    vector<int> arr4 = {1, 2, 3, 4, 5};
    ListNode* head4 = Create_List(arr4);
    ListNode* result4 = solution.First_Node_Hashing(head4);
    cout << "\nTest 3 (No loop): First node value = " << (result4 ? result4->data : -1) << endl;
}

int main() {
    Test_First_Node_In_Loop();
    return 0;
}
