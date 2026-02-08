/*
Problem: Remove Loop in Linked List
URL: https://practice.geeksforgeeks.org/problems/remove-loop-in-linked-list/1

Problem Statement:
Remove the loop from a linked list if it exists.

Sample Input/Output:
Input: 1->2->3->4->5->2 (loop at node 2)
Output: 1->2->3->4->5->NULL
Explanation: Loop is removed from the linked list.
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

void Print_List(ListNode* head, int maxNodes = 10) {
    int count = 0;
    while (head && count < maxNodes) {
        cout << head->data;
        if (head->next && count < maxNodes - 1) cout << "->";
        head = head->next;
        count++;
    }
    if (head) cout << "->...";
    cout << "->NULL" << endl;
}

class Solution {
public:
    void Remove_Loop_Hashing(ListNode* head) {
        /*
        Hashing approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        unordered_set<ListNode*> visited;
        ListNode* prev = NULL;
        ListNode* curr = head;
        
        while (curr) {
            if (visited.find(curr) != visited.end()) {
                prev->next = NULL;
                return;
            }
            visited.insert(curr);
            prev = curr;
            curr = curr->next;
        }
    }
    
    void Remove_Loop_Floyd_Detect_Remove(ListNode* head) {
        /*
        Floyd's detect and remove with counting
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (!head || !head->next) return;
        
        ListNode* slow = head;
        ListNode* fast = head;
        
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            
            if (slow == fast) {
                break;
            }
        }
        
        if (slow != fast) return;
        
        int loopLength = 1;
        ListNode* temp = slow->next;
        while (temp != slow) {
            loopLength++;
            temp = temp->next;
        }
        
        ListNode* ptr1 = head;
        ListNode* ptr2 = head;
        
        for (int i = 0; i < loopLength; i++) {
            ptr2 = ptr2->next;
        }
        
        while (ptr1 != ptr2) {
            ptr1 = ptr1->next;
            ptr2 = ptr2->next;
        }
        
        while (ptr2->next != ptr1) {
            ptr2 = ptr2->next;
        }
        
        ptr2->next = NULL;
    }
    
    void Remove_Loop_Floyd_Optimized(ListNode* head) {
        /*
        Floyd's optimized without counting
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (!head || !head->next) return;
        
        ListNode* slow = head;
        ListNode* fast = head;
        
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            
            if (slow == fast) {
                break;
            }
        }
        
        if (slow != fast) return;
        
        if (slow == head) {
            while (fast->next != head) {
                fast = fast->next;
            }
            fast->next = NULL;
            return;
        }
        
        slow = head;
        while (slow->next != fast->next) {
            slow = slow->next;
            fast = fast->next;
        }
        
        fast->next = NULL;
    }
};

void Test_Remove_Loop() {
    Solution solution;
    
    vector<int> arr1 = {1, 2, 3, 4, 5};
    ListNode* head1 = Create_List(arr1);
    Create_Loop(head1, 1);
    cout << "Before removal (with loop): ";
    Print_List(head1);
    solution.Remove_Loop_Hashing(head1);
    cout << "After removal (Hashing): ";
    Print_List(head1);
    
    vector<int> arr2 = {1, 2, 3, 4, 5};
    ListNode* head2 = Create_List(arr2);
    Create_Loop(head2, 0);
    cout << "\nBefore removal (loop at head): ";
    Print_List(head2);
    solution.Remove_Loop_Floyd_Detect_Remove(head2);
    cout << "After removal (Floyd with counting): ";
    Print_List(head2);
    
    vector<int> arr3 = {1, 2, 3, 4, 5, 6};
    ListNode* head3 = Create_List(arr3);
    Create_Loop(head3, 2);
    cout << "\nBefore removal: ";
    Print_List(head3);
    solution.Remove_Loop_Floyd_Optimized(head3);
    cout << "After removal (Floyd optimized): ";
    Print_List(head3);
}

int main() {
    Test_Remove_Loop();
    return 0;
}
