/*
Problem: Merge K Sorted Linked Lists
URL: https://practice.geeksforgeeks.org/problems/merge-k-sorted-linked-lists/1

Problem Statement:
Merge K sorted linked lists into one sorted linked list.

Sample Input/Output:
Input: [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
*/

#include <bits/stdc++.h>
using namespace std;

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

class Solution {
public:
    ListNode* Merge_K_Lists_Min_Heap(vector<ListNode*>& lists) {
        /*
        Min Heap Approach
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        */
        auto cmp = [](ListNode* a, ListNode* b) {
            return a->val > b->val;
        };
        priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq(cmp);
        
        for (ListNode* list : lists) {
            if (list) pq.push(list);
        }
        
        ListNode* dummy = new ListNode(0);
        ListNode* current = dummy;
        
        while (!pq.empty()) {
            ListNode* node = pq.top();
            pq.pop();
            current->next = node;
            current = current->next;
            if (node->next) {
                pq.push(node->next);
            }
        }
        
        return dummy->next;
    }
    
    ListNode* Merge_K_Lists_Divide_Conquer(vector<ListNode*>& lists) {
        /*
        Divide and Conquer Approach
        Time Complexity: O(n log k)
        Space Complexity: O(1)
        */
        if (lists.empty()) return nullptr;
        
        while (lists.size() > 1) {
            vector<ListNode*> merged;
            for (int i = 0; i < lists.size(); i += 2) {
                ListNode* l1 = lists[i];
                ListNode* l2 = (i + 1 < lists.size()) ? lists[i + 1] : nullptr;
                merged.push_back(MergeTwoLists(l1, l2));
            }
            lists = merged;
        }
        
        return lists[0];
    }
    
private:
    ListNode* MergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* dummy = new ListNode(0);
        ListNode* current = dummy;
        
        while (l1 && l2) {
            if (l1->val <= l2->val) {
                current->next = l1;
                l1 = l1->next;
            } else {
                current->next = l2;
                l2 = l2->next;
            }
            current = current->next;
        }
        
        current->next = l1 ? l1 : l2;
        return dummy->next;
    }
};

void Test_Merge_K_Lists() {
    Solution solution;
    
    ListNode* list1 = new ListNode(1);
    list1->next = new ListNode(4);
    list1->next->next = new ListNode(5);
    
    ListNode* list2 = new ListNode(1);
    list2->next = new ListNode(3);
    list2->next->next = new ListNode(4);
    
    ListNode* list3 = new ListNode(2);
    list3->next = new ListNode(6);
    
    vector<ListNode*> lists1 = {list1, list2, list3};
    ListNode* result1 = solution.Merge_K_Lists_Min_Heap(lists1);
    cout << "Min Heap Result: ";
    while (result1) {
        cout << result1->val << " ";
        result1 = result1->next;
    }
    cout << endl;
    
    ListNode* list4 = new ListNode(1);
    list4->next = new ListNode(4);
    list4->next->next = new ListNode(5);
    
    ListNode* list5 = new ListNode(1);
    list5->next = new ListNode(3);
    list5->next->next = new ListNode(4);
    
    ListNode* list6 = new ListNode(2);
    list6->next = new ListNode(6);
    
    vector<ListNode*> lists2 = {list4, list5, list6};
    ListNode* result2 = solution.Merge_K_Lists_Divide_Conquer(lists2);
    cout << "Divide Conquer Result: ";
    while (result2) {
        cout << result2->val << " ";
        result2 = result2->next;
    }
    cout << endl;
}

int main() {
    Test_Merge_K_Lists();
    return 0;
}
