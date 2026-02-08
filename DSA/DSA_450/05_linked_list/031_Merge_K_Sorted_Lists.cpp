/*
Problem: Merge K Sorted Linked Lists
URL: https://practice.geeksforgeeks.org/problems/merge-k-sorted-linked-lists/1

Problem Statement:
Given K sorted linked lists of different sizes. Merge them in such a way that after merging they will be a single sorted linked list.

Sample Input/Output:
Input: K = 3, Lists: 1->3->5->7, 2->4->6, 0->8->9
Output: 0->1->2->3->4->5->6->7->8->9
Explanation: All lists merged into one sorted list.
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
    ListNode* Merge_K_Sorted_Min_Heap(vector<ListNode*>& lists) {
        /*
        Use min heap to always get smallest element
        Time Complexity: O(n log k) where n is total nodes, k is number of lists
        Space Complexity: O(k) for heap
        */
        if (lists.empty()) return NULL;
        priority_queue<pair<int, ListNode*>, vector<pair<int, ListNode*>>, greater<pair<int, ListNode*>>> pq;
        for (ListNode* list : lists) {
            if (list) {
                pq.push({list->data, list});
            }
        }
        ListNode* dummy = new ListNode(0);
        ListNode* current = dummy;
        while (!pq.empty()) {
            ListNode* node = pq.top().second;
            pq.pop();
            current->next = node;
            current = current->next;
            if (node->next) {
                pq.push({node->next->data, node->next});
            }
        }
        ListNode* result = dummy->next;
        delete dummy;
        return result;
    }

    ListNode* Merge_K_Sorted_Divide_Conquer(vector<ListNode*>& lists) {
        /*
        Divide and conquer: merge pairs recursively
        Time Complexity: O(n log k)
        Space Complexity: O(log k) for recursion stack
        */
        if (lists.empty()) return NULL;
        return Merge_Divide_Conquer_Helper(lists, 0, lists.size() - 1);
    }

private:
    ListNode* Merge_Divide_Conquer_Helper(vector<ListNode*>& lists, int left, int right) {
        if (left == right) return lists[left];
        if (left < right) {
            int mid = left + (right - left) / 2;
            ListNode* left_list = Merge_Divide_Conquer_Helper(lists, left, mid);
            ListNode* right_list = Merge_Divide_Conquer_Helper(lists, mid + 1, right);
            return Merge_Two_Sorted(left_list, right_list);
        }
        return NULL;
    }

    ListNode* Merge_Two_Sorted(ListNode* a, ListNode* b) {
        if (a == NULL) return b;
        if (b == NULL) return a;
        ListNode* result;
        if (a->data < b->data) {
            result = a;
            result->next = Merge_Two_Sorted(a->next, b);
        } else {
            result = b;
            result->next = Merge_Two_Sorted(a, b->next);
        }
        return result;
    }
};

void Test_Merge_K_Sorted_Lists() {
    Solution solution;
    
    vector<ListNode*> test1;
    test1.push_back(Create_List({1, 3, 5, 7}));
    test1.push_back(Create_List({2, 4, 6}));
    test1.push_back(Create_List({0, 8, 9}));
    ListNode* result1 = solution.Merge_K_Sorted_Min_Heap(test1);
    cout << "Test 1 Min Heap: ";
    Print_List(result1);
    
    vector<ListNode*> test2;
    test2.push_back(Create_List({1, 4, 5}));
    test2.push_back(Create_List({1, 3, 4}));
    test2.push_back(Create_List({2, 6}));
    ListNode* result2 = solution.Merge_K_Sorted_Divide_Conquer(test2);
    cout << "Test 2 Divide Conquer: ";
    Print_List(result2);
    
    vector<ListNode*> test3;
    test3.push_back(Create_List({1}));
    test3.push_back(Create_List({0}));
    ListNode* result3 = solution.Merge_K_Sorted_Min_Heap(test3);
    cout << "Test 3 Two Lists: ";
    Print_List(result3);
}

int main() {
    Test_Merge_K_Sorted_Lists();
    return 0;
}
