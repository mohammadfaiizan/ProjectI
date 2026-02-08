/*
Problem: Flatten a Linked List
URL: https://practice.geeksforgeeks.org/problems/flattening-a-linked-list/1

Problem Statement:
Given a linked list where every node has a next pointer and a bottom/down pointer. All bottom lists are sorted. Flatten the list into a single sorted list using bottom pointers.

Sample Input/Output:
Input: 5 -> 10 -> 19 -> 28
       |    |     |     |
       V    V     V     V
       7    20    22    35
       |          |     |
       V          V     V
       8          50    40
       |                |
       V                V
       30               45
Output: 5 -> 7 -> 8 -> 10 -> 19 -> 20 -> 22 -> 28 -> 30 -> 35 -> 40 -> 45 -> 50
Explanation: All nodes are merged using bottom pointers into a single sorted list.
*/

#include <bits/stdc++.h>
using namespace std;

struct FlatNode {
    int data;
    FlatNode* next;
    FlatNode* bottom;
    FlatNode(int x) : data(x), next(NULL), bottom(NULL) {}
};

FlatNode* Create_Flat_List(vector<vector<int>> lists) {
    if (lists.empty()) return NULL;
    FlatNode* head = new FlatNode(lists[0][0]);
    FlatNode* current = head;
    for (int i = 0; i < lists.size(); i++) {
        FlatNode* row_head = NULL;
        FlatNode* row_current = NULL;
        for (int j = 0; j < lists[i].size(); j++) {
            FlatNode* node = new FlatNode(lists[i][j]);
            if (row_head == NULL) {
                row_head = node;
                row_current = node;
            } else {
                row_current->bottom = node;
                row_current = row_current->bottom;
            }
        }
        if (i == 0) {
            head = row_head;
        } else {
            current->next = row_head;
        }
        current = row_head;
        while (current->bottom) current = current->bottom;
    }
    return head;
}

void Print_Flattened_List(FlatNode* head) {
    while (head) {
        cout << head->data;
        if (head->bottom) cout << " -> ";
        head = head->bottom;
    }
    cout << endl;
}

class Solution {
public:
    FlatNode* Flatten_Recursive_Merge(FlatNode* root) {
        /*
        Recursive merge approach: Merge bottom lists recursively
        Time Complexity: O(n) where n is total nodes
        Space Complexity: O(1) excluding recursion stack
        */
        if (root == NULL || root->next == NULL) return root;
        root->next = Flatten_Recursive_Merge(root->next);
        root = Merge_Two_Sorted(root, root->next);
        return root;
    }

    FlatNode* Flatten_Min_Heap(FlatNode* root) {
        /*
        Min heap approach: Use priority queue to merge all lists
        Time Complexity: O(n log k) where k is number of lists
        Space Complexity: O(k) for heap
        */
        if (root == NULL) return NULL;
        priority_queue<pair<int, FlatNode*>, vector<pair<int, FlatNode*>>, greater<pair<int, FlatNode*>>> pq;
        FlatNode* current = root;
        while (current) {
            FlatNode* temp = current;
            while (temp) {
                pq.push({temp->data, temp});
                temp = temp->bottom;
            }
            current = current->next;
        }
        FlatNode* dummy = new FlatNode(0);
        FlatNode* result = dummy;
        while (!pq.empty()) {
            FlatNode* node = pq.top().second;
            pq.pop();
            dummy->bottom = node;
            dummy = dummy->bottom;
            dummy->next = NULL;
        }
        return result->bottom;
    }

private:
    FlatNode* Merge_Two_Sorted(FlatNode* a, FlatNode* b) {
        if (a == NULL) return b;
        if (b == NULL) return a;
        FlatNode* result;
        if (a->data < b->data) {
            result = a;
            result->bottom = Merge_Two_Sorted(a->bottom, b);
        } else {
            result = b;
            result->bottom = Merge_Two_Sorted(a, b->bottom);
        }
        result->next = NULL;
        return result;
    }
};

void Test_Flatten_Linked_List() {
    Solution solution;
    
    vector<vector<int>> test1 = {{5, 7, 8, 30}, {10, 20}, {19, 22, 50}, {28, 35, 40, 45}};
    FlatNode* list1 = Create_Flat_List(test1);
    FlatNode* result1 = solution.Flatten_Recursive_Merge(list1);
    cout << "Test 1 Recursive: ";
    Print_Flattened_List(result1);
    
    vector<vector<int>> test2 = {{1, 3, 5}, {2, 4}};
    FlatNode* list2 = Create_Flat_List(test2);
    FlatNode* result2 = solution.Flatten_Min_Heap(list2);
    cout << "Test 2 Min Heap: ";
    Print_Flattened_List(result2);
    
    vector<vector<int>> test3 = {{1}};
    FlatNode* list3 = Create_Flat_List(test3);
    FlatNode* result3 = solution.Flatten_Recursive_Merge(list3);
    cout << "Test 3 Single: ";
    Print_Flattened_List(result3);
}

int main() {
    Test_Flatten_Linked_List();
    return 0;
}
