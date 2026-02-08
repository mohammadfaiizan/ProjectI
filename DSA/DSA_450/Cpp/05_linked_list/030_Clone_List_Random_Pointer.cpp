/*
Problem: Clone a Linked List with Next and Random Pointer
URL: https://practice.geeksforgeeks.org/problems/clone-a-linked-list-with-next-and-random-pointer/1

Problem Statement:
Given a linked list where each node has a next pointer and a random pointer pointing to any node in the list or NULL. Clone this linked list.

Sample Input/Output:
Input: 1 -> 2 -> 3 -> 4 -> NULL
       |    |    |    |
       NULL 1    3    2
Output: 1 -> 2 -> 3 -> 4 -> NULL
        |    |    |    |
        NULL 1    3    2
Explanation: Create a deep copy with same structure and random pointers.
*/

#include <bits/stdc++.h>
using namespace std;

struct RandomNode {
    int data;
    RandomNode* next;
    RandomNode* random;
    RandomNode(int x) : data(x), next(NULL), random(NULL) {}
};

RandomNode* Create_Random_List(vector<int> arr, vector<int> random_indices) {
    if (arr.empty()) return NULL;
    vector<RandomNode*> nodes;
    RandomNode* head = new RandomNode(arr[0]);
    RandomNode* current = head;
    nodes.push_back(head);
    for (int i = 1; i < arr.size(); i++) {
        current->next = new RandomNode(arr[i]);
        current = current->next;
        nodes.push_back(current);
    }
    for (int i = 0; i < random_indices.size(); i++) {
        if (random_indices[i] != -1) {
            nodes[i]->random = nodes[random_indices[i]];
        }
    }
    return head;
}

void Print_Random_List(RandomNode* head) {
    RandomNode* current = head;
    while (current) {
        cout << current->data << " (random: ";
        if (current->random) cout << current->random->data;
        else cout << "NULL";
        cout << ")";
        if (current->next) cout << " -> ";
        current = current->next;
    }
    cout << endl;
}

class Solution {
public:
    RandomNode* Clone_Hash_Map(RandomNode* head) {
        /*
        Use hash map to store original to clone mapping
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (head == NULL) return NULL;
        unordered_map<RandomNode*, RandomNode*> map;
        RandomNode* current = head;
        while (current) {
            map[current] = new RandomNode(current->data);
            current = current->next;
        }
        current = head;
        while (current) {
            map[current]->next = current->next ? map[current->next] : NULL;
            map[current]->random = current->random ? map[current->random] : NULL;
            current = current->next;
        }
        return map[head];
    }

    RandomNode* Clone_Interleaving_Nodes(RandomNode* head) {
        /*
        Interleave original and clone nodes, then separate
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (head == NULL) return NULL;
        RandomNode* current = head;
        while (current) {
            RandomNode* clone = new RandomNode(current->data);
            clone->next = current->next;
            current->next = clone;
            current = clone->next;
        }
        current = head;
        while (current) {
            if (current->random) {
                current->next->random = current->random->next;
            }
            current = current->next->next;
        }
        current = head;
        RandomNode* clone_head = head->next;
        RandomNode* clone_current = clone_head;
        while (current) {
            current->next = current->next->next;
            if (clone_current->next) {
                clone_current->next = clone_current->next->next;
            }
            current = current->next;
            clone_current = clone_current->next;
        }
        return clone_head;
    }
};

void Test_Clone_List_Random_Pointer() {
    Solution solution;
    
    vector<int> test1_arr = {1, 2, 3, 4};
    vector<int> test1_random = {-1, 0, 2, 1};
    RandomNode* list1 = Create_Random_List(test1_arr, test1_random);
    RandomNode* clone1 = solution.Clone_Hash_Map(list1);
    cout << "Test 1 Hash Map Clone: ";
    Print_Random_List(clone1);
    
    vector<int> test2_arr = {1, 3, 5};
    vector<int> test2_random = {2, -1, 0};
    RandomNode* list2 = Create_Random_List(test2_arr, test2_random);
    RandomNode* clone2 = solution.Clone_Interleaving_Nodes(list2);
    cout << "Test 2 Interleaving Clone: ";
    Print_Random_List(clone2);
    
    vector<int> test3_arr = {7};
    vector<int> test3_random = {-1};
    RandomNode* list3 = Create_Random_List(test3_arr, test3_random);
    RandomNode* clone3 = solution.Clone_Hash_Map(list3);
    cout << "Test 3 Single Node: ";
    Print_Random_List(clone3);
}

int main() {
    Test_Clone_List_Random_Pointer();
    return 0;
}
