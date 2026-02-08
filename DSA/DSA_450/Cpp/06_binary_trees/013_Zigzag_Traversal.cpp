/*
Problem: Zigzag Traversal
URL: https://practice.geeksforgeeks.org/problems/zigzag-tree-traversal/1

Problem Statement:
Given a Binary Tree. Find the Zig-Zag Level Order Traversal of the Binary Tree. Zig-Zag traversal means starting from level 0 for the root node, for all the even levels we print the node's value from left to right and for all the odd levels we print the node's value from right to left.

Sample Input/Output:
Input:
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output: 1 3 2 4 5 6 7
Explanation: Level 0: 1 (left to right), Level 1: 3 2 (right to left), Level 2: 4 5 6 7 (left to right)
*/

#include <bits/stdc++.h>
using namespace std;

struct TreeNode {
    int data;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : data(x), left(NULL), right(NULL) {}
};

TreeNode* Create_Tree(vector<int> vals) {
    if (vals.empty() || vals[0] == -1) return NULL;
    TreeNode* root = new TreeNode(vals[0]);
    queue<TreeNode*> q;
    q.push(root);
    int i = 1;
    while (!q.empty() && i < vals.size()) {
        TreeNode* node = q.front();
        q.pop();
        if (i < vals.size() && vals[i] != -1) {
            node->left = new TreeNode(vals[i]);
            q.push(node->left);
        }
        i++;
        if (i < vals.size() && vals[i] != -1) {
            node->right = new TreeNode(vals[i]);
            q.push(node->right);
        }
        i++;
    }
    return root;
}

void Print_Inorder(TreeNode* root) {
    if (!root) return;
    Print_Inorder(root->left);
    cout << root->data << " ";
    Print_Inorder(root->right);
}

class Solution {
public:
    vector<int> Zigzag_Queue_Stack(TreeNode* root) {
        /*
        Queue + Stack approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<int> result;
        if (!root) return result;
        queue<TreeNode*> q;
        stack<int> s;
        bool leftToRight = true;
        q.push(root);
        while (!q.empty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode* node = q.front();
                q.pop();
                if (leftToRight) {
                    result.push_back(node->data);
                } else {
                    s.push(node->data);
                }
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            while (!s.empty()) {
                result.push_back(s.top());
                s.pop();
            }
            leftToRight = !leftToRight;
        }
        return result;
    }

    vector<int> Zigzag_Two_Stacks(TreeNode* root) {
        /*
        Two Stacks approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<int> result;
        if (!root) return result;
        stack<TreeNode*> s1, s2;
        s1.push(root);
        while (!s1.empty() || !s2.empty()) {
            while (!s1.empty()) {
                TreeNode* node = s1.top();
                s1.pop();
                result.push_back(node->data);
                if (node->left) s2.push(node->left);
                if (node->right) s2.push(node->right);
            }
            while (!s2.empty()) {
                TreeNode* node = s2.top();
                s2.pop();
                result.push_back(node->data);
                if (node->right) s1.push(node->right);
                if (node->left) s1.push(node->left);
            }
        }
        return result;
    }

    vector<int> Zigzag_Deque(TreeNode* root) {
        /*
        Deque approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<int> result;
        if (!root) return result;
        deque<TreeNode*> dq;
        dq.push_back(root);
        bool leftToRight = true;
        while (!dq.empty()) {
            int size = dq.size();
            for (int i = 0; i < size; i++) {
                TreeNode* node;
                if (leftToRight) {
                    node = dq.front();
                    dq.pop_front();
                    result.push_back(node->data);
                    if (node->left) dq.push_back(node->left);
                    if (node->right) dq.push_back(node->right);
                } else {
                    node = dq.back();
                    dq.pop_back();
                    result.push_back(node->data);
                    if (node->right) dq.push_front(node->right);
                    if (node->left) dq.push_front(node->left);
                }
            }
            leftToRight = !leftToRight;
        }
        return result;
    }
};

void Test_Zigzag_Traversal() {
    Solution solution;
    
    vector<int> vals1 = {1, 2, 3, 4, 5, 6, 7};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Test 1 - Queue+Stack: ";
    vector<int> result1 = solution.Zigzag_Queue_Stack(root1);
    for (int val : result1) cout << val << " ";
    cout << endl;
    
    cout << "Test 1 - Two Stacks: ";
    vector<int> result2 = solution.Zigzag_Two_Stacks(root1);
    for (int val : result2) cout << val << " ";
    cout << endl;
    
    cout << "Test 1 - Deque: ";
    vector<int> result3 = solution.Zigzag_Deque(root1);
    for (int val : result3) cout << val << " ";
    cout << endl;
}

int main() {
    Test_Zigzag_Traversal();
    return 0;
}
