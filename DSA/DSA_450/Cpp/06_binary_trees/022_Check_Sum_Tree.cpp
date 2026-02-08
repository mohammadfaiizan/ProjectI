/*
Problem: Check Sum Tree
URL: https://practice.geeksforgeeks.org/problems/sum-tree/1

Problem Statement:
Check if binary tree is a sum tree (each node = sum of left + right subtree).

Sample Input/Output:
Input: 
        26
      /    \
    10      3
   /  \    / \
  4    6  1   2

Output: true
Explanation: 26 = 10 + 3 + 13, 10 = 4 + 6, 3 = 1 + 2
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
    int Sum_Tree_Optimized(TreeNode* root, bool& is_sum_tree) {
        /*
        Optimized single pass: Return sum and check condition simultaneously
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        */
        if (!root) return 0;
        if (!root->left && !root->right) return root->data;
        int left_sum = Sum_Tree_Optimized(root->left, is_sum_tree);
        int right_sum = Sum_Tree_Optimized(root->right, is_sum_tree);
        if (root->data != left_sum + right_sum) {
            is_sum_tree = false;
        }
        return root->data + left_sum + right_sum;
    }

    bool Is_Sum_Tree_Optimized(TreeNode* root) {
        bool is_sum_tree = true;
        Sum_Tree_Optimized(root, is_sum_tree);
        return is_sum_tree;
    }

    int Get_Sum(TreeNode* root) {
        if (!root) return 0;
        return root->data + Get_Sum(root->left) + Get_Sum(root->right);
    }

    bool Is_Sum_Tree_Naive(TreeNode* root) {
        /*
        Naive with separate sum function: Check each node separately
        Time Complexity: O(n^2) worst case
        Space Complexity: O(h) where h is height of tree
        */
        if (!root || (!root->left && !root->right)) return true;
        int left_sum = Get_Sum(root->left);
        int right_sum = Get_Sum(root->right);
        return (root->data == left_sum + right_sum) && 
               Is_Sum_Tree_Naive(root->left) && 
               Is_Sum_Tree_Naive(root->right);
    }
};

void Test_Check_Sum_Tree() {
    Solution solution;
    
    vector<int> vals1 = {26, 10, 3, 4, 6, 1, 2};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Tree 1 (optimized): " << solution.Is_Sum_Tree_Optimized(root1) << endl;
    cout << "Tree 1 (naive): " << solution.Is_Sum_Tree_Naive(root1) << endl;
    
    vector<int> vals2 = {10, 4, 6};
    TreeNode* root2 = Create_Tree(vals2);
    cout << "Tree 2 (optimized): " << solution.Is_Sum_Tree_Optimized(root2) << endl;
    cout << "Tree 2 (naive): " << solution.Is_Sum_Tree_Naive(root2) << endl;
    
    vector<int> vals3 = {10, 3, 5};
    TreeNode* root3 = Create_Tree(vals3);
    cout << "Tree 3 (optimized): " << solution.Is_Sum_Tree_Optimized(root3) << endl;
    cout << "Tree 3 (naive): " << solution.Is_Sum_Tree_Naive(root3) << endl;
}

int main() {
    Test_Check_Sum_Tree();
    return 0;
}
