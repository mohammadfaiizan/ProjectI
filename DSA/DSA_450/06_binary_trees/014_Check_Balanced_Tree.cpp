/*
Problem: Check Balanced Tree
URL: https://practice.geeksforgeeks.org/problems/check-for-balanced-tree/1

Problem Statement:
Given a binary tree, find if it is height balanced or not. A tree is height balanced if difference between heights of left and right subtrees is not more than one for all nodes of tree.

Sample Input/Output:
Input:
        1
      /   \
     2     3
    / \
   4   5

Output: true
Explanation: Height difference at each node is at most 1.

Input:
        1
      /
     2
    /
   3

Output: false
Explanation: Height difference exceeds 1.
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
    bool Is_Balanced_Optimized(TreeNode* root) {
        /*
        Optimized single pass approach
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height
        */
        return Check_Balanced_Helper(root) != -1;
    }

    bool Is_Balanced_Naive(TreeNode* root) {
        /*
        Naive height check approach
        Time Complexity: O(n^2)
        Space Complexity: O(h) where h is height
        */
        if (!root) return true;
        int leftHeight = Height(root->left);
        int rightHeight = Height(root->right);
        return abs(leftHeight - rightHeight) <= 1 && 
               Is_Balanced_Naive(root->left) && 
               Is_Balanced_Naive(root->right);
    }

private:
    int Check_Balanced_Helper(TreeNode* root) {
        if (!root) return 0;
        int leftHeight = Check_Balanced_Helper(root->left);
        if (leftHeight == -1) return -1;
        int rightHeight = Check_Balanced_Helper(root->right);
        if (rightHeight == -1) return -1;
        if (abs(leftHeight - rightHeight) > 1) return -1;
        return 1 + max(leftHeight, rightHeight);
    }

    int Height(TreeNode* root) {
        if (!root) return 0;
        return 1 + max(Height(root->left), Height(root->right));
    }
};

void Test_Check_Balanced_Tree() {
    Solution solution;
    
    vector<int> vals1 = {1, 2, 3, 4, 5};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Test 1 - Optimized: " << (solution.Is_Balanced_Optimized(root1) ? "true" : "false") << endl;
    cout << "Test 1 - Naive: " << (solution.Is_Balanced_Naive(root1) ? "true" : "false") << endl;
    
    vector<int> vals2 = {1, 2, -1, 3};
    TreeNode* root2 = Create_Tree(vals2);
    cout << "Test 2 - Optimized: " << (solution.Is_Balanced_Optimized(root2) ? "true" : "false") << endl;
    cout << "Test 2 - Naive: " << (solution.Is_Balanced_Naive(root2) ? "true" : "false") << endl;
}

int main() {
    Test_Check_Balanced_Tree();
    return 0;
}
