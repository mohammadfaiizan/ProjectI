/*
Problem: Convert To Sum Tree
URL: https://practice.geeksforgeeks.org/problems/transform-to-sum-tree/1

Problem Statement:
Convert a binary tree such that each node contains sum of left and right subtree values. Leaf nodes become 0.

Sample Input/Output:
Input: 
        10
      /    \
     -2     6
    /  \   / \
   8   -4 7   5

Output:
        20
      /    \
     4     12
    /  \   / \
   0   0  0   0

Explanation: Each node is replaced with sum of its left and right subtree values.
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
    int Convert_To_Sum_Tree_Postorder(TreeNode* root) {
        /*
        Post-order recursion: Process children first, then update current node
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        */
        if (!root) return 0;
        int left_sum = Convert_To_Sum_Tree_Postorder(root->left);
        int right_sum = Convert_To_Sum_Tree_Postorder(root->right);
        int old_data = root->data;
        root->data = left_sum + right_sum;
        return old_data + root->data;
    }
};

void Test_Convert_To_Sum_Tree() {
    Solution solution;
    
    vector<int> vals1 = {10, -2, 6, 8, -4, 7, 5};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Before conversion: ";
    Print_Inorder(root1);
    cout << endl;
    solution.Convert_To_Sum_Tree_Postorder(root1);
    cout << "After conversion: ";
    Print_Inorder(root1);
    cout << endl;
    
    vector<int> vals2 = {1, 2, 3};
    TreeNode* root2 = Create_Tree(vals2);
    cout << "Before conversion: ";
    Print_Inorder(root2);
    cout << endl;
    solution.Convert_To_Sum_Tree_Postorder(root2);
    cout << "After conversion: ";
    Print_Inorder(root2);
    cout << endl;
}

int main() {
    Test_Convert_To_Sum_Tree();
    return 0;
}
