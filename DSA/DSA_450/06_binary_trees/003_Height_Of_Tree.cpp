/*
Problem: Height Of Tree
URL: https://practice.geeksforgeeks.org/problems/height-of-binary-tree/1

Problem Statement:
Given a binary tree, find its height. Height of a tree is the number of edges in the longest path from root to a leaf node.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output: 2
Explanation: Longest path from root to leaf has 2 edges (e.g., 1->2->4).
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

class Solution {
public:
    int Height_Recursive(TreeNode* root) {
        /*
        Recursive approach: Height is max of left and right subtree heights + 1
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        */
        if (!root) return 0;
        return 1 + max(Height_Recursive(root->left), Height_Recursive(root->right));
    }

    int Height_Iterative(TreeNode* root) {
        /*
        Iterative BFS approach: Count levels using queue
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for queue
        */
        if (!root) return 0;
        queue<TreeNode*> q;
        q.push(root);
        int height = 0;
        while (!q.empty()) {
            int size = q.size();
            height++;
            for (int i = 0; i < size; i++) {
                TreeNode* node = q.front();
                q.pop();
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
        }
        return height;
    }
};

void Test_Height_Of_Tree() {
    Solution solution;
    
    vector<int> vals1 = {1, 2, 3, 4, 5, 6, 7};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Test 1 - Recursive: " << solution.Height_Recursive(root1) << endl;
    cout << "Test 1 - Iterative: " << solution.Height_Iterative(root1) << endl;
    
    vector<int> vals2 = {1, 2, 3, -1, -1, 4, 5};
    TreeNode* root2 = Create_Tree(vals2);
    cout << "Test 2 - Recursive: " << solution.Height_Recursive(root2) << endl;
    cout << "Test 2 - Iterative: " << solution.Height_Iterative(root2) << endl;
    
    vector<int> vals3 = {1};
    TreeNode* root3 = Create_Tree(vals3);
    cout << "Test 3 - Recursive: " << solution.Height_Recursive(root3) << endl;
    cout << "Test 3 - Iterative: " << solution.Height_Iterative(root3) << endl;
}

int main() {
    Test_Height_Of_Tree();
    return 0;
}
