/*
Problem: Diameter Of Tree
URL: https://practice.geeksforgeeks.org/problems/diameter-of-binary-tree/1

Problem Statement:
Given a binary tree, find its diameter. Diameter of a tree is the number of nodes on the longest path between any two nodes in the tree.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output: 5
Explanation: Longest path is 4->2->1->3->7 with 5 nodes.
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
    int Diameter_Optimized(TreeNode* root) {
        /*
        Optimized single pass: Calculate height and diameter together
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        */
        int diameter = 0;
        Height_Helper(root, diameter);
        return diameter;
    }

    int Diameter_Naive(TreeNode* root) {
        /*
        Naive approach: Calculate height at each node
        Time Complexity: O(n^2) worst case
        Space Complexity: O(h) where h is height of tree
        */
        if (!root) return 0;
        int left_height = Height(root->left);
        int right_height = Height(root->right);
        int diameter_through_root = left_height + right_height + 1;
        int left_diameter = Diameter_Naive(root->left);
        int right_diameter = Diameter_Naive(root->right);
        return max({diameter_through_root, left_diameter, right_diameter});
    }

private:
    int Height_Helper(TreeNode* root, int& diameter) {
        if (!root) return 0;
        int left_height = Height_Helper(root->left, diameter);
        int right_height = Height_Helper(root->right, diameter);
        diameter = max(diameter, left_height + right_height + 1);
        return 1 + max(left_height, right_height);
    }

    int Height(TreeNode* root) {
        if (!root) return 0;
        return 1 + max(Height(root->left), Height(root->right));
    }
};

void Test_Diameter_Of_Tree() {
    Solution solution;
    
    vector<int> vals1 = {1, 2, 3, 4, 5, 6, 7};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Test 1 - Optimized: " << solution.Diameter_Optimized(root1) << endl;
    cout << "Test 1 - Naive: " << solution.Diameter_Naive(root1) << endl;
    
    vector<int> vals2 = {1, 2, 3, -1, -1, 4, 5};
    TreeNode* root2 = Create_Tree(vals2);
    cout << "Test 2 - Optimized: " << solution.Diameter_Optimized(root2) << endl;
    cout << "Test 2 - Naive: " << solution.Diameter_Naive(root2) << endl;
    
    vector<int> vals3 = {1, 2, -1, 3, -1, 4};
    TreeNode* root3 = Create_Tree(vals3);
    cout << "Test 3 - Optimized: " << solution.Diameter_Optimized(root3) << endl;
    cout << "Test 3 - Naive: " << solution.Diameter_Naive(root3) << endl;
}

int main() {
    Test_Diameter_Of_Tree();
    return 0;
}
