/*
Problem: Largest Independent Set in Binary Tree
URL: https://www.geeksforgeeks.org/largest-independent-set-problem-dp-26/

Problem Statement:
Given a binary tree, find the size of the largest independent set (LIS). An independent set is a set of nodes such that no two nodes in the set are adjacent (parent-child relationship).

Sample Input/Output:
Input: Binary tree
Output: Size of largest independent set
*/

#include <bits/stdc++.h>
using namespace std;

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    int LIS_Tree_Recursive(TreeNode* root) {
        /*
        Recursive approach
        Time Complexity: O(n)
        Space Complexity: O(h)
        */
        if (!root) return 0;
        
        int exclude = LIS_Tree_Recursive(root->left) + LIS_Tree_Recursive(root->right);
        
        int include = 1;
        if (root->left) {
            include += LIS_Tree_Recursive(root->left->left) + 
                      LIS_Tree_Recursive(root->left->right);
        }
        if (root->right) {
            include += LIS_Tree_Recursive(root->right->left) + 
                      LIS_Tree_Recursive(root->right->right);
        }
        
        return max(include, exclude);
    }
    
    int LIS_Tree_Memo(TreeNode* root, unordered_map<TreeNode*, int>& dp) {
        /*
        Memoization approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (!root) return 0;
        if (dp.find(root) != dp.end()) return dp[root];
        
        int exclude = LIS_Tree_Memo(root->left, dp) + LIS_Tree_Memo(root->right, dp);
        
        int include = 1;
        if (root->left) {
            include += LIS_Tree_Memo(root->left->left, dp) + 
                      LIS_Tree_Memo(root->left->right, dp);
        }
        if (root->right) {
            include += LIS_Tree_Memo(root->right->left, dp) + 
                      LIS_Tree_Memo(root->right->right, dp);
        }
        
        dp[root] = max(include, exclude);
        return dp[root];
    }
};

void Test_LIS_Tree() {
    Solution solution;
    
    TreeNode* root = new TreeNode(10);
    root->left = new TreeNode(20);
    root->right = new TreeNode(30);
    root->left->left = new TreeNode(40);
    root->left->right = new TreeNode(50);
    root->right->right = new TreeNode(60);
    root->left->right->left = new TreeNode(70);
    root->left->right->right = new TreeNode(80);
    
    cout << "Recursive: " << solution.LIS_Tree_Recursive(root) << endl;
    
    unordered_map<TreeNode*, int> dp;
    cout << "Memo: " << solution.LIS_Tree_Memo(root, dp) << endl;
}

int main() {
    Test_LIS_Tree();
    return 0;
}
