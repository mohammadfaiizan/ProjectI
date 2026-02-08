/*
Problem: Maximum Sum of Nodes with No Two Adjacent
URL: https://www.geeksforgeeks.org/maximum-sum-nodes-binary-tree-no-two-adjacent/

Problem Statement:
Find the maximum sum of nodes in a binary tree such that no two selected nodes are adjacent (parent-child relationship).

Sample Input/Output:
Input: [1, 2, 3]
Output: 5
Explanation: Select nodes with values 2 and 3 (sum = 5), avoiding adjacent nodes.
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
    pair<int, int> Max_Sum_No_Adjacent_Pair(TreeNode* root) {
        /*
        Pair-based recursion (include/exclude)
        Time Complexity: O(n)
        Space Complexity: O(h)
        */
        if (!root) return {0, 0};
        pair<int, int> left = Max_Sum_No_Adjacent_Pair(root->left);
        pair<int, int> right = Max_Sum_No_Adjacent_Pair(root->right);
        int include = root->data + left.second + right.second;
        int exclude = max(left.first, left.second) + max(right.first, right.second);
        return {include, exclude};
    }
    
    int Max_Sum_No_Adjacent_Recursion(TreeNode* root) {
        pair<int, int> result = Max_Sum_No_Adjacent_Pair(root);
        return max(result.first, result.second);
    }
    
    int Max_Sum_No_Adjacent_Memoization(TreeNode* root, map<TreeNode*, int>& memo) {
        /*
        Memoization with map
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (!root) return 0;
        if (memo.find(root) != memo.end()) return memo[root];
        int include = root->data;
        if (root->left) {
            include += Max_Sum_No_Adjacent_Memoization(root->left->left, memo);
            include += Max_Sum_No_Adjacent_Memoization(root->left->right, memo);
        }
        if (root->right) {
            include += Max_Sum_No_Adjacent_Memoization(root->right->left, memo);
            include += Max_Sum_No_Adjacent_Memoization(root->right->right, memo);
        }
        int exclude = Max_Sum_No_Adjacent_Memoization(root->left, memo) + 
                      Max_Sum_No_Adjacent_Memoization(root->right, memo);
        memo[root] = max(include, exclude);
        return memo[root];
    }
    
    int Find_Max_Sum_No_Adjacent(TreeNode* root) {
        return Max_Sum_No_Adjacent_Recursion(root);
    }
};

void Test_Max_Sum_No_Adjacent() {
    Solution solution;
    
    vector<int> vals1 = {1, 2, 3};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Test 1: " << solution.Find_Max_Sum_No_Adjacent(root1) << endl;
    
    vector<int> vals2 = {10, 1, 2, 3, 4, 5, 6};
    TreeNode* root2 = Create_Tree(vals2);
    cout << "Test 2: " << solution.Find_Max_Sum_No_Adjacent(root2) << endl;
    
    vector<int> vals3 = {1, 2, 3, 1, 3, 5};
    TreeNode* root3 = Create_Tree(vals3);
    cout << "Test 3: " << solution.Find_Max_Sum_No_Adjacent(root3) << endl;
}

int main() {
    Test_Max_Sum_No_Adjacent();
    return 0;
}
