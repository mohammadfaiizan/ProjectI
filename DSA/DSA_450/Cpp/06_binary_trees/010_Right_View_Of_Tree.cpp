/*
Problem: Right View Of Tree
URL: https://practice.geeksforgeeks.org/problems/right-view-of-binary-tree/1

Problem Statement:
Given a Binary Tree, print Right view of it. Right view of a Binary Tree is set of nodes visible when tree is visited from Right side.

Sample Input/Output:
Input:
        1
      /   \
     2     3
    / \   / \
   4   5 6   7
      /
     8

Output: 1 3 7 8
Explanation: Right view shows the rightmost node at each level.
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
    vector<int> Right_View_Recursive(TreeNode* root) {
        /*
        Recursive approach with level tracking
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height
        */
        vector<int> result;
        Right_View_Helper(root, 0, result);
        return result;
    }

    vector<int> Right_View_BFS(TreeNode* root) {
        /*
        Queue BFS approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<int> result;
        if (!root) return result;
        queue<TreeNode*> q;
        q.push(root);
        while (!q.empty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode* node = q.front();
                q.pop();
                if (i == size - 1) {
                    result.push_back(node->data);
                }
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
        }
        return result;
    }

private:
    void Right_View_Helper(TreeNode* root, int level, vector<int>& result) {
        if (!root) return;
        if (level == result.size()) {
            result.push_back(root->data);
        }
        Right_View_Helper(root->right, level + 1, result);
        Right_View_Helper(root->left, level + 1, result);
    }
};

void Test_Right_View_Of_Tree() {
    Solution solution;
    
    vector<int> vals1 = {1, 2, 3, 4, 5, 6, 7, -1, -1, 8};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Test 1 - Recursive: ";
    vector<int> result1 = solution.Right_View_Recursive(root1);
    for (int val : result1) cout << val << " ";
    cout << endl;
    
    cout << "Test 1 - BFS: ";
    vector<int> result2 = solution.Right_View_BFS(root1);
    for (int val : result2) cout << val << " ";
    cout << endl;
}

int main() {
    Test_Right_View_Of_Tree();
    return 0;
}
