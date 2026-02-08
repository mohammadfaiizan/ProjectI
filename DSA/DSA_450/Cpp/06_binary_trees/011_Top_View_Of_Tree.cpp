/*
Problem: Top View Of Tree
URL: https://practice.geeksforgeeks.org/problems/top-view-of-binary-tree/1

Problem Statement:
Given a binary tree, print the top view of it. Top view means when you look the tree from the top, the nodes you will see will be called the top view of the tree.

Sample Input/Output:
Input:
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output: 4 2 1 3 7
Explanation: Top view shows nodes at each horizontal distance from root.
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
    vector<int> Top_View_BFS(TreeNode* root) {
        /*
        BFS with horizontal distance and map
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<int> result;
        if (!root) return result;
        map<int, int> m;
        queue<pair<TreeNode*, int>> q;
        q.push({root, 0});
        while (!q.empty()) {
            TreeNode* node = q.front().first;
            int hd = q.front().second;
            q.pop();
            if (m.find(hd) == m.end()) {
                m[hd] = node->data;
            }
            if (node->left) q.push({node->left, hd - 1});
            if (node->right) q.push({node->right, hd + 1});
        }
        for (auto it : m) {
            result.push_back(it.second);
        }
        return result;
    }

    vector<int> Top_View_Recursive(TreeNode* root) {
        /*
        Recursive with map
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        map<int, pair<int, int>> m;
        Top_View_Helper(root, 0, 0, m);
        vector<int> result;
        for (auto it : m) {
            result.push_back(it.second.second);
        }
        return result;
    }

private:
    void Top_View_Helper(TreeNode* root, int hd, int level, map<int, pair<int, int>>& m) {
        if (!root) return;
        if (m.find(hd) == m.end() || level < m[hd].first) {
            m[hd] = {level, root->data};
        }
        Top_View_Helper(root->left, hd - 1, level + 1, m);
        Top_View_Helper(root->right, hd + 1, level + 1, m);
    }
};

void Test_Top_View_Of_Tree() {
    Solution solution;
    
    vector<int> vals1 = {1, 2, 3, 4, 5, 6, 7};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Test 1 - BFS: ";
    vector<int> result1 = solution.Top_View_BFS(root1);
    for (int val : result1) cout << val << " ";
    cout << endl;
    
    cout << "Test 1 - Recursive: ";
    vector<int> result2 = solution.Top_View_Recursive(root1);
    for (int val : result2) cout << val << " ";
    cout << endl;
}

int main() {
    Test_Top_View_Of_Tree();
    return 0;
}
