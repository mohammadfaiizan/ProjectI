/*
Problem: Diagonal Traversal
URL: https://www.geeksforgeeks.org/diagonal-traversal-of-binary-tree/

Problem Statement:
Given a Binary Tree, print the diagonal traversal of the binary tree. Diagonal traversal means traversing nodes diagonally from top-left to bottom-right.

Sample Input/Output:
Input:
        8
      /   \
     3    10
    / \     \
   1   6    14
      / \   /
     4   7 13

Output: 8 10 14 3 6 7 13 1 4
Explanation: Nodes are printed diagonally from top-left to bottom-right.
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
    vector<int> Diagonal_Traversal_Map(TreeNode* root) {
        /*
        Map-based recursion approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        map<int, vector<int>> m;
        Diagonal_Traversal_Helper(root, 0, m);
        vector<int> result;
        for (auto it : m) {
            for (int val : it.second) {
                result.push_back(val);
            }
        }
        return result;
    }

    vector<int> Diagonal_Traversal_Queue(TreeNode* root) {
        /*
        Queue-based BFS approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<int> result;
        if (!root) return result;
        queue<TreeNode*> q;
        q.push(root);
        while (!q.empty()) {
            TreeNode* node = q.front();
            q.pop();
            while (node) {
                result.push_back(node->data);
                if (node->left) q.push(node->left);
                node = node->right;
            }
        }
        return result;
    }

private:
    void Diagonal_Traversal_Helper(TreeNode* root, int diagonal, map<int, vector<int>>& m) {
        if (!root) return;
        m[diagonal].push_back(root->data);
        Diagonal_Traversal_Helper(root->left, diagonal + 1, m);
        Diagonal_Traversal_Helper(root->right, diagonal, m);
    }
};

void Test_Diagonal_Traversal() {
    Solution solution;
    
    vector<int> vals1 = {8, 3, 10, 1, 6, -1, 14, -1, -1, 4, 7, 13};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Test 1 - Map: ";
    vector<int> result1 = solution.Diagonal_Traversal_Map(root1);
    for (int val : result1) cout << val << " ";
    cout << endl;
    
    cout << "Test 1 - Queue: ";
    vector<int> result2 = solution.Diagonal_Traversal_Queue(root1);
    for (int val : result2) cout << val << " ";
    cout << endl;
}

int main() {
    Test_Diagonal_Traversal();
    return 0;
}
