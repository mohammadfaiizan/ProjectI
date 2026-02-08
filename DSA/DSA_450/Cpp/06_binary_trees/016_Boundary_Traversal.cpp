/*
Problem: Boundary Traversal
URL: https://practice.geeksforgeeks.org/problems/boundary-traversal-of-binary-tree/1

Problem Statement:
Given a Binary Tree, find its Boundary Traversal. The traversal should be in the following order: Left boundary nodes, Leaf nodes, Right boundary nodes in reverse order.

Sample Input/Output:
Input:
        1
      /   \
     2     3
    / \   / \
   4   5 6   7
      /
     8

Output: 1 2 4 8 5 6 7 3
Explanation: Left boundary: 1 2 4, Leaves: 8 5 6 7, Right boundary (reverse): 3
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
    vector<int> Boundary_Traversal_Recursive(TreeNode* root) {
        /*
        Recursive approach (left boundary + leaves + right boundary)
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height
        */
        vector<int> result;
        if (!root) return result;
        if (root->left || root->right) {
            result.push_back(root->data);
        }
        Left_Boundary(root->left, result);
        Leaves(root, result);
        Right_Boundary(root->right, result);
        return result;
    }

    vector<int> Boundary_Traversal_Iterative(TreeNode* root) {
        /*
        Iterative approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<int> result;
        if (!root) return result;
        if (root->left || root->right) {
            result.push_back(root->data);
        }
        TreeNode* node = root->left;
        while (node && (node->left || node->right)) {
            result.push_back(node->data);
            node = node->left ? node->left : node->right;
        }
        stack<TreeNode*> s;
        s.push(root);
        while (!s.empty()) {
            TreeNode* curr = s.top();
            s.pop();
            if (curr->right) s.push(curr->right);
            if (curr->left) s.push(curr->left);
            if (!curr->left && !curr->right && curr != root) {
                result.push_back(curr->data);
            }
        }
        vector<int> rightBoundary;
        node = root->right;
        while (node && (node->left || node->right)) {
            rightBoundary.push_back(node->data);
            node = node->right ? node->right : node->left;
        }
        reverse(rightBoundary.begin(), rightBoundary.end());
        for (int val : rightBoundary) {
            result.push_back(val);
        }
        return result;
    }

private:
    void Left_Boundary(TreeNode* root, vector<int>& result) {
        if (!root || (!root->left && !root->right)) return;
        result.push_back(root->data);
        if (root->left) {
            Left_Boundary(root->left, result);
        } else {
            Left_Boundary(root->right, result);
        }
    }

    void Right_Boundary(TreeNode* root, vector<int>& result) {
        if (!root || (!root->left && !root->right)) return;
        if (root->right) {
            Right_Boundary(root->right, result);
        } else {
            Right_Boundary(root->left, result);
        }
        result.push_back(root->data);
    }

    void Leaves(TreeNode* root, vector<int>& result) {
        if (!root) return;
        if (!root->left && !root->right) {
            result.push_back(root->data);
            return;
        }
        Leaves(root->left, result);
        Leaves(root->right, result);
    }
};

void Test_Boundary_Traversal() {
    Solution solution;
    
    vector<int> vals1 = {1, 2, 3, 4, 5, 6, 7, -1, -1, 8};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Test 1 - Recursive: ";
    vector<int> result1 = solution.Boundary_Traversal_Recursive(root1);
    for (int val : result1) cout << val << " ";
    cout << endl;
    
    cout << "Test 1 - Iterative: ";
    vector<int> result2 = solution.Boundary_Traversal_Iterative(root1);
    for (int val : result2) cout << val << " ";
    cout << endl;
}

int main() {
    Test_Boundary_Traversal();
    return 0;
}
