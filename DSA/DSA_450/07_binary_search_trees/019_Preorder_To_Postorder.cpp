/*
Problem: Preorder to Postorder
URL: https://practice.geeksforgeeks.org/problems/preorder-to-postorder4423/1

Problem Statement:
Given preorder of BST, find postorder without constructing tree.

Sample Input/Output:
Input: pre[] = {40, 30, 35, 80, 100}
Output: post[] = {35, 30, 100, 80, 40}
Explanation: BST preorder converted to postorder.
*/

#include <bits/stdc++.h>
using namespace std;

struct TreeNode {
    int data;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : data(x), left(NULL), right(NULL) {}
};

TreeNode* Insert_BST(TreeNode* root, int key) {
    if (root == NULL) return new TreeNode(key);
    if (key < root->data) root->left = Insert_BST(root->left, key);
    else root->right = Insert_BST(root->right, key);
    return root;
}

void Print_Inorder(TreeNode* root) {
    if (root == NULL) return;
    Print_Inorder(root->left);
    cout << root->data << " ";
    Print_Inorder(root->right);
}

class Solution {
public:
    void Preorder_To_Postorder_Range(vector<int>& pre, int& idx, int min_val, int max_val, vector<int>& post) {
        if (idx >= pre.size()) return;
        if (pre[idx] < min_val || pre[idx] > max_val) return;
        int val = pre[idx];
        idx++;
        Preorder_To_Postorder_Range(pre, idx, min_val, val, post);
        Preorder_To_Postorder_Range(pre, idx, val, max_val, post);
        post.push_back(val);
    }

    vector<int> Preorder_To_Postorder_Range_Based(vector<int>& pre) {
        /*
        Range-based recursion approach
        Time Complexity: O(n)
        Space Complexity: O(h)
        */
        vector<int> post;
        int idx = 0;
        Preorder_To_Postorder_Range(pre, idx, INT_MIN, INT_MAX, post);
        return post;
    }

    void Postorder_Traversal(TreeNode* root, vector<int>& post) {
        if (root == NULL) return;
        Postorder_Traversal(root->left, post);
        Postorder_Traversal(root->right, post);
        post.push_back(root->data);
    }

    vector<int> Preorder_To_Postorder_Construct_BST(vector<int>& pre) {
        /*
        Construct BST then postorder approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        TreeNode* root = NULL;
        for (int val : pre) {
            root = Insert_BST(root, val);
        }
        vector<int> post;
        Postorder_Traversal(root, post);
        return post;
    }
};

void Test_Preorder_To_Postorder() {
    Solution solution;
    vector<int> pre = {40, 30, 35, 80, 100};
    vector<int> post1 = solution.Preorder_To_Postorder_Range_Based(pre);
    vector<int> post2 = solution.Preorder_To_Postorder_Construct_BST(pre);
    cout << "Postorder (Range): ";
    for (int val : post1) cout << val << " ";
    cout << endl;
    cout << "Postorder (Construct BST): ";
    for (int val : post2) cout << val << " ";
    cout << endl;
}

int main() {
    Test_Preorder_To_Postorder();
    return 0;
}
