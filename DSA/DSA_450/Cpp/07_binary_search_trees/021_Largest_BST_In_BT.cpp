/*
Problem: Largest BST in Binary Tree
URL: https://practice.geeksforgeeks.org/problems/largest-bst/1

Problem Statement:
Find size of largest BST subtree in a binary tree.

Sample Input/Output:
Input: root = [10,5,15,1,8,null,7]
Output: 3
Explanation: Largest BST subtree has size 3 (rooted at node 5).
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
    struct BSTInfo {
        bool is_bst;
        int size;
        int min_val;
        int max_val;
        BSTInfo(bool b, int s, int min_v, int max_v) : is_bst(b), size(s), min_val(min_v), max_val(max_v) {}
    };

    BSTInfo Largest_BST_Helper(TreeNode* root, int& max_size) {
        if (root == NULL) {
            return BSTInfo(true, 0, INT_MAX, INT_MIN);
        }
        BSTInfo left_info = Largest_BST_Helper(root->left, max_size);
        BSTInfo right_info = Largest_BST_Helper(root->right, max_size);
        if (left_info.is_bst && right_info.is_bst &&
            root->data > left_info.max_val && root->data < right_info.min_val) {
            int size = left_info.size + right_info.size + 1;
            max_size = max(max_size, size);
            int min_val = (left_info.size == 0) ? root->data : left_info.min_val;
            int max_val = (right_info.size == 0) ? root->data : right_info.max_val;
            return BSTInfo(true, size, min_val, max_val);
        }
        return BSTInfo(false, 0, 0, 0);
    }

    int Largest_BST_Size(TreeNode* root) {
        /*
        Bottom-up with min/max/size tracking approach
        Time Complexity: O(n)
        Space Complexity: O(h)
        */
        int max_size = 0;
        Largest_BST_Helper(root, max_size);
        return max_size;
    }
};

void Test_Largest_BST_In_BT() {
    Solution solution;
    TreeNode* root = new TreeNode(10);
    root->left = new TreeNode(5);
    root->right = new TreeNode(15);
    root->left->left = new TreeNode(1);
    root->left->right = new TreeNode(8);
    root->right->right = new TreeNode(7);
    cout << "Largest BST Size: " << solution.Largest_BST_Size(root) << endl;
}

int main() {
    Test_Largest_BST_In_BT();
    return 0;
}
