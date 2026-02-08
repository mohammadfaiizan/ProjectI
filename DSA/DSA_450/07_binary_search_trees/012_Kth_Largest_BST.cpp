/*
Problem: Kth Largest Element in BST
URL: https://practice.geeksforgeeks.org/problems/kth-largest-element-in-bst/1

Problem Statement:
Find kth largest element in BST.

Sample Input/Output:
Input: root = [5,3,6,2,4,null,null,1], k = 3
Output: 4
Explanation: The 3rd largest element is 4.
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
    int Kth_Largest_Reverse_Inorder(TreeNode* root, int& k) {
        /*
        Reverse inorder traversal approach
        Time Complexity: O(h + k)
        Space Complexity: O(h)
        */
        if (root == NULL) return -1;
        int right = Kth_Largest_Reverse_Inorder(root->right, k);
        if (right != -1) return right;
        k--;
        if (k == 0) return root->data;
        return Kth_Largest_Reverse_Inorder(root->left, k);
    }

    int Kth_Largest_Morris(TreeNode* root, int k) {
        /*
        Morris reverse inorder traversal approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        TreeNode* curr = root;
        int count = 0;
        int result = -1;
        while (curr != NULL) {
            if (curr->right == NULL) {
                count++;
                if (count == k) result = curr->data;
                curr = curr->left;
            } else {
                TreeNode* prev = curr->right;
                while (prev->left != NULL && prev->left != curr) {
                    prev = prev->left;
                }
                if (prev->left == NULL) {
                    prev->left = curr;
                    curr = curr->right;
                } else {
                    prev->left = NULL;
                    count++;
                    if (count == k) result = curr->data;
                    curr = curr->left;
                }
            }
        }
        return result;
    }
};

void Test_Kth_Largest_BST() {
    Solution solution;
    TreeNode* root = NULL;
    root = Insert_BST(root, 5);
    root = Insert_BST(root, 3);
    root = Insert_BST(root, 6);
    root = Insert_BST(root, 2);
    root = Insert_BST(root, 4);
    root = Insert_BST(root, 1);
    int k = 3;
    int k_val = k;
    cout << "Kth Largest (Reverse Inorder): " << solution.Kth_Largest_Reverse_Inorder(root, k_val) << endl;
    cout << "Kth Largest (Morris): " << solution.Kth_Largest_Morris(root, k) << endl;
}

int main() {
    Test_Kth_Largest_BST();
    return 0;
}
