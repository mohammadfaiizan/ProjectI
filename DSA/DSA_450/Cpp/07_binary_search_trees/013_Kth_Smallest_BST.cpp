/*
Problem: Kth Smallest Element in BST
URL: https://practice.geeksforgeeks.org/problems/find-k-th-smallest-element-in-bst/1

Problem Statement:
Find kth smallest element in BST.

Sample Input/Output:
Input: root = [5,3,6,2,4,null,null,1], k = 3
Output: 3
Explanation: The 3rd smallest element is 3.
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
    int Kth_Smallest_Inorder(TreeNode* root, int& k) {
        /*
        Inorder traversal approach
        Time Complexity: O(h + k)
        Space Complexity: O(h)
        */
        if (root == NULL) return -1;
        int left = Kth_Smallest_Inorder(root->left, k);
        if (left != -1) return left;
        k--;
        if (k == 0) return root->data;
        return Kth_Smallest_Inorder(root->right, k);
    }

    int Kth_Smallest_Morris(TreeNode* root, int k) {
        /*
        Morris inorder traversal approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        TreeNode* curr = root;
        int count = 0;
        int result = -1;
        while (curr != NULL) {
            if (curr->left == NULL) {
                count++;
                if (count == k) result = curr->data;
                curr = curr->right;
            } else {
                TreeNode* prev = curr->left;
                while (prev->right != NULL && prev->right != curr) {
                    prev = prev->right;
                }
                if (prev->right == NULL) {
                    prev->right = curr;
                    curr = curr->left;
                } else {
                    prev->right = NULL;
                    count++;
                    if (count == k) result = curr->data;
                    curr = curr->right;
                }
            }
        }
        return result;
    }
};

void Test_Kth_Smallest_BST() {
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
    cout << "Kth Smallest (Inorder): " << solution.Kth_Smallest_Inorder(root, k_val) << endl;
    cout << "Kth Smallest (Morris): " << solution.Kth_Smallest_Morris(root, k) << endl;
}

int main() {
    Test_Kth_Smallest_BST();
    return 0;
}
