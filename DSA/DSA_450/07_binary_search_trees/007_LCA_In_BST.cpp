/*
Problem: Lowest Common Ancestor in BST
URL: https://practice.geeksforgeeks.org/problems/lowest-common-ancestor-in-a-bst/1

Problem Statement:
Given a Binary Search Tree (with all values unique) and two node values. Find the Lowest Common Ancestors (LCA) of the two nodes in the BST. LCA is the node which is the ancestor of both nodes.

Sample Input/Output:
Input: BST with root 20, left 8, right 22, left of 8 is 4, right of 8 is 12. Nodes: 4 and 12
Output: 8
Explanation: LCA of 4 and 12 is 8, which is their common ancestor
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
    if (root == NULL) {
        return new TreeNode(key);
    }
    if (key < root->data) {
        root->left = Insert_BST(root->left, key);
    } else if (key > root->data) {
        root->right = Insert_BST(root->right, key);
    }
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
    TreeNode* LCA_Iterative(TreeNode* root, int Node1, int Node2) {
        /*
        Iterative approach using BST property
        Time Complexity: O(h) where h is height
        Space Complexity: O(1) constant space
        */
        while (root != NULL) {
            if (root->data > Node1 && root->data > Node2) {
                root = root->left;
            } else if (root->data < Node1 && root->data < Node2) {
                root = root->right;
            } else {
                break;
            }
        }
        return root;
    }

    TreeNode* LCA_Recursive(TreeNode* root, int Node1, int Node2) {
        /*
        Recursive approach using BST property
        Time Complexity: O(h) where h is height
        Space Complexity: O(h) for recursion stack
        */
        if (root == NULL) {
            return NULL;
        }
        if (root->data > Node1 && root->data > Node2) {
            return LCA_Recursive(root->left, Node1, Node2);
        }
        if (root->data < Node1 && root->data < Node2) {
            return LCA_Recursive(root->right, Node1, Node2);
        }
        return root;
    }
};

void Test_LCA_In_BST() {
    Solution solution;
    TreeNode* Root = NULL;
    Root = Insert_BST(Root, 20);
    Root = Insert_BST(Root, 8);
    Root = Insert_BST(Root, 22);
    Root = Insert_BST(Root, 4);
    Root = Insert_BST(Root, 12);
    Root = Insert_BST(Root, 10);
    Root = Insert_BST(Root, 14);
    
    cout << "BST Inorder: ";
    Print_Inorder(Root);
    cout << endl;
    
    TreeNode* LCA_Iter = solution.LCA_Iterative(Root, 4, 12);
    cout << "LCA of 4 and 12 (Iterative): " << (LCA_Iter ? LCA_Iter->data : -1) << endl;
    
    TreeNode* LCA_Rec = solution.LCA_Recursive(Root, 10, 14);
    cout << "LCA of 10 and 14 (Recursive): " << (LCA_Rec ? LCA_Rec->data : -1) << endl;
    
    TreeNode* LCA_Mixed = solution.LCA_Iterative(Root, 4, 22);
    cout << "LCA of 4 and 22 (Iterative): " << (LCA_Mixed ? LCA_Mixed->data : -1) << endl;
}

int main() {
    Test_LCA_In_BST();
    return 0;
}
