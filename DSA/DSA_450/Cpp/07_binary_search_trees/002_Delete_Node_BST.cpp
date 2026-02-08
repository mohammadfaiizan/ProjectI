/*
Problem: Delete Node in BST
URL: https://leetcode.com/problems/delete-node-in-a-bst/

Problem Statement:
Given a root node reference of a BST and a key, delete the node with the given key in the BST. Return the root node reference (possibly updated) of the BST. Handle three cases: node is a leaf, node has one child, node has two children (replace with inorder successor).

Sample Input/Output:
Input: Root = [5,3,6,2,4,null,7], key = 3
Output: [5,4,6,2,null,null,7]
Explanation: Node 3 is deleted and replaced with its inorder successor 4
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
    TreeNode* Find_Min(TreeNode* root) {
        while (root->left != NULL) {
            root = root->left;
        }
        return root;
    }

    TreeNode* Delete_Node_Recursive(TreeNode* root, int key) {
        /*
        Recursive deletion handling three cases: leaf, one child, two children
        Time Complexity: O(h) where h is height
        Space Complexity: O(h) for recursion stack
        */
        if (root == NULL) {
            return root;
        }
        if (key < root->data) {
            root->left = Delete_Node_Recursive(root->left, key);
        } else if (key > root->data) {
            root->right = Delete_Node_Recursive(root->right, key);
        } else {
            if (root->left == NULL) {
                TreeNode* Temp = root->right;
                delete root;
                return Temp;
            } else if (root->right == NULL) {
                TreeNode* Temp = root->left;
                delete root;
                return Temp;
            }
            TreeNode* Inorder_Successor = Find_Min(root->right);
            root->data = Inorder_Successor->data;
            root->right = Delete_Node_Recursive(root->right, Inorder_Successor->data);
        }
        return root;
    }
};

void Test_Delete_Node_BST() {
    Solution solution;
    TreeNode* Root = NULL;
    Root = Insert_BST(Root, 50);
    Root = Insert_BST(Root, 30);
    Root = Insert_BST(Root, 70);
    Root = Insert_BST(Root, 20);
    Root = Insert_BST(Root, 40);
    Root = Insert_BST(Root, 60);
    Root = Insert_BST(Root, 80);
    
    cout << "Before deletion: ";
    Print_Inorder(Root);
    cout << endl;
    
    Root = solution.Delete_Node_Recursive(Root, 20);
    cout << "After deleting 20 (leaf): ";
    Print_Inorder(Root);
    cout << endl;
    
    Root = solution.Delete_Node_Recursive(Root, 30);
    cout << "After deleting 30 (one child): ";
    Print_Inorder(Root);
    cout << endl;
    
    Root = solution.Delete_Node_Recursive(Root, 50);
    cout << "After deleting 50 (two children): ";
    Print_Inorder(Root);
    cout << endl;
}

int main() {
    Test_Delete_Node_BST();
    return 0;
}
