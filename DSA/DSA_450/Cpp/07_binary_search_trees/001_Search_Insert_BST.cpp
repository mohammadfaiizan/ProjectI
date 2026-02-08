/*
Problem: Search and Insert a Node in BST
URL: https://practice.geeksforgeeks.org/problems/insert-a-node-in-a-bst/1

Problem Statement:
Given a BST and a key K. If K is not present in the BST, Insert a new Node with a value equal to K into the BST. If K is already present in the BST, don't modify the BST.

Sample Input/Output:
Input: BST with root 2, left 1, right 3. Key = 4
Output: BST with 4 inserted as right child of 3
Explanation: 4 is not present, so insert it maintaining BST property
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
    TreeNode* Search_Recursive(TreeNode* root, int key) {
        /*
        Recursive search using BST property
        Time Complexity: O(h) where h is height
        Space Complexity: O(h) for recursion stack
        */
        if (root == NULL || root->data == key) {
            return root;
        }
        if (key < root->data) {
            return Search_Recursive(root->left, key);
        }
        return Search_Recursive(root->right, key);
    }

    TreeNode* Insert_Recursive(TreeNode* root, int key) {
        /*
        Recursive insert maintaining BST property
        Time Complexity: O(h) where h is height
        Space Complexity: O(h) for recursion stack
        */
        if (root == NULL) {
            return new TreeNode(key);
        }
        if (key < root->data) {
            root->left = Insert_Recursive(root->left, key);
        } else if (key > root->data) {
            root->right = Insert_Recursive(root->right, key);
        }
        return root;
    }

    TreeNode* Insert_Iterative(TreeNode* root, int key) {
        /*
        Iterative insert using while loop
        Time Complexity: O(h) where h is height
        Space Complexity: O(1) constant space
        */
        TreeNode* New_Node = new TreeNode(key);
        if (root == NULL) {
            return New_Node;
        }
        TreeNode* Current = root;
        TreeNode* Parent = NULL;
        while (Current != NULL) {
            Parent = Current;
            if (key < Current->data) {
                Current = Current->left;
            } else if (key > Current->data) {
                Current = Current->right;
            } else {
                return root;
            }
        }
        if (key < Parent->data) {
            Parent->left = New_Node;
        } else {
            Parent->right = New_Node;
        }
        return root;
    }
};

void Test_Search_Insert_BST() {
    Solution solution;
    TreeNode* Root = NULL;
    Root = Insert_BST(Root, 50);
    Root = Insert_BST(Root, 30);
    Root = Insert_BST(Root, 70);
    Root = Insert_BST(Root, 20);
    Root = Insert_BST(Root, 40);
    Root = Insert_BST(Root, 60);
    Root = Insert_BST(Root, 80);
    
    TreeNode* Found = solution.Search_Recursive(Root, 40);
    cout << "Search 40: " << (Found ? "Found" : "Not Found") << endl;
    
    Root = solution.Insert_Recursive(Root, 35);
    Root = solution.Insert_Iterative(Root, 90);
    
    cout << "Inorder after inserts: ";
    Print_Inorder(Root);
    cout << endl;
}

int main() {
    Test_Search_Insert_BST();
    return 0;
}
