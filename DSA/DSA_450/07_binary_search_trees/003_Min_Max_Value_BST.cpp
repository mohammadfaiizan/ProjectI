/*
Problem: Find Minimum and Maximum Values in BST
URL: https://practice.geeksforgeeks.org/problems/minimum-element-in-bst/1

Problem Statement:
Given a Binary Search Tree, find the minimum and maximum values in the BST.

Sample Input/Output:
Input: BST with root 5, left 3, right 7, left of 3 is 2, right of 3 is 4
Output: Min = 2, Max = 7
Explanation: Leftmost node has minimum value, rightmost node has maximum value
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
    int Min_Value_Iterative(TreeNode* root) {
        /*
        Iterative approach: traverse to leftmost node
        Time Complexity: O(h) where h is height
        Space Complexity: O(1) constant space
        */
        if (root == NULL) {
            return -1;
        }
        while (root->left != NULL) {
            root = root->left;
        }
        return root->data;
    }

    int Max_Value_Iterative(TreeNode* root) {
        /*
        Iterative approach: traverse to rightmost node
        Time Complexity: O(h) where h is height
        Space Complexity: O(1) constant space
        */
        if (root == NULL) {
            return -1;
        }
        while (root->right != NULL) {
            root = root->right;
        }
        return root->data;
    }

    int Min_Value_Recursive(TreeNode* root) {
        /*
        Recursive approach: base case and recurse left
        Time Complexity: O(h) where h is height
        Space Complexity: O(h) for recursion stack
        */
        if (root == NULL) {
            return -1;
        }
        if (root->left == NULL) {
            return root->data;
        }
        return Min_Value_Recursive(root->left);
    }

    int Max_Value_Recursive(TreeNode* root) {
        /*
        Recursive approach: base case and recurse right
        Time Complexity: O(h) where h is height
        Space Complexity: O(h) for recursion stack
        */
        if (root == NULL) {
            return -1;
        }
        if (root->right == NULL) {
            return root->data;
        }
        return Max_Value_Recursive(root->right);
    }
};

void Test_Min_Max_Value_BST() {
    Solution solution;
    TreeNode* Root = NULL;
    Root = Insert_BST(Root, 50);
    Root = Insert_BST(Root, 30);
    Root = Insert_BST(Root, 70);
    Root = Insert_BST(Root, 20);
    Root = Insert_BST(Root, 40);
    Root = Insert_BST(Root, 60);
    Root = Insert_BST(Root, 80);
    
    cout << "BST Inorder: ";
    Print_Inorder(Root);
    cout << endl;
    
    cout << "Min (Iterative): " << solution.Min_Value_Iterative(Root) << endl;
    cout << "Max (Iterative): " << solution.Max_Value_Iterative(Root) << endl;
    cout << "Min (Recursive): " << solution.Min_Value_Recursive(Root) << endl;
    cout << "Max (Recursive): " << solution.Max_Value_Recursive(Root) << endl;
}

int main() {
    Test_Min_Max_Value_BST();
    return 0;
}
