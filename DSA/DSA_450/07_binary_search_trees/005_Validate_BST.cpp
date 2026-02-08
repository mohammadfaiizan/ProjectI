/*
Problem: Check if Binary Tree is BST
URL: https://practice.geeksforgeeks.org/problems/check-for-bst/1

Problem Statement:
Given a binary tree, check whether it is a valid Binary Search Tree (BST). A valid BST is defined as follows: The left subtree of a node contains only nodes with keys less than the node's key. The right subtree of a node contains only nodes with keys greater than the node's key. Both the left and right subtrees must also be binary search trees.

Sample Input/Output:
Input: Root = [2,1,3]
Output: true
Explanation: All nodes satisfy BST property
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
    bool Is_Valid_BST_Min_Max_Range(TreeNode* root, long long Min_Value, long long Max_Value) {
        if (root == NULL) {
            return true;
        }
        if (root->data <= Min_Value || root->data >= Max_Value) {
            return false;
        }
        return Is_Valid_BST_Min_Max_Range(root->left, Min_Value, root->data) &&
               Is_Valid_BST_Min_Max_Range(root->right, root->data, Max_Value);
    }

    bool Validate_BST_Min_Max_Range(TreeNode* root) {
        /*
        Using min-max range approach: each node must be in valid range
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height for recursion stack
        */
        return Is_Valid_BST_Min_Max_Range(root, LLONG_MIN, LLONG_MAX);
    }

    void Inorder_Traversal_Check(TreeNode* root, vector<int>& Inorder_List) {
        if (root == NULL) return;
        Inorder_Traversal_Check(root->left, Inorder_List);
        Inorder_List.push_back(root->data);
        Inorder_Traversal_Check(root->right, Inorder_List);
    }

    bool Validate_BST_Inorder_Traversal(TreeNode* root) {
        /*
        Using inorder traversal: BST inorder should be strictly increasing
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) for recursion stack + O(n) for storing inorder
        */
        vector<int> Inorder_List;
        Inorder_Traversal_Check(root, Inorder_List);
        for (int i = 1; i < Inorder_List.size(); i++) {
            if (Inorder_List[i] <= Inorder_List[i - 1]) {
                return false;
            }
        }
        return true;
    }
};

void Test_Validate_BST() {
    Solution solution;
    TreeNode* Valid_BST = NULL;
    Valid_BST = Insert_BST(Valid_BST, 50);
    Valid_BST = Insert_BST(Valid_BST, 30);
    Valid_BST = Insert_BST(Valid_BST, 70);
    Valid_BST = Insert_BST(Valid_BST, 20);
    Valid_BST = Insert_BST(Valid_BST, 40);
    
    cout << "Valid BST check (Min-Max): " << solution.Validate_BST_Min_Max_Range(Valid_BST) << endl;
    cout << "Valid BST check (Inorder): " << solution.Validate_BST_Inorder_Traversal(Valid_BST) << endl;
    
    TreeNode* Invalid_BST = new TreeNode(10);
    Invalid_BST->left = new TreeNode(5);
    Invalid_BST->right = new TreeNode(15);
    Invalid_BST->right->left = new TreeNode(6);
    Invalid_BST->right->right = new TreeNode(20);
    
    cout << "Invalid BST check (Min-Max): " << solution.Validate_BST_Min_Max_Range(Invalid_BST) << endl;
    cout << "Invalid BST check (Inorder): " << solution.Validate_BST_Inorder_Traversal(Invalid_BST) << endl;
}

int main() {
    Test_Validate_BST();
    return 0;
}
