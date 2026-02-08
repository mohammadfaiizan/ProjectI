/*
Problem: Convert Binary Tree to BST
URL: https://practice.geeksforgeeks.org/problems/binary-tree-to-bst/1

Problem Statement:
Given a Binary Tree, convert it to Binary Search Tree in such a way that keeps the original structure of Binary Tree intact. Store inorder traversal, sort it, then assign back using inorder traversal.

Sample Input/Output:
Input: BT with root 10, left 2, right 7, left of 2 is 8, right of 2 is 4
Output: BST with same structure but values rearranged: root 8, left 4, right 10, left of 4 is 2, right of 4 is 7
Explanation: Inorder of BT: [8,2,4,10,7], sorted: [2,4,7,8,10], reassigned maintaining structure
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
    void Store_Inorder(TreeNode* root, vector<int>& Inorder_List) {
        if (root == NULL) return;
        Store_Inorder(root->left, Inorder_List);
        Inorder_List.push_back(root->data);
        Store_Inorder(root->right, Inorder_List);
    }

    void Assign_Inorder(TreeNode* root, vector<int>& Inorder_List, int& Index) {
        if (root == NULL) return;
        Assign_Inorder(root->left, Inorder_List, Index);
        root->data = Inorder_List[Index++];
        Assign_Inorder(root->right, Inorder_List, Index);
    }

    TreeNode* Convert_BT_To_BST(TreeNode* root) {
        /*
        Approach: Store inorder, sort, assign back maintaining structure
        Time Complexity: O(n log n) for sorting
        Space Complexity: O(n) for storing inorder list
        */
        if (root == NULL) {
            return root;
        }
        vector<int> Inorder_List;
        Store_Inorder(root, Inorder_List);
        sort(Inorder_List.begin(), Inorder_List.end());
        int Index = 0;
        Assign_Inorder(root, Inorder_List, Index);
        return root;
    }
};

void Test_Convert_BT_To_BST() {
    Solution solution;
    TreeNode* Root = new TreeNode(10);
    Root->left = new TreeNode(2);
    Root->right = new TreeNode(7);
    Root->left->left = new TreeNode(8);
    Root->left->right = new TreeNode(4);
    
    cout << "Binary Tree Inorder (before conversion): ";
    Print_Inorder(Root);
    cout << endl;
    
    Root = solution.Convert_BT_To_BST(Root);
    
    cout << "BST Inorder (after conversion): ";
    Print_Inorder(Root);
    cout << endl;
}

int main() {
    Test_Convert_BT_To_BST();
    return 0;
}
