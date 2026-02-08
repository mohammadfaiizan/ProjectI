/*
Problem: Populate Inorder Successor for All Nodes
URL: https://practice.geeksforgeeks.org/problems/populate-inorder-successor-for-all-nodes/1

Problem Statement:
Given a Binary Tree, write a function to populate next pointer for all nodes. The next pointer for every node should be set to point to inorder successor. Use reverse inorder traversal approach.

Sample Input/Output:
Input: Root with data 10, left 8, right 12
Output: Node 8's next points to 10, Node 10's next points to 12, Node 12's next is NULL
Explanation: Inorder is 8->10->12, so next pointers follow this order
*/

#include <bits/stdc++.h>
using namespace std;

struct TreeNode_With_Next {
    int data;
    TreeNode_With_Next* left;
    TreeNode_With_Next* right;
    TreeNode_With_Next* next;
    TreeNode_With_Next(int x) : data(x), left(NULL), right(NULL), next(NULL) {}
};

class Solution {
public:
    void Populate_Inorder_Successor_Reverse_Inorder(TreeNode_With_Next* root, TreeNode_With_Next*& Next_Node) {
        /*
        Reverse inorder traversal: process right, root, left
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height for recursion stack
        */
        if (root == NULL) {
            return;
        }
        Populate_Inorder_Successor_Reverse_Inorder(root->right, Next_Node);
        root->next = Next_Node;
        Next_Node = root;
        Populate_Inorder_Successor_Reverse_Inorder(root->left, Next_Node);
    }

    void Populate_Next_Pointers(TreeNode_With_Next* root) {
        TreeNode_With_Next* Next_Node = NULL;
        Populate_Inorder_Successor_Reverse_Inorder(root, Next_Node);
    }
};

void Print_Inorder_With_Next(TreeNode_With_Next* root) {
    if (root == NULL) return;
    Print_Inorder_With_Next(root->left);
    cout << root->data << " -> " << (root->next ? root->next->data : -1) << endl;
    Print_Inorder_With_Next(root->right);
}

void Test_Populate_Inorder_Successor() {
    Solution solution;
    TreeNode_With_Next* Root = new TreeNode_With_Next(10);
    Root->left = new TreeNode_With_Next(8);
    Root->right = new TreeNode_With_Next(12);
    Root->left->left = new TreeNode_With_Next(3);
    
    solution.Populate_Next_Pointers(Root);
    
    cout << "Inorder traversal with next pointers:" << endl;
    Print_Inorder_With_Next(Root);
}

int main() {
    Test_Populate_Inorder_Successor();
    return 0;
}
