/*
Problem: Count BST Nodes That Lie in a Given Range
URL: https://practice.geeksforgeeks.org/problems/count-bst-nodes-that-lie-in-a-given-range/1

Problem Statement:
Count BST nodes that lie in a given range [low, high].

Sample Input/Output:
Input: root = [10,5,50,1,null,40,100], low = 5, high = 45
Output: 3
Explanation: Nodes in range are 5, 10, 40.
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
    int Count_Nodes_Pruned(TreeNode* root, int low, int high) {
        /*
        Pruned traversal approach
        Time Complexity: O(h + k) where k is nodes in range
        Space Complexity: O(h)
        */
        if (root == NULL) return 0;
        if (root->data >= low && root->data <= high) {
            return 1 + Count_Nodes_Pruned(root->left, low, high) + 
                   Count_Nodes_Pruned(root->right, low, high);
        } else if (root->data < low) {
            return Count_Nodes_Pruned(root->right, low, high);
        } else {
            return Count_Nodes_Pruned(root->left, low, high);
        }
    }

    int Count_Nodes_Full(TreeNode* root, int low, int high) {
        /*
        Full traversal approach
        Time Complexity: O(n)
        Space Complexity: O(h)
        */
        if (root == NULL) return 0;
        int count = 0;
        if (root->data >= low && root->data <= high) count++;
        count += Count_Nodes_Full(root->left, low, high);
        count += Count_Nodes_Full(root->right, low, high);
        return count;
    }
};

void Test_Count_Nodes_In_Range() {
    Solution solution;
    TreeNode* root = NULL;
    root = Insert_BST(root, 10);
    root = Insert_BST(root, 5);
    root = Insert_BST(root, 50);
    root = Insert_BST(root, 1);
    root = Insert_BST(root, 40);
    root = Insert_BST(root, 100);
    int low = 5, high = 45;
    cout << "Count Nodes (Pruned): " << solution.Count_Nodes_Pruned(root, low, high) << endl;
    cout << "Count Nodes (Full): " << solution.Count_Nodes_Full(root, low, high) << endl;
}

int main() {
    Test_Count_Nodes_In_Range();
    return 0;
}
