/*
Problem: Construct BST from Given Preorder Traversal
URL: https://www.geeksforgeeks.org/construct-bst-from-given-preorder-traversa/

Problem Statement:
Given preorder traversal of a binary search tree, construct the BST. Preorder traversal is Root, Left, Right.

Sample Input/Output:
Input: [10, 5, 1, 7, 40, 50]
Output: BST with root 10, left subtree [5,1,7], right subtree [40,50]
Explanation: First element is root, then elements less than root form left subtree, greater form right subtree
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
    TreeNode* Construct_BST_Array_Splitting(vector<int>& Preorder, int Start, int End) {
        if (Start > End) {
            return NULL;
        }
        TreeNode* Root = new TreeNode(Preorder[Start]);
        int Right_Start = Start + 1;
        while (Right_Start <= End && Preorder[Right_Start] < Preorder[Start]) {
            Right_Start++;
        }
        Root->left = Construct_BST_Array_Splitting(Preorder, Start + 1, Right_Start - 1);
        Root->right = Construct_BST_Array_Splitting(Preorder, Right_Start, End);
        return Root;
    }

    TreeNode* Build_BST_Array_Splitting(vector<int>& Preorder) {
        /*
        Array splitting approach: find split point for left and right subtrees
        Time Complexity: O(n^2) worst case when tree is skewed
        Space Complexity: O(h) where h is height for recursion stack
        */
        return Construct_BST_Array_Splitting(Preorder, 0, Preorder.size() - 1);
    }

    TreeNode* Construct_BST_Range_Based(vector<int>& Preorder, int& Index, int Min_Value, int Max_Value) {
        if (Index >= Preorder.size()) {
            return NULL;
        }
        int Value = Preorder[Index];
        if (Value < Min_Value || Value > Max_Value) {
            return NULL;
        }
        TreeNode* Root = new TreeNode(Value);
        Index++;
        Root->left = Construct_BST_Range_Based(Preorder, Index, Min_Value, Value - 1);
        Root->right = Construct_BST_Range_Based(Preorder, Index, Value + 1, Max_Value);
        return Root;
    }

    TreeNode* Build_BST_Range_Based(vector<int>& Preorder) {
        /*
        Range-based approach: use min-max range to validate each node
        Time Complexity: O(n) single pass through array
        Space Complexity: O(h) where h is height for recursion stack
        */
        int Index = 0;
        return Construct_BST_Range_Based(Preorder, Index, INT_MIN, INT_MAX);
    }
};

void Test_Construct_BST_From_Preorder() {
    Solution solution;
    vector<int> Preorder = {10, 5, 1, 7, 40, 50};
    
    TreeNode* Root_Array_Split = solution.Build_BST_Array_Splitting(Preorder);
    cout << "BST from Array Splitting (Inorder): ";
    Print_Inorder(Root_Array_Split);
    cout << endl;
    
    TreeNode* Root_Range_Based = solution.Build_BST_Range_Based(Preorder);
    cout << "BST from Range Based (Inorder): ";
    Print_Inorder(Root_Range_Based);
    cout << endl;
}

int main() {
    Test_Construct_BST_From_Preorder();
    return 0;
}
