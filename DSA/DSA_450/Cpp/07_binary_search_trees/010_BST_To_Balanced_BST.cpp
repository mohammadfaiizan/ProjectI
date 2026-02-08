/*
Problem: Convert Normal BST to Balanced BST
URL: https://www.geeksforgeeks.org/convert-normal-bst-balanced-bst/

Problem Statement:
Given a BST (Binary Search Tree) that may be unbalanced, convert it into a balanced BST that has minimum possible height. Store inorder traversal to get sorted array, then build balanced BST from sorted array.

Sample Input/Output:
Input: Skewed BST: 1->2->3->4->5->6->7 (all right children)
Output: Balanced BST with root 4, left subtree [1,2,3], right subtree [5,6,7]
Explanation: Inorder gives sorted array, then build balanced tree from middle element
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
    void Store_Inorder_To_Array(TreeNode* root, vector<int>& Inorder_Array) {
        if (root == NULL) return;
        Store_Inorder_To_Array(root->left, Inorder_Array);
        Inorder_Array.push_back(root->data);
        Store_Inorder_To_Array(root->right, Inorder_Array);
    }

    TreeNode* Build_Balanced_BST_From_Sorted_Array(vector<int>& Sorted_Array, int Start, int End) {
        if (Start > End) {
            return NULL;
        }
        int Mid = Start + (End - Start) / 2;
        TreeNode* Root = new TreeNode(Sorted_Array[Mid]);
        Root->left = Build_Balanced_BST_From_Sorted_Array(Sorted_Array, Start, Mid - 1);
        Root->right = Build_Balanced_BST_From_Sorted_Array(Sorted_Array, Mid + 1, End);
        return Root;
    }

    TreeNode* Convert_To_Balanced_BST(TreeNode* root) {
        /*
        Approach: Inorder to sorted array + build balanced BST from sorted array
        Time Complexity: O(n) for traversal and building
        Space Complexity: O(n) for storing inorder array
        */
        vector<int> Inorder_Array;
        Store_Inorder_To_Array(root, Inorder_Array);
        return Build_Balanced_BST_From_Sorted_Array(Inorder_Array, 0, Inorder_Array.size() - 1);
    }
};

void Test_BST_To_Balanced_BST() {
    Solution solution;
    TreeNode* Skewed_BST = NULL;
    Skewed_BST = Insert_BST(Skewed_BST, 1);
    Skewed_BST = Insert_BST(Skewed_BST, 2);
    Skewed_BST = Insert_BST(Skewed_BST, 3);
    Skewed_BST = Insert_BST(Skewed_BST, 4);
    Skewed_BST = Insert_BST(Skewed_BST, 5);
    Skewed_BST = Insert_BST(Skewed_BST, 6);
    Skewed_BST = Insert_BST(Skewed_BST, 7);
    
    cout << "Skewed BST Inorder: ";
    Print_Inorder(Skewed_BST);
    cout << endl;
    
    TreeNode* Balanced_BST = solution.Convert_To_Balanced_BST(Skewed_BST);
    
    cout << "Balanced BST Inorder: ";
    Print_Inorder(Balanced_BST);
    cout << endl;
}

int main() {
    Test_BST_To_Balanced_BST();
    return 0;
}
