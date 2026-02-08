/*
Problem: Merge Two Balanced Binary Search Trees
URL: https://www.geeksforgeeks.org/merge-two-balanced-binary-search-trees/

Problem Statement:
Given two Balanced Binary Search Trees (BSTs), merge them into a single balanced BST. Get inorder traversal of both BSTs, merge the two sorted arrays, then construct balanced BST from merged sorted array.

Sample Input/Output:
Input: BST1: [1,2,3], BST2: [4,5,6]
Output: Balanced BST: root 3, left subtree [1,2], right subtree [4,5,6]
Explanation: Merged sorted array [1,2,3,4,5,6], then build balanced BST from middle
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

    vector<int> Merge_Sorted_Arrays(vector<int>& Array1, vector<int>& Array2) {
        vector<int> Merged_Array;
        int i = 0, j = 0;
        while (i < Array1.size() && j < Array2.size()) {
            if (Array1[i] < Array2[j]) {
                Merged_Array.push_back(Array1[i++]);
            } else {
                Merged_Array.push_back(Array2[j++]);
            }
        }
        while (i < Array1.size()) {
            Merged_Array.push_back(Array1[i++]);
        }
        while (j < Array2.size()) {
            Merged_Array.push_back(Array2[j++]);
        }
        return Merged_Array;
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

    TreeNode* Merge_Two_BST(TreeNode* Root1, TreeNode* Root2) {
        /*
        Approach: Inorder + merge sorted arrays + sorted array to BST
        Time Complexity: O(m+n) where m and n are sizes of two BSTs
        Space Complexity: O(m+n) for storing inorder lists and merged array
        */
        vector<int> Inorder_List1;
        vector<int> Inorder_List2;
        Store_Inorder(Root1, Inorder_List1);
        Store_Inorder(Root2, Inorder_List2);
        vector<int> Merged_Array = Merge_Sorted_Arrays(Inorder_List1, Inorder_List2);
        return Build_Balanced_BST_From_Sorted_Array(Merged_Array, 0, Merged_Array.size() - 1);
    }
};

void Test_Merge_Two_BST() {
    Solution solution;
    TreeNode* BST1 = NULL;
    BST1 = Insert_BST(BST1, 1);
    BST1 = Insert_BST(BST1, 2);
    BST1 = Insert_BST(BST1, 3);
    
    TreeNode* BST2 = NULL;
    BST2 = Insert_BST(BST2, 4);
    BST2 = Insert_BST(BST2, 5);
    BST2 = Insert_BST(BST2, 6);
    
    cout << "BST1 Inorder: ";
    Print_Inorder(BST1);
    cout << endl;
    
    cout << "BST2 Inorder: ";
    Print_Inorder(BST2);
    cout << endl;
    
    TreeNode* Merged_BST = solution.Merge_Two_BST(BST1, BST2);
    
    cout << "Merged Balanced BST Inorder: ";
    Print_Inorder(Merged_BST);
    cout << endl;
}

int main() {
    Test_Merge_Two_BST();
    return 0;
}
