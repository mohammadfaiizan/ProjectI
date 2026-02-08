/*
Problem: Find Inorder Predecessor and Successor
URL: https://practice.geeksforgeeks.org/problems/predecessor-and-successor/1

Problem Statement:
Given a BST and a key, find the inorder predecessor and successor of the given key in the BST. If the key does not exist in BST, return the two values between which this key will lie.

Sample Input/Output:
Input: BST with root 50, left 30, right 70. Key = 65
Output: Predecessor = 50, Successor = 70
Explanation: 65 is not present, so predecessor is largest value less than 65, successor is smallest value greater than 65
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
    void Find_Predecessor_Successor_BST_Property(TreeNode* root, int key, TreeNode*& Predecessor, TreeNode*& Successor) {
        /*
        Using BST property to find predecessor and successor
        Time Complexity: O(h) where h is height
        Space Complexity: O(1) constant space
        */
        Predecessor = NULL;
        Successor = NULL;
        while (root != NULL) {
            if (root->data == key) {
                if (root->left != NULL) {
                    TreeNode* Temp = root->left;
                    while (Temp->right != NULL) {
                        Temp = Temp->right;
                    }
                    Predecessor = Temp;
                }
                if (root->right != NULL) {
                    TreeNode* Temp = root->right;
                    while (Temp->left != NULL) {
                        Temp = Temp->left;
                    }
                    Successor = Temp;
                }
                return;
            } else if (root->data > key) {
                Successor = root;
                root = root->left;
            } else {
                Predecessor = root;
                root = root->right;
            }
        }
    }

    void Inorder_Traversal(TreeNode* root, vector<int>& Inorder_List) {
        if (root == NULL) return;
        Inorder_Traversal(root->left, Inorder_List);
        Inorder_List.push_back(root->data);
        Inorder_Traversal(root->right, Inorder_List);
    }

    pair<int, int> Find_Predecessor_Successor_Inorder(TreeNode* root, int key) {
        /*
        Using inorder traversal to get sorted list, then find predecessor and successor
        Time Complexity: O(n) for traversal
        Space Complexity: O(n) for storing inorder list
        */
        vector<int> Inorder_List;
        Inorder_Traversal(root, Inorder_List);
        int Predecessor = -1;
        int Successor = -1;
        for (int i = 0; i < Inorder_List.size(); i++) {
            if (Inorder_List[i] == key) {
                if (i > 0) Predecessor = Inorder_List[i - 1];
                if (i < Inorder_List.size() - 1) Successor = Inorder_List[i + 1];
                break;
            } else if (Inorder_List[i] < key) {
                Predecessor = Inorder_List[i];
            } else if (Inorder_List[i] > key && Successor == -1) {
                Successor = Inorder_List[i];
                break;
            }
        }
        return {Predecessor, Successor};
    }
};

void Test_Inorder_Successor_Predecessor() {
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
    
    TreeNode* Predecessor = NULL;
    TreeNode* Successor = NULL;
    solution.Find_Predecessor_Successor_BST_Property(Root, 65, Predecessor, Successor);
    cout << "Key 65 - Predecessor: " << (Predecessor ? Predecessor->data : -1) << ", Successor: " << (Successor ? Successor->data : -1) << endl;
    
    pair<int, int> Result = solution.Find_Predecessor_Successor_Inorder(Root, 40);
    cout << "Key 40 - Predecessor: " << Result.first << ", Successor: " << Result.second << endl;
}

int main() {
    Test_Inorder_Successor_Predecessor();
    return 0;
}
