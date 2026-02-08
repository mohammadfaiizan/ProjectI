/*
Problem: Flatten BST to Sorted List
URL: https://www.geeksforgeeks.org/flatten-bst-to-sorted-list-increasing-order/

Problem Statement:
Flatten BST to sorted linked list using right pointers.

Sample Input/Output:
Input: root = [5,3,7,2,4,6,8]
Output: 2->3->4->5->6->7->8
Explanation: BST flattened to sorted linked list.
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
    void Flatten_BST_Inorder(TreeNode* root, TreeNode*& prev) {
        if (root == NULL) return;
        Flatten_BST_Inorder(root->left, prev);
        if (prev != NULL) {
            prev->right = root;
            prev->left = NULL;
        }
        prev = root;
        Flatten_BST_Inorder(root->right, prev);
    }

    TreeNode* Flatten_BST_Inorder_Approach(TreeNode* root) {
        /*
        Inorder with prev pointer approach
        Time Complexity: O(n)
        Space Complexity: O(h)
        */
        TreeNode* dummy = new TreeNode(0);
        TreeNode* prev = dummy;
        Flatten_BST_Inorder(root, prev);
        prev->left = NULL;
        prev->right = NULL;
        TreeNode* result = dummy->right;
        delete dummy;
        return result;
    }

    TreeNode* Flatten_BST_Morris(TreeNode* root) {
        /*
        Morris traversal approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        TreeNode* curr = root;
        TreeNode* prev = NULL;
        TreeNode* head = NULL;
        while (curr != NULL) {
            if (curr->left == NULL) {
                if (head == NULL) head = curr;
                if (prev != NULL) {
                    prev->right = curr;
                    prev->left = NULL;
                }
                prev = curr;
                curr = curr->right;
            } else {
                TreeNode* predecessor = curr->left;
                while (predecessor->right != NULL && predecessor->right != curr) {
                    predecessor = predecessor->right;
                }
                if (predecessor->right == NULL) {
                    predecessor->right = curr;
                    curr = curr->left;
                } else {
                    predecessor->right = NULL;
                    if (head == NULL) head = curr;
                    if (prev != NULL) {
                        prev->right = curr;
                        prev->left = NULL;
                    }
                    prev = curr;
                    curr = curr->right;
                }
            }
        }
        if (prev != NULL) {
            prev->left = NULL;
            prev->right = NULL;
        }
        return head;
    }
};

void Test_Flatten_BST_Sorted_List() {
    Solution solution;
    TreeNode* root = NULL;
    root = Insert_BST(root, 5);
    root = Insert_BST(root, 3);
    root = Insert_BST(root, 7);
    root = Insert_BST(root, 2);
    root = Insert_BST(root, 4);
    root = Insert_BST(root, 6);
    root = Insert_BST(root, 8);
    TreeNode* flattened1 = solution.Flatten_BST_Inorder_Approach(root);
    TreeNode* temp = flattened1;
    cout << "Flattened (Inorder): ";
    while (temp != NULL) {
        cout << temp->data << " ";
        temp = temp->right;
    }
    cout << endl;
}

int main() {
    Test_Flatten_BST_Sorted_List();
    return 0;
}
