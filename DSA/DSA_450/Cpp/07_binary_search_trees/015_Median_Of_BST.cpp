/*
Problem: Find Median of BST
URL: https://www.geeksforgeeks.org/find-median-bst-time-o1-space/

Problem Statement:
Find median of BST in O(n) time and O(1) space.

Sample Input/Output:
Input: root = [6,3,8,1,5,7,9]
Output: 6
Explanation: Inorder: [1,3,5,6,7,8,9], median is 6.
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
    int Count_Nodes(TreeNode* root) {
        int count = 0;
        TreeNode* curr = root;
        while (curr != NULL) {
            if (curr->left == NULL) {
                count++;
                curr = curr->right;
            } else {
                TreeNode* prev = curr->left;
                while (prev->right != NULL && prev->right != curr) {
                    prev = prev->right;
                }
                if (prev->right == NULL) {
                    prev->right = curr;
                    curr = curr->left;
                } else {
                    prev->right = NULL;
                    count++;
                    curr = curr->right;
                }
            }
        }
        return count;
    }

    int Find_Kth_Node_Morris(TreeNode* root, int k) {
        TreeNode* curr = root;
        int count = 0;
        while (curr != NULL) {
            if (curr->left == NULL) {
                count++;
                if (count == k) return curr->data;
                curr = curr->right;
            } else {
                TreeNode* prev = curr->left;
                while (prev->right != NULL && prev->right != curr) {
                    prev = prev->right;
                }
                if (prev->right == NULL) {
                    prev->right = curr;
                    curr = curr->left;
                } else {
                    prev->right = NULL;
                    count++;
                    if (count == k) return curr->data;
                    curr = curr->right;
                }
            }
        }
        return -1;
    }

    double Median_Morris(TreeNode* root) {
        /*
        Morris traversal with node counting approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int n = Count_Nodes(root);
        if (n % 2 == 1) {
            return Find_Kth_Node_Morris(root, (n + 1) / 2);
        } else {
            int first = Find_Kth_Node_Morris(root, n / 2);
            int second = Find_Kth_Node_Morris(root, n / 2 + 1);
            return (first + second) / 2.0;
        }
    }

    void Inorder_To_Array(TreeNode* root, vector<int>& arr) {
        if (root == NULL) return;
        Inorder_To_Array(root->left, arr);
        arr.push_back(root->data);
        Inorder_To_Array(root->right, arr);
    }

    double Median_Array(TreeNode* root) {
        /*
        Inorder to array approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<int> arr;
        Inorder_To_Array(root, arr);
        int n = arr.size();
        if (n % 2 == 1) {
            return arr[n / 2];
        } else {
            return (arr[n / 2 - 1] + arr[n / 2]) / 2.0;
        }
    }
};

void Test_Median_Of_BST() {
    Solution solution;
    TreeNode* root = NULL;
    root = Insert_BST(root, 6);
    root = Insert_BST(root, 3);
    root = Insert_BST(root, 8);
    root = Insert_BST(root, 1);
    root = Insert_BST(root, 5);
    root = Insert_BST(root, 7);
    root = Insert_BST(root, 9);
    cout << "Median (Morris): " << solution.Median_Morris(root) << endl;
    cout << "Median (Array): " << solution.Median_Array(root) << endl;
}

int main() {
    Test_Median_Of_BST();
    return 0;
}
