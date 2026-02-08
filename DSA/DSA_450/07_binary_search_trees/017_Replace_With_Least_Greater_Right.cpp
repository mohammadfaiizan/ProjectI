/*
Problem: Replace Every Element with the Least Greater Element on Its Right
URL: https://www.geeksforgeeks.org/replace-every-element-with-the-least-greater-element-on-its-right/

Problem Statement:
Replace every element with the least greater element on its right side. If none, use -1.

Sample Input/Output:
Input: [8, 58, 71, 18, 31, 32, 63, 92, 43, 3, 91, 93, 25, 80, 28]
Output: [18, 63, 80, 25, 32, 43, 80, 93, 80, 25, 93, -1, 28, -1, -1]
Explanation: For 8, least greater on right is 18.
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

TreeNode* Insert_BST_With_Successor(TreeNode* root, int key, TreeNode*& successor) {
    if (root == NULL) return new TreeNode(key);
    if (key < root->data) {
        successor = root;
        root->left = Insert_BST_With_Successor(root->left, key, successor);
    } else {
        root->right = Insert_BST_With_Successor(root->right, key, successor);
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
    vector<int> Replace_Least_Greater_BST(vector<int>& arr) {
        /*
        BST insertion from right with successor tracking approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        int n = arr.size();
        vector<int> result(n, -1);
        TreeNode* root = NULL;
        for (int i = n - 1; i >= 0; i--) {
            TreeNode* successor = NULL;
            root = Insert_BST_With_Successor(root, arr[i], successor);
            if (successor != NULL) {
                result[i] = successor->data;
            }
        }
        return result;
    }

    vector<int> Replace_Least_Greater_Brute(vector<int>& arr) {
        /*
        Brute force approach
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        int n = arr.size();
        vector<int> result(n, -1);
        for (int i = 0; i < n; i++) {
            int min_greater = INT_MAX;
            for (int j = i + 1; j < n; j++) {
                if (arr[j] > arr[i] && arr[j] < min_greater) {
                    min_greater = arr[j];
                }
            }
            if (min_greater != INT_MAX) {
                result[i] = min_greater;
            }
        }
        return result;
    }
};

void Test_Replace_With_Least_Greater_Right() {
    Solution solution;
    vector<int> arr = {8, 58, 71, 18, 31, 32, 63, 92, 43, 3, 91, 93, 25, 80, 28};
    vector<int> result1 = solution.Replace_Least_Greater_BST(arr);
    vector<int> result2 = solution.Replace_Least_Greater_Brute(arr);
    cout << "Replace (BST): ";
    for (int val : result1) cout << val << " ";
    cout << endl;
    cout << "Replace (Brute): ";
    for (int val : result2) cout << val << " ";
    cout << endl;
}

int main() {
    Test_Replace_With_Least_Greater_Right();
    return 0;
}
