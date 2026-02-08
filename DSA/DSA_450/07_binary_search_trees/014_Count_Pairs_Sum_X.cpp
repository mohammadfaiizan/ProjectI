/*
Problem: Count Pairs from Two BSTs Whose Sum Equals X
URL: https://practice.geeksforgeeks.org/problems/brothers-from-different-root/1

Problem Statement:
Count pairs from two BSTs whose sum equals X.

Sample Input/Output:
Input: root1 = [5,3,7,2,4,6,8], root2 = [10,6,15,3,8,11,18], X = 16
Output: 3
Explanation: Pairs are (5,11), (6,10), (8,8)
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
    bool Search_BST(TreeNode* root, int key) {
        if (root == NULL) return false;
        if (root->data == key) return true;
        if (key < root->data) return Search_BST(root->left, key);
        return Search_BST(root->right, key);
    }

    int Count_Pairs_BST_Search(TreeNode* root1, TreeNode* root2, int X) {
        /*
        Inorder traversal + BST search approach
        Time Complexity: O(n log m)
        Space Complexity: O(h)
        */
        if (root1 == NULL) return 0;
        int count = Count_Pairs_BST_Search(root1->left, root2, X);
        if (Search_BST(root2, X - root1->data)) count++;
        count += Count_Pairs_BST_Search(root1->right, root2, X);
        return count;
    }

    void Inorder_To_Array(TreeNode* root, vector<int>& arr) {
        if (root == NULL) return;
        Inorder_To_Array(root->left, arr);
        arr.push_back(root->data);
        Inorder_To_Array(root->right, arr);
    }

    int Count_Pairs_Two_Pointer(TreeNode* root1, TreeNode* root2, int X) {
        /*
        Inorder both trees + two pointer approach
        Time Complexity: O(m + n)
        Space Complexity: O(m + n)
        */
        vector<int> arr1, arr2;
        Inorder_To_Array(root1, arr1);
        Inorder_To_Array(root2, arr2);
        int i = 0, j = arr2.size() - 1;
        int count = 0;
        while (i < arr1.size() && j >= 0) {
            int sum = arr1[i] + arr2[j];
            if (sum == X) {
                count++;
                i++;
                j--;
            } else if (sum < X) {
                i++;
            } else {
                j--;
            }
        }
        return count;
    }
};

void Test_Count_Pairs_Sum_X() {
    Solution solution;
    TreeNode* root1 = NULL;
    root1 = Insert_BST(root1, 5);
    root1 = Insert_BST(root1, 3);
    root1 = Insert_BST(root1, 7);
    root1 = Insert_BST(root1, 2);
    root1 = Insert_BST(root1, 4);
    root1 = Insert_BST(root1, 6);
    root1 = Insert_BST(root1, 8);
    TreeNode* root2 = NULL;
    root2 = Insert_BST(root2, 10);
    root2 = Insert_BST(root2, 6);
    root2 = Insert_BST(root2, 15);
    root2 = Insert_BST(root2, 3);
    root2 = Insert_BST(root2, 8);
    root2 = Insert_BST(root2, 11);
    root2 = Insert_BST(root2, 18);
    int X = 16;
    cout << "Count Pairs (BST Search): " << solution.Count_Pairs_BST_Search(root1, root2, X) << endl;
    cout << "Count Pairs (Two Pointer): " << solution.Count_Pairs_Two_Pointer(root1, root2, X) << endl;
}

int main() {
    Test_Count_Pairs_Sum_X();
    return 0;
}
