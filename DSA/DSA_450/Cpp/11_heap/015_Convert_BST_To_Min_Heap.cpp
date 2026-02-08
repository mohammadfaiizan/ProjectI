/*
Problem: Convert BST to Min Heap
URL: https://www.geeksforgeeks.org/convert-bst-min-heap/

Problem Statement:
Given a BST (with property that each node has either 0 or 2 children), convert it to a Min Heap such that all values in left subtree < all values in right subtree.

Sample Input/Output:
Input: BST structure
Output: Min Heap structure
*/

#include <bits/stdc++.h>
using namespace std;

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
private:
    vector<int> inorder;
    int index;
    
public:
    TreeNode* Convert_BST_Heap_Inorder_Preorder(TreeNode* root) {
        /*
        Inorder-Preorder Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        inorder.clear();
        index = 0;
        InorderTraversal(root);
        PreorderFill(root);
        return root;
    }
    
private:
    void InorderTraversal(TreeNode* root) {
        if (!root) return;
        InorderTraversal(root->left);
        inorder.push_back(root->val);
        InorderTraversal(root->right);
    }
    
    void PreorderFill(TreeNode* root) {
        if (!root) return;
        root->val = inorder[index++];
        PreorderFill(root->left);
        PreorderFill(root->right);
    }
};

void PrintLevelOrder(TreeNode* root) {
    if (!root) return;
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        TreeNode* node = q.front();
        q.pop();
        cout << node->val << " ";
        if (node->left) q.push(node->left);
        if (node->right) q.push(node->right);
    }
    cout << endl;
}

void Test_Convert_BST_Heap() {
    Solution solution;
    
    TreeNode* root1 = new TreeNode(4);
    root1->left = new TreeNode(2);
    root1->right = new TreeNode(6);
    root1->left->left = new TreeNode(1);
    root1->left->right = new TreeNode(3);
    root1->right->left = new TreeNode(5);
    root1->right->right = new TreeNode(7);
    
    cout << "Original BST (Level Order): ";
    PrintLevelOrder(root1);
    
    TreeNode* result1 = solution.Convert_BST_Heap_Inorder_Preorder(root1);
    cout << "Converted Min Heap (Level Order): ";
    PrintLevelOrder(result1);
    
    TreeNode* root2 = new TreeNode(8);
    root2->left = new TreeNode(4);
    root2->right = new TreeNode(12);
    root2->left->left = new TreeNode(2);
    root2->left->right = new TreeNode(6);
    root2->right->left = new TreeNode(10);
    root2->right->right = new TreeNode(14);
    
    cout << "Original BST 2 (Level Order): ";
    PrintLevelOrder(root2);
    
    TreeNode* result2 = solution.Convert_BST_Heap_Inorder_Preorder(root2);
    cout << "Converted Min Heap 2 (Level Order): ";
    PrintLevelOrder(result2);
}

int main() {
    Test_Convert_BST_Heap();
    return 0;
}
