/*
Problem: Mirror Of Tree
URL: https://www.geeksforgeeks.org/create-a-mirror-tree-from-the-given-binary-tree/

Problem Statement:
Given a binary tree, create its mirror tree. Mirror of a tree is obtained by swapping left and right children of all nodes.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output (Mirror):
        1
      /   \
     3     2
    / \   / \
   7   6 5   4

Explanation: Left and right children of all nodes are swapped.
*/

#include <bits/stdc++.h>
using namespace std;

struct TreeNode {
    int data;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : data(x), left(NULL), right(NULL) {}
};

TreeNode* Create_Tree(vector<int> vals) {
    if (vals.empty() || vals[0] == -1) return NULL;
    TreeNode* root = new TreeNode(vals[0]);
    queue<TreeNode*> q;
    q.push(root);
    int i = 1;
    while (!q.empty() && i < vals.size()) {
        TreeNode* node = q.front();
        q.pop();
        if (i < vals.size() && vals[i] != -1) {
            node->left = new TreeNode(vals[i]);
            q.push(node->left);
        }
        i++;
        if (i < vals.size() && vals[i] != -1) {
            node->right = new TreeNode(vals[i]);
            q.push(node->right);
        }
        i++;
    }
    return root;
}

void Print_Inorder(TreeNode* root) {
    if (!root) return;
    Print_Inorder(root->left);
    cout << root->data << " ";
    Print_Inorder(root->right);
}

class Solution {
public:
    TreeNode* Mirror_In_Place(TreeNode* root) {
        /*
        In-place recursive swap: Swap left and right children recursively
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        */
        if (!root) return NULL;
        TreeNode* temp = root->left;
        root->left = root->right;
        root->right = temp;
        Mirror_In_Place(root->left);
        Mirror_In_Place(root->right);
        return root;
    }

    TreeNode* Mirror_Separate_Tree(TreeNode* root) {
        /*
        Create separate mirror tree: Build new tree with swapped children
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for new tree
        */
        if (!root) return NULL;
        TreeNode* mirror = new TreeNode(root->data);
        mirror->left = Mirror_Separate_Tree(root->right);
        mirror->right = Mirror_Separate_Tree(root->left);
        return mirror;
    }
};

void Test_Mirror_Of_Tree() {
    Solution solution;
    
    vector<int> vals1 = {1, 2, 3, 4, 5, 6, 7};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Original Inorder: ";
    Print_Inorder(root1);
    cout << endl;
    
    TreeNode* root1_copy = Create_Tree(vals1);
    solution.Mirror_In_Place(root1_copy);
    cout << "Mirror In-Place Inorder: ";
    Print_Inorder(root1_copy);
    cout << endl;
    
    TreeNode* mirror1 = solution.Mirror_Separate_Tree(root1);
    cout << "Mirror Separate Tree Inorder: ";
    Print_Inorder(mirror1);
    cout << endl;
    
    vector<int> vals2 = {1, 2, 3, -1, -1, 4, 5};
    TreeNode* root2 = Create_Tree(vals2);
    cout << "\nOriginal Inorder: ";
    Print_Inorder(root2);
    cout << endl;
    
    TreeNode* root2_copy = Create_Tree(vals2);
    solution.Mirror_In_Place(root2_copy);
    cout << "Mirror In-Place Inorder: ";
    Print_Inorder(root2_copy);
    cout << endl;
}

int main() {
    Test_Mirror_Of_Tree();
    return 0;
}
