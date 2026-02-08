/*
Problem: Check Whether BST Contains Dead End
URL: https://practice.geeksforgeeks.org/problems/check-whether-bst-contains-dead-end/1

Problem Statement:
Check if BST contains a dead end (leaf where no new node can be inserted).

Sample Input/Output:
Input: root = [8, 5, 9, 2, 7, null, null, null, null, null, 3]
Output: true
Explanation: Node 3 is a dead end (leaf with value 3, parent is 2, cannot insert 1 or 4).
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
    void Store_Nodes(TreeNode* root, unordered_set<int>& all_nodes, unordered_set<int>& leaf_nodes) {
        if (root == NULL) return;
        all_nodes.insert(root->data);
        if (root->left == NULL && root->right == NULL) {
            leaf_nodes.insert(root->data);
        }
        Store_Nodes(root->left, all_nodes, leaf_nodes);
        Store_Nodes(root->right, all_nodes, leaf_nodes);
    }

    bool Contains_Dead_End_Hash(TreeNode* root) {
        /*
        Hash set approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        unordered_set<int> all_nodes, leaf_nodes;
        Store_Nodes(root, all_nodes, leaf_nodes);
        for (int leaf : leaf_nodes) {
            if ((leaf == 1 || all_nodes.find(leaf - 1) == all_nodes.end()) &&
                all_nodes.find(leaf + 1) == all_nodes.end()) {
                return true;
            }
        }
        return false;
    }

    bool Contains_Dead_End_Range(TreeNode* root, int min_val, int max_val) {
        /*
        Range-based recursion approach
        Time Complexity: O(n)
        Space Complexity: O(h)
        */
        if (root == NULL) return false;
        if (root->left == NULL && root->right == NULL) {
            if (min_val == max_val || (min_val == INT_MIN && max_val == root->data - 1) ||
                (max_val == INT_MAX && min_val == root->data + 1) ||
                (min_val == root->data + 1 && max_val == root->data - 1)) {
                return true;
            }
        }
        bool left_dead = Contains_Dead_End_Range(root->left, min_val, root->data - 1);
        bool right_dead = Contains_Dead_End_Range(root->right, root->data + 1, max_val);
        return left_dead || right_dead;
    }
};

void Test_Dead_End_In_BST() {
    Solution solution;
    TreeNode* root = NULL;
    root = Insert_BST(root, 8);
    root = Insert_BST(root, 5);
    root = Insert_BST(root, 9);
    root = Insert_BST(root, 2);
    root = Insert_BST(root, 7);
    root = Insert_BST(root, 3);
    cout << "Dead End (Hash): " << solution.Contains_Dead_End_Hash(root) << endl;
    cout << "Dead End (Range): " << solution.Contains_Dead_End_Range(root, INT_MIN, INT_MAX) << endl;
}

int main() {
    Test_Dead_End_In_BST();
    return 0;
}
