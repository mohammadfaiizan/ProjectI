/*
Problem: Lowest Common Ancestor in Binary Tree
URL: https://practice.geeksforgeeks.org/problems/lowest-common-ancestor-in-a-binary-tree/1

Problem Statement:
Find the Lowest Common Ancestor (LCA) of two nodes in a binary tree. LCA is the lowest node that has both nodes as descendants.

Sample Input/Output:
Input: Tree [1, 2, 3, 4, 5], nodes 4 and 5
Output: 2
Explanation: Node 2 is the LCA of nodes 4 and 5.
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
    bool Find_Path(TreeNode* root, int target, vector<TreeNode*>& path) {
        if (!root) return false;
        path.push_back(root);
        if (root->data == target) return true;
        if (Find_Path(root->left, target, path) || Find_Path(root->right, target, path)) {
            return true;
        }
        path.pop_back();
        return false;
    }
    
    TreeNode* LCA_Path_Storage(TreeNode* root, int n1, int n2) {
        /*
        Path storage and comparison
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<TreeNode*> path1, path2;
        if (!Find_Path(root, n1, path1) || !Find_Path(root, n2, path2)) {
            return NULL;
        }
        int i = 0;
        while (i < path1.size() && i < path2.size() && path1[i] == path2[i]) {
            i++;
        }
        return path1[i - 1];
    }
    
    TreeNode* LCA_Single_Traversal(TreeNode* root, int n1, int n2) {
        /*
        Single traversal recursion
        Time Complexity: O(n)
        Space Complexity: O(h)
        */
        if (!root) return NULL;
        if (root->data == n1 || root->data == n2) return root;
        TreeNode* left_lca = LCA_Single_Traversal(root->left, n1, n2);
        TreeNode* right_lca = LCA_Single_Traversal(root->right, n1, n2);
        if (left_lca && right_lca) return root;
        return left_lca ? left_lca : right_lca;
    }
    
    TreeNode* Find_LCA(TreeNode* root, int n1, int n2) {
        return LCA_Single_Traversal(root, n1, n2);
    }
};

void Test_LCA_Binary_Tree() {
    Solution solution;
    
    vector<int> vals1 = {1, 2, 3, 4, 5};
    TreeNode* root1 = Create_Tree(vals1);
    TreeNode* lca1 = solution.Find_LCA(root1, 4, 5);
    cout << "Test 1 - LCA of 4 and 5: " << (lca1 ? lca1->data : -1) << endl;
    
    TreeNode* lca2 = solution.Find_LCA(root1, 2, 3);
    cout << "Test 2 - LCA of 2 and 3: " << (lca2 ? lca2->data : -1) << endl;
    
    TreeNode* lca3 = solution.Find_LCA(root1, 4, 3);
    cout << "Test 3 - LCA of 4 and 3: " << (lca3 ? lca3->data : -1) << endl;
}

int main() {
    Test_LCA_Binary_Tree();
    return 0;
}
