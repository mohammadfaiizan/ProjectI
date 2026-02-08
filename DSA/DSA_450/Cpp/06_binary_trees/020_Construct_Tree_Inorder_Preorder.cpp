/*
Problem: Construct Tree Inorder Preorder
URL: https://practice.geeksforgeeks.org/problems/construct-tree-1/1

Problem Statement:
Construct binary tree from inorder and preorder traversals.

Sample Input/Output:
Input: 
Inorder: [9, 3, 15, 20, 7]
Preorder: [3, 9, 20, 15, 7]

Output:
        3
      /   \
     9    20
         /  \
       15    7

Explanation: Root is first in preorder, then left and right subtrees are constructed recursively.
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
    TreeNode* Build_Tree_Linear_Search(vector<int>& inorder, vector<int>& preorder, int& pre_idx, int in_start, int in_end) {
        /*
        Recursive with linear search: Find root in inorder array using linear search
        Time Complexity: O(n^2) worst case
        Space Complexity: O(h) where h is height of tree
        */
        if (in_start > in_end || pre_idx >= preorder.size()) return NULL;
        TreeNode* root = new TreeNode(preorder[pre_idx++]);
        int in_idx = in_start;
        for (int i = in_start; i <= in_end; i++) {
            if (inorder[i] == root->data) {
                in_idx = i;
                break;
            }
        }
        root->left = Build_Tree_Linear_Search(inorder, preorder, pre_idx, in_start, in_idx - 1);
        root->right = Build_Tree_Linear_Search(inorder, preorder, pre_idx, in_idx + 1, in_end);
        return root;
    }

    TreeNode* Build_Tree_Hashmap(vector<int>& inorder, vector<int>& preorder, int& pre_idx, int in_start, int in_end, unordered_map<int, int>& in_map) {
        /*
        Recursive with hashmap: Use hashmap to find root index in O(1)
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for hashmap
        */
        if (in_start > in_end || pre_idx >= preorder.size()) return NULL;
        TreeNode* root = new TreeNode(preorder[pre_idx++]);
        int in_idx = in_map[root->data];
        root->left = Build_Tree_Hashmap(inorder, preorder, pre_idx, in_start, in_idx - 1, in_map);
        root->right = Build_Tree_Hashmap(inorder, preorder, pre_idx, in_idx + 1, in_end, in_map);
        return root;
    }

    TreeNode* Construct_Tree_Linear_Search(vector<int>& inorder, vector<int>& preorder) {
        int pre_idx = 0;
        return Build_Tree_Linear_Search(inorder, preorder, pre_idx, 0, inorder.size() - 1);
    }

    TreeNode* Construct_Tree_Hashmap(vector<int>& inorder, vector<int>& preorder) {
        unordered_map<int, int> in_map;
        for (int i = 0; i < inorder.size(); i++) {
            in_map[inorder[i]] = i;
        }
        int pre_idx = 0;
        return Build_Tree_Hashmap(inorder, preorder, pre_idx, 0, inorder.size() - 1, in_map);
    }
};

void Test_Construct_Tree_Inorder_Preorder() {
    Solution solution;
    
    vector<int> inorder1 = {9, 3, 15, 20, 7};
    vector<int> preorder1 = {3, 9, 20, 15, 7};
    TreeNode* root1 = solution.Construct_Tree_Hashmap(inorder1, preorder1);
    cout << "Constructed tree (hashmap): ";
    Print_Inorder(root1);
    cout << endl;
    
    vector<int> inorder2 = {4, 2, 5, 1, 6, 3, 7};
    vector<int> preorder2 = {1, 2, 4, 5, 3, 6, 7};
    TreeNode* root2 = solution.Construct_Tree_Linear_Search(inorder2, preorder2);
    cout << "Constructed tree (linear search): ";
    Print_Inorder(root2);
    cout << endl;
}

int main() {
    Test_Construct_Tree_Inorder_Preorder();
    return 0;
}
