/*
Problem: Postorder Traversal
URL: https://www.techiedelight.com/postorder-tree-traversal-iterative-recursive/

Problem Statement:
Given a binary tree, perform postorder traversal. Postorder traversal visits left subtree, right subtree, then root.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output: 4 5 2 6 7 3 1
Explanation: Left subtree -> Right subtree -> Root
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

class Solution {
public:
    vector<int> Postorder_Recursive(TreeNode* root) {
        /*
        Recursive approach: Visit left, right, root
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        */
        vector<int> result;
        Postorder_Helper(root, result);
        return result;
    }

    vector<int> Postorder_Iterative(TreeNode* root) {
        /*
        Iterative two stacks: Use two stacks for postorder traversal
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for stacks
        */
        vector<int> result;
        if (!root) return result;
        stack<TreeNode*> st1, st2;
        st1.push(root);
        while (!st1.empty()) {
            TreeNode* node = st1.top();
            st1.pop();
            st2.push(node);
            if (node->left) st1.push(node->left);
            if (node->right) st1.push(node->right);
        }
        while (!st2.empty()) {
            result.push_back(st2.top()->data);
            st2.pop();
        }
        return result;
    }

private:
    void Postorder_Helper(TreeNode* root, vector<int>& result) {
        if (!root) return;
        Postorder_Helper(root->left, result);
        Postorder_Helper(root->right, result);
        result.push_back(root->data);
    }
};

void Test_Postorder_Traversal() {
    Solution solution;
    
    vector<int> vals1 = {1, 2, 3, 4, 5, 6, 7};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Test 1 - Recursive: ";
    vector<int> result1 = solution.Postorder_Recursive(root1);
    for (int val : result1) cout << val << " ";
    cout << endl;
    
    cout << "Test 1 - Iterative: ";
    vector<int> result2 = solution.Postorder_Iterative(root1);
    for (int val : result2) cout << val << " ";
    cout << endl;
    
    vector<int> vals2 = {1, 2, 3, -1, -1, 4, 5};
    TreeNode* root2 = Create_Tree(vals2);
    cout << "Test 2 - Recursive: ";
    vector<int> result3 = solution.Postorder_Recursive(root2);
    for (int val : result3) cout << val << " ";
    cout << endl;
    
    cout << "Test 2 - Iterative: ";
    vector<int> result4 = solution.Postorder_Iterative(root2);
    for (int val : result4) cout << val << " ";
    cout << endl;
}

int main() {
    Test_Postorder_Traversal();
    return 0;
}
