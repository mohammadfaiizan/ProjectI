/*
Problem: Inorder Traversal
URL: https://www.techiedelight.com/inorder-tree-traversal-iterative-recursive/

Problem Statement:
Given a binary tree, perform inorder traversal. Inorder traversal visits left subtree, root, then right subtree.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output: 4 2 5 1 6 3 7
Explanation: Left subtree -> Root -> Right subtree
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
    vector<int> Inorder_Recursive(TreeNode* root) {
        /*
        Recursive approach: Visit left, root, right
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        */
        vector<int> result;
        Inorder_Helper(root, result);
        return result;
    }

    vector<int> Inorder_Iterative(TreeNode* root) {
        /*
        Iterative with stack: Use stack to simulate recursion
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        */
        vector<int> result;
        if (!root) return result;
        stack<TreeNode*> st;
        TreeNode* current = root;
        while (current || !st.empty()) {
            while (current) {
                st.push(current);
                current = current->left;
            }
            current = st.top();
            st.pop();
            result.push_back(current->data);
            current = current->right;
        }
        return result;
    }

private:
    void Inorder_Helper(TreeNode* root, vector<int>& result) {
        if (!root) return;
        Inorder_Helper(root->left, result);
        result.push_back(root->data);
        Inorder_Helper(root->right, result);
    }
};

void Test_Inorder_Traversal() {
    Solution solution;
    
    vector<int> vals1 = {1, 2, 3, 4, 5, 6, 7};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Test 1 - Recursive: ";
    vector<int> result1 = solution.Inorder_Recursive(root1);
    for (int val : result1) cout << val << " ";
    cout << endl;
    
    cout << "Test 1 - Iterative: ";
    vector<int> result2 = solution.Inorder_Iterative(root1);
    for (int val : result2) cout << val << " ";
    cout << endl;
    
    vector<int> vals2 = {1, 2, 3, -1, -1, 4, 5};
    TreeNode* root2 = Create_Tree(vals2);
    cout << "Test 2 - Recursive: ";
    vector<int> result3 = solution.Inorder_Recursive(root2);
    for (int val : result3) cout << val << " ";
    cout << endl;
    
    cout << "Test 2 - Iterative: ";
    vector<int> result4 = solution.Inorder_Iterative(root2);
    for (int val : result4) cout << val << " ";
    cout << endl;
}

int main() {
    Test_Inorder_Traversal();
    return 0;
}
