/*
Problem: Preorder Traversal
URL: https://www.techiedelight.com/preorder-tree-traversal-iterative-recursive/

Problem Statement:
Given a binary tree, perform preorder traversal. Preorder traversal visits root, left subtree, then right subtree.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output: 1 2 4 5 3 6 7
Explanation: Root -> Left subtree -> Right subtree
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
    vector<int> Preorder_Recursive(TreeNode* root) {
        /*
        Recursive approach: Visit root, left, right
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        */
        vector<int> result;
        Preorder_Helper(root, result);
        return result;
    }

    vector<int> Preorder_Iterative(TreeNode* root) {
        /*
        Iterative with stack: Use stack to simulate recursion
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        */
        vector<int> result;
        if (!root) return result;
        stack<TreeNode*> st;
        st.push(root);
        while (!st.empty()) {
            TreeNode* node = st.top();
            st.pop();
            result.push_back(node->data);
            if (node->right) st.push(node->right);
            if (node->left) st.push(node->left);
        }
        return result;
    }

private:
    void Preorder_Helper(TreeNode* root, vector<int>& result) {
        if (!root) return;
        result.push_back(root->data);
        Preorder_Helper(root->left, result);
        Preorder_Helper(root->right, result);
    }
};

void Test_Preorder_Traversal() {
    Solution solution;
    
    vector<int> vals1 = {1, 2, 3, 4, 5, 6, 7};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Test 1 - Recursive: ";
    vector<int> result1 = solution.Preorder_Recursive(root1);
    for (int val : result1) cout << val << " ";
    cout << endl;
    
    cout << "Test 1 - Iterative: ";
    vector<int> result2 = solution.Preorder_Iterative(root1);
    for (int val : result2) cout << val << " ";
    cout << endl;
    
    vector<int> vals2 = {1, 2, 3, -1, -1, 4, 5};
    TreeNode* root2 = Create_Tree(vals2);
    cout << "Test 2 - Recursive: ";
    vector<int> result3 = solution.Preorder_Recursive(root2);
    for (int val : result3) cout << val << " ";
    cout << endl;
    
    cout << "Test 2 - Iterative: ";
    vector<int> result4 = solution.Preorder_Iterative(root2);
    for (int val : result4) cout << val << " ";
    cout << endl;
}

int main() {
    Test_Preorder_Traversal();
    return 0;
}
