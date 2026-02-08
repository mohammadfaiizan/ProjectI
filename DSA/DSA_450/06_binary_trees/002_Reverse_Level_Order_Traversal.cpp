/*
Problem: Reverse Level Order Traversal
URL: https://practice.geeksforgeeks.org/problems/reverse-level-order-traversal/1

Problem Statement:
Given a binary tree, print its reverse level order traversal. Reverse level order means visiting nodes level by level from bottom to top, left to right within each level.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output: 4 5 6 7 2 3 1
Explanation: Nodes are printed level by level from bottom to top.
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

int Height(TreeNode* root) {
    if (!root) return 0;
    return 1 + max(Height(root->left), Height(root->right));
}

class Solution {
public:
    vector<int> Reverse_Level_Order_Recursive(TreeNode* root) {
        /*
        Recursive approach: Print levels in reverse order
        Time Complexity: O(n^2) worst case for skewed tree
        Space Complexity: O(n) for recursion stack
        */
        vector<int> result;
        if (!root) return result;
        int h = Height(root);
        for (int i = h; i >= 1; i--) {
            Print_Level_Helper(root, i, result);
        }
        return result;
    }

    vector<int> Reverse_Level_Order_Queue_Stack(TreeNode* root) {
        /*
        Queue + Stack approach: Use queue for BFS and stack to reverse
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for queue and stack
        */
        vector<int> result;
        if (!root) return result;
        queue<TreeNode*> q;
        stack<int> st;
        q.push(root);
        while (!q.empty()) {
            TreeNode* node = q.front();
            q.pop();
            st.push(node->data);
            if (node->right) q.push(node->right);
            if (node->left) q.push(node->left);
        }
        while (!st.empty()) {
            result.push_back(st.top());
            st.pop();
        }
        return result;
    }

private:
    void Print_Level_Helper(TreeNode* root, int level, vector<int>& result) {
        if (!root) return;
        if (level == 1) {
            result.push_back(root->data);
            return;
        }
        Print_Level_Helper(root->left, level - 1, result);
        Print_Level_Helper(root->right, level - 1, result);
    }
};

void Test_Reverse_Level_Order_Traversal() {
    Solution solution;
    
    vector<int> vals1 = {1, 2, 3, 4, 5, 6, 7};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Test 1 - Queue+Stack Approach: ";
    vector<int> result1 = solution.Reverse_Level_Order_Queue_Stack(root1);
    for (int val : result1) cout << val << " ";
    cout << endl;
    
    cout << "Test 1 - Recursive Approach: ";
    vector<int> result2 = solution.Reverse_Level_Order_Recursive(root1);
    for (int val : result2) cout << val << " ";
    cout << endl;
    
    vector<int> vals2 = {1, 2, 3, -1, -1, 4, 5};
    TreeNode* root2 = Create_Tree(vals2);
    cout << "Test 2 - Queue+Stack Approach: ";
    vector<int> result3 = solution.Reverse_Level_Order_Queue_Stack(root2);
    for (int val : result3) cout << val << " ";
    cout << endl;
}

int main() {
    Test_Reverse_Level_Order_Traversal();
    return 0;
}
