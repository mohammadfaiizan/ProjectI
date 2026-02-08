/*
Problem: Left View Of Tree
URL: https://practice.geeksforgeeks.org/problems/left-view-of-binary-tree/1

Problem Statement:
Given a binary tree, print its left view. Left view of a tree is the set of nodes visible when the tree is viewed from the left side.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output: 1 2 4
Explanation: When viewed from left, nodes 1, 2, and 4 are visible.
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
    vector<int> Left_View_Recursive(TreeNode* root) {
        /*
        Recursive level tracking: Track max level reached
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        */
        vector<int> result;
        int max_level = 0;
        Left_View_Helper(root, 1, max_level, result);
        return result;
    }

    vector<int> Left_View_Queue(TreeNode* root) {
        /*
        Queue BFS approach: First node of each level
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for queue
        */
        vector<int> result;
        if (!root) return result;
        queue<TreeNode*> q;
        q.push(root);
        while (!q.empty()) {
            int size = q.size();
            result.push_back(q.front()->data);
            for (int i = 0; i < size; i++) {
                TreeNode* node = q.front();
                q.pop();
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
        }
        return result;
    }

private:
    void Left_View_Helper(TreeNode* root, int level, int& max_level, vector<int>& result) {
        if (!root) return;
        if (level > max_level) {
            result.push_back(root->data);
            max_level = level;
        }
        Left_View_Helper(root->left, level + 1, max_level, result);
        Left_View_Helper(root->right, level + 1, max_level, result);
    }
};

void Test_Left_View_Of_Tree() {
    Solution solution;
    
    vector<int> vals1 = {1, 2, 3, 4, 5, 6, 7};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Test 1 - Recursive: ";
    vector<int> result1 = solution.Left_View_Recursive(root1);
    for (int val : result1) cout << val << " ";
    cout << endl;
    
    cout << "Test 1 - Queue: ";
    vector<int> result2 = solution.Left_View_Queue(root1);
    for (int val : result2) cout << val << " ";
    cout << endl;
    
    vector<int> vals2 = {1, 2, 3, -1, -1, 4, 5};
    TreeNode* root2 = Create_Tree(vals2);
    cout << "Test 2 - Recursive: ";
    vector<int> result3 = solution.Left_View_Recursive(root2);
    for (int val : result3) cout << val << " ";
    cout << endl;
    
    cout << "Test 2 - Queue: ";
    vector<int> result4 = solution.Left_View_Queue(root2);
    for (int val : result4) cout << val << " ";
    cout << endl;
    
    vector<int> vals3 = {1, 2, -1, 3};
    TreeNode* root3 = Create_Tree(vals3);
    cout << "Test 3 - Recursive: ";
    vector<int> result5 = solution.Left_View_Recursive(root3);
    for (int val : result5) cout << val << " ";
    cout << endl;
    
    cout << "Test 3 - Queue: ";
    vector<int> result6 = solution.Left_View_Queue(root3);
    for (int val : result6) cout << val << " ";
    cout << endl;
}

int main() {
    Test_Left_View_Of_Tree();
    return 0;
}
