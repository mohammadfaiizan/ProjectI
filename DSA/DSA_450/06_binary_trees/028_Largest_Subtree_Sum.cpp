/*
Problem: Find Largest Subtree Sum
URL: https://www.geeksforgeeks.org/find-largest-subtree-sum-tree/

Problem Statement:
Given a binary tree, find the largest subtree sum. The subtree sum of a node is the sum of all node values in the subtree rooted at that node.

Sample Input/Output:
Input: [1, 2, 3, 4, 5, -6, 2]
Output: 7
Explanation: Subtree rooted at node with value 2 has sum 2+4+5-6+2 = 7, which is the maximum.
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
    int Largest_Subtree_Sum_Postorder(TreeNode* root, int& max_sum) {
        /*
        Post-order recursion tracking max sum
        Time Complexity: O(n)
        Space Complexity: O(h)
        */
        if (!root) return 0;
        int left_sum = Largest_Subtree_Sum_Postorder(root->left, max_sum);
        int right_sum = Largest_Subtree_Sum_Postorder(root->right, max_sum);
        int subtree_sum = root->data + left_sum + right_sum;
        max_sum = max(max_sum, subtree_sum);
        return subtree_sum;
    }
    
    int Find_Largest_Subtree_Sum(TreeNode* root) {
        int max_sum = INT_MIN;
        Largest_Subtree_Sum_Postorder(root, max_sum);
        return max_sum;
    }
};

void Test_Largest_Subtree_Sum() {
    Solution solution;
    
    vector<int> vals1 = {1, 2, 3, 4, 5, -6, 2};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Test 1: " << solution.Find_Largest_Subtree_Sum(root1) << endl;
    
    vector<int> vals2 = {1, -2, 3, 4, 5, -6, 2};
    TreeNode* root2 = Create_Tree(vals2);
    cout << "Test 2: " << solution.Find_Largest_Subtree_Sum(root2) << endl;
    
    vector<int> vals3 = {-5, 2, 3};
    TreeNode* root3 = Create_Tree(vals3);
    cout << "Test 3: " << solution.Find_Largest_Subtree_Sum(root3) << endl;
}

int main() {
    Test_Largest_Subtree_Sum();
    return 0;
}
