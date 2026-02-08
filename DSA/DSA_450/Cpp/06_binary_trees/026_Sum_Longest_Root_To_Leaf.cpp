/*
Problem: Sum Longest Root To Leaf
URL: https://practice.geeksforgeeks.org/problems/sum-of-the-longest-bloodline-of-a-tree/1

Problem Statement:
Find sum of nodes on the longest path from root to leaf.

Sample Input/Output:
Input: 
        4
      /   \
     2     5
    / \   / \
   7   1 2   3
      /
     6

Output: 13
Explanation: Longest path is 4->2->1->6 with sum 13.
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
    void Sum_Longest_Path_Recursive(TreeNode* root, int level, int sum, int& max_level, int& max_sum) {
        /*
        Recursive with level and sum tracking: Track level and sum simultaneously
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        */
        if (!root) return;
        sum += root->data;
        if (!root->left && !root->right) {
            if (level > max_level) {
                max_level = level;
                max_sum = sum;
            } else if (level == max_level) {
                max_sum = max(max_sum, sum);
            }
            return;
        }
        Sum_Longest_Path_Recursive(root->left, level + 1, sum, max_level, max_sum);
        Sum_Longest_Path_Recursive(root->right, level + 1, sum, max_level, max_sum);
    }

    int Sum_Of_Longest_Bloodline_Recursive(TreeNode* root) {
        int max_level = 0;
        int max_sum = 0;
        Sum_Longest_Path_Recursive(root, 1, 0, max_level, max_sum);
        return max_sum;
    }

    int Sum_Of_Longest_Bloodline_BFS(TreeNode* root) {
        /*
        BFS with level tracking: Use level order traversal
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for queue
        */
        if (!root) return 0;
        queue<pair<TreeNode*, pair<int, int>>> q;
        q.push({root, {1, root->data}});
        int max_level = 0;
        int max_sum = 0;
        while (!q.empty()) {
            TreeNode* node = q.front().first;
            int level = q.front().second.first;
            int sum = q.front().second.second;
            q.pop();
            if (!node->left && !node->right) {
                if (level > max_level) {
                    max_level = level;
                    max_sum = sum;
                } else if (level == max_level) {
                    max_sum = max(max_sum, sum);
                }
            }
            if (node->left) {
                q.push({node->left, {level + 1, sum + node->left->data}});
            }
            if (node->right) {
                q.push({node->right, {level + 1, sum + node->right->data}});
            }
        }
        return max_sum;
    }
};

void Test_Sum_Longest_Root_To_Leaf() {
    Solution solution;
    
    vector<int> vals1 = {4, 2, 5, 7, 1, 2, 3, -1, -1, 6};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Tree 1 (recursive): " << solution.Sum_Of_Longest_Bloodline_Recursive(root1) << endl;
    cout << "Tree 1 (BFS): " << solution.Sum_Of_Longest_Bloodline_BFS(root1) << endl;
    
    vector<int> vals2 = {1, 2, 3};
    TreeNode* root2 = Create_Tree(vals2);
    cout << "Tree 2 (recursive): " << solution.Sum_Of_Longest_Bloodline_Recursive(root2) << endl;
    cout << "Tree 2 (BFS): " << solution.Sum_Of_Longest_Bloodline_BFS(root2) << endl;
}

int main() {
    Test_Sum_Longest_Root_To_Leaf();
    return 0;
}
