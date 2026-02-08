/*
Problem: Print K Sum Paths in Binary Tree
URL: https://www.geeksforgeeks.org/print-k-sum-paths-binary-tree/

Problem Statement:
Print all paths in a binary tree whose sum equals k. A path can start and end at any node but must be downward.

Sample Input/Output:
Input: k=5, tree [1, 3, -1, 2, 1, 4, 5]
Output: [3 2], [3 1 1], [4 1], [1 3 1], [5]
Explanation: Multiple paths sum to 5.
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
    void K_Sum_Paths_Backtracking(TreeNode* root, int k, vector<int>& path, vector<vector<int>>& result) {
        /*
        Recursion with path vector and backtracking
        Time Complexity: O(n^2)
        Space Complexity: O(h)
        */
        if (!root) return;
        path.push_back(root->data);
        int sum = 0;
        for (int i = path.size() - 1; i >= 0; i--) {
            sum += path[i];
            if (sum == k) {
                vector<int> valid_path(path.begin() + i, path.end());
                result.push_back(valid_path);
            }
        }
        K_Sum_Paths_Backtracking(root->left, k, path, result);
        K_Sum_Paths_Backtracking(root->right, k, path, result);
        path.pop_back();
    }
    
    void K_Sum_Paths_Prefix_Sum(TreeNode* root, int k, int current_sum, unordered_map<int, int>& prefix_map, vector<int>& path, vector<vector<int>>& result) {
        /*
        Prefix sum with hashmap
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (!root) return;
        current_sum += root->data;
        path.push_back(root->data);
        if (current_sum == k) {
            result.push_back(path);
        }
        if (prefix_map.find(current_sum - k) != prefix_map.end()) {
            int start_idx = prefix_map[current_sum - k];
            vector<int> valid_path(path.begin() + start_idx + 1, path.end());
            result.push_back(valid_path);
        }
        prefix_map[current_sum] = path.size() - 1;
        K_Sum_Paths_Prefix_Sum(root->left, k, current_sum, prefix_map, path, result);
        K_Sum_Paths_Prefix_Sum(root->right, k, current_sum, prefix_map, path, result);
        prefix_map.erase(current_sum);
        path.pop_back();
    }
    
    vector<vector<int>> Find_K_Sum_Paths(TreeNode* root, int k) {
        vector<vector<int>> result;
        vector<int> path;
        K_Sum_Paths_Backtracking(root, k, path, result);
        return result;
    }
    
    vector<vector<int>> Find_K_Sum_Paths_Optimized(TreeNode* root, int k) {
        vector<vector<int>> result;
        vector<int> path;
        unordered_map<int, int> prefix_map;
        K_Sum_Paths_Prefix_Sum(root, k, 0, prefix_map, path, result);
        return result;
    }
};

void Test_K_Sum_Paths() {
    Solution solution;
    
    vector<int> vals1 = {1, 3, -1, 2, 1, 4, 5};
    TreeNode* root1 = Create_Tree(vals1);
    vector<vector<int>> paths1 = solution.Find_K_Sum_Paths(root1, 5);
    cout << "Test 1 - Paths with sum 5:" << endl;
    for (auto& path : paths1) {
        for (int val : path) cout << val << " ";
        cout << endl;
    }
    
    vector<int> vals2 = {1, 2, 3, 4, 5};
    TreeNode* root2 = Create_Tree(vals2);
    vector<vector<int>> paths2 = solution.Find_K_Sum_Paths(root2, 3);
    cout << "Test 2 - Paths with sum 3:" << endl;
    for (auto& path : paths2) {
        for (int val : path) cout << val << " ";
        cout << endl;
    }
}

int main() {
    Test_K_Sum_Paths();
    return 0;
}
