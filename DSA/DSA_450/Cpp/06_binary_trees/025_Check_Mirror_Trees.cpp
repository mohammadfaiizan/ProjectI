/*
Problem: Check Mirror Trees
URL: https://practice.geeksforgeeks.org/problems/check-mirror-in-n-ary-tree1528/1

Problem Statement:
Check if two N-ary trees are mirror of each other. Given as edge lists.

Sample Input/Output:
Input: 
Tree 1: (1,2), (1,3), (1,4)
Tree 2: (1,4), (1,3), (1,2)

Output: true
Explanation: Tree 2 is mirror of Tree 1.
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
    bool Check_Mirror_Stack(vector<int>& tree1, vector<int>& tree2, int n, int e) {
        /*
        Stack-based comparison: Compare children order using stacks
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for stacks
        */
        unordered_map<int, stack<int>> adj1;
        unordered_map<int, stack<int>> adj2;
        for (int i = 0; i < 2 * e; i += 2) {
            adj1[tree1[i]].push(tree1[i + 1]);
        }
        for (int i = 0; i < 2 * e; i += 2) {
            adj2[tree2[i]].push(tree2[i + 1]);
        }
        for (int i = 1; i <= n; i++) {
            if (adj1[i].size() != adj2[i].size()) {
                return false;
            }
            while (!adj1[i].empty()) {
                if (adj1[i].top() != adj2[i].top()) {
                    return false;
                }
                adj1[i].pop();
                adj2[i].pop();
            }
        }
        return true;
    }
};

void Test_Check_Mirror_Trees() {
    Solution solution;
    
    int n1 = 3, e1 = 3;
    vector<int> tree1_1 = {1, 2, 1, 3, 1, 4};
    vector<int> tree1_2 = {1, 4, 1, 3, 1, 2};
    cout << "Test 1: " << solution.Check_Mirror_Stack(tree1_1, tree1_2, n1, e1) << endl;
    
    int n2 = 3, e2 = 3;
    vector<int> tree2_1 = {1, 2, 1, 3, 1, 4};
    vector<int> tree2_2 = {1, 2, 1, 3, 1, 4};
    cout << "Test 2: " << solution.Check_Mirror_Stack(tree2_1, tree2_2, n2, e2) << endl;
}

int main() {
    Test_Check_Mirror_Trees();
    return 0;
}
