/*
Problem: Minimum Distance Between Two Nodes
URL: https://practice.geeksforgeeks.org/problems/min-distance-between-two-given-nodes-of-a-binary-tree/1

Problem Statement:
Find the minimum distance between two nodes in a binary tree. Distance is the number of edges between them.
Formula: dist(a,b) = dist(root,a) + dist(root,b) - 2*dist(root,lca)

Sample Input/Output:
Input: Tree [1, 2, 3, 4, 5], nodes 4 and 5
Output: 2
Explanation: Path from 4 to 5: 4->2->5, distance = 2 edges.
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
    TreeNode* Find_LCA(TreeNode* root, int n1, int n2) {
        if (!root) return NULL;
        if (root->data == n1 || root->data == n2) return root;
        TreeNode* left_lca = Find_LCA(root->left, n1, n2);
        TreeNode* right_lca = Find_LCA(root->right, n1, n2);
        if (left_lca && right_lca) return root;
        return left_lca ? left_lca : right_lca;
    }
    
    int Find_Level(TreeNode* root, int target, int level) {
        if (!root) return -1;
        if (root->data == target) return level;
        int left_level = Find_Level(root->left, target, level + 1);
        if (left_level != -1) return left_level;
        return Find_Level(root->right, target, level + 1);
    }
    
    int Distance_LCA_Level(TreeNode* root, int a, int b) {
        /*
        LCA + level finding
        Time Complexity: O(n)
        Space Complexity: O(h)
        */
        TreeNode* lca = Find_LCA(root, a, b);
        if (!lca) return -1;
        int dist_a = Find_Level(root, a, 0);
        int dist_b = Find_Level(root, b, 0);
        int dist_lca = Find_Level(root, lca->data, 0);
        return dist_a + dist_b - 2 * dist_lca;
    }
    
    int Distance_Single_Traversal(TreeNode* root, int a, int b, int& dist) {
        /*
        Single traversal
        Time Complexity: O(n)
        Space Complexity: O(h)
        */
        if (!root) return 0;
        int left = Distance_Single_Traversal(root->left, a, b, dist);
        int right = Distance_Single_Traversal(root->right, a, b, dist);
        if (root->data == a || root->data == b) {
            if (left || right) {
                dist = max(left, right);
                return 0;
            }
            return 1;
        }
        if (left && right) {
            dist = left + right;
            return 0;
        }
        if (left || right) {
            return max(left, right) + 1;
        }
        return 0;
    }
    
    int Find_Distance_Between_Nodes(TreeNode* root, int a, int b) {
        int dist = 0;
        Distance_Single_Traversal(root, a, b, dist);
        return dist;
    }
};

void Test_Distance_Between_Two_Nodes() {
    Solution solution;
    
    vector<int> vals1 = {1, 2, 3, 4, 5};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Test 1 - Distance between 4 and 5: " << solution.Find_Distance_Between_Nodes(root1, 4, 5) << endl;
    
    cout << "Test 2 - Distance between 2 and 3: " << solution.Find_Distance_Between_Nodes(root1, 2, 3) << endl;
    
    cout << "Test 3 - Distance between 4 and 3: " << solution.Find_Distance_Between_Nodes(root1, 4, 3) << endl;
}

int main() {
    Test_Distance_Between_Two_Nodes();
    return 0;
}
