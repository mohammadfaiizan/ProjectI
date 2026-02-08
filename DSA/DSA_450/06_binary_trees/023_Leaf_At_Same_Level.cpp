/*
Problem: Leaf At Same Level
URL: https://practice.geeksforgeeks.org/problems/leaf-at-same-level/1

Problem Statement:
Check if all leaf nodes are at the same level.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    /
   4

Output: false
Explanation: Leaf nodes 4 and 3 are at different levels.
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
    bool Check_Leaf_Level_Recursive(TreeNode* root, int level, int& leaf_level) {
        /*
        Recursive level tracking: Track level of first leaf, compare with others
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        */
        if (!root) return true;
        if (!root->left && !root->right) {
            if (leaf_level == -1) {
                leaf_level = level;
                return true;
            }
            return level == leaf_level;
        }
        return Check_Leaf_Level_Recursive(root->left, level + 1, leaf_level) &&
               Check_Leaf_Level_Recursive(root->right, level + 1, leaf_level);
    }

    bool Check_Leaf_Same_Level_Recursive(TreeNode* root) {
        int leaf_level = -1;
        return Check_Leaf_Level_Recursive(root, 0, leaf_level);
    }

    bool Check_Leaf_Same_Level_BFS(TreeNode* root) {
        /*
        BFS iterative: Use level order traversal to check leaf levels
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for queue
        */
        if (!root) return true;
        queue<TreeNode*> q;
        q.push(root);
        int leaf_level = -1;
        int level = 0;
        while (!q.empty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode* node = q.front();
                q.pop();
                if (!node->left && !node->right) {
                    if (leaf_level == -1) {
                        leaf_level = level;
                    } else if (leaf_level != level) {
                        return false;
                    }
                }
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            level++;
        }
        return true;
    }
};

void Test_Leaf_At_Same_Level() {
    Solution solution;
    
    vector<int> vals1 = {1, 2, 3, 4};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Tree 1 (recursive): " << solution.Check_Leaf_Same_Level_Recursive(root1) << endl;
    cout << "Tree 1 (BFS): " << solution.Check_Leaf_Same_Level_BFS(root1) << endl;
    
    vector<int> vals2 = {1, 2, 3};
    TreeNode* root2 = Create_Tree(vals2);
    cout << "Tree 2 (recursive): " << solution.Check_Leaf_Same_Level_Recursive(root2) << endl;
    cout << "Tree 2 (BFS): " << solution.Check_Leaf_Same_Level_BFS(root2) << endl;
    
    vector<int> vals3 = {10, 20, 30, 40, 50};
    TreeNode* root3 = Create_Tree(vals3);
    cout << "Tree 3 (recursive): " << solution.Check_Leaf_Same_Level_Recursive(root3) << endl;
    cout << "Tree 3 (BFS): " << solution.Check_Leaf_Same_Level_BFS(root3) << endl;
}

int main() {
    Test_Leaf_At_Same_Level();
    return 0;
}
