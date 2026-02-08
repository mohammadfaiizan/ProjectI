/*
Problem: Duplicate Subtree Size 2
URL: https://practice.geeksforgeeks.org/problems/duplicate-subtree-in-binary-tree/1

Problem Statement:
Check if binary tree contains duplicate subtrees of size 2 or more.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    / \   / \
   4   5 2   4
      /
     4

Output: true
Explanation: Subtree with root 2 and children 4,5 appears twice.
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
    string Serialize_Subtree(TreeNode* root, unordered_map<string, int>& subtree_map, bool& found_duplicate) {
        /*
        Hashing with serialization using unordered_map: Serialize each subtree
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for storing serializations
        */
        if (!root) return "#";
        string serial = to_string(root->data) + "," + 
                        Serialize_Subtree(root->left, subtree_map, found_duplicate) + "," +
                        Serialize_Subtree(root->right, subtree_map, found_duplicate);
        if (root->left || root->right) {
            subtree_map[serial]++;
            if (subtree_map[serial] == 2) {
                found_duplicate = true;
            }
        }
        return serial;
    }

    bool Has_Duplicate_Subtree_Map(TreeNode* root) {
        unordered_map<string, int> subtree_map;
        bool found_duplicate = false;
        Serialize_Subtree(root, subtree_map, found_duplicate);
        return found_duplicate;
    }

    string Serialize_Subtree_Set(TreeNode* root, unordered_set<string>& subtree_set) {
        /*
        Using unordered_set: Track seen subtrees
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for storing serializations
        */
        if (!root) return "#";
        string serial = to_string(root->data) + "," + 
                        Serialize_Subtree_Set(root->left, subtree_set) + "," +
                        Serialize_Subtree_Set(root->right, subtree_set);
        if (root->left || root->right) {
            if (subtree_set.find(serial) != subtree_set.end()) {
                return "DUPLICATE";
            }
            subtree_set.insert(serial);
        }
        return serial;
    }

    bool Has_Duplicate_Subtree_Set(TreeNode* root) {
        unordered_set<string> subtree_set;
        string result = Serialize_Subtree_Set(root, subtree_set);
        return result == "DUPLICATE";
    }
};

void Test_Duplicate_Subtree_Size_2() {
    Solution solution;
    
    vector<int> vals1 = {1, 2, 3, 4, 5, 2, 4, -1, -1, -1, -1, 4};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Tree 1 (map): " << solution.Has_Duplicate_Subtree_Map(root1) << endl;
    cout << "Tree 1 (set): " << solution.Has_Duplicate_Subtree_Set(root1) << endl;
    
    vector<int> vals2 = {1, 2, 3};
    TreeNode* root2 = Create_Tree(vals2);
    cout << "Tree 2 (map): " << solution.Has_Duplicate_Subtree_Map(root2) << endl;
    cout << "Tree 2 (set): " << solution.Has_Duplicate_Subtree_Set(root2) << endl;
}

int main() {
    Test_Duplicate_Subtree_Size_2();
    return 0;
}
