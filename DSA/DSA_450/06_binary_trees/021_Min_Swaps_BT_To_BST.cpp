/*
Problem: Min Swaps BT To BST
URL: https://www.geeksforgeeks.org/minimum-swap-required-convert-binary-tree-binary-search-tree/

Problem Statement:
Find minimum swaps required to convert a binary tree to BST. Get inorder, then find min swaps to sort.

Sample Input/Output:
Input: 
        5
      /   \
     6     7
    / \   / \
   8   9 10  11

Output: 3
Explanation: Inorder: [8, 6, 9, 5, 10, 7, 11]. Need 3 swaps to sort.
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
    void Get_Inorder(TreeNode* root, vector<int>& inorder) {
        if (!root) return;
        Get_Inorder(root->left, inorder);
        inorder.push_back(root->data);
        Get_Inorder(root->right, inorder);
    }

    int Min_Swaps_Cycle_Detection(vector<int>& arr) {
        /*
        Inorder traversal + min swaps using cycle detection
        Time Complexity: O(n log n) for sorting
        Space Complexity: O(n) for storing pairs and visited array
        */
        int n = arr.size();
        vector<pair<int, int>> pairs;
        for (int i = 0; i < n; i++) {
            pairs.push_back({arr[i], i});
        }
        sort(pairs.begin(), pairs.end());
        vector<bool> visited(n, false);
        int swaps = 0;
        for (int i = 0; i < n; i++) {
            if (visited[i] || pairs[i].second == i) continue;
            int cycle_size = 0;
            int j = i;
            while (!visited[j]) {
                visited[j] = true;
                j = pairs[j].second;
                cycle_size++;
            }
            if (cycle_size > 0) {
                swaps += (cycle_size - 1);
            }
        }
        return swaps;
    }

    int Min_Swaps_BT_To_BST(TreeNode* root) {
        vector<int> inorder;
        Get_Inorder(root, inorder);
        return Min_Swaps_Cycle_Detection(inorder);
    }
};

void Test_Min_Swaps_BT_To_BST() {
    Solution solution;
    
    vector<int> vals1 = {5, 6, 7, 8, 9, 10, 11};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Inorder: ";
    Print_Inorder(root1);
    cout << endl;
    int swaps1 = solution.Min_Swaps_BT_To_BST(root1);
    cout << "Minimum swaps needed: " << swaps1 << endl;
    
    vector<int> vals2 = {1, 2, 3};
    TreeNode* root2 = Create_Tree(vals2);
    cout << "Inorder: ";
    Print_Inorder(root2);
    cout << endl;
    int swaps2 = solution.Min_Swaps_BT_To_BST(root2);
    cout << "Minimum swaps needed: " << swaps2 << endl;
}

int main() {
    Test_Min_Swaps_BT_To_BST();
    return 0;
}
