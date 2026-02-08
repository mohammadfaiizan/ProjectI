/*
Problem: Check if All Levels of Two Trees are Anagrams
URL: https://www.geeksforgeeks.org/check-if-all-levels-of-two-trees-are-anagrams-or-not/

Problem Statement:
Given two binary trees, check if all levels of one tree are anagrams of the corresponding levels in the other tree.

Sample Input/Output:
Input: Tree1:      Tree2:
        1              1
       / \            / \
      3   2          2   3
     / \            / \
    5   4          4   5
Output: Yes (Level 0: [1] = [1], Level 1: [3,2] = [2,3], Level 2: [5,4] = [4,5])
*/

#include <bits/stdc++.h>
using namespace std;

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    bool Level_Anagram_Trees_BFS(TreeNode* root1, TreeNode* root2) {
        /*
        BFS level order with sort comparison
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        if (!root1 && !root2) return true;
        if (!root1 || !root2) return false;
        
        queue<TreeNode*> q1, q2;
        q1.push(root1);
        q2.push(root2);
        
        while (!q1.empty() && !q2.empty()) {
            int size1 = q1.size();
            int size2 = q2.size();
            
            if (size1 != size2) return false;
            
            vector<int> level1, level2;
            
            for (int i = 0; i < size1; i++) {
                TreeNode* node1 = q1.front();
                TreeNode* node2 = q2.front();
                q1.pop();
                q2.pop();
                
                level1.push_back(node1->val);
                level2.push_back(node2->val);
                
                if (node1->left) q1.push(node1->left);
                if (node1->right) q1.push(node1->right);
                if (node2->left) q2.push(node2->left);
                if (node2->right) q2.push(node2->right);
            }
            
            sort(level1.begin(), level1.end());
            sort(level2.begin(), level2.end());
            
            if (level1 != level2) return false;
        }
        
        return q1.empty() && q2.empty();
    }
};

void Test_Level_Anagram_Trees() {
    Solution solution;
    
    TreeNode* root1 = new TreeNode(1);
    root1->left = new TreeNode(3);
    root1->right = new TreeNode(2);
    root1->left->left = new TreeNode(5);
    root1->left->right = new TreeNode(4);
    
    TreeNode* root2 = new TreeNode(1);
    root2->left = new TreeNode(2);
    root2->right = new TreeNode(3);
    root2->left->left = new TreeNode(4);
    root2->left->right = new TreeNode(5);
    
    cout << "Test 1 - Level Anagrams: " << (solution.Level_Anagram_Trees_BFS(root1, root2) ? "Yes" : "No") << endl;
    
    TreeNode* root3 = new TreeNode(1);
    root3->left = new TreeNode(2);
    root3->right = new TreeNode(3);
    
    TreeNode* root4 = new TreeNode(1);
    root4->left = new TreeNode(3);
    root4->right = new TreeNode(2);
    root4->left->left = new TreeNode(4);
    
    cout << "Test 2 - Level Anagrams: " << (solution.Level_Anagram_Trees_BFS(root3, root4) ? "Yes" : "No") << endl;
    
    TreeNode* root5 = new TreeNode(1);
    root5->left = new TreeNode(2);
    root5->right = new TreeNode(3);
    
    TreeNode* root6 = new TreeNode(1);
    root6->left = new TreeNode(2);
    root6->right = new TreeNode(4);
    
    cout << "Test 3 - Level Anagrams: " << (solution.Level_Anagram_Trees_BFS(root5, root6) ? "Yes" : "No") << endl;
}

int main() {
    Test_Level_Anagram_Trees();
    return 0;
}
