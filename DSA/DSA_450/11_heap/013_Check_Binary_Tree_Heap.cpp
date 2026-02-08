/*
Problem: Check if a Binary Tree is a Heap
URL: https://practice.geeksforgeeks.org/problems/is-binary-tree-heap/1

Problem Statement:
Check if a given binary tree satisfies max heap properties: completeness + max heap ordering.

Sample Input/Output:
Input: Tree structure
Output: true/false
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
    bool Check_Heap_Recursive(TreeNode* root) {
        /*
        Recursive Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        int nodeCount = CountNodes(root);
        return IsComplete(root, 0, nodeCount) && IsMaxHeap(root);
    }
    
    bool Check_Heap_Level_Order(TreeNode* root) {
        /*
        Level Order Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (!root) return true;
        
        queue<TreeNode*> q;
        q.push(root);
        bool foundNull = false;
        
        while (!q.empty()) {
            TreeNode* node = q.front();
            q.pop();
            
            if (!node) {
                foundNull = true;
            } else {
                if (foundNull) return false;
                
                if (node->left) {
                    if (node->left->val > node->val) return false;
                    q.push(node->left);
                } else {
                    q.push(nullptr);
                }
                
                if (node->right) {
                    if (node->right->val > node->val) return false;
                    q.push(node->right);
                } else {
                    q.push(nullptr);
                }
            }
        }
        
        return true;
    }
    
private:
    int CountNodes(TreeNode* root) {
        if (!root) return 0;
        return 1 + CountNodes(root->left) + CountNodes(root->right);
    }
    
    bool IsComplete(TreeNode* root, int index, int nodeCount) {
        if (!root) return true;
        if (index >= nodeCount) return false;
        return IsComplete(root->left, 2 * index + 1, nodeCount) &&
               IsComplete(root->right, 2 * index + 2, nodeCount);
    }
    
    bool IsMaxHeap(TreeNode* root) {
        if (!root) return true;
        
        bool leftValid = true;
        bool rightValid = true;
        
        if (root->left) {
            if (root->left->val > root->val) return false;
            leftValid = IsMaxHeap(root->left);
        }
        
        if (root->right) {
            if (root->right->val > root->val) return false;
            rightValid = IsMaxHeap(root->right);
        }
        
        return leftValid && rightValid;
    }
};

void Test_Check_Heap() {
    Solution solution;
    
    TreeNode* root1 = new TreeNode(10);
    root1->left = new TreeNode(9);
    root1->right = new TreeNode(8);
    root1->left->left = new TreeNode(7);
    root1->left->right = new TreeNode(6);
    root1->right->left = new TreeNode(5);
    
    cout << "Test 1 (Valid Heap): " << solution.Check_Heap_Recursive(root1) << endl;
    cout << "Test 1 Level Order: " << solution.Check_Heap_Level_Order(root1) << endl;
    
    TreeNode* root2 = new TreeNode(10);
    root2->left = new TreeNode(15);
    root2->right = new TreeNode(8);
    
    cout << "Test 2 (Invalid Heap): " << solution.Check_Heap_Recursive(root2) << endl;
    cout << "Test 2 Level Order: " << solution.Check_Heap_Level_Order(root2) << endl;
    
    TreeNode* root3 = new TreeNode(10);
    root3->left = new TreeNode(9);
    root3->right = new TreeNode(8);
    root3->left->left = new TreeNode(7);
    
    cout << "Test 3 (Valid Heap): " << solution.Check_Heap_Recursive(root3) << endl;
    cout << "Test 3 Level Order: " << solution.Check_Heap_Level_Order(root3) << endl;
}

int main() {
    Test_Check_Heap();
    return 0;
}
