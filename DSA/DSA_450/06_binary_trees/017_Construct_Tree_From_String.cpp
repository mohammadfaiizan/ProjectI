/*
Problem: Construct Tree From String
URL: https://www.geeksforgeeks.org/construct-binary-tree-string-bracket-representation/

Problem Statement:
Construct a binary tree from a string consisting of parenthesis and integers. The whole input represents a binary tree. It contains an integer followed by zero, one or two pairs of parenthesis. The integer represents the root's value and a pair of parenthesis contains a child binary tree with the same structure.

Sample Input/Output:
Input: "4(2(3)(1))(6(5))"
Output:
        4
      /   \
     2     6
    / \   /
   3   1 5

Explanation: Root is 4, left subtree is 2(3)(1), right subtree is 6(5).
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
    TreeNode* Construct_Tree_Stack(string s) {
        /*
        Recursive with stack for bracket matching
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        if (s.empty()) return NULL;
        int i = 0;
        return Construct_Helper_Stack(s, i);
    }

    TreeNode* Construct_Tree_Pointer(string s) {
        /*
        Recursive with pointer index
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height
        */
        if (s.empty()) return NULL;
        int index = 0;
        return Construct_Helper_Pointer(s, index);
    }

private:
    TreeNode* Construct_Helper_Stack(string& s, int& i) {
        if (i >= s.length()) return NULL;
        int num = 0;
        while (i < s.length() && isdigit(s[i])) {
            num = num * 10 + (s[i] - '0');
            i++;
        }
        TreeNode* root = new TreeNode(num);
        if (i < s.length() && s[i] == '(') {
            i++;
            root->left = Construct_Helper_Stack(s, i);
            i++;
        }
        if (i < s.length() && s[i] == '(') {
            i++;
            root->right = Construct_Helper_Stack(s, i);
            i++;
        }
        return root;
    }

    TreeNode* Construct_Helper_Pointer(string& s, int& index) {
        if (index >= s.length()) return NULL;
        bool negative = false;
        if (s[index] == '-') {
            negative = true;
            index++;
        }
        int num = 0;
        while (index < s.length() && isdigit(s[index])) {
            num = num * 10 + (s[index] - '0');
            index++;
        }
        if (negative) num = -num;
        TreeNode* root = new TreeNode(num);
        if (index < s.length() && s[index] == '(') {
            index++;
            root->left = Construct_Helper_Pointer(s, index);
            index++;
        }
        if (index < s.length() && s[index] == '(') {
            index++;
            root->right = Construct_Helper_Pointer(s, index);
            index++;
        }
        return root;
    }
};

void Test_Construct_Tree_From_String() {
    Solution solution;
    
    string s1 = "4(2(3)(1))(6(5))";
    cout << "Test 1 - Stack: ";
    TreeNode* root1 = solution.Construct_Tree_Stack(s1);
    Print_Inorder(root1);
    cout << endl;
    
    cout << "Test 1 - Pointer: ";
    TreeNode* root2 = solution.Construct_Tree_Pointer(s1);
    Print_Inorder(root2);
    cout << endl;
}

int main() {
    Test_Construct_Tree_From_String();
    return 0;
}
