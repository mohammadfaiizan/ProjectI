/*
Problem: Binary Tree To DLL
URL: https://practice.geeksforgeeks.org/problems/binary-tree-to-dll/1

Problem Statement:
Given a Binary Tree (BT), convert it to a Doubly Linked List(DLL) In-Place. The left and right pointers in nodes are to be used as previous and next pointers respectively in converted DLL. The order of nodes in DLL must be same as Inorder of the given Binary Tree. The first node of Inorder traversal (leftmost node in BT) must be head node of the DLL.

Sample Input/Output:
Input:
        10
      /    \
     12    15
    / \    /
   25 30  36

Output: 25 12 30 10 36 15
Explanation: DLL is created using inorder traversal. Left pointer becomes prev, right pointer becomes next.
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
    TreeNode* Binary_Tree_To_DLL_Inorder(TreeNode* root) {
        /*
        Inorder recursion with head/prev tracking
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height
        */
        TreeNode* head = NULL;
        TreeNode* prev = NULL;
        Convert_To_DLL(root, head, prev);
        return head;
    }

    TreeNode* Binary_Tree_To_DLL_Inplace(TreeNode* root) {
        /*
        In-place conversion approach
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height
        */
        if (!root) return NULL;
        TreeNode* head = NULL;
        TreeNode* prev = NULL;
        Convert_Inplace(root, head, prev);
        return head;
    }

private:
    void Convert_To_DLL(TreeNode* root, TreeNode*& head, TreeNode*& prev) {
        if (!root) return;
        Convert_To_DLL(root->left, head, prev);
        if (!prev) {
            head = root;
        } else {
            root->left = prev;
            prev->right = root;
        }
        prev = root;
        Convert_To_DLL(root->right, head, prev);
    }

    void Convert_Inplace(TreeNode* root, TreeNode*& head, TreeNode*& prev) {
        if (!root) return;
        Convert_Inplace(root->left, head, prev);
        if (!head) {
            head = root;
        } else {
            prev->right = root;
            root->left = prev;
        }
        prev = root;
        Convert_Inplace(root->right, head, prev);
    }
};

void Print_DLL(TreeNode* head) {
    TreeNode* curr = head;
    while (curr) {
        cout << curr->data << " ";
        curr = curr->right;
    }
    cout << endl;
}

void Test_Binary_Tree_To_DLL() {
    Solution solution;
    
    vector<int> vals1 = {10, 12, 15, 25, 30, 36};
    TreeNode* root1 = Create_Tree(vals1);
    cout << "Test 1 - Inorder: ";
    TreeNode* head1 = solution.Binary_Tree_To_DLL_Inorder(root1);
    Print_DLL(head1);
    
    vector<int> vals2 = {10, 12, 15, 25, 30, 36};
    TreeNode* root2 = Create_Tree(vals2);
    cout << "Test 1 - Inplace: ";
    TreeNode* head2 = solution.Binary_Tree_To_DLL_Inplace(root2);
    Print_DLL(head2);
}

int main() {
    Test_Binary_Tree_To_DLL();
    return 0;
}
