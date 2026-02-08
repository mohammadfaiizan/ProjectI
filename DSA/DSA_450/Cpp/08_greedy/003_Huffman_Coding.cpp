/*
Problem: Huffman Coding
URL: https://practice.geeksforgeeks.org/problems/huffman-encoding3345/1

Problem Statement:
Given a string S with distinct character frequencies, build a Huffman tree and generate Huffman codes for each character. Return the codes sorted by character.

Sample Input/Output:
Input: S = "abcdef", freq[] = {5, 9, 12, 13, 16, 45}
Output: {"0", "100", "101", "1100", "1101", "111"}
Explanation: Huffman tree built from frequencies, codes assigned based on path from root.
*/

#include <bits/stdc++.h>
using namespace std;

struct Node {
    char data;
    int freq;
    Node* left;
    Node* right;
    
    Node(char d, int f) : data(d), freq(f), left(nullptr), right(nullptr) {}
};

struct Compare {
    bool operator()(Node* a, Node* b) {
        return a->freq > b->freq;
    }
};

class Solution {
public:
    vector<string> Huffman_Codes_MinHeap(string S, vector<int>& f, int n) {
        /*
        Build Huffman tree using min-heap, traverse to generate codes
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        priority_queue<Node*, vector<Node*>, Compare> pq;
        
        for (int i = 0; i < n; i++) {
            pq.push(new Node(S[i], f[i]));
        }
        
        while (pq.size() > 1) {
            Node* left = pq.top();
            pq.pop();
            Node* right = pq.top();
            pq.pop();
            
            Node* merged = new Node('$', left->freq + right->freq);
            merged->left = left;
            merged->right = right;
            pq.push(merged);
        }
        
        vector<string> codes(n);
        string code = "";
        Generate_Codes(pq.top(), code, codes, S);
        
        return codes;
    }
    
private:
    void Generate_Codes(Node* root, string code, vector<string>& codes, string& S) {
        if (!root) return;
        
        if (root->data != '$') {
            int idx = S.find(root->data);
            codes[idx] = code;
            return;
        }
        
        Generate_Codes(root->left, code + "0", codes, S);
        Generate_Codes(root->right, code + "1", codes, S);
    }
};

void Test_Huffman_Coding() {
    Solution solution;
    string S = "abcdef";
    vector<int> f = {5, 9, 12, 13, 16, 45};
    vector<string> codes = solution.Huffman_Codes_MinHeap(S, f, S.length());
    cout << "Huffman codes:" << endl;
    for (int i = 0; i < codes.size(); i++) {
        cout << S[i] << ": " << codes[i] << endl;
    }
}

int main() {
    Test_Huffman_Coding();
    return 0;
}
