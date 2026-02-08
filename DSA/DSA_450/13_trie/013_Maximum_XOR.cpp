/*
Problem: Maximum XOR of Two Numbers in an Array
URL: https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/

Problem Statement:
Given an integer array nums, return the maximum result of nums[i] XOR nums[j],
where 0 <= i <= j < n.

Sample Input/Output:
Input: nums = [3, 10, 5, 25, 2, 8]
Output: 28
Explanation: The maximum XOR is 5 XOR 25 = 28.

Input: nums = [14, 70, 53, 83, 49, 91, 36, 80, 92, 51, 66, 70]
Output: 127
*/

#include <bits/stdc++.h>
using namespace std;

struct TrieNode {
    TrieNode* children[2];
    TrieNode() {
        children[0] = nullptr;
        children[1] = nullptr;
    }
};

class BinaryTrie {
public:
    TrieNode* root;

    BinaryTrie() {
        root = new TrieNode();
    }

    void Insert(int num) {
        TrieNode* curr = root;
        for (int i = 31; i >= 0; i--) {
            int bit = (num >> i) & 1;
            if (!curr->children[bit])
                curr->children[bit] = new TrieNode();
            curr = curr->children[bit];
        }
    }

    int Get_Max_XOR(int num) {
        TrieNode* curr = root;
        int maxXor = 0;
        for (int i = 31; i >= 0; i--) {
            int bit = (num >> i) & 1;
            int oppBit = 1 - bit;
            if (curr->children[oppBit]) {
                maxXor |= (1 << i);
                curr = curr->children[oppBit];
            } else {
                curr = curr->children[bit];
            }
        }
        return maxXor;
    }
};

class Solution {
public:
    int Max_XOR_Binary_Trie(vector<int>& nums) {
        /*
        Binary Trie - Insert all, then greedily pick opposite bits
        Time Complexity: O(32 * n)
        Space Complexity: O(32 * n)
        */
        BinaryTrie trie;
        for (int num : nums)
            trie.Insert(num);
        int result = 0;
        for (int num : nums)
            result = max(result, trie.Get_Max_XOR(num));
        return result;
    }

    int Max_XOR_Brute(vector<int>& nums) {
        /*
        Brute Force - Check all pairs
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        int result = 0;
        int n = nums.size();
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                result = max(result, nums[i] ^ nums[j]);
        return result;
    }
};

void Test_Maximum_XOR() {
    Solution solution;

    vector<int> nums1 = {3, 10, 5, 25, 2, 8};
    cout << "Array: [3,10,5,25,2,8]" << endl;
    cout << "Binary Trie: " << solution.Max_XOR_Binary_Trie(nums1) << endl;
    cout << "Brute Force: " << solution.Max_XOR_Brute(nums1) << endl;
    cout << string(50, '-') << endl;

    vector<int> nums2 = {14, 70, 53, 83, 49, 91, 36, 80, 92, 51, 66, 70};
    cout << "Array: [14,70,53,83,49,91,36,80,92,51,66,70]" << endl;
    cout << "Binary Trie: " << solution.Max_XOR_Binary_Trie(nums2) << endl;
    cout << "Brute Force: " << solution.Max_XOR_Brute(nums2) << endl;
    cout << string(50, '-') << endl;

    vector<int> nums3 = {0};
    cout << "Array: [0]" << endl;
    cout << "Binary Trie: " << solution.Max_XOR_Binary_Trie(nums3) << endl;
    cout << "Brute Force: " << solution.Max_XOR_Brute(nums3) << endl;
    cout << string(50, '-') << endl;

    vector<int> nums4 = {1, 2, 3, 4, 5};
    cout << "Array: [1,2,3,4,5]" << endl;
    cout << "Binary Trie: " << solution.Max_XOR_Binary_Trie(nums4) << endl;
    cout << "Brute Force: " << solution.Max_XOR_Brute(nums4) << endl;
}

int main() {
    Test_Maximum_XOR();
    return 0;
}
