/*
Problem: Find Two Non-Repeating Elements
URL: https://practice.geeksforgeeks.org/problems/finding-the-numbers0702/1

Problem Statement:
Given an array where every element appears twice except two elements, find those two unique elements.

Sample Input/Output:
Input: [2,4,7,9,2,4]
Output: {7,9}

Input: [1,1,2,3,3,4,4,5]
Output: {2,5}
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    pair<int, int> Non_Repeating_XOR(vector<int>& nums) {
        /*
        XOR approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int xor_all = 0;
        for (int num : nums) {
            xor_all ^= num;
        }
        
        int rightmost_set_bit = xor_all & (-xor_all);
        
        int group1 = 0, group2 = 0;
        for (int num : nums) {
            if (num & rightmost_set_bit) {
                group1 ^= num;
            } else {
                group2 ^= num;
            }
        }
        
        return {min(group1, group2), max(group1, group2)};
    }

    pair<int, int> Non_Repeating_Hash(vector<int>& nums) {
        /*
        Frequency map approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        unordered_map<int, int> freq;
        for (int num : nums) {
            freq[num]++;
        }
        
        vector<int> result;
        for (auto& p : freq) {
            if (p.second == 1) {
                result.push_back(p.first);
            }
        }
        
        return {min(result[0], result[1]), max(result[0], result[1])};
    }
};

void Test_Non_Repeating_Elements() {
    Solution solution;
    
    vector<int> test1 = {2, 4, 7, 9, 2, 4};
    pair<int, int> result1 = solution.Non_Repeating_XOR(test1);
    cout << "Test 1 XOR: [" << test1[0] << "," << test1[1] << "," << test1[2] << "," << test1[3] << "," << test1[4] << "," << test1[5] << "] -> {" << result1.first << "," << result1.second << "} (expected: {7,9})" << endl;
    
    pair<int, int> result1_hash = solution.Non_Repeating_Hash(test1);
    cout << "Test 1 Hash: [" << test1[0] << "," << test1[1] << "," << test1[2] << "," << test1[3] << "," << test1[4] << "," << test1[5] << "] -> {" << result1_hash.first << "," << result1_hash.second << "} (expected: {7,9})" << endl;
    
    vector<int> test2 = {1, 1, 2, 3, 3, 4, 4, 5};
    pair<int, int> result2 = solution.Non_Repeating_XOR(test2);
    cout << "Test 2 XOR: [" << test2[0] << "," << test2[1] << "," << test2[2] << "," << test2[3] << "," << test2[4] << "," << test2[5] << "," << test2[6] << "," << test2[7] << "] -> {" << result2.first << "," << result2.second << "} (expected: {2,5})" << endl;
    
    pair<int, int> result2_hash = solution.Non_Repeating_Hash(test2);
    cout << "Test 2 Hash: [" << test2[0] << "," << test2[1] << "," << test2[2] << "," << test2[3] << "," << test2[4] << "," << test2[5] << "," << test2[6] << "," << test2[7] << "] -> {" << result2_hash.first << "," << result2_hash.second << "} (expected: {2,5})" << endl;
}

int main() {
    Test_Non_Repeating_Elements();
    return 0;
}
