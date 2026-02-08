/*
Problem: Kth Permutation Sequence
URL: https://leetcode.com/problems/permutation-sequence/

Problem Statement:
Given N and K, find the Kth permutation sequence of numbers 1 to N.

Sample Input/Output:
Input: N = 3, K = 3
Output: "213"
Explanation: Permutations: "123", "132", "213", "231", "312", "321"
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string Kth_Permutation_Math_Based(int n, int k) {
        /*
        Math-based factorial number system
        Time Complexity: O(N^2)
        Space Complexity: O(N)
        */
        vector<int> factorial(n + 1, 1);
        for (int i = 1; i <= n; i++) {
            factorial[i] = factorial[i - 1] * i;
        }
        
        vector<int> numbers;
        for (int i = 1; i <= n; i++) {
            numbers.push_back(i);
        }
        
        string result;
        k--;
        
        for (int i = n; i >= 1; i--) {
            int idx = k / factorial[i - 1];
            result += to_string(numbers[idx]);
            numbers.erase(numbers.begin() + idx);
            k %= factorial[i - 1];
        }
        
        return result;
    }
    
    string Kth_Permutation_Generate_All(int n, int k) {
        /*
        Generate all permutations
        Time Complexity: O(N!)
        Space Complexity: O(N!)
        */
        vector<int> nums;
        for (int i = 1; i <= n; i++) {
            nums.push_back(i);
        }
        
        vector<string> permutations;
        
        function<void(vector<int>&, int)> backtrack = [&](vector<int>& arr, int idx) {
            if (idx == arr.size()) {
                string perm;
                for (int num : arr) {
                    perm += to_string(num);
                }
                permutations.push_back(perm);
                return;
            }
            
            for (int i = idx; i < arr.size(); i++) {
                swap(arr[idx], arr[i]);
                backtrack(arr, idx + 1);
                swap(arr[idx], arr[i]);
                
                if (permutations.size() >= k) return;
            }
        };
        
        backtrack(nums, 0);
        return permutations[k - 1];
    }
};

void Test_Kth_Permutation_Sequence() {
    Solution solution;
    int n = 3, k = 3;
    cout << "Math-Based Approach: " << solution.Kth_Permutation_Math_Based(n, k) << endl;
    cout << "Generate All Approach: " << solution.Kth_Permutation_Generate_All(n, k) << endl;
}

int main() {
    Test_Kth_Permutation_Sequence();
    return 0;
}
