/*
Problem: Palindrome Partitioning
URL: https://www.geeksforgeeks.org/given-a-string-print-all-possible-palindromic-partition/

Problem Statement:
Given a string, find all possible palindromic partitions.

Sample Input/Output:
Input: s="aab"
Output: [["a","a","b"],["aa","b"]]
Explanation: Two ways to partition into palindromes
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<string>> Palindrome_Partitioning_Backtracking(string s) {
        /*
        Backtracking with palindrome check
        Time Complexity: O(n * 2^n)
        Space Complexity: O(n)
        */
        vector<vector<string>> result;
        vector<string> current_partition;
        
        function<bool(string)> Is_Palindrome = [&](string str) {
            int left = 0, right = str.length() - 1;
            while (left < right) {
                if (str[left++] != str[right--]) {
                    return false;
                }
            }
            return true;
        };
        
        function<void(int)> backtrack = [&](int start) {
            if (start == s.length()) {
                result.push_back(current_partition);
                return;
            }
            
            for (int end = start + 1; end <= s.length(); end++) {
                string substring = s.substr(start, end - start);
                if (Is_Palindrome(substring)) {
                    current_partition.push_back(substring);
                    backtrack(end);
                    current_partition.pop_back();
                }
            }
        };
        
        backtrack(0);
        return result;
    }
};

void Test_Palindrome_Partitioning() {
    Solution solution;
    
    string s = "aab";
    vector<vector<string>> partitions = solution.Palindrome_Partitioning_Backtracking(s);
    
    cout << "Palindromic partitions:" << endl;
    for (const auto &partition : partitions) {
        for (const string &str : partition) {
            cout << str << " ";
        }
        cout << endl;
    }
}

int main() {
    Test_Palindrome_Partitioning();
    return 0;
}
