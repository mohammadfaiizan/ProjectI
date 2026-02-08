/*
Problem: All Permutations of String
URL: https://practice.geeksforgeeks.org/problems/permutations-of-a-given-string2041/1

Problem Statement:
Print all permutations of a given string.

Sample Input/Output:
Input: str = "ABC"
Output: ABC ACB BAC BCA CAB CBA
Explanation: All permutations of ABC
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<string> All_Permutations_Swap_Based(string str) {
        /*
        Swap-based backtracking
        Time Complexity: O(n*n!)
        Space Complexity: O(n)
        */
        vector<string> result;
        
        function<void(string&, int)> backtrack = [&](string& s, int idx) {
            if (idx == s.length()) {
                result.push_back(s);
                return;
            }
            
            for (int i = idx; i < s.length(); i++) {
                swap(s[idx], s[i]);
                backtrack(s, idx + 1);
                swap(s[idx], s[i]);
            }
        };
        
        string temp = str;
        backtrack(temp, 0);
        return result;
    }
    
    vector<string> All_Permutations_Build_Exclude(string str) {
        /*
        Build by excluding characters
        Time Complexity: O(n*n!)
        Space Complexity: O(n)
        */
        vector<string> result;
        string current;
        vector<bool> used(str.length(), false);
        
        function<void()> backtrack = [&]() {
            if (current.length() == str.length()) {
                result.push_back(current);
                return;
            }
            
            for (int i = 0; i < str.length(); i++) {
                if (!used[i]) {
                    used[i] = true;
                    current.push_back(str[i]);
                    backtrack();
                    current.pop_back();
                    used[i] = false;
                }
            }
        };
        
        backtrack();
        return result;
    }
};

void Test_All_Permutations_String() {
    Solution solution;
    string str = "ABC";
    vector<string> result1 = solution.All_Permutations_Swap_Based(str);
    vector<string> result2 = solution.All_Permutations_Build_Exclude(str);
    
    cout << "Swap-Based Approach:" << endl;
    for (string& perm : result1) {
        cout << perm << " ";
    }
    cout << endl;
    
    cout << "Build-Exclude Approach:" << endl;
    for (string& perm : result2) {
        cout << perm << " ";
    }
    cout << endl;
}

int main() {
    Test_All_Permutations_String();
    return 0;
}
