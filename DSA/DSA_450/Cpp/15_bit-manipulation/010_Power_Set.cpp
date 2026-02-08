/*
Problem: Power Set (Generate All Subsets)
URL: https://practice.geeksforgeeks.org/problems/power-set4302/1

Problem Statement:
Given a string/array, generate all possible subsets using bit manipulation.

Sample Input/Output:
Input: "abc"
Output: ["","a","b","ab","c","ac","bc","abc"]

Input: [1,2,3]
Output: All subsets
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<string> Power_Set_Bitmask(string s) {
        /*
        Iterate 0 to 2^n - 1, include element if bit is set
        Time Complexity: O(2^n * n)
        Space Complexity: O(2^n * n)
        */
        int n = s.length();
        vector<string> result;
        for (int i = 0; i < (1 << n); i++) {
            string subset = "";
            for (int j = 0; j < n; j++) {
                if (i & (1 << j)) {
                    subset += s[j];
                }
            }
            result.push_back(subset);
        }
        return result;
    }

    void Power_Set_Recursive_Helper(string s, int index, string current, vector<string>& result) {
        if (index == s.length()) {
            result.push_back(current);
            return;
        }
        Power_Set_Recursive_Helper(s, index + 1, current, result);
        Power_Set_Recursive_Helper(s, index + 1, current + s[index], result);
    }

    vector<string> Power_Set_Recursive(string s) {
        /*
        Recursive include/exclude
        Time Complexity: O(2^n * n)
        Space Complexity: O(2^n * n)
        */
        vector<string> result;
        Power_Set_Recursive_Helper(s, 0, "", result);
        return result;
    }

    vector<vector<int>> Power_Set_Bitmask_Array(vector<int>& arr) {
        /*
        Iterate 0 to 2^n - 1, include element if bit is set
        Time Complexity: O(2^n * n)
        Space Complexity: O(2^n * n)
        */
        int n = arr.size();
        vector<vector<int>> result;
        for (int i = 0; i < (1 << n); i++) {
            vector<int> subset;
            for (int j = 0; j < n; j++) {
                if (i & (1 << j)) {
                    subset.push_back(arr[j]);
                }
            }
            result.push_back(subset);
        }
        return result;
    }
};

void Test_Power_Set() {
    Solution solution;
    
    cout << "Testing Power_Set_Bitmask:" << endl;
    string s = "abc";
    vector<string> result1 = solution.Power_Set_Bitmask(s);
    cout << "Input: \"abc\"" << endl;
    cout << "Output: ";
    for (int i = 0; i < result1.size(); i++) {
        cout << "\"" << result1[i] << "\"";
        if (i < result1.size() - 1) cout << ", ";
    }
    cout << endl;
    
    cout << "\nTesting Power_Set_Recursive:" << endl;
    vector<string> result2 = solution.Power_Set_Recursive(s);
    cout << "Input: \"abc\"" << endl;
    cout << "Output: ";
    for (int i = 0; i < result2.size(); i++) {
        cout << "\"" << result2[i] << "\"";
        if (i < result2.size() - 1) cout << ", ";
    }
    cout << endl;
    
    cout << "\nTesting Power_Set_Bitmask_Array:" << endl;
    vector<int> arr = {1, 2, 3};
    vector<vector<int>> result3 = solution.Power_Set_Bitmask_Array(arr);
    cout << "Input: [1,2,3]" << endl;
    cout << "Output: ";
    for (int i = 0; i < result3.size(); i++) {
        cout << "[";
        for (int j = 0; j < result3[i].size(); j++) {
            cout << result3[i][j];
            if (j < result3[i].size() - 1) cout << ",";
        }
        cout << "]";
        if (i < result3.size() - 1) cout << ", ";
    }
    cout << endl;
}

int main() {
    Test_Power_Set();
    return 0;
}
