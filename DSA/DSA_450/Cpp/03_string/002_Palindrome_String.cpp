/*
Problem: Palindrome String
URL: https://practice.geeksforgeeks.org/problems/palindrome-string0817/1

Problem Statement:
Given a string S, check if it is palindrome or not.

Sample Input/Output:
Input: S = "abba"
Output: 1

Input: S = "abc"
Output: 0
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Is_Palindrome_Two_Pointer(string s) {
        /*
        Two Pointer - compare from both ends
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int len = s.size();
        for (int i = 0; i < len / 2; i++) {
            if (s[i] != s[len - i - 1]) return 0;
        }
        return 1;
    }

    int Is_Palindrome_Reverse(string s) {
        /*
        Reverse and compare
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        string temp = s;
        reverse(temp.begin(), temp.end());
        return (temp == s);
    }

    int Is_Palindrome_Recursive(string& s, int left, int right) {
        /*
        Recursive check
        Time Complexity: O(n)
        Space Complexity: O(n) recursion stack
        */
        if (left >= right) return 1;
        if (s[left] != s[right]) return 0;
        return Is_Palindrome_Recursive(s, left + 1, right - 1);
    }
};

void Test_Palindrome_String() {
    Solution sol;
    vector<string> tests = {"abba", "abc", "a", "aa", "racecar", "abcba", "abcd"};

    for (auto& s : tests) {
        cout << "Input: " << s << endl;
        cout << "Two Pointer: " << sol.Is_Palindrome_Two_Pointer(s) << endl;
        cout << "Reverse: " << sol.Is_Palindrome_Reverse(s) << endl;
        cout << "Recursive: " << sol.Is_Palindrome_Recursive(s, 0, s.size() - 1) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Palindrome_String();
    return 0;
}
