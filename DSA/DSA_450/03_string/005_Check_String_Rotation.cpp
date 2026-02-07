/*
Problem: Check if Strings are Rotations of Each Other
URL: https://www.geeksforgeeks.org/a-program-to-check-if-strings-are-rotations-of-each-other/

Problem Statement:
Given two strings s1 and s2, check whether s2 is a rotation of s1.

Sample Input/Output:
Input: s1 = "AACD", s2 = "ACDA"
Output: YES

Input: s1 = "ABCD", s2 = "ACBD"
Output: NO
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Check_Rotation_Concatenation(string s1, string s2) {
        /*
        Concatenate s1 with itself and search for s2
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (s1.size() != s2.size()) return false;
        string temp = s1 + s1;
        return temp.find(s2) != string::npos;
    }

    bool Check_Rotation_One_By_One(string s1, string s2) {
        /*
        Try all rotations one by one
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        if (s1.size() != s2.size()) return false;
        int n = s1.size();
        for (int i = 0; i < n; i++) {
            string rotated = s1.substr(i) + s1.substr(0, i);
            if (rotated == s2) return true;
        }
        return false;
    }

    bool Check_Rotation_Queue(string s1, string s2) {
        /*
        Using queue - dequeue from front and enqueue to back
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        if (s1.size() != s2.size()) return false;
        queue<char> q1, q2;
        for (char c : s1) q1.push(c);
        for (char c : s2) q2.push(c);
        int n = s1.size();
        for (int i = 0; i < n; i++) {
            char front = q1.front();
            q1.pop();
            q1.push(front);
            if (q1 == q2) return true;
        }
        return false;
    }
};

void Test_Check_String_Rotation() {
    Solution sol;
    vector<pair<string, string>> tests = {
        {"AACD", "ACDA"},
        {"ABCD", "ACBD"},
        {"abcde", "cdeab"},
        {"abc", "abc"},
        {"abc", "ab"}
    };

    for (auto& [s1, s2] : tests) {
        cout << "s1: " << s1 << ", s2: " << s2 << endl;
        cout << "Concatenation: " << (sol.Check_Rotation_Concatenation(s1, s2) ? "YES" : "NO") << endl;
        cout << "One By One: " << (sol.Check_Rotation_One_By_One(s1, s2) ? "YES" : "NO") << endl;
        cout << "Queue: " << (sol.Check_Rotation_Queue(s1, s2) ? "YES" : "NO") << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Check_String_Rotation();
    return 0;
}
