/*
Problem: Isomorphic Strings
URL: https://practice.geeksforgeeks.org/problems/isomorphic-strings-1587115620/1

Problem Statement:
Two strings str1 and str2 are called isomorphic if there is a one-to-one mapping
possible for every character of str1 to every character of str2.

Sample Input/Output:
Input: str1 = "egg", str2 = "add"
Output: true (e->a, g->d)

Input: str1 = "foo", str2 = "bar"
Output: false
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Isomorphic_Two_Maps(string str1, string str2) {
        /*
        Two hashmaps for bidirectional mapping
        Time Complexity: O(n)
        Space Complexity: O(k) where k = unique chars
        */
        if (str1.size() != str2.size()) return false;
        unordered_map<char, char> mp1, mp2;

        for (int i = 0; i < (int)str1.size(); i++) {
            if (mp1.find(str1[i]) == mp1.end()) mp1[str1[i]] = str2[i];
            if (mp2.find(str2[i]) == mp2.end()) mp2[str2[i]] = str1[i];

            if (mp1[str1[i]] != str2[i] || mp2[str2[i]] != str1[i])
                return false;
        }
        return true;
    }

    bool Isomorphic_Array(string str1, string str2) {
        /*
        Using array for mapping + marked array
        Time Complexity: O(n)
        Space Complexity: O(1) - fixed 256
        */
        if (str1.size() != str2.size()) return false;
        int map1[256], map2[256];
        memset(map1, -1, sizeof(map1));
        memset(map2, -1, sizeof(map2));

        for (int i = 0; i < (int)str1.size(); i++) {
            if (map1[(int)str1[i]] == -1 && map2[(int)str2[i]] == -1) {
                map1[(int)str1[i]] = str2[i];
                map2[(int)str2[i]] = str1[i];
            } else if (map1[(int)str1[i]] != str2[i] || map2[(int)str2[i]] != str1[i]) {
                return false;
            }
        }
        return true;
    }

    bool Isomorphic_Transform(string str1, string str2) {
        /*
        Transform both strings to canonical form and compare
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (str1.size() != str2.size()) return false;

        auto Transform = [](string& s) -> string {
            unordered_map<char, int> mp;
            string result = "";
            int counter = 0;
            for (char c : s) {
                if (mp.find(c) == mp.end()) mp[c] = counter++;
                result += to_string(mp[c]) + " ";
            }
            return result;
        };

        return Transform(str1) == Transform(str2);
    }
};

void Test_Isomorphic_Strings() {
    Solution sol;
    struct TestCase { string s1, s2; };
    vector<TestCase> tests = {
        {"egg", "add"},
        {"foo", "bar"},
        {"paper", "title"},
        {"ab", "aa"},
        {"abc", "abc"},
        {"", ""}
    };

    for (auto& t : tests) {
        cout << "s1: \"" << t.s1 << "\", s2: \"" << t.s2 << "\"" << endl;
        cout << "Two Maps: " << (sol.Isomorphic_Two_Maps(t.s1, t.s2) ? "true" : "false") << endl;
        cout << "Array: " << (sol.Isomorphic_Array(t.s1, t.s2) ? "true" : "false") << endl;
        cout << "Transform: " << (sol.Isomorphic_Transform(t.s1, t.s2) ? "true" : "false") << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Isomorphic_Strings();
    return 0;
}
