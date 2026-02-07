/*
Problem: Reverse String
URL: https://leetcode.com/problems/reverse-string/

Problem Statement:
Write a function that reverses a string. The input string is given as an array of characters.
You must do this by modifying the input array in-place with O(1) extra memory.

Sample Input/Output:
Input: s = ["h","e","l","l","o"]
Output: ["o","l","l","e","h"]

Input: s = ["H","a","n","n","a","h"]
Output: ["h","a","n","n","a","H"]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Reverse_String_Two_Pointer(vector<char>& s) {
        /*
        Two Pointer - swap from both ends towards center
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int left = 0, right = s.size() - 1;
        while (left < right) {
            swap(s[left], s[right]);
            left++;
            right--;
        }
    }

    void Reverse_String_STL(vector<char>& s) {
        /*
        Using STL reverse
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        reverse(s.begin(), s.end());
    }

    void Reverse_String_Recursive(vector<char>& s, int left, int right) {
        /*
        Recursive approach
        Time Complexity: O(n)
        Space Complexity: O(n) recursion stack
        */
        if (left >= right) return;
        swap(s[left], s[right]);
        Reverse_String_Recursive(s, left + 1, right - 1);
    }
};

void Test_Reverse_String() {
    Solution sol;
    vector<vector<char>> tests = {
        {'h','e','l','l','o'},
        {'H','a','n','n','a','h'},
        {'a'},
        {'a','b'}
    };

    for (auto& s : tests) {
        vector<char> s1 = s, s2 = s, s3 = s;
        cout << "Input: ";
        for (char c : s) cout << c;
        cout << endl;

        sol.Reverse_String_Two_Pointer(s1);
        cout << "Two Pointer: ";
        for (char c : s1) cout << c;
        cout << endl;

        sol.Reverse_String_STL(s2);
        cout << "STL: ";
        for (char c : s2) cout << c;
        cout << endl;

        sol.Reverse_String_Recursive(s3, 0, s3.size() - 1);
        cout << "Recursive: ";
        for (char c : s3) cout << c;
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Reverse_String();
    return 0;
}
