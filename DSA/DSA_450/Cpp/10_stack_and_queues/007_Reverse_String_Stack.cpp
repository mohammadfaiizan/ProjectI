/*
Problem: Reverse a String using Stack
URL: https://practice.geeksforgeeks.org/problems/reverse-a-string-using-stack/1

Problem Statement:
Reverse a string using stack data structure. Push all characters to stack then pop them back.

Sample Input/Output:
Input: "hello"
Output: "olleh"
Input: "abc"
Output: "cba"
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string Reverse_Stack(string s) {
        stack<char> st;
        for (char c : s) {
            st.push(c);
        }
        string result = "";
        while (!st.empty()) {
            result += st.top();
            st.pop();
        }
        return result;
    }

    string Reverse_TwoPointer(string s) {
        int left = 0;
        int right = s.length() - 1;
        while (left < right) {
            swap(s[left], s[right]);
            left++;
            right--;
        }
        return s;
    }
};

void Test_Reverse_String_Stack() {
    Solution solution;
    cout << "Reverse String Stack Tests:" << endl;
    
    cout << "\"hello\" -> \"" << solution.Reverse_Stack("hello") << "\"" << endl;
    cout << "\"abc\" -> \"" << solution.Reverse_Stack("abc") << "\"" << endl;
    cout << "\"\" -> \"" << solution.Reverse_Stack("") << "\"" << endl;
    cout << "\"a\" -> \"" << solution.Reverse_Stack("a") << "\"" << endl;
    cout << "\"racecar\" -> \"" << solution.Reverse_Stack("racecar") << "\"" << endl;
    
    cout << "\nTwo Pointer Comparison:" << endl;
    cout << "\"hello\" -> \"" << solution.Reverse_TwoPointer("hello") << "\"" << endl;
    cout << "\"abc\" -> \"" << solution.Reverse_TwoPointer("abc") << "\"" << endl;
}

int main() {
    Test_Reverse_String_Stack();
    return 0;
}
