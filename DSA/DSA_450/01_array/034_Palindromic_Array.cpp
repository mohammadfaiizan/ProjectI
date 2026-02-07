/*
Problem: Palindromic Array
URL: https://practice.geeksforgeeks.org/problems/palindromic-array-1587115620/1

Problem Statement:
Given a positive integer array arr of size N, check if every element of the array
is a palindrome or not. Return 1 if all elements are palindromes, otherwise return 0.

Sample Input/Output:
Input: arr = [111, 222, 333, 444, 555]
Output: 1
Explanation: All elements are palindromes.

Input: arr = [121, 131, 20]
Output: 0
Explanation: 20 is not a palindrome.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Palindromic_Array_String_Optimal(vector<int>& arr) {
        /*
        String Conversion - Convert each number to string and check
        Time Complexity: O(n * d) where d is max digits
        Space Complexity: O(d)
        */
        for (int x : arr) {
            string s = to_string(x);
            int i = 0, j = s.size() - 1;
            while (i < j) {
                if (s[i] != s[j]) return 0;
                i++;
                j--;
            }
        }
        return 1;
    }

    int Palindromic_Array_Digit_Reversal(vector<int>& arr) {
        /*
        Digit Reversal - Reverse digits mathematically and compare
        Time Complexity: O(n * d)
        Space Complexity: O(1)
        */
        for (int x : arr) {
            if (!Is_Palindrome_Number(x)) return 0;
        }
        return 1;
    }

private:
    bool Is_Palindrome_Number(int n) {
        int original = n, reversed = 0;
        while (n > 0) {
            reversed = reversed * 10 + n % 10;
            n /= 10;
        }
        return original == reversed;
    }
};

void Test_Palindromic_Array() {
    Solution solution;

    struct TestCase {
        vector<int> arr;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{111, 222, 333, 444, 555}, 1},
        {{121, 131, 20}, 0},
        {{1, 2, 3, 4, 5}, 1},
        {{12321, 45654, 78987}, 1}
    };

    for (auto& tc : test_cases) {
        cout << "Array: ";
        for (int x : tc.arr) cout << x << " ";
        cout << ", Expected: " << tc.expected << endl;

        cout << "String: " << solution.Palindromic_Array_String_Optimal(tc.arr) << endl;
        cout << "Digit Reversal: " << solution.Palindromic_Array_Digit_Reversal(tc.arr) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Palindromic_Array();
    return 0;
}
