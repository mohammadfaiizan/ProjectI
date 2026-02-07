/*
Problem: Factorial of a Large Number
URL: https://practice.geeksforgeeks.org/problems/factorials-of-large-numbers2508/1

Problem Statement:
Given an integer N, find its factorial. The factorial can be very large,
so return the result as a vector of digits.

Sample Input/Output:
Input: N = 5
Output: [1, 2, 0]
Explanation: 5! = 120.

Input: N = 10
Output: [3, 6, 2, 8, 8, 0, 0]
Explanation: 10! = 3628800.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Factorial_Array_Multiplication_Optimal(int n) {
        /*
        Array Multiplication - Multiply digit by digit with carry
        Time Complexity: O(n * digits)
        Space Complexity: O(digits)
        */
        vector<int> result = {1};
        int size = 1;
        for (int i = 2; i <= n; i++) {
            int carry = 0;
            for (int j = 0; j < size; j++) {
                int prod = result[j] * i + carry;
                result[j] = prod % 10;
                carry = prod / 10;
            }
            while (carry) {
                result.push_back(carry % 10);
                carry /= 10;
                size++;
            }
        }
        reverse(result.begin(), result.end());
        return result;
    }

    vector<int> Factorial_String_Multiplication(int n) {
        /*
        String Based - Use string to handle large number multiplication
        Time Complexity: O(n * digits)
        Space Complexity: O(digits)
        */
        string result = "1";
        for (int i = 2; i <= n; i++) {
            result = Multiply_String(result, i);
        }
        vector<int> digits;
        for (char c : result) digits.push_back(c - '0');
        return digits;
    }

private:
    string Multiply_String(string num, int x) {
        int carry = 0;
        for (int i = num.size() - 1; i >= 0; i--) {
            int prod = (num[i] - '0') * x + carry;
            num[i] = (prod % 10) + '0';
            carry = prod / 10;
        }
        string prefix = "";
        while (carry) {
            prefix = char(carry % 10 + '0') + prefix;
            carry /= 10;
        }
        return prefix + num;
    }
};

void Test_Factorial_Large_Number() {
    Solution solution;

    vector<int> test_cases = {5, 10, 20, 25};

    for (int n : test_cases) {
        cout << "N=" << n << endl;

        auto r1 = solution.Factorial_Array_Multiplication_Optimal(n);
        cout << "Array: ";
        for (int d : r1) cout << d;
        cout << endl;

        auto r2 = solution.Factorial_String_Multiplication(n);
        cout << "String: ";
        for (int d : r2) cout << d;
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Factorial_Large_Number();
    return 0;
}
