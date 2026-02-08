/*
Problem: Divide Two Integers Without Using Multiplication, Division, or Mod
URL: https://leetcode.com/problems/divide-two-integers/

Problem Statement:
Given dividend and divisor, compute quotient without *, /, %.

Sample Input/Output:
Input: 43, 8
Output: 5

Input: 10, 3
Output: 3

Input: -7, 2
Output: -3

Input: INT_MIN, -1
Output: INT_MAX
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Divide_Bit_Shift(int dividend, int divisor) {
        /*
        Double divisor using left shift until > dividend, subtract and accumulate
        Time Complexity: O(log^2 n)
        Space Complexity: O(1)
        */
        if (divisor == 0) return INT_MAX;
        if (dividend == INT_MIN && divisor == -1) return INT_MAX;
        
        bool negative = (dividend < 0) ^ (divisor < 0);
        long long dvd = abs((long long)dividend);
        long long dvs = abs((long long)divisor);
        
        int result = 0;
        while (dvd >= dvs) {
            long long temp = dvs;
            int multiple = 1;
            while (dvd >= (temp << 1)) {
                temp <<= 1;
                multiple <<= 1;
            }
            dvd -= temp;
            result += multiple;
        }
        
        return negative ? -result : result;
    }

    int Divide_Subtract(int dividend, int divisor) {
        /*
        Repeated subtraction
        Time Complexity: O(dividend/divisor)
        Space Complexity: O(1)
        */
        if (divisor == 0) return INT_MAX;
        if (dividend == INT_MIN && divisor == -1) return INT_MAX;
        
        bool negative = (dividend < 0) ^ (divisor < 0);
        long long dvd = abs((long long)dividend);
        long long dvs = abs((long long)divisor);
        
        int result = 0;
        while (dvd >= dvs) {
            dvd -= dvs;
            result++;
        }
        
        return negative ? -result : result;
    }
};

void Test_Divide_Without_Operators() {
    Solution solution;
    
    cout << "Testing Divide_Bit_Shift:" << endl;
    cout << "43 / 8 -> " << solution.Divide_Bit_Shift(43, 8) << " (expected: 5)" << endl;
    cout << "10 / 3 -> " << solution.Divide_Bit_Shift(10, 3) << " (expected: 3)" << endl;
    cout << "-7 / 2 -> " << solution.Divide_Bit_Shift(-7, 2) << " (expected: -3)" << endl;
    cout << "INT_MIN / -1 -> " << solution.Divide_Bit_Shift(INT_MIN, -1) << " (expected: INT_MAX)" << endl;
    
    cout << "\nTesting Divide_Subtract:" << endl;
    cout << "43 / 8 -> " << solution.Divide_Subtract(43, 8) << " (expected: 5)" << endl;
    cout << "10 / 3 -> " << solution.Divide_Subtract(10, 3) << " (expected: 3)" << endl;
    cout << "-7 / 2 -> " << solution.Divide_Subtract(-7, 2) << " (expected: -3)" << endl;
    cout << "INT_MIN / -1 -> " << solution.Divide_Subtract(INT_MIN, -1) << " (expected: INT_MAX)" << endl;
}

int main() {
    Test_Divide_Without_Operators();
    return 0;
}
