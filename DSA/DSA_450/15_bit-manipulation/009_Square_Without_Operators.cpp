/*
Problem: Calculate Square of a Number Without *, /, pow
URL: https://www.geeksforgeeks.org/calculate-square-of-a-number-without-using-and-pow/

Problem Statement:
Calculate n^2 without using multiplication, division, or pow.

Sample Input/Output:
Input: 5
Output: 25

Input: -7
Output: 49

Input: 0
Output: 0

Input: 12
Output: 144
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Square_Bit_Shift(int n) {
        /*
        For each set bit i in |n|, add n << i; works because n*n = n * sum(2^i for set bits)
        Time Complexity: O(log n)
        Space Complexity: O(1)
        */
        if (n == 0) return 0;
        int num = abs(n);
        int result = 0;
        int temp = num;
        int i = 0;
        while (num) {
            if (num & 1) {
                result += (temp << i);
            }
            num >>= 1;
            i++;
        }
        return result;
    }

    int Square_Odd_Sum(int n) {
        /*
        n^2 = sum of first n odd numbers
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (n == 0) return 0;
        int num = abs(n);
        int result = 0;
        int odd = 1;
        for (int i = 0; i < num; i++) {
            result += odd;
            odd += 2;
        }
        return result;
    }
};

void Test_Square_Without_Operators() {
    Solution solution;
    
    cout << "Testing Square_Bit_Shift:" << endl;
    cout << "5 -> " << solution.Square_Bit_Shift(5) << " (expected: 25)" << endl;
    cout << "-7 -> " << solution.Square_Bit_Shift(-7) << " (expected: 49)" << endl;
    cout << "0 -> " << solution.Square_Bit_Shift(0) << " (expected: 0)" << endl;
    cout << "12 -> " << solution.Square_Bit_Shift(12) << " (expected: 144)" << endl;
    
    cout << "\nTesting Square_Odd_Sum:" << endl;
    cout << "5 -> " << solution.Square_Odd_Sum(5) << " (expected: 25)" << endl;
    cout << "-7 -> " << solution.Square_Odd_Sum(-7) << " (expected: 49)" << endl;
    cout << "0 -> " << solution.Square_Odd_Sum(0) << " (expected: 0)" << endl;
    cout << "12 -> " << solution.Square_Odd_Sum(12) << " (expected: 144)" << endl;
}

int main() {
    Test_Square_Without_Operators();
    return 0;
}
