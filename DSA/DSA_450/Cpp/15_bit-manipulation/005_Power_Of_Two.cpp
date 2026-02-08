/*
Problem: Check if Number is Power of Two
URL: https://practice.geeksforgeeks.org/problems/power-of-2-1587115620/1

Problem Statement:
Check if a given positive number is a power of 2.

Sample Input/Output:
Input: 1
Output: true

Input: 16
Output: true

Input: 18
Output: false

Input: 0
Output: false
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Power_Two_Bit_Trick(int n) {
        /*
        Bit trick: n > 0 && (n & (n-1)) == 0
        Time Complexity: O(1)
        Space Complexity: O(1)
        */
        return n > 0 && (n & (n - 1)) == 0;
    }

    bool Power_Two_Count_Bits(int n) {
        /*
        Count set bits, must be exactly 1
        Time Complexity: O(log n)
        Space Complexity: O(1)
        */
        if (n <= 0) return false;
        int count = 0;
        while (n) {
            n &= (n - 1);
            count++;
        }
        return count == 1;
    }

    bool Power_Two_Log(int n) {
        /*
        Use log2
        Time Complexity: O(1)
        Space Complexity: O(1)
        */
        if (n <= 0) return false;
        double log_val = log2(n);
        return log_val == floor(log_val);
    }
};

void Test_Power_Of_Two() {
    Solution solution;
    
    cout << "Testing Power_Two_Bit_Trick:" << endl;
    cout << "1 -> " << (solution.Power_Two_Bit_Trick(1) ? "true" : "false") << " (expected: true)" << endl;
    cout << "16 -> " << (solution.Power_Two_Bit_Trick(16) ? "true" : "false") << " (expected: true)" << endl;
    cout << "18 -> " << (solution.Power_Two_Bit_Trick(18) ? "true" : "false") << " (expected: false)" << endl;
    cout << "0 -> " << (solution.Power_Two_Bit_Trick(0) ? "true" : "false") << " (expected: false)" << endl;
    
    cout << "\nTesting Power_Two_Count_Bits:" << endl;
    cout << "1 -> " << (solution.Power_Two_Count_Bits(1) ? "true" : "false") << " (expected: true)" << endl;
    cout << "16 -> " << (solution.Power_Two_Count_Bits(16) ? "true" : "false") << " (expected: true)" << endl;
    cout << "18 -> " << (solution.Power_Two_Count_Bits(18) ? "true" : "false") << " (expected: false)" << endl;
    cout << "0 -> " << (solution.Power_Two_Count_Bits(0) ? "true" : "false") << " (expected: false)" << endl;
    
    cout << "\nTesting Power_Two_Log:" << endl;
    cout << "1 -> " << (solution.Power_Two_Log(1) ? "true" : "false") << " (expected: true)" << endl;
    cout << "16 -> " << (solution.Power_Two_Log(16) ? "true" : "false") << " (expected: true)" << endl;
    cout << "18 -> " << (solution.Power_Two_Log(18) ? "true" : "false") << " (expected: false)" << endl;
    cout << "0 -> " << (solution.Power_Two_Log(0) ? "true" : "false") << " (expected: false)" << endl;
}

int main() {
    Test_Power_Of_Two();
    return 0;
}
