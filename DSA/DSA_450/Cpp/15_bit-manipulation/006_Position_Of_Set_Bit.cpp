/*
Problem: Find Position of the Only Set Bit
URL: https://practice.geeksforgeeks.org/problems/find-position-of-set-bit3706/1

Problem Statement:
If a number has exactly one set bit, return its position (1-indexed). Otherwise return -1.

Sample Input/Output:
Input: 2
Output: 2

Input: 5
Output: -1

Input: 32
Output: 6

Input: 0
Output: -1
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Position_Bit_Log(int n) {
        /*
        Use log2 to find position
        Time Complexity: O(1)
        Space Complexity: O(1)
        */
        if (n == 0 || (n & (n - 1)) != 0) {
            return -1;
        }
        return (int)log2(n) + 1;
    }

    int Position_Bit_Loop(int n) {
        /*
        Shift right and count position
        Time Complexity: O(log n)
        Space Complexity: O(1)
        */
        if (n == 0) return -1;
        if ((n & (n - 1)) != 0) return -1;
        int pos = 0;
        while (n) {
            pos++;
            n >>= 1;
        }
        return pos;
    }

    int Position_Bit_Power_Check(int n) {
        /*
        First check power of 2, then find position
        Time Complexity: O(log n)
        Space Complexity: O(1)
        */
        if (n == 0) return -1;
        if ((n & (n - 1)) != 0) return -1;
        int pos = 1;
        while (n != 1) {
            n >>= 1;
            pos++;
        }
        return pos;
    }
};

void Test_Position_Of_Set_Bit() {
    Solution solution;
    
    cout << "Testing Position_Bit_Log:" << endl;
    cout << "2 -> " << solution.Position_Bit_Log(2) << " (expected: 2)" << endl;
    cout << "5 -> " << solution.Position_Bit_Log(5) << " (expected: -1)" << endl;
    cout << "32 -> " << solution.Position_Bit_Log(32) << " (expected: 6)" << endl;
    cout << "0 -> " << solution.Position_Bit_Log(0) << " (expected: -1)" << endl;
    
    cout << "\nTesting Position_Bit_Loop:" << endl;
    cout << "2 -> " << solution.Position_Bit_Loop(2) << " (expected: 2)" << endl;
    cout << "5 -> " << solution.Position_Bit_Loop(5) << " (expected: -1)" << endl;
    cout << "32 -> " << solution.Position_Bit_Loop(32) << " (expected: 6)" << endl;
    cout << "0 -> " << solution.Position_Bit_Loop(0) << " (expected: -1)" << endl;
    
    cout << "\nTesting Position_Bit_Power_Check:" << endl;
    cout << "2 -> " << solution.Position_Bit_Power_Check(2) << " (expected: 2)" << endl;
    cout << "5 -> " << solution.Position_Bit_Power_Check(5) << " (expected: -1)" << endl;
    cout << "32 -> " << solution.Position_Bit_Power_Check(32) << " (expected: 6)" << endl;
    cout << "0 -> " << solution.Position_Bit_Power_Check(0) << " (expected: -1)" << endl;
}

int main() {
    Test_Position_Of_Set_Bit();
    return 0;
}
