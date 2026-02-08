/*
Problem: Count Set Bits in a Number
URL: https://practice.geeksforgeeks.org/problems/set-bits0143/1

Problem Statement:
Count number of 1s in binary representation of a given number.

Sample Input/Output:
Input: 6
Output: 2

Input: 13
Output: 3

Input: 0
Output: 0

Input: 255
Output: 8
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Count_Set_Bits_Builtin(int n) {
        /*
        Built-in function
        Time Complexity: O(1)
        Space Complexity: O(1)
        */
        return __builtin_popcount(n);
    }

    int Count_Set_Bits_Brian_Kernighan(int n) {
        /*
        Brian Kernighan's algorithm
        Time Complexity: O(log n)
        Space Complexity: O(1)
        */
        int count = 0;
        while (n) {
            n &= (n - 1);
            count++;
        }
        return count;
    }

    int Count_Set_Bits_Loop(int n) {
        /*
        Check each bit
        Time Complexity: O(log n)
        Space Complexity: O(1)
        */
        int count = 0;
        while (n) {
            if (n & 1) count++;
            n >>= 1;
        }
        return count;
    }
};

void Test_Count_Set_Bits() {
    Solution solution;
    
    cout << "Testing Count_Set_Bits_Builtin:" << endl;
    cout << "6 -> " << solution.Count_Set_Bits_Builtin(6) << " (expected: 2)" << endl;
    cout << "13 -> " << solution.Count_Set_Bits_Builtin(13) << " (expected: 3)" << endl;
    cout << "0 -> " << solution.Count_Set_Bits_Builtin(0) << " (expected: 0)" << endl;
    cout << "255 -> " << solution.Count_Set_Bits_Builtin(255) << " (expected: 8)" << endl;
    
    cout << "\nTesting Count_Set_Bits_Brian_Kernighan:" << endl;
    cout << "6 -> " << solution.Count_Set_Bits_Brian_Kernighan(6) << " (expected: 2)" << endl;
    cout << "13 -> " << solution.Count_Set_Bits_Brian_Kernighan(13) << " (expected: 3)" << endl;
    cout << "0 -> " << solution.Count_Set_Bits_Brian_Kernighan(0) << " (expected: 0)" << endl;
    cout << "255 -> " << solution.Count_Set_Bits_Brian_Kernighan(255) << " (expected: 8)" << endl;
    
    cout << "\nTesting Count_Set_Bits_Loop:" << endl;
    cout << "6 -> " << solution.Count_Set_Bits_Loop(6) << " (expected: 2)" << endl;
    cout << "13 -> " << solution.Count_Set_Bits_Loop(13) << " (expected: 3)" << endl;
    cout << "0 -> " << solution.Count_Set_Bits_Loop(0) << " (expected: 0)" << endl;
    cout << "255 -> " << solution.Count_Set_Bits_Loop(255) << " (expected: 8)" << endl;
}

int main() {
    Test_Count_Set_Bits();
    return 0;
}
