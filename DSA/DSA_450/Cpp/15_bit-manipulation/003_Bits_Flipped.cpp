/*
Problem: Count Number of Bits to Flip to Convert A to B
URL: https://practice.geeksforgeeks.org/problems/bit-difference-1587115620/1

Problem Statement:
Count the number of bits that need to be flipped to convert number A to number B.

Sample Input/Output:
Input: A=10, B=20
Output: 4

Input: A=7, B=10
Output: 3
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Bits_Flipped_XOR_Count(int A, int B) {
        /*
        XOR then count set bits using Brian Kernighan
        Time Complexity: O(log n)
        Space Complexity: O(1)
        */
        int diff = A ^ B;
        int count = 0;
        while (diff) {
            diff &= (diff - 1);
            count++;
        }
        return count;
    }

    int Bits_Flipped_Loop(int A, int B) {
        /*
        Check each bit position
        Time Complexity: O(32)
        Space Complexity: O(1)
        */
        int count = 0;
        for (int i = 0; i < 32; i++) {
            if (((A >> i) & 1) != ((B >> i) & 1)) {
                count++;
            }
        }
        return count;
    }
};

void Test_Bits_Flipped() {
    Solution solution;
    
    cout << "Testing Bits_Flipped_XOR_Count:" << endl;
    cout << "A=10, B=20 -> " << solution.Bits_Flipped_XOR_Count(10, 20) << " (expected: 4)" << endl;
    cout << "A=7, B=10 -> " << solution.Bits_Flipped_XOR_Count(7, 10) << " (expected: 3)" << endl;
    
    cout << "\nTesting Bits_Flipped_Loop:" << endl;
    cout << "A=10, B=20 -> " << solution.Bits_Flipped_Loop(10, 20) << " (expected: 4)" << endl;
    cout << "A=7, B=10 -> " << solution.Bits_Flipped_Loop(7, 10) << " (expected: 3)" << endl;
}

int main() {
    Test_Bits_Flipped();
    return 0;
}
