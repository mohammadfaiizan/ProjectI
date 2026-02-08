/*
Problem: Count Total Set Bits from 1 to N
URL: https://practice.geeksforgeeks.org/problems/count-total-set-bits-1587115620/1

Problem Statement:
Count the total number of set bits in all numbers from 1 to N.

Sample Input/Output:
Input: N=4
Output: 5

Input: N=17
Output: 35
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Total_Bits_Recursive(int N) {
        /*
        Recursive using power-of-2 pattern
        Time Complexity: O(log N)
        Space Complexity: O(log N)
        */
        if (N <= 0) return 0;
        if (N == 1) return 1;
        
        int x = 0;
        while ((1 << x) <= N) {
            x++;
        }
        x--;
        
        int bits_upto_2x = x * (1 << (x - 1));
        int msb_from_2x_to_N = N - (1 << x) + 1;
        int rest = N - (1 << x);
        
        return bits_upto_2x + msb_from_2x_to_N + Total_Bits_Recursive(rest);
    }

    int Total_Bits_Brute(int N) {
        /*
        Count each number
        Time Complexity: O(N log N)
        Space Complexity: O(1)
        */
        int total = 0;
        for (int i = 1; i <= N; i++) {
            int num = i;
            while (num) {
                num &= (num - 1);
                total++;
            }
        }
        return total;
    }
};

void Test_Total_Set_Bits_1_To_N() {
    Solution solution;
    
    cout << "Testing Total_Bits_Recursive:" << endl;
    cout << "N=4 -> " << solution.Total_Bits_Recursive(4) << " (expected: 5)" << endl;
    cout << "N=17 -> " << solution.Total_Bits_Recursive(17) << " (expected: 35)" << endl;
    
    cout << "\nTesting Total_Bits_Brute:" << endl;
    cout << "N=4 -> " << solution.Total_Bits_Brute(4) << " (expected: 5)" << endl;
    cout << "N=17 -> " << solution.Total_Bits_Brute(17) << " (expected: 35)" << endl;
}

int main() {
    Test_Total_Set_Bits_1_To_N();
    return 0;
}
