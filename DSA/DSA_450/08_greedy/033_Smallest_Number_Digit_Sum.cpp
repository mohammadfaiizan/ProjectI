/*
Problem: Smallest Number With Given Digit Sum
URL: https://practice.geeksforgeeks.org/problems/smallest-number5829/1

Problem Statement:
Find smallest number with M digits and digit sum S.

Sample Input/Output:
Input: M=2, S=9
Output: 18
Explanation: Greedy fill from right with max digits approach.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string Smallest_Number(int M, int S) {
        /*
        Greedy fill from right with max digits approach
        Time Complexity: O(m)
        Space Complexity: O(m)
        */
        if (S == 0) {
            if (M == 1) return "0";
            return "-1";
        }
        
        if (S > 9 * M) {
            return "-1";
        }
        
        string result(M, '0');
        result[0] = '1';
        S -= 1;
        
        for (int i = M - 1; i >= 0; i--) {
            if (S >= 9) {
                result[i] = '9';
                S -= 9;
            } else {
                if (i == 0) {
                    result[i] = char('0' + S + 1);
                } else {
                    result[i] = char('0' + S);
                }
                S = 0;
            }
        }
        
        return result;
    }
};

void Test_Smallest_Number_Digit_Sum() {
    Solution solution;
    
    cout << "Test 1: " << solution.Smallest_Number(2, 9) << endl;
    cout << "Test 2: " << solution.Smallest_Number(3, 20) << endl;
    cout << "Test 3: " << solution.Smallest_Number(1, 9) << endl;
    cout << "Test 4: " << solution.Smallest_Number(2, 0) << endl;
}

int main() {
    Test_Smallest_Number_Digit_Sum();
    return 0;
}
