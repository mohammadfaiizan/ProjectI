/*
Problem: Middle of Three
URL: https://practice.geeksforgeeks.org/problems/middle-of-three2926/1

Problem Statement:
Given three distinct numbers A, B and C. Find the number with value in middle (Try to do it with minimum comparisons).

Sample Input/Output:
Input: A = 978, B = 518, C = 300
Output: 518

Input: A = 162, B = 934, C = 200
Output: 200
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Middle_Of_Three_Sum_Method(int A, int B, int C) {
        /*
        Using sum minus min minus max to find middle
        Time Complexity: O(1)
        Space Complexity: O(1)
        */
        return A + B + C - min({A, B, C}) - max({A, B, C});
    }

    int Middle_Of_Three_Comparisons(int A, int B, int C) {
        /*
        Using comparisons to find middle element
        Time Complexity: O(1)
        Space Complexity: O(1)
        */
        if ((A > B && A < C) || (A < B && A > C)) {
            return A;
        } else if ((B > A && B < C) || (B < A && B > C)) {
            return B;
        } else {
            return C;
        }
    }
};

void Test_Middle_Of_Three() {
    Solution sol;
    vector<vector<int>> tests = {
        {978, 518, 300},
        {162, 934, 200},
        {1, 2, 3},
        {10, 5, 8},
        {100, 50, 75}
    };

    for (auto& test : tests) {
        int A = test[0], B = test[1], C = test[2];
        cout << "A = " << A << ", B = " << B << ", C = " << C << endl;
        
        int res1 = sol.Middle_Of_Three_Sum_Method(A, B, C);
        cout << "Sum Method: " << res1 << endl;
        
        int res2 = sol.Middle_Of_Three_Comparisons(A, B, C);
        cout << "Comparisons: " << res2 << endl;
        
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Middle_Of_Three();
    return 0;
}
