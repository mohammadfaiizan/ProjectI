/*
Problem: Count Squares / Square Root of Integer
URL: https://practice.geeksforgeeks.org/problems/count-squares3649/1

Problem Statement:
Consider a sample space S consisting of all perfect squares starting from 1, 4, 9 and so on. You are given a number N, you have to output the number of integers less than N in the sample space S. This is equivalent to finding floor(sqrt(N-1)).

Sample Input/Output:
Input: N = 9
Output: 2

Input: N = 3
Output: 1
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Count_Squares_Math_Sqrt(int N) {
        /*
        Using built-in sqrt function
        Time Complexity: O(1)
        Space Complexity: O(1)
        */
        return (int)sqrt(N - 1);
    }

    int Count_Squares_Linear(int N) {
        /*
        Linear iteration checking perfect squares
        Time Complexity: O(sqrt(n))
        Space Complexity: O(1)
        */
        int count = 0;
        for (int i = 1; i * i < N; i++) {
            count++;
        }
        return count;
    }

    int Count_Squares_Binary_Search(int N) {
        /*
        Binary search to find floor of sqrt(N-1)
        Time Complexity: O(log n)
        Space Complexity: O(1)
        */
        if (N <= 1) return 0;
        
        int left = 1, right = N - 1;
        int result = 0;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (mid * mid < N) {
                result = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return result;
    }
};

void Test_Square_Root_Integer() {
    Solution sol;
    vector<int> tests = {9, 3, 1, 16, 25, 100};

    for (int N : tests) {
        cout << "N = " << N << endl;
        
        int res1 = sol.Count_Squares_Math_Sqrt(N);
        cout << "Math sqrt: " << res1 << endl;
        
        int res2 = sol.Count_Squares_Linear(N);
        cout << "Linear: " << res2 << endl;
        
        int res3 = sol.Count_Squares_Binary_Search(N);
        cout << "Binary Search: " << res3 << endl;
        
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Square_Root_Integer();
    return 0;
}
