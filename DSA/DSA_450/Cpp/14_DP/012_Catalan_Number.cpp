/*
Problem: Nth Catalan Number
URL: https://practice.geeksforgeeks.org/problems/nth-catalan-number0817/1

Problem Statement:
Catalan numbers are a sequence of natural numbers that occurs in many interesting counting problems. The first few Catalan numbers for n = 0, 1, 2, 3, … are 1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, …

Sample Input/Output:
Input: n = 5
Output: 42
Input: n = 10
Output: 16796
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Catalan_Number_Catalan_Recursive(int n) {
        /*
        Recursive approach
        Time Complexity: O(4^n/sqrt(n))
        Space Complexity: O(n)
        */
        if (n <= 1) return 1;
        long long res = 0;
        for (int i = 0; i < n; i++) {
            res += Catalan_Number_Catalan_Recursive(i) * Catalan_Number_Catalan_Recursive(n-1-i);
        }
        return res;
    }

    long long Catalan_Number_Catalan_DP(int n) {
        /*
        DP approach
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        vector<long long> dp(n+1, 0);
        dp[0] = dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                dp[i] += dp[j] * dp[i-1-j];
            }
        }
        return dp[n];
    }

    long long Catalan_Number_Catalan_Binomial(int n) {
        /*
        Binomial coefficient approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        long long res = 1;
        for (int i = 0; i < n; i++) {
            res = res * (2LL * n - i);
            res = res / (i + 1);
        }
        return res / (n + 1);
    }
};

void Test_Catalan_Number() {
    Solution solution;
    
    cout << "n=5 Recursive: " << solution.Catalan_Number_Catalan_Recursive(5) << endl;
    cout << "n=5 DP: " << solution.Catalan_Number_Catalan_DP(5) << endl;
    cout << "n=5 Binomial: " << solution.Catalan_Number_Catalan_Binomial(5) << endl;
    cout << "n=10 DP: " << solution.Catalan_Number_Catalan_DP(10) << endl;
    cout << "n=10 Binomial: " << solution.Catalan_Number_Catalan_Binomial(10) << endl;
}

int main() {
    Test_Catalan_Number();
    return 0;
}
