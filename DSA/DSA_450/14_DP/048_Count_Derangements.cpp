/*
Problem: Count Derangements
URL: https://www.geeksforgeeks.org/count-derangements-permutation-such-that-no-element-appears-in-its-original-position/

Problem Statement:
Count the number of derangements of n elements. A derangement is a permutation where no element appears in its original position. D(n) = (n-1) * (D(n-1) + D(n-2))

Sample Input/Output:
Input: n = 4
Output: 9
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Derange_DP(int n) {
        /*
        DP approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        if (n == 0 || n == 1) return 0;
        if (n == 2) return 1;
        
        vector<long long> dp(n + 1);
        dp[0] = 0;
        dp[1] = 0;
        dp[2] = 1;
        
        for (int i = 3; i <= n; i++) {
            dp[i] = (i - 1) * (dp[i - 1] + dp[i - 2]);
        }
        
        return dp[n];
    }
    
    long long Derange_Space(int n) {
        /*
        Space optimized approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (n == 0 || n == 1) return 0;
        if (n == 2) return 1;
        
        long long prev2 = 0;
        long long prev1 = 1;
        
        for (int i = 3; i <= n; i++) {
            long long current = (i - 1) * (prev1 + prev2);
            prev2 = prev1;
            prev1 = current;
        }
        
        return prev1;
    }
};

void Test_Derange() {
    Solution solution;
    
    int n = 4;
    cout << "DP: n=" << n << " -> " << solution.Derange_DP(n) << endl;
    cout << "Space Optimized: n=" << n << " -> " << solution.Derange_Space(n) << endl;
}

int main() {
    Test_Derange();
    return 0;
}
